//! Prompt prefill on the GPU (`kernels/prefill.metal`). The prompt goes
//! through the model one layer at a time, in chunks of tokens, so a layer's
//! weights are read from memory once per chunk and paged in from disk only
//! once per prompt. The states it leaves are the same as feeding the tokens
//! one by one.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::{Config, LayerType};
    use crate::cpu::attention::AttentionConfig;
    use crate::cpu::deltanet::DeltaNetConfig;
    use crate::cpu::model::{attention_config, deltanet_config};
    use crate::gpu::{Gpu, contents, dispatch_1d};
    use crate::gpu_layers::{GpuDeltaNetState, GpuKvCache, Norm};
    use crate::gpu_model::State;
    use crate::kv_quant::KvBits;
    use crate::moe::Experts;
    use crate::plan::TokenParams;
    use crate::qmv::{GpuQuant, View};
    use crate::sample_gpu::GpuSampler;
    use crate::weights::Weights;
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
    use std::cell::Cell;
    use std::time::Instant;

    const PREFIX: &str = "language_model.model";
    /// Rows and columns of a qmm tile; must match PREFILL_MM_BM and PREFILL_MM_BN.
    const TILE: usize = 64;
    /// Key block of attention_rows2 (attention_rows takes 32); the KV
    /// cache needs this much slack.
    const KEY_BLOCK: usize = 64;
    const _: () = assert!(KEY_BLOCK.is_multiple_of(crate::attention_prefill::KEY_BLOCK));

    /// The fast-mode attention: gemm (src/attention_gemm.rs, 9 TF/s at long
    /// context, as MLX) unless SAETTA_ATT=rows2, fa8 or fa4 (B's flash
    /// kernels), 1 or 8 (older ones).
    fn attention_half_kernel() -> &'static str {
        match std::env::var("SAETTA_ATT").as_deref() {
            Ok("1") => "attention_rows_half",
            Ok("8") => "attention_rows2_half8",
            Ok("rows2") => "attention_rows2_half",
            Ok("fa8") => "attention_fa8",
            Ok("fa4") => "attention_fa4",
            _ => "gemm",
        }
    }

    fn at(buffer: &Buffer, floats: usize) -> View {
        View {
            buffer: buffer.clone(),
            offset: (floats * 4) as u64,
        }
    }

    fn whole(buffer: &Buffer) -> View {
        View::new(buffer.clone())
    }

    fn params<T>(enc: &ComputeCommandEncoderRef, index: u64, p: &T) {
        enc.set_bytes(index, size_of::<T>() as u64, (p as *const T).cast());
    }

    fn groups(enc: &ComputeCommandEncoderRef, grid: (usize, usize), threads: usize) {
        enc.dispatch_thread_groups(
            MTLSize::new(grid.0 as u64, grid.1 as u64, 1),
            MTLSize::new(threads as u64, 1, 1),
        );
    }

    /// Binds `views` at consecutive buffer indices from `first`.
    fn set_views(enc: &ComputeCommandEncoderRef, first: u64, views: &[&View]) {
        for (i, v) in views.iter().enumerate() {
            enc.set_buffer(first + i as u64, Some(&v.buffer), v.offset);
        }
    }

    /// Binds the words, scales and biases of `w` at `first` to `first + 2`.
    fn set_quant(enc: &ComputeCommandEncoderRef, first: u64, w: &GpuQuant) {
        set_views(enc, first, &[&w.words, &w.scales, &w.biases]);
    }

    /// Must match NormParams in kernels/layers.metal.
    #[repr(C)]
    struct NormParams {
        dim: u32,
        in_stride: u32,
        out_stride: u32,
        eps: f32,
        scale: f32,
        weighted: u32,
        gated: u32,
    }

    /// `out = rms_norm(x)` as `n` says, like `Norm::record`.
    fn norm(gpu: &Gpu, enc: &ComputeCommandEncoderRef, n: &Norm, x: &View, out: &View) {
        enc.set_compute_pipeline_state(&gpu.pipeline("rms_norm"));
        set_views(
            enc,
            0,
            &[x, n.weight.unwrap_or(x), n.gate.unwrap_or(x), out],
        );
        let p = NormParams {
            dim: n.dim as u32,
            in_stride: n.in_stride as u32,
            out_stride: n.out_stride as u32,
            eps: n.eps,
            scale: n.scale,
            weighted: n.weight.is_some() as u32,
            gated: n.gate.is_some() as u32,
        };
        params(enc, 4, &p);
        let threads = n.dim.next_multiple_of(32).min(256);
        groups(enc, (n.rows, 1), threads);
    }

    /// y = W x for one vector, with qmv.metal (8 rows per threadgroup).
    fn qmv(gpu: &Gpu, enc: &ComputeCommandEncoderRef, w: &GpuQuant, x: &View, y: &View) {
        let name = match w.bits {
            5 => "qmv_5bit",
            8 => "qmv_8bit",
            b => panic!("no qmv kernel for {b} bits"),
        };
        enc.set_compute_pipeline_state(&gpu.pipeline(name));
        set_quant(enc, 0, w);
        set_views(enc, 3, &[x, y]);
        params(enc, 5, &[w.rows as u32, w.cols as u32]);
        groups(enc, (w.rows.div_ceil(8), 1), 64);
    }

    #[repr(C)]
    struct MmShape {
        m: u32,
        n: u32,
        k: u32,
        gather: u32,
    }

    /// The kernel for more than FEW_MAX rows: with `split`, qmm_<bits>bit_split
    /// (x split into two halves, close to f32), else qmm_dq_<bits>bit (w
    /// dequantized to half, as MLX).
    fn qmm_name(w: &GpuQuant, split: bool) -> String {
        assert!(matches!(w.bits, 5 | 8), "no qmm kernel for {} bits", w.bits);
        if split {
            format!("qmm_{}bit_split", w.bits)
        } else {
            format!("qmm_dq_{}bit", w.bits)
        }
    }

    fn qmm_pipeline(gpu: &Gpu, w: &GpuQuant, split: bool) -> metal::ComputePipelineState {
        gpu.pipeline(&qmm_name(w, split))
    }

    /// y = x W^T for `m` rows of x.
    fn qmm(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        w: &GpuQuant,
        x: &View,
        y: &View,
        m: usize,
        split: bool,
    ) {
        qmm_with(gpu, enc, w, x, y, m, &qmm_name(w, split));
    }

    /// qmm with the named large-m kernel.
    fn qmm_with(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        w: &GpuQuant,
        x: &View,
        y: &View,
        m: usize,
        kernel: &str,
    ) {
        assert_eq!(w.experts, 1);
        let shape = MmShape {
            m: m as u32,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: 0,
        };
        // The exact (split) kernels keep qmm_few's f32 up to FEW_MAX rows.
        let few = m <= FEW_MAX && (kernel.ends_with("_split") || m <= fast_few_max());
        if few {
            enc.set_compute_pipeline_state(&few_pipeline(gpu, w));
        } else {
            enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
        }
        set_quant(enc, 0, w);
        set_views(enc, 3, &[x, y, x, x, x]);
        params(enc, 5, &shape);
        if few {
            groups(enc, (w.rows.div_ceil(FEW_OUT), m.div_ceil(FEW)), 64);
        } else {
            groups(enc, (w.rows.div_ceil(TILE), m.div_ceil(TILE)), 128);
        }
    }

    /// Up to this many rows, qmm uses qmm_few: weights are read once per 8
    /// rows of x instead of computing a 64-row tile.
    const FEW_MAX: usize = 64;

    /// The FEW_MAX of the fast (qmm_dq) dense products: more rows go to the
    /// 64-row tiles, which read the weights once where qmm_few reads them
    /// once per 8 rows. bench-prefill (batch 6): qmm_few wins at 17 tokens
    /// (0.093 against 0.110 s), qmm_dq at 48 (0.142 against 0.186 s); the
    /// lines cross near 26. SAETTA_FEW_MAX=<n> sets it, at most FEW_MAX.
    const FAST_FEW_MAX: usize = 24;

    fn fast_few_max() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("SAETTA_FEW_MAX")
                .ok()
                .and_then(|v| v.parse().ok())
                .map_or(FAST_FEW_MAX, |n: usize| n.min(FEW_MAX))
        })
    }
    /// x rows per qmm_few threadgroup; must match PREFILL_FEW.
    const FEW: usize = 8;
    /// y columns per qmm_few threadgroup: 2 simdgroups of 4.
    const FEW_OUT: usize = 8;

    /// Rows per pass of encode_few_experts: the rows kernels take 8.
    const FEW_MOE_ROWS: usize = crate::rows::MAX_ROWS;
    /// Simdgroups of moe_gate_up_mma3 and moe_down_mma3, as rows::MOE_SPLIT.
    const FEW_MOE_SPLIT: usize = 8;

    /// Up to this many tokens a chunk's experts run on encode_few_experts:
    /// 32 (at 48 the grouped path is faster), or SAETTA_FEW_MOE=<n> (0 turns
    /// it off).
    fn few_moe_max() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("SAETTA_FEW_MOE")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(32)
        })
    }

    fn few_pipeline(gpu: &Gpu, w: &GpuQuant) -> metal::ComputePipelineState {
        match w.bits {
            5 => gpu.pipeline("qmm_few_5bit"),
            8 => gpu.pipeline("qmm_few_8bit"),
            b => panic!("no qmm_few kernel for {b} bits"),
        }
    }

    /// The (token, expert) pairs of a chunk sorted by expert, from moe_group.
    struct Grouped<'v> {
        tiles: &'v View,
        rows: &'v View,
        tile_count: &'v View,
        max_tiles: usize,
        /// Tokens of the chunk, which pick the tile size.
        tokens: usize,
    }

    #[allow(clippy::too_many_arguments)]
    /// qmm over the tiles of `g`. With `gather_x`, input row i is
    /// x[rows[i]]; without, x[i]. Output row i is y[i].
    fn qmm_grouped(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        w: &GpuQuant,
        x: &View,
        y: &View,
        g: &Grouped,
        gather_x: bool,
        split: bool,
    ) {
        enc.set_compute_pipeline_state(&qmm_pipeline(gpu, w, split));
        set_quant(enc, 0, w);
        set_views(enc, 3, &[x, y]);
        let shape = MmShape {
            m: 0,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: if gather_x { 1 } else { 2 },
        };
        params(enc, 5, &shape);
        set_views(enc, 6, &[g.tiles, g.rows, g.tile_count]);
        groups(enc, (w.rows.div_ceil(TILE), g.max_tiles), 128);
        // The tiles of FEW rows or fewer, which qmm skips.
        enc.set_compute_pipeline_state(&few_pipeline(gpu, w));
        groups(enc, (w.rows.div_ceil(FEW_OUT), g.max_tiles), 64);
    }

    /// Fast-mode experts run gate, up and silu in prefill_gate_up unless
    /// SAETTA_GATE_UP=0 or SAETTA_EXPERT_VARIANT picks other tile rows.
    fn fused_gate_up(t: usize) -> bool {
        std::env::var("SAETTA_GATE_UP").map_or(true, |v| v != "0") && expert_variant(t).rows == 32
    }

    /// Experts' silu(gate) * up into `g` (f32) and `g_half`: prefill_gate_up
    /// for the tiles of more than FEW rows, qmm_few for gate and up of the
    /// others and prefill_silu_few on those rows. `x` is the input, `x_half`
    /// its half, `u` scratch for up of the small tiles.
    #[allow(clippy::too_many_arguments)]
    fn gate_up_grouped(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        gate: &GpuQuant,
        up: &GpuQuant,
        x: &View,
        x_half: &View,
        g_out: &View,
        u: &View,
        g_half: &View,
        g: &Grouped,
        rows64: bool,
    ) {
        let shape = MmShape {
            m: 0,
            n: gate.rows as u32,
            k: gate.cols as u32,
            gather: 1,
        };
        let threads = bind_gate_up(gpu, enc, gate, up, x_half, g_out, g_half, &shape, rows64);
        set_views(enc, 10, &[g.tiles, g.rows, g.tile_count]);
        groups(enc, (gate.rows.div_ceil(64), g.max_tiles), threads);

        few_grouped(gpu, enc, gate, x, g_out, g, true);
        few_grouped(gpu, enc, up, x, u, g, true);
        enc.set_compute_pipeline_state(&gpu.pipeline("prefill_silu_few"));
        set_views(enc, 0, &[g_out, u, g_half]);
        params(enc, 3, &(gate.rows as u32));
        set_views(enc, 4, &[g.tiles, g.tile_count]);
        groups(enc, (g.max_tiles, 1), 256);
    }

    /// Binds prefill_gate_up (32-row tiles) or, with `rows64`,
    /// prefill_gate_up64 and their buffers up to the shape; returns the
    /// threads per threadgroup.
    #[allow(clippy::too_many_arguments)]
    fn bind_gate_up(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        gate: &GpuQuant,
        up: &GpuQuant,
        x_half: &View,
        g_out: &View,
        g_half: &View,
        shape: &MmShape,
        rows64: bool,
    ) -> usize {
        assert!(
            matches!(gate.bits, 5 | 8),
            "no prefill_gate_up kernel for {} bits",
            gate.bits
        );
        let (name, threads) = if rows64 {
            ("prefill_gate_up64", 256)
        } else {
            ("prefill_gate_up", 128)
        };
        enc.set_compute_pipeline_state(&gpu.pipeline(&format!("{name}_{}bit", gate.bits)));
        set_quant(enc, 0, gate);
        set_quant(enc, 3, up);
        set_views(enc, 6, &[x_half, g_out, g_half]);
        params(enc, 9, shape);
        threads
    }

    /// g = silu(x gate^T) * (x up^T) for `m` rows of half x, in f32 and in
    /// half, with prefill_gate_up64 in plain mode.
    #[allow(clippy::too_many_arguments)]
    fn gate_up_plain(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        gate: &GpuQuant,
        up: &GpuQuant,
        x_half: &View,
        g_out: &View,
        g_half: &View,
        m: usize,
    ) {
        let shape = MmShape {
            m: m as u32,
            n: gate.rows as u32,
            k: gate.cols as u32,
            gather: 0,
        };
        let threads = bind_gate_up(gpu, enc, gate, up, x_half, g_out, g_half, &shape, true);
        // Unused in plain mode, but bound so the kernel sees valid buffers.
        set_views(enc, 10, &[x_half, x_half, x_half]);
        groups(enc, (gate.rows.div_ceil(64), m.div_ceil(64)), threads);
    }

    /// qmm_few on the tiles of FEW rows or fewer of a grouped product.
    fn few_grouped(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        w: &GpuQuant,
        x: &View,
        y: &View,
        g: &Grouped,
        gather_x: bool,
    ) {
        enc.set_compute_pipeline_state(&few_pipeline(gpu, w));
        set_quant(enc, 0, w);
        set_views(enc, 3, &[x, y]);
        let shape = MmShape {
            m: 0,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: if gather_x { 1 } else { 2 },
        };
        params(enc, 5, &shape);
        set_views(enc, 6, &[g.tiles, g.rows, g.tile_count]);
        groups(enc, (w.rows.div_ceil(FEW_OUT), g.max_tiles), 64);
    }

    /// qmm_grouped on qmm2 (src/qmm.rs), which takes `x_half`, the half of
    /// `x`; the tiles of FEW rows or fewer still go to qmm_few on `x`.
    /// The qmm2 variant for the experts: SAETTA_EXPERT_VARIANT=<index into
    /// qmm::VARIANTS> (0 to 3) to compare, else HALVES for the 64-row tiles
    /// of a chunk of `t` tokens or DEFAULT.
    fn expert_variant(t: usize) -> crate::qmm::Variant {
        std::env::var("SAETTA_EXPERT_VARIANT")
            .ok()
            .and_then(|v| v.parse::<usize>().ok())
            .filter(|&i| i < 4)
            .map_or(
                if gate_up64(t) {
                    // SAETTA_DOWN64=1: down on qmm2 64 x 64 instead (slower
                    // in batch 6: 201 against 174 ms per 2k chunk).
                    if std::env::var("SAETTA_DOWN64").is_ok_and(|v| v == "1") {
                        crate::qmm::VARIANTS[0]
                    } else {
                        crate::qmm::HALVES
                    }
                } else {
                    crate::qmm::DEFAULT
                },
                |i| crate::qmm::VARIANTS[i],
            )
    }

    #[allow(clippy::too_many_arguments)]
    fn qmm2_grouped(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        w: &GpuQuant,
        x: &View,
        x_half: &View,
        y: &View,
        g: &Grouped,
        gather_x: bool,
    ) {
        assert_eq!(crate::qmm::FEW, FEW);
        crate::qmm::encode_gather(
            enc,
            gpu,
            expert_variant(g.tokens),
            w,
            x_half,
            y,
            g.tiles,
            g.rows,
            g.tile_count,
            g.max_tiles,
            gather_x,
        );
        few_grouped(gpu, enc, w, x, y, g, gather_x);
    }

    /// Fast-mode dense matrices run on qmm_dq (w rounded to half), 6% faster
    /// than qmm2 at 1800 tokens; SAETTA_QMM2=1 picks qmm2 (x in half).
    fn dense_qmm2() -> bool {
        std::env::var("SAETTA_QMM2").is_ok_and(|v| v == "1")
    }

    /// Expert tiles of up to 64 rows, gate and up in prefill_gate_up64,
    /// down on qmm2 32 x 128 over each half of a tile (3-4% faster at 1k
    /// to 15k tokens than 32-row tiles, batch 6), unless SAETTA_GATE_UP64=0
    /// or SAETTA_EXPERT_VARIANT picks a variant. Chunks of fewer than
    /// gate_up64_min() tokens keep the 32-row tiles.
    fn gate_up64(t: usize) -> bool {
        std::env::var("SAETTA_GATE_UP64").map_or(true, |v| v != "0")
            && std::env::var("SAETTA_EXPERT_VARIANT").is_err()
            && t >= gate_up64_min()
    }

    /// Fewest tokens of a chunk for the 64-row expert tiles: 0, or
    /// SAETTA_GU64_MIN=<n>.
    fn gate_up64_min() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("SAETTA_GU64_MIN")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        })
    }

    /// The shared expert's gate, up and silu in one prefill_gate_up64 pass
    /// on normed in half, unless SAETTA_SHARED_GU=0.
    fn shared_gate_up() -> bool {
        std::env::var("SAETTA_SHARED_GU").map_or(true, |v| v != "0")
    }

    /// The fast path runs the router on the fast kernels too (about 70 ms
    /// less per 2k chunk than qmm_split), unless SAETTA_ROUTER_FAST=0.
    fn fast_router() -> bool {
        std::env::var("SAETTA_ROUTER_FAST").map_or(true, |v| v != "0")
    }

    /// SAETTA_CONV=1 runs the DeltaNet conv on conv_rows, one thread per
    /// channel, instead of conv_rows2 and conv_state, to compare.
    fn old_conv() -> bool {
        std::env::var("SAETTA_CONV").is_ok_and(|v| v == "1")
    }

    /// SAETTA_DELTA2=1 runs the delta rule on delta_prep and delta_rows2
    /// instead of delta_rows: measured slower so far.
    fn delta2() -> bool {
        std::env::var("SAETTA_DELTA2").is_ok_and(|v| v == "1")
    }

    /// The delta rule runs on delta_prep plus delta_rows3 with this many
    /// simdgroups (8 state rows each) per threadgroup: 8 (24% faster than
    /// delta_rows in batch 10), or SAETTA_DELTA3=<n>, 2 to 16; 0 runs
    /// delta_rows.
    fn delta3() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("SAETTA_DELTA3")
                .ok()
                .and_then(|v| v.parse().ok())
                .filter(|&n| n == 0 || (2..=16).contains(&n))
                .unwrap_or(8)
        })
    }

    fn vector(gpu: &Gpu, weights: &Weights, name: &str) -> Result<View, String> {
        Ok(View::new(gpu.buffer_from(&weights.tensor(name)?.to_f32())))
    }

    struct DeltaNetWeights<'a> {
        qkv: GpuQuant<'a>,
        z: GpuQuant<'a>,
        b: GpuQuant<'a>,
        a: GpuQuant<'a>,
        out: GpuQuant<'a>,
        conv1d: View,
        a_log: View,
        dt_bias: View,
        norm: View,
    }

    struct AttentionWeights<'a> {
        q: GpuQuant<'a>,
        k: GpuQuant<'a>,
        v: GpuQuant<'a>,
        o: GpuQuant<'a>,
        q_norm: View,
        k_norm: View,
    }

    enum Mixer<'a> {
        Linear(DeltaNetWeights<'a>),
        Full(AttentionWeights<'a>),
    }

    struct Layer<'a> {
        input_norm: View,
        post_norm: View,
        mixer: Mixer<'a>,
        router: GpuQuant<'a>,
        experts: Experts<'a>,
        shared: [GpuQuant<'a>; 3],
        shared_gate: GpuQuant<'a>,
    }

    /// Encodes y[i] = half(x[i]) for i < n.
    fn to_half(gpu: &Gpu, enc: &ComputeCommandEncoderRef, x: &View, y: &View, n: usize) {
        if n > 0 {
            dispatch_1d(
                enc,
                &gpu.pipeline("to_half"),
                &[(&x.buffer, x.offset), (&y.buffer, y.offset)],
                n,
            );
        }
    }

    fn at_half(buffer: &Buffer, halves: usize) -> View {
        View {
            buffer: buffer.clone(),
            offset: (halves * 2) as u64,
        }
    }

    /// Scratch for one chunk of up to `chunk` tokens.
    struct Scratch {
        normed: Buffer,
        /// in_proj_qkv or q_proj output, `[chunk, 8192]`
        wide: Buffer,
        /// conv output, `[chunk, 8192]`, or the new keys and values in f32
        mixed: Buffer,
        /// in_proj_z output, or q after norm and rope
        z: Buffer,
        /// delta rule or attention output, before out_proj
        y: Buffer,
        a: Buffer,
        b: Buffer,
        mixer_out: Buffer,
        /// q after norm and rope, in half
        q_half: Buffer,
        router: Buffer,
        shared_gate: Buffer,
        shared_g: Buffer,
        shared_u: Buffer,
        shared_out: Buffer,
        ids: Buffer,
        expert_weights: Buffer,
        order: Buffer,
        rows: Buffer,
        tiles: Buffer,
        tile_count: Buffer,
        expert_g: Buffer,
        expert_u: Buffer,
        expert_out: Buffer,
        /// normed and expert_g in half, the input of qmm2 in fast mode
        normed_half: Buffer,
        expert_half: Buffer,
        /// the x of a dense qmm2 in fast mode
        x_half: Buffer,
        /// moe_union's distinct experts and slots for encode_few_experts
        few_list: Buffer,
        few_slots: Buffer,
        max_tiles: usize,
    }

    impl Scratch {
        fn new(gpu: &Gpu, cfg: &Config, chunk: usize) -> Scratch {
            let f = |n: usize| gpu.buffer(n.max(1) * 4);
            let h = cfg.hidden_size;
            let a = attention_config(cfg);
            let d = deltanet_config(cfg);
            let wide =
                (2 * d.k_heads * d.k_dim + d.v_heads * d.v_dim).max(2 * a.heads * a.head_dim);
            let narrow = (d.v_heads * d.v_dim).max(a.heads * a.head_dim);
            let pairs = chunk * cfg.num_experts_per_tok;
            let size = cfg.moe_intermediate_size;
            let max_tiles = pairs.div_ceil(TILE.min(crate::qmm::tile_rows())) + cfg.num_experts;
            Scratch {
                normed: f(chunk * h),
                wide: f(chunk * wide),
                mixed: f(chunk * wide),
                z: f(chunk * narrow),
                y: f(chunk * narrow),
                a: f(chunk * d.v_heads),
                b: f(chunk * d.v_heads),
                mixer_out: f(chunk * h),
                q_half: f(chunk * narrow / 2),
                router: f(chunk * cfg.num_experts),
                shared_gate: f(chunk),
                shared_g: f(chunk * cfg.shared_expert_intermediate_size),
                shared_u: f(chunk * cfg.shared_expert_intermediate_size),
                shared_out: f(chunk * h),
                ids: f(pairs),
                expert_weights: f(pairs),
                order: f(pairs),
                rows: f(pairs),
                tiles: f(4 * max_tiles),
                tile_count: f(1),
                expert_g: f(pairs * size),
                expert_u: f(pairs * size),
                expert_out: f(pairs * h),
                normed_half: f(chunk * h / 2),
                x_half: f(chunk * h.max(narrow) / 2),
                expert_half: f(pairs * size / 2),
                few_list: f(1 + FEW_MOE_ROWS * cfg.num_experts_per_tok),
                few_slots: f(FEW_MOE_ROWS * FEW_MOE_ROWS * cfg.num_experts_per_tok),
                max_tiles,
            }
        }
    }

    #[repr(C)]
    struct EmbedParams {
        cols: u32,
    }

    #[repr(C)]
    struct RopeRowsParams {
        stride: u32,
        half_dims: u32,
        pos: u32,
        log2_theta: f32,
        heads: u32,
    }

    #[repr(C)]
    struct ConvRowsParams {
        channels: u32,
        kernel_size: u32,
        tokens: u32,
        snapshots: u32,
    }

    #[repr(C)]
    struct DeltaRowsParams {
        k_heads: u32,
        v_heads: u32,
        v_dim: u32,
        tokens: u32,
        channels: u32,
        eps: f32,
        snapshots: u32,
    }

    #[repr(C)]
    struct AttentionParams {
        heads: u32,
        kv_heads: u32,
        pos: u32,
        tokens: u32,
        scale: f32,
    }

    #[repr(C)]
    struct RouteParams {
        experts: u32,
        top_k: u32,
    }

    #[repr(C)]
    struct GroupParams {
        pairs: u32,
        top_k: u32,
        experts: u32,
        tile_rows: u32,
    }

    #[repr(C)]
    struct CombineParams {
        dim: u32,
        top_k: u32,
    }

    pub struct Prefill<'a> {
        cfg: &'a Config,
        attention: AttentionConfig,
        deltanet: DeltaNetConfig,
        embed: GpuQuant<'a>,
        layers: Vec<Layer<'a>>,
        norm: View,
        lm_head: GpuQuant<'a>,
        chunk: usize,
        scratch: Scratch,
        /// Use the split qmm kernels and f32 attention: within f32 rounding
        /// of decode. With false, w and x in half, as MLX: the matmuls are
        /// about 6 times faster, 1e-3 off. serve runs with false.
        pub exact: bool,
        /// Largest |x| seen at each of `RANGES`, when tracked.
        ranges: Option<Buffer>,
        /// Buffers of `verify`, once `enable_verify` made them.
        verify_rows: Option<VerifyRows>,
        /// Set while `verify` encodes, so the DeltaNet layers write snapshots.
        snapshot_run: Cell<bool>,
        /// Set while `verify` encodes: its rows run on the exact kernels
        /// whatever `exact` says, so they match decode.
        force_exact: Cell<bool>,
        /// The buffers of the matrix-product attention, with SAETTA_ATT=gemm.
        gemm: Option<crate::attention_gemm::AttGemm>,
        /// Set by `profile`: only the stages up to this index in STAGES run.
        only: Cell<Option<usize>>,
    }

    /// Per-row outputs of `verify`: the DeltaNet state after each row, one
    /// buffer per DeltaNet layer laid out as `rows` copies of
    /// GpuDeltaNetState's buffer, and the final-norm hidden states, logits
    /// and argmax of each row.
    struct VerifyRows {
        rows: usize,
        conv: Vec<Buffer>,
        recurrent: Vec<Buffer>,
        hidden: Buffer,
        logits: Buffer,
        argmax: Buffer,
    }

    /// What `verify` leaves on the GPU for its n rows.
    pub struct Verify {
        /// `[n, hidden]` after the final norm.
        pub hidden: View,
        /// `[n, vocab]`
        pub logits: View,
        /// The best token of each row, lowest id on ties.
        pub argmax: Vec<u32>,
        hidden_size: usize,
    }

    impl Verify {
        /// Row `i` of `hidden`.
        pub fn hidden_row(&self, i: usize) -> View {
            View {
                buffer: self.hidden.buffer.clone(),
                offset: self.hidden.offset + (i * self.hidden_size * 4) as u64,
            }
        }
    }

    /// The stages `profile` times, in the order they run.
    pub const STAGES: [&str; 18] = [
        "norm",
        "deltanet in_proj",
        "deltanet conv",
        "deltanet delta",
        "deltanet norm",
        "deltanet out_proj",
        "attention q/k/v",
        "attention norm, rope",
        "attention",
        "attention o_proj",
        "add, post norm",
        "router",
        "shared expert",
        "route, group",
        "experts gate, up",
        "experts silu",
        "experts down",
        "combine",
    ];

    /// The inputs whose range `track_ranges` records: half tops out at 65504.
    pub const RANGES: [&str; 5] = [
        "qmm input after RMSNorm",
        "DeltaNet out_proj input",
        "attention o_proj input",
        "shared expert down_proj input",
        "expert down_proj input",
    ];

    impl<'a> Prefill<'a> {
        /// `chunk` is the number of tokens per dispatch.
        pub fn new(
            gpu: &Gpu,
            cfg: &'a Config,
            weights: &'a Weights,
            chunk: usize,
        ) -> Result<Prefill<'a>, String> {
            let attention = attention_config(cfg);
            let deltanet = deltanet_config(cfg);
            if attention.head_dim != 256 || deltanet.k_dim != 128 || cfg.num_experts > 256 {
                return Err(
                    "prefill kernels need head_dim 256, k_dim 128, at most 256 experts".into(),
                );
            }
            let matrix = |m: &str| -> Result<GpuQuant<'a>, String> {
                Ok(GpuQuant::new(
                    gpu,
                    &weights.quantized(m, &cfg.quantization)?,
                ))
            };
            let mut layers = Vec::new();
            for (i, kind) in cfg.layer_types.iter().enumerate() {
                let p = format!("{PREFIX}.layers.{i}");
                let mixer = match kind {
                    LayerType::LinearAttention => {
                        let l = format!("{p}.linear_attn");
                        Mixer::Linear(DeltaNetWeights {
                            qkv: matrix(&format!("{l}.in_proj_qkv"))?,
                            z: matrix(&format!("{l}.in_proj_z"))?,
                            b: matrix(&format!("{l}.in_proj_b"))?,
                            a: matrix(&format!("{l}.in_proj_a"))?,
                            out: matrix(&format!("{l}.out_proj"))?,
                            conv1d: vector(gpu, weights, &format!("{l}.conv1d.weight"))?,
                            a_log: vector(gpu, weights, &format!("{l}.A_log"))?,
                            dt_bias: vector(gpu, weights, &format!("{l}.dt_bias"))?,
                            norm: vector(gpu, weights, &format!("{l}.norm.weight"))?,
                        })
                    }
                    LayerType::FullAttention => {
                        let s = format!("{p}.self_attn");
                        Mixer::Full(AttentionWeights {
                            q: matrix(&format!("{s}.q_proj"))?,
                            k: matrix(&format!("{s}.k_proj"))?,
                            v: matrix(&format!("{s}.v_proj"))?,
                            o: matrix(&format!("{s}.o_proj"))?,
                            q_norm: vector(gpu, weights, &format!("{s}.q_norm.weight"))?,
                            k_norm: vector(gpu, weights, &format!("{s}.k_norm.weight"))?,
                        })
                    }
                };
                let shared = |n: &str| matrix(&format!("{p}.mlp.shared_expert.{n}"));
                layers.push(Layer {
                    input_norm: vector(gpu, weights, &format!("{p}.input_layernorm.weight"))?,
                    post_norm: vector(
                        gpu,
                        weights,
                        &format!("{p}.post_attention_layernorm.weight"),
                    )?,
                    mixer,
                    router: matrix(&format!("{p}.mlp.gate"))?,
                    experts: Experts::new(gpu, weights, &cfg.quantization, &p)?,
                    shared: [
                        shared("gate_proj")?,
                        shared("up_proj")?,
                        shared("down_proj")?,
                    ],
                    shared_gate: matrix(&format!("{p}.mlp.shared_expert_gate"))?,
                });
            }
            let chunk = chunk.next_multiple_of(TILE);
            let gemm = (attention_half_kernel() == "gemm")
                .then(|| crate::attention_gemm::AttGemm::new(gpu, chunk, attention.kv_heads));
            Ok(Prefill {
                gemm,
                verify_rows: None,
                snapshot_run: Cell::new(false),
                only: Cell::new(None),
                force_exact: Cell::new(false),
                cfg,
                attention,
                deltanet,
                embed: matrix(&format!("{PREFIX}.embed_tokens"))?,
                layers,
                norm: vector(gpu, weights, &format!("{PREFIX}.norm.weight"))?,
                lm_head: matrix("language_model.lm_head")?,
                scratch: Scratch::new(gpu, cfg, chunk),
                chunk,
                exact: true,
                ranges: None,
            })
        }

        /// Starts recording the largest |x| at each of `RANGES`.
        pub fn track_ranges(&mut self, gpu: &Gpu) {
            self.ranges = Some(gpu.buffer(RANGES.len() * 4));
        }

        /// The largest |x| at each of `RANGES` so far.
        pub fn ranges(&self) -> Vec<f32> {
            self.ranges
                .as_ref()
                .map(|b| contents::<f32>(b, RANGES.len()).to_vec())
                .unwrap_or_default()
        }

        fn record(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            slot: usize,
            x: &View,
            n: usize,
        ) {
            let Some(ranges) = &self.ranges else { return };
            enc.set_compute_pipeline_state(&gpu.pipeline("max_abs"));
            set_views(enc, 0, &[x, &whole(ranges)]);
            params(enc, 2, &(slot as u32));
            // n is a multiple of 32 everywhere it is called.
            enc.dispatch_threads(MTLSize::new(n as u64, 1, 1), MTLSize::new(256, 1, 1));
        }

        /// Feeds `tokens` at positions `pos ..` into `states`, which hold
        /// the tokens before, and returns the logits of the last token. One
        /// command buffer runs the whole prompt.
        pub fn run(&self, gpu: &Gpu, tokens: &[u32], pos: usize, states: &[State]) -> Vec<f32> {
            self.run_hidden(gpu, tokens, pos, states).0
        }

        /// `run`, which also returns the hidden states `[n, hidden]` after
        /// the last layer, before the final norm.
        pub fn run_hidden(
            &self,
            gpu: &Gpu,
            tokens: &[u32],
            pos: usize,
            states: &[State],
        ) -> (Vec<f32>, Buffer) {
            let n = tokens.len();
            assert!(n > 0);
            let x = gpu.buffer(n * self.cfg.hidden_size * 4);
            let ids = gpu.buffer_from(tokens);
            let logits = gpu.buffer(self.lm_head.rows * 4);
            gpu.run(|enc| {
                self.encode_embed(gpu, enc, &ids, &x, n);
                for (l, state) in states.iter().enumerate() {
                    self.encode_layer(gpu, enc, l, &x, n, pos, state);
                }
                self.encode_logits(gpu, enc, &x, n, &logits);
            });
            (contents::<f32>(&logits, self.lm_head.rows).to_vec(), x)
        }

        /// The hidden states `[tokens, hidden]` after the embedding.
        pub fn embed(&self, gpu: &Gpu, tokens: &[u32]) -> Buffer {
            let x = gpu.buffer(tokens.len() * self.cfg.hidden_size * 4);
            let ids = gpu.buffer_from(tokens);
            gpu.run(|enc| self.encode_embed(gpu, enc, &ids, &x, tokens.len()));
            x
        }

        fn encode_embed(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            ids: &Buffer,
            x: &Buffer,
            n: usize,
        ) {
            let h = self.cfg.hidden_size;
            let name = match self.embed.bits {
                5 => "embed_rows_5bit",
                8 => "embed_rows_8bit",
                b => panic!("no embed kernel for {b} bits"),
            };
            enc.set_compute_pipeline_state(&gpu.pipeline(name));
            set_quant(enc, 0, &self.embed);
            set_views(enc, 3, &[&whole(ids), &whole(x)]);
            params(enc, 5, &EmbedParams { cols: h as u32 });
            enc.dispatch_threads(MTLSize::new(h as u64, n as u64, 1), MTLSize::new(256, 1, 1));
        }

        /// Runs layer `l` over the `n` hidden states in `x`, in place, in
        /// one command buffer. The first is at position `pos`.
        pub fn layer(&self, gpu: &Gpu, l: usize, x: &Buffer, n: usize, pos: usize, state: &State) {
            gpu.run(|enc| self.encode_layer(gpu, enc, l, x, n, pos, state));
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_layer(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            l: usize,
            x: &Buffer,
            n: usize,
            pos: usize,
            state: &State,
        ) {
            if let State::Full(cache) = state {
                assert!(pos + n <= cache.capacity, "KV cache is full");
                // attention_rows reads keys in blocks of KEY_BLOCK.
                assert!(cache.capacity.is_multiple_of(KEY_BLOCK));
            }
            for start in (0..n).step_by(self.chunk) {
                let t = self.chunk.min(n - start);
                self.encode_chunk(gpu, enc, l, x, start, t, pos + start, state);
            }
        }

        /// y = x W^T for `m` rows of x, with the kernels `exact` picks: in
        /// fast mode, more than FEW_MAX rows go to qmm2 on x in half.
        #[allow(clippy::too_many_arguments)]
        fn mm(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &GpuQuant,
            x: &View,
            y: &View,
            m: usize,
        ) {
            if self.exact_now() || m <= FEW_MAX || !dense_qmm2() {
                return qmm(gpu, enc, w, x, y, m, self.exact_now());
            }
            let x_half = whole(&self.scratch.x_half);
            to_half(gpu, enc, x, &x_half, m * w.cols);
            crate::qmm::encode(enc, gpu, crate::qmm::DEFAULT, w, &x_half, y, m);
        }

        /// `exact`, or true while `verify` runs.
        fn exact_now(&self) -> bool {
            self.exact || self.force_exact.get()
        }

        /// Whether the dispatches of `stage` run: all do unless `profile`
        /// picked one.
        fn on(&self, stage: &str) -> bool {
            self.only.get().is_none_or(|last| {
                STAGES
                    .iter()
                    .position(|&s| s == stage)
                    .expect("a stage in STAGES")
                    <= last
            })
        }

        /// Runs `tokens` from position `pos` as `run` does, and returns the GPU
        /// time of each of `STAGES`, summed over the layers, then the time
        /// of whole layers and of the logits. Each layer runs once for real,
        /// then once for every stage k with only stages 0 ..= k, from the
        /// same input; stage k gets the difference to the run up to k - 1.
        /// So every stage sees the data the earlier ones made for its chunk.
        pub fn profile(
            &self,
            gpu: &Gpu,
            tokens: &[u32],
            pos: usize,
            states: &[State],
        ) -> Vec<(&'static str, f64)> {
            let n = tokens.len();
            let x = self.embed(gpu, tokens);
            let bytes = n * self.cfg.hidden_size * 4;
            let ptr = x.contents() as *mut u8;
            let (mut input, mut output) = (vec![0u8; bytes], vec![0u8; bytes]);
            let mut times: Vec<(&'static str, f64)> = STAGES.iter().map(|&s| (s, 0.0)).collect();
            let mut layers = 0.0;
            let serial = metal::MTLDispatchType::Serial;
            for (l, state) in states.iter().enumerate() {
                unsafe { std::ptr::copy_nonoverlapping(ptr, input.as_mut_ptr(), bytes) };
                let run = || {
                    gpu.run_timed(serial, |enc| {
                        self.encode_layer(gpu, enc, l, &x, n, pos, state)
                    })
                    .1
                };
                layers += run();
                unsafe { std::ptr::copy_nonoverlapping(ptr, output.as_mut_ptr(), bytes) };
                let mut before = 0.0;
                for (k, (_, total)) in times.iter_mut().enumerate() {
                    unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, bytes) };
                    self.only.set(Some(k));
                    let upto = run();
                    *total += upto - before;
                    before = upto;
                }
                self.only.set(None);
                unsafe { std::ptr::copy_nonoverlapping(output.as_ptr(), ptr, bytes) };
            }
            let logits = gpu.buffer(self.lm_head.rows * 4);
            let head = gpu
                .run_timed(metal::MTLDispatchType::Serial, |enc| {
                    self.encode_logits(gpu, enc, &x, n, &logits)
                })
                .1;
            times.push(("whole layers", layers));
            times.push(("final norm, lm_head", head));
            times
        }

        /// Which DeltaNet layer `l` is.
        fn linear_index(&self, l: usize) -> usize {
            self.layers[..l]
                .iter()
                .filter(|l| matches!(l.mixer, Mixer::Linear(_)))
                .count()
        }

        /// Makes the buffers `verify` needs for up to `rows` tokens: about
        /// 2 MB per row per DeltaNet layer.
        pub fn enable_verify(&mut self, gpu: &Gpu, rows: usize) {
            assert!(rows <= self.chunk);
            let c = &self.deltanet;
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let conv = (c.kernel - 1) * channels * 4;
            let recurrent = c.v_heads * c.v_dim * c.k_dim * 4;
            let linear = self
                .layers
                .iter()
                .filter(|l| matches!(l.mixer, Mixer::Linear(_)))
                .count();
            self.verify_rows = Some(VerifyRows {
                rows,
                conv: (0..linear).map(|_| gpu.buffer(rows * conv)).collect(),
                recurrent: (0..linear).map(|_| gpu.buffer(rows * recurrent)).collect(),
                hidden: gpu.buffer(rows * self.cfg.hidden_size * 4),
                logits: gpu.buffer(rows * self.lm_head.rows * 4),
                argmax: gpu.buffer(rows * 4),
            });
        }

        /// Runs `tokens` at positions `pos ..` as `run` does, and keeps for
        /// every row the final-norm hidden state, the logits, the argmax and
        /// the DeltaNet state after it, for speculative decoding. Runs on the
        /// exact kernels even when `exact` is false, so rows match decode.
        /// Needs `enable_verify`.
        /// With `sampler`, also samples every row into its ring slot
        /// (row i at position pos + i fills slot pos + i + 1), against the
        /// drafts set in the ring.
        pub fn verify(
            &self,
            gpu: &Gpu,
            tokens: &[u32],
            pos: usize,
            states: &[State],
            sampler: Option<&GpuSampler>,
        ) -> Verify {
            let v = self.verify_rows.as_ref().expect("enable_verify first");
            let n = tokens.len();
            assert!(n > 0 && n <= v.rows, "verify takes 1 to {} tokens", v.rows);
            let h = self.cfg.hidden_size;
            let x = gpu.buffer(n * h * 4);
            let ids = gpu.buffer_from(tokens);
            let hidden = whole(&v.hidden);
            let logits = whole(&v.logits);
            self.snapshot_run.set(true);
            self.force_exact.set(true);
            gpu.run(|enc| {
                self.encode_embed(gpu, enc, &ids, &x, n);
                for (l, state) in states.iter().enumerate() {
                    self.encode_layer(gpu, enc, l, &x, n, pos, state);
                }
                norm(
                    gpu,
                    enc,
                    &Norm::rows(n, h, self.cfg.rms_norm_eps, &self.norm),
                    &whole(&x),
                    &hidden,
                );
                qmm(gpu, enc, &self.lm_head, &hidden, &logits, n, true);
                enc.set_compute_pipeline_state(&gpu.pipeline("argmax_rows"));
                set_views(enc, 0, &[&logits, &whole(&v.argmax)]);
                params(enc, 2, &(self.lm_head.rows as u32));
                groups(enc, (n, 1), 1024);
                if let Some(s) = sampler {
                    let vocab = self.lm_head.rows;
                    let rows: Vec<TokenParams> = (0..n)
                        .map(|i| TokenParams::new(0, (pos + i) as u32))
                        .collect();
                    let tp = gpu.buffer_from(&rows);
                    for i in 0..n {
                        let row = View {
                            buffer: logits.buffer.clone(),
                            offset: logits.offset + (i * vocab * 4) as u64,
                        };
                        let params = View {
                            buffer: tp.clone(),
                            offset: (i * size_of::<TokenParams>()) as u64,
                        };
                        s.encode(gpu, enc, &row, &params);
                    }
                }
            });
            self.snapshot_run.set(false);
            self.force_exact.set(false);
            Verify {
                hidden,
                logits,
                argmax: contents::<u32>(&v.argmax, n).to_vec(),
                hidden_size: h,
            }
        }

        /// Copies the DeltaNet states after row `row` of the last `verify`
        /// into `states`. The caller then sets the position to the verify
        /// position + row + 1; the KV caches need nothing.
        pub fn restore(&self, gpu: &Gpu, row: usize, states: &[State]) {
            let v = self.verify_rows.as_ref().expect("enable_verify first");
            assert!(row < v.rows);
            let copy = gpu.pipeline("copy_u32");
            gpu.run(|enc| {
                let linear = states.iter().filter_map(|s| match s {
                    State::Linear(s) => Some(s),
                    State::Full(_) => None,
                });
                for (i, s) in linear.enumerate() {
                    for (from, to) in [(&v.conv[i], &s.conv), (&v.recurrent[i], &s.recurrent)] {
                        let n = from.length() as usize / 4 / v.rows;
                        dispatch_1d(enc, &copy, &[(from, (row * n * 4) as u64), (to, 0)], n);
                    }
                }
            });
        }

        /// The logits of the last of the `n` hidden states in `x`.
        pub fn logits(&self, gpu: &Gpu, x: &Buffer, n: usize) -> Vec<f32> {
            let logits = gpu.buffer(self.lm_head.rows * 4);
            gpu.run(|enc| self.encode_logits(gpu, enc, x, n, &logits));
            contents::<f32>(&logits, self.lm_head.rows).to_vec()
        }

        fn encode_logits(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &Buffer,
            n: usize,
            logits: &Buffer,
        ) {
            let h = self.cfg.hidden_size;
            let last = whole(&self.scratch.normed);
            norm(
                gpu,
                enc,
                &Norm::rows(1, h, self.cfg.rms_norm_eps, &self.norm),
                &at(x, (n - 1) * h),
                &last,
            );
            qmv(gpu, enc, &self.lm_head, &last, &whole(logits));
        }

        /// Tokens `start .. start + t` of `x` through layer `l`; the first of
        /// them is at position `pos`.
        #[allow(clippy::too_many_arguments)]
        fn encode_chunk(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            l: usize,
            x: &Buffer,
            start: usize,
            t: usize,
            pos: usize,
            state: &State,
        ) {
            let layer = &self.layers[l];
            let s = &self.scratch;
            let h = self.cfg.hidden_size;
            let eps = self.cfg.rms_norm_eps;
            let xt = at(x, start * h);
            let normed = whole(&s.normed);
            let mixer_out = whole(&s.mixer_out);

            if self.on("norm") {
                norm(
                    gpu,
                    enc,
                    &Norm::rows(t, h, eps, &layer.input_norm),
                    &xt,
                    &normed,
                );
                self.record(gpu, enc, 0, &normed, t * h);
            }
            match (&layer.mixer, state) {
                (Mixer::Linear(w), State::Linear(st)) => {
                    let snaps = self
                        .verify_rows
                        .as_ref()
                        .filter(|_| self.snapshot_run.get())
                        .map(|v| {
                            let i = self.linear_index(l);
                            (&v.conv[i], &v.recurrent[i])
                        });
                    self.encode_deltanet(gpu, enc, w, st, &normed, &mixer_out, t, snaps)
                }
                (Mixer::Full(w), State::Full(cache)) => {
                    self.encode_attention(gpu, enc, w, cache, &normed, &mixer_out, t, pos)
                }
                _ => panic!("layer {l}: state does not match the layer type"),
            }
            if self.on("add, post norm") {
                dispatch_1d(
                    enc,
                    &gpu.pipeline("add_in_place"),
                    &[(&xt.buffer, xt.offset), (&s.mixer_out, 0)],
                    t * h,
                );
                norm(
                    gpu,
                    enc,
                    &Norm::rows(t, h, eps, &layer.post_norm),
                    &xt,
                    &normed,
                );
            }
            self.encode_moe(gpu, enc, layer, &normed, &xt, t);
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_deltanet(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &DeltaNetWeights,
            state: &GpuDeltaNetState,
            x: &View,
            out: &View,
            t: usize,
            snaps: Option<(&Buffer, &Buffer)>,
        ) {
            let c = &self.deltanet;
            let s = &self.scratch;
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let [wide, mixed, z, y, a, b] = [&s.wide, &s.mixed, &s.z, &s.y, &s.a, &s.b].map(whole);
            if self.on("deltanet in_proj") {
                self.mm(gpu, enc, &w.qkv, x, &wide, t);
                self.mm(gpu, enc, &w.z, x, &z, t);
                self.mm(gpu, enc, &w.b, x, &b, t);
                self.mm(gpu, enc, &w.a, x, &a, t);
            }

            let (conv_snaps, delta_snaps) = snaps.unwrap_or((&state.conv, &state.recurrent));
            if self.on("deltanet conv") {
                enc.set_compute_pipeline_state(&gpu.pipeline("conv_rows"));
                set_views(enc, 0, &[&wide, &w.conv1d, &whole(&state.conv), &mixed]);
                let p = ConvRowsParams {
                    channels: channels as u32,
                    kernel_size: c.kernel as u32,
                    tokens: t as u32,
                    snapshots: snaps.is_some() as u32,
                };
                params(enc, 4, &p);
                set_views(enc, 5, &[&whole(conv_snaps)]);
                if old_conv() {
                    enc.dispatch_threads(
                        MTLSize::new(channels as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                } else {
                    assert!(c.kernel <= 9);
                    enc.set_compute_pipeline_state(&gpu.pipeline("conv_rows2"));
                    enc.dispatch_threads(
                        MTLSize::new(channels as u64, t as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                    enc.set_compute_pipeline_state(&gpu.pipeline("conv_state"));
                    enc.dispatch_threads(
                        MTLSize::new(channels as u64, 1, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
            }

            if !self.on("deltanet delta") {
                return self.encode_deltanet_out(gpu, enc, w, &y, &z, out, t);
            }
            let p = DeltaRowsParams {
                k_heads: c.k_heads as u32,
                v_heads: c.v_heads as u32,
                v_dim: c.v_dim as u32,
                tokens: t as u32,
                channels: channels as u32,
                eps: c.eps,
                snapshots: snaps.is_some() as u32,
            };
            // SAETTA_DELTA2=1 wins over delta_rows3.
            let d3 = if delta2() { 0 } else { delta3() };
            if !delta2() && d3 == 0 {
                enc.set_compute_pipeline_state(&gpu.pipeline("delta_rows"));
                set_views(
                    enc,
                    0,
                    &[
                        &mixed,
                        &a,
                        &b,
                        &w.a_log,
                        &w.dt_bias,
                        &whole(&state.recurrent),
                        &y,
                    ],
                );
                params(enc, 7, &p);
                set_views(enc, 8, &[&whole(delta_snaps)]);
                // 2 simdgroups of 8 rows each per threadgroup.
                groups(enc, (c.v_dim.div_ceil(16), c.v_heads), 64);
            } else {
                assert!(c.k_dim == 128 && c.v_heads.is_multiple_of(32));
                // Normalizes q and k in place, a and b become g and beta.
                enc.set_compute_pipeline_state(&gpu.pipeline("delta_prep"));
                set_views(enc, 0, &[&mixed, &a, &b, &w.a_log, &w.dt_bias]);
                params(enc, 5, &p);
                groups(enc, (t, 2 * c.k_heads + 1), 32);

                let kernel = if d3 > 0 { "delta_rows3" } else { "delta_rows2" };
                let pipeline = gpu.pipeline(kernel);
                assert!(
                    pipeline.max_total_threads_per_threadgroup() as usize >= 32 * d3,
                    "delta_rows3 takes at most {} threads per threadgroup",
                    pipeline.max_total_threads_per_threadgroup()
                );
                enc.set_compute_pipeline_state(&pipeline);
                set_views(enc, 0, &[&mixed, &a, &b]);
                set_views(enc, 5, &[&whole(&state.recurrent), &y]);
                params(enc, 7, &p);
                set_views(enc, 8, &[&whole(delta_snaps)]);
                if d3 > 0 {
                    // d3 simdgroups of 8 rows each per threadgroup.
                    groups(enc, (c.v_dim.div_ceil(8 * d3), c.v_heads), 32 * d3);
                } else {
                    // 2 simdgroups of 4 rows each per threadgroup.
                    groups(enc, (c.v_dim.div_ceil(8), c.v_heads), 64);
                }
            }
            self.encode_deltanet_out(gpu, enc, w, &y, &z, out, t);
        }

        /// The gated norm of the delta rule output y, and out_proj.
        #[allow(clippy::too_many_arguments)]
        fn encode_deltanet_out(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &DeltaNetWeights,
            y: &View,
            z: &View,
            out: &View,
            t: usize,
        ) {
            let c = &self.deltanet;
            if self.on("deltanet norm") {
                let mut gated = Norm::rows(t * c.v_heads, c.v_dim, c.eps, &w.norm);
                gated.gate = Some(z);
                norm(gpu, enc, &gated, y, y);
                self.record(gpu, enc, 1, y, t * c.v_heads * c.v_dim);
            }
            if self.on("deltanet out_proj") {
                self.mm(gpu, enc, &w.out, y, out, t);
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_attention(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &AttentionWeights,
            cache: &GpuKvCache,
            x: &View,
            out: &View,
            t: usize,
            pos: usize,
        ) {
            let c = &self.attention;
            let s = &self.scratch;
            let d = c.head_dim;
            let kv_size = c.kv_heads * d;
            let (q_and_gate, q) = (whole(&s.wide), whole(&s.z));
            let (keys, values) = (whole(&s.mixed), at(&s.mixed, t * kv_size));
            if self.on("attention q/k/v") {
                self.mm(gpu, enc, &w.q, x, &q_and_gate, t);
                self.mm(gpu, enc, &w.k, x, &keys, t);
                self.mm(gpu, enc, &w.v, x, &values, t);
            }
            if !self.on("attention norm, rope") {
                return self.encode_attention_rows(gpu, enc, w, cache, out, t, pos);
            }

            let mut q_norm = Norm::rows(t * c.heads, d, c.eps, &w.q_norm);
            q_norm.in_stride = 2 * d;
            norm(gpu, enc, &q_norm, &q_and_gate, &q);
            norm(
                gpu,
                enc,
                &Norm::rows(t * c.kv_heads, d, c.eps, &w.k_norm),
                &keys,
                &keys,
            );
            for (view, heads) in [(&q, c.heads), (&keys, c.kv_heads)] {
                enc.set_compute_pipeline_state(&gpu.pipeline("rope_rows"));
                set_views(enc, 0, &[view]);
                let half = c.rope_dims / 2;
                let p = RopeRowsParams {
                    stride: d as u32,
                    half_dims: half as u32,
                    pos: pos as u32,
                    log2_theta: c.rope_theta.log2(),
                    heads: heads as u32,
                };
                params(enc, 1, &p);
                enc.dispatch_threads(
                    MTLSize::new(half as u64, (t * heads) as u64, 1),
                    MTLSize::new(half.min(64) as u64, 1, 1),
                );
            }
            self.encode_attention_rows(gpu, enc, w, cache, out, t, pos);
        }

        /// The new keys and values into the cache (half, or quantized), attention over the
        /// cache for the `t` new queries, and o_proj.
        #[allow(clippy::too_many_arguments)]
        fn encode_attention_rows(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &AttentionWeights,
            cache: &GpuKvCache,
            out: &View,
            t: usize,
            pos: usize,
        ) {
            let c = &self.attention;
            let s = &self.scratch;
            let d = c.head_dim;
            let kv_size = c.kv_heads * d;
            let (q_and_gate, q, y) = (whole(&s.wide), whole(&s.z), whole(&s.y));
            let (keys, values) = (whole(&s.mixed), at(&s.mixed, t * kv_size));
            match &cache.quant {
                None => {
                    for (new, cached) in [(&keys, &cache.keys), (&values, &cache.values)] {
                        to_half(gpu, enc, new, &at_half(cached, pos * kv_size), t * kv_size);
                    }
                }
                Some((kq, vq)) => {
                    kq.encode(gpu, enc, &keys, pos, t);
                    vq.encode(gpu, enc, &values, pos, t);
                }
            }
            if !self.on("attention") {
                if self.on("attention o_proj") {
                    self.mm(gpu, enc, &w.o, &y, out, t);
                }
                return;
            }
            let exact = self.exact_now();
            let kernel = match (&cache.quant, exact) {
                (None, true) => "attention_rows",
                (None, false) => attention_half_kernel(),
                (Some((kq, _)), true) => match kq.bits {
                    KvBits::Eight => "attention_rows_q8",
                    _ => "attention_rows_q4",
                },
                (Some(_), false) if attention_half_kernel() == "gemm" => "gemm",
                (Some((kq, _)), false) => match kq.bits {
                    KvBits::Eight => "attention_rows2_half_q8",
                    _ => "attention_rows2_half_q4",
                },
            };
            let q_in = if exact {
                q.clone()
            } else {
                let q_half = whole(&s.q_half);
                to_half(gpu, enc, &q, &q_half, t * c.heads * d);
                q_half
            };
            // The kernel match picks flash only over a half cache; gemm reads
            // a quantized one in place.
            if kernel.starts_with("attention_fa") || kernel == "gemm" {
                let p = crate::attention_prefill::FlashParams {
                    heads: c.heads as u32,
                    kv_heads: c.kv_heads as u32,
                    pos: pos as u32,
                    tokens: t as u32,
                    scale: 1.0 / (d as f32).sqrt(),
                };
                let (k, v) = (whole(&cache.keys), whole(&cache.values));
                match &self.gemm {
                    Some(gemm) if kernel == "gemm" => gemm.encode(
                        enc,
                        gpu,
                        &q_in,
                        &k,
                        &v,
                        cache.quant.as_ref(),
                        &q_and_gate,
                        &y,
                        p,
                    ),
                    _ => crate::attention_prefill::encode(
                        enc,
                        gpu,
                        kernel,
                        &q_in,
                        &k,
                        &v,
                        &q_and_gate,
                        &y,
                        p,
                    ),
                }
                self.record(gpu, enc, 2, &y, t * c.heads * d);
                if self.on("attention o_proj") {
                    self.mm(gpu, enc, &w.o, &y, out, t);
                }
                return;
            }
            enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
            // A quantized cache is read in place (kernels/attention_rows_q.metal).
            let next = match &cache.quant {
                None => {
                    set_views(enc, 0, &[&q_in, &whole(&cache.keys), &whole(&cache.values)]);
                    3
                }
                Some((kq, vq)) => {
                    let w = |b: &Buffer| whole(b);
                    set_views(
                        enc,
                        0,
                        &[
                            &q_in,
                            &w(&kq.data),
                            &w(&vq.data),
                            &w(&kq.scales),
                            &w(&kq.biases),
                            &w(&vq.scales),
                            &w(&vq.biases),
                        ],
                    );
                    7
                }
            };
            set_views(enc, next, &[&q_and_gate, &y]);
            let p = AttentionParams {
                heads: c.heads as u32,
                kv_heads: c.kv_heads as u32,
                pos: pos as u32,
                tokens: t as u32,
                scale: 1.0 / (d as f32).sqrt(),
            };
            params(enc, next + 2, &p);
            // Heads of one kv head per threadgroup.
            let heads = if kernel == "attention_rows2_half8" {
                8
            } else {
                4
            };
            assert!((c.heads / c.kv_heads).is_multiple_of(heads));
            groups(enc, (t.div_ceil(8), c.heads / heads), 32 * heads);
            self.record(gpu, enc, 2, &y, t * c.heads * d);
            if self.on("attention o_proj") {
                self.mm(gpu, enc, &w.o, &y, out, t);
            }
        }

        /// The experts of `t` <= few_moe_max() tokens, 8 rows at a time on
        /// the verify rows' kernels (moe_union, moe_gate_up_mma3,
        /// moe_down_mma3), in f32 as decode: each distinct expert of the 8
        /// rows is read once, where the grouped path launches a tile grid
        /// for all 256 experts. Leaves sum over k of weight * expert out in
        /// expert_out `[t, hidden]`.
        fn encode_few_experts(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            e: &Experts,
            normed: &View,
            t: usize,
        ) {
            let s = &self.scratch;
            let top_k = self.cfg.num_experts_per_tok;
            let h = self.cfg.hidden_size;
            let (g, up, d) = (&e.gate, &e.up, &e.down);
            assert!(g.bits == 5 && d.bits == 5, "the rows MoE kernels are 5-bit");
            let (list, slots) = (whole(&s.few_list), whole(&s.few_slots));
            let eg = whole(&s.expert_g);
            let split = FEW_MOE_SPLIT;
            let threads = MTLSize::new(32 * split as u64, 1, 1);
            for r0 in (0..t).step_by(FEW_MOE_ROWS) {
                let m = FEW_MOE_ROWS.min(t - r0);
                let ids = at(&s.ids, r0 * top_k);
                let weights = at(&s.expert_weights, r0 * top_k);
                // x reads FEW_MOE_ROWS rows from r0: the chunk scratch has them.
                let x = View {
                    buffer: normed.buffer.clone(),
                    offset: normed.offset + (r0 * h * 4) as u64,
                };
                let y = at(&s.expert_out, r0 * h);

                enc.set_compute_pipeline_state(&gpu.pipeline("moe_union"));
                set_views(enc, 0, &[&ids, &list, &slots]);
                params(enc, 3, &[m as u32, top_k as u32]);
                groups(enc, (1, 1), 32);

                enc.set_compute_pipeline_state(&gpu.pipeline(&format!("moe_gate_up_mma3_{split}")));
                set_quant(enc, 0, g);
                set_quant(enc, 3, up);
                set_views(enc, 6, &[&x, &eg]);
                params(enc, 8, &[g.rows as u32, g.cols as u32, top_k as u32]);
                set_views(enc, 9, &[&list, &slots]);
                params(enc, 11, &(m as u32));
                enc.dispatch_thread_groups(
                    MTLSize::new((g.rows / 8) as u64, (m * top_k) as u64, 1),
                    threads,
                );

                enc.set_compute_pipeline_state(&gpu.pipeline(&format!("moe_down_mma3_{split}")));
                set_quant(enc, 0, d);
                set_views(enc, 3, &[&eg, &y]);
                params(enc, 5, &[d.rows as u32, d.cols as u32, top_k as u32]);
                set_views(enc, 6, &[&list, &slots, &weights]);
                params(enc, 9, &(m as u32));
                enc.dispatch_thread_groups(MTLSize::new((d.rows / 8) as u64, 1, 1), threads);
            }
        }

        /// x += moe(normed) for `t` tokens.
        fn encode_moe(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            layer: &Layer,
            normed: &View,
            x: &View,
            t: usize,
        ) {
            let s = &self.scratch;
            let cfg = self.cfg;
            let top_k = cfg.num_experts_per_tok;
            let h = cfg.hidden_size;
            let [router, shared_gate, sg, su, sd] = [
                &s.router,
                &s.shared_gate,
                &s.shared_g,
                &s.shared_u,
                &s.shared_out,
            ]
            .map(whole);
            let fast = !self.exact_now();
            if self.on("router") {
                // In half like the other matrices: the quality gate held. With
                // SAETTA_ROUTER_FAST=0 it stays close to f32, as decode.
                if fast && fast_router() {
                    self.mm(gpu, enc, &layer.router, normed, &router, t);
                } else {
                    qmm(gpu, enc, &layer.router, normed, &router, t, true);
                }
                self.mm(gpu, enc, &layer.shared_gate, normed, &shared_gate, t);
            }
            let silu_mul = gpu.pipeline("silu_mul_rows");
            let normed_half = whole(&s.normed_half);
            // normed in half, for the fused shared expert and the experts.
            let half_early = fast && shared_gate_up();
            if half_early && self.on("router") {
                to_half(gpu, enc, normed, &normed_half, t * h);
            }
            if self.on("shared expert") {
                if half_early && t > FEW_MAX {
                    let [gate, up, _] = &layer.shared;
                    let shared_half = whole(&s.expert_half);
                    gate_up_plain(gpu, enc, gate, up, &normed_half, &sg, &shared_half, t);
                } else {
                    self.mm(gpu, enc, &layer.shared[0], normed, &sg, t);
                    self.mm(gpu, enc, &layer.shared[1], normed, &su, t);
                    dispatch_1d(
                        enc,
                        &silu_mul,
                        &[(&s.shared_g, 0), (&s.shared_u, 0)],
                        t * layer.shared[0].rows,
                    );
                }
                self.record(gpu, enc, 3, &sg, t * layer.shared[0].rows);
                self.mm(gpu, enc, &layer.shared[2], &sg, &sd, t);
            }

            // Fast mode runs the experts on qmm2, with its tile size.
            let tile_rows = if self.exact_now() {
                TILE
            } else {
                expert_variant(t).rows
            };
            let [ids, weights, order, rows, tiles, tile_count] = [
                &s.ids,
                &s.expert_weights,
                &s.order,
                &s.rows,
                &s.tiles,
                &s.tile_count,
            ]
            .map(whole);
            if self.on("route, group") {
                enc.set_compute_pipeline_state(&gpu.pipeline("moe_route"));
                set_views(enc, 0, &[&router, &ids, &weights]);
                let p = RouteParams {
                    experts: cfg.num_experts as u32,
                    top_k: top_k as u32,
                };
                params(enc, 3, &p);
                groups(enc, (t, 1), 32);
            }
            if t <= few_moe_max() {
                let e = &layer.experts;
                if self.on("experts gate, up") {
                    self.encode_few_experts(gpu, enc, e, normed, t);
                }
                if self.on("combine") {
                    enc.set_compute_pipeline_state(&gpu.pipeline("moe_add_rows"));
                    set_views(enc, 0, &[x, &sd, &shared_gate, &whole(&s.expert_out)]);
                    params(enc, 4, &(h as u32));
                    enc.dispatch_threads(
                        MTLSize::new(h as u64, t as u64, 1),
                        MTLSize::new(256, 1, 1),
                    );
                }
                return;
            }
            if self.on("route, group") {
                enc.set_compute_pipeline_state(&gpu.pipeline("moe_group"));
                set_views(enc, 0, &[&ids, &order, &rows, &tiles, &tile_count]);
                let p = GroupParams {
                    pairs: (t * top_k) as u32,
                    top_k: top_k as u32,
                    experts: cfg.num_experts as u32,
                    tile_rows: tile_rows as u32,
                };
                params(enc, 5, &p);
                groups(enc, (1, 1), 1024);
            }

            let g = Grouped {
                tiles: &tiles,
                rows: &rows,
                tile_count: &tile_count,
                // Every tile has a row: no more tiles than pairs.
                max_tiles: ((t * top_k).div_ceil(tile_rows) + cfg.num_experts).min(t * top_k),
                tokens: t,
            };
            assert!(g.max_tiles <= s.max_tiles);
            let e = &layer.experts;
            let (eg, eu, eo) = (whole(&s.expert_g), whole(&s.expert_u), whole(&s.expert_out));
            let expert_half = whole(&s.expert_half);
            let fused64 = fast && gate_up64(t);
            let fused = fast && (fused64 || fused_gate_up(t));
            if self.on("experts gate, up") {
                if self.exact_now() {
                    qmm_grouped(gpu, enc, &e.gate, normed, &eg, &g, true, true);
                    qmm_grouped(gpu, enc, &e.up, normed, &eu, &g, true, true);
                } else if fused {
                    if !half_early {
                        to_half(gpu, enc, normed, &normed_half, t * h);
                    }
                    gate_up_grouped(
                        gpu,
                        enc,
                        &e.gate,
                        &e.up,
                        normed,
                        &normed_half,
                        &eg,
                        &eu,
                        &expert_half,
                        &g,
                        fused64,
                    );
                } else {
                    if !half_early {
                        to_half(gpu, enc, normed, &normed_half, t * h);
                    }
                    qmm2_grouped(gpu, enc, &e.gate, normed, &normed_half, &eg, &g, true);
                    qmm2_grouped(gpu, enc, &e.up, normed, &normed_half, &eu, &g, true);
                }
            }
            if self.on("experts silu") {
                if !fused {
                    dispatch_1d(
                        enc,
                        &silu_mul,
                        &[(&s.expert_g, 0), (&s.expert_u, 0)],
                        t * top_k * e.size(),
                    );
                }
                self.record(gpu, enc, 4, &eg, t * top_k * e.size());
            }
            if self.on("experts down") {
                if self.exact_now() {
                    qmm_grouped(gpu, enc, &e.down, &eg, &eo, &g, false, true);
                } else {
                    if !fused {
                        to_half(gpu, enc, &eg, &expert_half, t * top_k * e.size());
                    }
                    qmm2_grouped(gpu, enc, &e.down, &eg, &expert_half, &eo, &g, false);
                }
            }

            if self.on("combine") {
                enc.set_compute_pipeline_state(&gpu.pipeline("moe_combine"));
                set_views(enc, 0, &[x, &sd, &shared_gate, &eo, &order, &weights]);
                let p = CombineParams {
                    dim: h as u32,
                    top_k: top_k as u32,
                };
                params(enc, 6, &p);
                enc.dispatch_threads(MTLSize::new(h as u64, t as u64, 1), MTLSize::new(256, 1, 1));
            }
        }
    }

    /// Prompt tokens for tests and benchmarks: a fixed spread of ids.
    pub fn test_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 7919 + 13) % 150_000) as u32).collect()
    }

    /// SAETTA_EXACT=1 picks the exact prefill kernels over the fast ones.
    pub fn exact_from_env() -> bool {
        std::env::var("SAETTA_EXACT").is_ok_and(|v| v == "1")
    }

    /// Layer states for prefill without a `GpuModel`, for tests and
    /// benchmarks.
    pub enum OwnedState {
        Linear(GpuDeltaNetState),
        Full(GpuKvCache),
    }

    impl OwnedState {
        pub fn new_all(gpu: &Gpu, cfg: &Config, capacity: usize) -> Vec<OwnedState> {
            OwnedState::with_bits(gpu, cfg, capacity, KvBits::Half)
        }

        /// As `new_all`, with the KV caches stored as `bits`.
        pub fn with_bits(
            gpu: &Gpu,
            cfg: &Config,
            capacity: usize,
            bits: KvBits,
        ) -> Vec<OwnedState> {
            let capacity = capacity.next_multiple_of(KEY_BLOCK);
            cfg.layer_types
                .iter()
                .map(|kind| match kind {
                    LayerType::LinearAttention => {
                        OwnedState::Linear(GpuDeltaNetState::new(gpu, &deltanet_config(cfg)))
                    }
                    LayerType::FullAttention => OwnedState::Full(GpuKvCache::with_bits(
                        gpu,
                        &attention_config(cfg),
                        capacity,
                        bits,
                    )),
                })
                .collect()
        }

        pub fn state(&self) -> State<'_> {
            match self {
                OwnedState::Linear(s) => State::Linear(s),
                OwnedState::Full(c) => State::Full(c),
            }
        }
    }

    /// Prefills `tokens` tokens `runs` times, after one warm-up run, and
    /// prints the best and median tokens per second.
    pub fn bench(
        dir: &std::path::Path,
        tokens: usize,
        chunk: usize,
        runs: usize,
        exact: bool,
        bits: KvBits,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let capacity = tokens.next_multiple_of(KEY_BLOCK);
        let mut prefill = Prefill::new(&gpu, &cfg, &weights, chunk)?;
        prefill.exact = exact;
        let prompt = test_tokens(tokens);
        let owned = OwnedState::with_bits(&gpu, &cfg, capacity, bits);
        let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
        let mut times: Vec<f64> = (0..=runs)
            .map(|_| {
                for s in &owned {
                    if let OwnedState::Linear(d) = s {
                        zero(&d.conv);
                        zero(&d.recurrent);
                    }
                }
                let start = Instant::now();
                prefill.run(&gpu, &prompt, 0, &states);
                start.elapsed().as_secs_f64()
            })
            .skip(1)
            .collect();
        times.sort_by(f64::total_cmp);
        println!(
            "prefill {tokens} tokens, chunk {}, exact {exact}, {bits:?} KV, {runs} runs: best {:.3} s ({:.0} tokens/s), median {:.3} s ({:.0} tokens/s)",
            prefill.chunk,
            times[0],
            tokens as f64 / times[0],
            times[runs / 2],
            tokens as f64 / times[runs / 2],
        );
        Ok(())
    }

    /// Looks for what the first prefill after a pause pays for: the driver
    /// gives back about 1.1 GB of its own memory between requests and the
    /// next prefill takes about 0.55 s longer. Prints the wall time of a
    /// whole prefill right after another and after `gap` seconds, then of
    /// each layer and of each stage of layers 0 (DeltaNet) and 3 (attention)
    /// after a gap and right after. A stage that costs much more after the
    /// gap is one that makes the driver allocate again.
    pub fn bench_gap(
        dir: &std::path::Path,
        tokens: usize,
        chunk: usize,
        gap: f64,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use std::time::{Duration, Instant};
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let capacity = tokens.next_multiple_of(KEY_BLOCK);
        let mut prefill = Prefill::new(&gpu, &cfg, &weights, chunk)?;
        // The kernels serve runs.
        prefill.exact = exact_from_env();
        let prompt = test_tokens(tokens);
        let owned = OwnedState::new_all(&gpu, &cfg, capacity);
        let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
        let pause = || std::thread::sleep(Duration::from_secs_f64(gap));
        let footprint = || {
            let pid = std::process::id().to_string();
            let out = std::process::Command::new("footprint").arg(&pid).output();
            out.ok()
                .and_then(|o| {
                    let text = String::from_utf8_lossy(&o.stdout).into_owned();
                    let line = text.lines().find(|l| l.contains("Footprint:"))?.to_string();
                    Some(line.split("Footprint:").nth(1)?.trim().to_string())
                })
                .unwrap_or_else(|| "?".into())
        };
        let whole = || {
            let start = Instant::now();
            prefill.run(&gpu, &prompt, 0, &states);
            start.elapsed().as_secs_f64() * 1e3
        };
        whole();
        let warm = whole();
        let before = footprint();
        pause();
        let after_gap = footprint();
        let cold = whole();
        println!(
            "whole prefill of {tokens} tokens: {warm:.0} ms right after one, {cold:.0} ms after {gap} s (footprint {before} before the pause, {after_gap} after it)"
        );

        let x = prefill.embed(&gpu, &prompt);
        let bytes = tokens * cfg.hidden_size * 4;
        let ptr = x.contents() as *mut u8;
        let mut input = vec![0u8; bytes];
        unsafe { std::ptr::copy_nonoverlapping(ptr, input.as_mut_ptr(), bytes) };
        let layer = |l: usize| {
            unsafe { std::ptr::copy_nonoverlapping(input.as_ptr(), ptr, bytes) };
            let start = Instant::now();
            prefill.layer(&gpu, l, &x, tokens, 0, &states[l]);
            start.elapsed().as_secs_f64() * 1e3
        };
        pause();
        let cold: Vec<f64> = (0..states.len()).map(layer).collect();
        let warm: Vec<f64> = (0..states.len()).map(layer).collect();
        println!("layers after {gap} s (ms, right after in brackets):");
        for (l, (c, w)) in cold.iter().zip(&warm).enumerate() {
            let mark = if c - w > 5.0 { "  <-" } else { "" };
            println!("  layer {l:2}: {c:7.1} ({w:6.1}){mark}");
        }

        let stages = |l: usize| -> Vec<f64> {
            (0..STAGES.len())
                .map(|k| {
                    prefill.only.set(Some(k));
                    let t = layer(l);
                    prefill.only.set(None);
                    t
                })
                .collect()
        };
        for l in [0, 3] {
            pause();
            let cold = stages(l);
            let warm = stages(l);
            println!(
                "layer {l} by stage after {gap} s, stages 0 ..= k (ms, right after in brackets):"
            );
            for (k, (c, w)) in cold.iter().zip(&warm).enumerate() {
                let mark = if c - w > 5.0 { "  <-" } else { "" };
                println!("  {:>22}: {c:7.1} ({w:6.1}){mark}", STAGES[k]);
            }
        }
        Ok(())
    }

    /// Prints the GPU time per stage of a `tokens`-token prefill (see
    /// `Prefill::profile`), after one warm-up run.
    /// Profiles the prefill of `tokens` tokens, or with `last`, only of the
    /// last `last` of them after a plain prefill of the others, to see what
    /// a chunk costs deep in a long prompt.
    pub fn bench_profile(
        dir: &std::path::Path,
        tokens: usize,
        chunk: usize,
        exact: bool,
        last: Option<usize>,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let capacity = tokens.next_multiple_of(KEY_BLOCK);
        let mut prefill = Prefill::new(&gpu, &cfg, &weights, chunk)?;
        prefill.exact = exact;
        let prompt = test_tokens(tokens);
        let owned = OwnedState::new_all(&gpu, &cfg, capacity);
        let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
        let first = tokens - last.unwrap_or(tokens).min(tokens);
        if first == 0 {
            // A warm-up, then the DeltaNet state of an empty prompt again.
            prefill.run(&gpu, &prompt, 0, &states);
            for s in &owned {
                if let OwnedState::Linear(d) = s {
                    zero(&d.conv);
                    zero(&d.recurrent);
                }
            }
        } else {
            // The keys and values of the first tokens for real; the DeltaNet
            // state the profiled runs then start from does not change the time.
            let start = std::time::Instant::now();
            prefill.run(&gpu, &prompt[..first], 0, &states);
            let secs = start.elapsed().as_secs_f64();
            println!("prefill of the first {first} tokens: {secs:.1} s");
        }
        let times = prefill.profile(&gpu, &prompt[first..], first, &states);
        let total: f64 = times.iter().rev().take(2).map(|t| t.1).sum();
        let n = tokens - first;
        println!(
            "prefill {n} tokens at position {first}, chunk {}, exact {exact}: GPU {:.3} s ({:.0} tokens/s)",
            prefill.chunk,
            total,
            n as f64 / total
        );
        for (stage, t) in &times {
            println!("{stage:>22} {:8.1} ms {:5.1}%", t * 1e3, 100.0 * t / total);
        }
        Ok(())
    }

    /// Times qmm on real matrices with `m` rows of x: `runs` command buffers
    /// of 5 calls each after one warm-up, for both kernel kinds. Prints the
    /// median TFLOP/s.
    pub fn bench_qmm(
        dir: &std::path::Path,
        m: usize,
        runs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        for module in [
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "language_model.model.layers.0.linear_attn.out_proj",
            "language_model.model.layers.0.mlp.shared_expert.gate_proj",
        ] {
            let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization)?);
            let x = View::new(gpu.buffer_from(&vec![0.01f32; m * w.cols]));
            let y = View::new(gpu.buffer(m * w.rows * 4));
            let b = w.bits;
            for name in [
                format!("qmm_dq_{b}bit"),
                format!("qmm_{b}bit"),
                format!("qmm_{b}bit_split"),
            ] {
                let mut times: Vec<f64> = (0..=runs)
                    .map(|_| {
                        gpu.run(|enc| {
                            for _ in 0..5 {
                                qmm_with(&gpu, enc, &w, &x, &y, m, &name);
                            }
                        })
                        .as_secs_f64()
                            / 5.0
                    })
                    .skip(1)
                    .collect();
                times.sort_by(f64::total_cmp);
                let flops = 2.0 * (m * w.rows * w.cols) as f64;
                println!(
                    "{name} {}x{} {}-bit, m {m}: median {:.2} ms, {:.1} TFLOP/s ({runs} runs)",
                    w.rows,
                    w.cols,
                    w.bits,
                    times[runs / 2] * 1e3,
                    flops / times[runs / 2] / 1e12,
                );
            }
        }
        Ok(())
    }

    fn zero(buffer: &Buffer) {
        unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, buffer.length() as usize) };
    }

    #[cfg(test)]
    mod tests {
        //! Prefill against the CPU model fed one token at a time:
        //!   scripts/mac.sh test --release --lib prefill -- --ignored --nocapture

        use super::*;
        use crate::cpu::matrix::Matrix;
        use crate::cpu::model::{LayerState, Model, argmax};
        use crate::cpu::ops::rms_norm;
        use crate::gpu_model::GpuModel;
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            assert_eq!(got.len(), want.len());
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn qmm_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let m = 70;
            for module in [
                "language_model.model.layers.0.linear_attn.in_proj_qkv",
                "language_model.model.layers.0.linear_attn.in_proj_a",
                "language_model.model.layers.0.mlp.gate",
                "language_model.model.layers.0.mlp.shared_expert_gate",
            ] {
                let q = weights.quantized(module, &cfg.quantization).unwrap();
                let w = GpuQuant::new(&gpu, &q);
                let cpu = Matrix::quant(q);
                let x: Vec<f32> = (0..m * w.cols)
                    .map(|i| ((i * 37) as f32 * 0.011).sin())
                    .collect();
                let want: Vec<f32> = x.chunks_exact(w.cols).flat_map(|r| cpu.matvec(r)).collect();
                let (xv, yv) = (
                    View::new(gpu.buffer_from(&x)),
                    View::new(gpu.buffer(want.len() * 4)),
                );
                for (split, bound) in [(true, 1e-5), (false, 5e-3)] {
                    gpu.run(|enc| qmm(&gpu, enc, &w, &xv, &yv, m, split));
                    let err = rel_rms(contents::<f32>(&yv.buffer, want.len()), &want);
                    println!(
                        "{module}: {}x{} {}-bit, split {split}, rel rms {err:.2e}",
                        w.rows, w.cols, w.bits
                    );
                    assert!(err < bound);
                }
            }
        }

        /// With the split kernels and the CPU rounding its KV cache to half
        /// as well, prefill is within f32 rounding of the CPU (1e-6 to 1e-4
        /// up to layer 19 of this prompt). The two round slightly different
        /// keys, so now and then a key lands on the next half value, and a
        /// token at a near tie of the router picks another expert: 3e-3 for
        /// one token at layer 20, which attention and the DeltaNet states
        /// spread from there (median token 2.2e-3, worst 1.6e-2 and states 8e-3
        /// by layer 37). A layout or slot bug is off by order 1, so bounds
        /// at twice that still catch it.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn exact_prefill_matches_token_by_token() {
            assert_passes(&check_prefill(true, 5e-3, 1e-2, KvBits::Half));
        }

        /// Prefill over an 8-bit and a 4-bit KV cache against the CPU storing
        /// its cache the same way and reading it through half. The codes match
        /// the CPU's for the same input (tests/kv_quant.rs); here the inputs
        /// differ by f32 rounding, so a value now and then lands on the next
        /// code, and the bounds are the half cache's times
        /// kv_quant::test_slack (8 bits: logits 1.3e-2, median token 2e-3).
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn quantized_prefill_matches_token_by_token() {
            let mut failed = Vec::new();
            for bits in [KvBits::Eight, KvBits::Four] {
                println!("{bits:?}:");
                failed.extend(check_prefill(true, 5e-3, 1e-2, bits));
            }
            assert_passes(&failed);
        }

        /// With half inputs the typical token is off by the rounding of x to
        /// half (2e-4), far below the bf16 MLX server's 1e-2 after a few
        /// layers (the golden test in src/cpu/model.rs). A token whose router
        /// hits a near tie picks another expert and is off by about 5e-2,
        /// which the DeltaNet states of later layers take in. So this test
        /// bounds the median token tightly and the states loosely; the KL of
        /// track F's eval is the gate for this path.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn fast_prefill_matches_token_by_token() {
            // With 128-row chunks on the half kernels a few near-tie tokens
            // pick another expert at layer 0 and spread through attention:
            // the median token peaks at 6.1e-2 (layer 26), state 7.4e-2,
            // logits 4.2e-2. MLX in bf16 is off by 1.6e-2 at layer 3.
            // Bounds are about twice what was measured.
            assert_passes(&check_prefill(false, 0.12, 0.1, KvBits::Half));
        }

        /// Fast prefill (gemm attention reading the codes in place) over an
        /// 8-bit and a 4-bit KV cache, with the fast bounds times
        /// kv_quant::test_slack.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn fast_quantized_prefill_matches_token_by_token() {
            let mut failed = Vec::new();
            for bits in [KvBits::Eight, KvBits::Four] {
                println!("{bits:?}:");
                failed.extend(check_prefill(false, 0.12, 0.1, bits));
            }
            assert_passes(&failed);
        }

        /// 166 tokens in two calls (150 then 16) with chunks of 128 rounded
        /// from 100, so the first call runs a 128-row chunk on the large-m
        /// kernels and a 22-row one on qmm_few: the hidden state after each layer and every layer state
        /// must be within `bound` of the CPU fed one token at a time, the
        /// logits within `logits_bound`, with the same argmax (over a
        /// quantized cache, a near tie may pick the other). Returns what
        /// went over instead of stopping, so one run shows every layer.
        fn check_prefill(exact: bool, bound: f32, logits_bound: f32, bits: KvBits) -> Vec<String> {
            let slack = crate::kv_quant::test_slack(bits);
            let (bound, logits_bound) = (bound * slack, logits_bound * slack);
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let model = Model::new(&cfg, &weights).unwrap();
            let tokens = test_tokens(166);
            let capacity = tokens.len().next_multiple_of(KEY_BLOCK);
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 100).unwrap();
            prefill.exact = exact;
            let parts = [&tokens[..150], &tokens[150..]];
            let h = cfg.hidden_size;

            let mut cpu_states = model.new_states();
            // The CPU stores its keys and values as the GPU cache does, and
            // reads a quantized one through half as prefill does.
            for s in &mut cpu_states {
                if let LayerState::Full(c) = s {
                    c.round_to_as_half(bits);
                }
            }
            let gpu_states = OwnedState::with_bits(&gpu, &cfg, capacity, bits);
            let mut cpu_x: Vec<Vec<Vec<f32>>> = parts
                .iter()
                .map(|p| p.iter().map(|&t| model.embed(t)).collect())
                .collect();
            let gpu_x: Vec<Buffer> = parts.iter().map(|p| prefill.embed(&gpu, p)).collect();
            let mut failed = Vec::new();
            for l in 0..cfg.num_layers() {
                let mut err = 0.0f32;
                let mut all = Vec::with_capacity(tokens.len());
                let mut last_median = 0.0f32;
                let mut pos = 0;
                for (i, part) in parts.iter().enumerate() {
                    for x in cpu_x[i].iter_mut() {
                        *x = model.layer(l).step(x, &mut cpu_states[l]).unwrap().hidden;
                    }
                    let state = gpu_states[l].state();
                    prefill.layer(&gpu, l, &gpu_x[i], part.len(), pos, &state);
                    pos += part.len();
                    let got = contents::<f32>(&gpu_x[i], part.len() * h);
                    err = err.max(rel_rms(got, &cpu_x[i].concat()));
                    let per_token: Vec<f32> = got
                        .chunks_exact(h)
                        .zip(&cpu_x[i])
                        .map(|(g, w)| rel_rms(g, w))
                        .collect();
                    let (worst, e) = per_token
                        .iter()
                        .enumerate()
                        .fold((0, 0.0f32), |b, (i, &e)| if e > b.1 { (i, e) } else { b });
                    let mut sorted = per_token.clone();
                    sorted.sort_by(f32::total_cmp);
                    last_median = sorted[sorted.len() / 2];
                    all.extend_from_slice(&per_token);
                    if e > bound {
                        println!(
                            "  part {i}: worst token {worst} at {e:.2e}, median {:.2e}",
                            sorted[sorted.len() / 2]
                        );
                    }
                }
                let state = match (&cpu_states[l], &gpu_states[l]) {
                    (LayerState::Linear(c), OwnedState::Linear(g)) => {
                        let conv = rel_rms(contents::<f32>(&g.conv, c.conv.len()), &c.conv);
                        let rec = rel_rms(
                            contents::<f32>(&g.recurrent, c.recurrent.len()),
                            &c.recurrent,
                        );
                        conv.max(rec)
                    }
                    (LayerState::Full(c), OwnedState::Full(g)) => {
                        let n = c.keys().len();
                        let f32s = |b: &Buffer| -> Vec<f32> {
                            contents::<half::f16>(b, n)
                                .iter()
                                .map(|v| v.to_f32())
                                .collect()
                        };
                        let dequantized = |q: &crate::kv_quant::QuantRows| {
                            let groups = n / crate::kv_quant::GROUP;
                            crate::kv_quant::dequantize(
                                contents::<u32>(&q.data, groups * q.bits.words_per_group()),
                                contents::<half::f16>(&q.scales, groups),
                                contents::<half::f16>(&q.biases, groups),
                                q.bits,
                            )
                        };
                        let (keys, values) = match &g.quant {
                            None => (f32s(&g.keys), f32s(&g.values)),
                            Some((k, v)) => (dequantized(k), dequantized(v)),
                        };
                        let k = rel_rms(&keys, c.keys());
                        let v = rel_rms(&values, c.values());
                        k.max(v)
                    }
                    _ => panic!("layer {l}: state types differ"),
                };
                all.sort_by(f32::total_cmp);
                let median = all[all.len() / 2];
                println!(
                    "layer {l}: hidden rel rms {err:.2e}, median token {median:.2e}, last call median {last_median:.2e}, state {state:.2e}"
                );
                // The half path may pick another expert at a near tie for a
                // token, as bf16 MLX does, so only the median token counts.
                // So does a quantized cache: a key on the next code moves a
                // few tokens far more than the rest (8 bits, layer 14: rms
                // 2.2e-2, median 5.1e-3), and F's KL gate covers the whole.
                // `bound` has the slack of a quantized cache in it.
                let by_median = !(exact && bits == KvBits::Half);
                let hidden = if by_median { median } else { err };
                // The state of an attention layer over a quantized cache is
                // its codes: a key that moved with the hidden state lands on
                // the next code, one step (1.9e-2 of the rms at 8 bits)
                // away, so it gets twice the room (8 bits, layer 15: 2.5e-2
                // while the median token is 5.3e-3).
                let full = matches!(cpu_states[l], LayerState::Full(_));
                let state_bound = if full && bits != KvBits::Half {
                    bound * 2.5
                } else {
                    bound * 1.25
                };
                if hidden >= bound || state >= state_bound {
                    failed.push(format!(
                        "{bits:?} layer {l}: hidden {hidden:.2e} (bound {bound:.2e}), state {state:.2e} (bound {state_bound:.2e})"
                    ));
                }
                // The median over all tokens is mostly the first call. The
                // 16 tokens of the second read the states the first left
                // and move with them, so they get twice the room, which
                // still catches an order 1 error from a wrong position or
                // slot. Measured on the exact path (batch 7), second call
                // median at 8 bits: layer 24 2.2e-2, 27 2.8e-2, 28 3.2e-2,
                // 29 3.0e-2, 30 2.7e-2, 31 2.6e-2, under 2e-2 elsewhere;
                // at 4 bits at most 3.1e-2 (layer 23) against 1.6e-1.
                // A batch 6 run had 2.8e-2 at layer 17 of 8 bits, where
                // batch 7 had 5.8e-3: the runs differ, so this is not
                // a tight fit to one of them.
                let second_bound = 2.0 * bound;
                if by_median && last_median >= second_bound {
                    failed.push(format!(
                        "{bits:?} layer {l}: second call median {last_median:.2e} (bound {second_bound:.2e})"
                    ));
                }
            }

            let norm = weights
                .tensor("language_model.model.norm.weight")
                .unwrap()
                .to_f32();
            let lm_head = Matrix::quant(
                weights
                    .quantized("language_model.lm_head", &cfg.quantization)
                    .unwrap(),
            );
            let mut last = cpu_x[1].last().unwrap().clone();
            rms_norm(&mut last, &norm, cfg.rms_norm_eps);
            let want = lm_head.matvec(&last);
            let got = prefill.logits(&gpu, &gpu_x[1], parts[1].len());
            let err = rel_rms(&got, &want);
            println!("logits: rel rms {err:.2e}");
            if err >= logits_bound {
                failed.push(format!("{bits:?} logits {err:.2e} (bound {logits_bound:.2e})"));
            }
            // Over a quantized cache the GPU and CPU round some codes
            // differently, and at a near tie the top token can flip from run
            // to run (batch 8, 4 bits: 95726 vs 96022 in one of two runs).
            // So the GPU's pick must only be within the logits' error of the
            // CPU's best: its CPU logit at most logits_bound times the rms of
            // all logits below the best, a few times the logits' error.
            let (g, w) = (argmax(&got) as usize, argmax(&want) as usize);
            if exact && g != w {
                let rms = (want.iter().map(|v| v * v).sum::<f32>() / want.len() as f32).sqrt();
                let gap = want[w] - want[g];
                println!("argmax {g} vs {w}: CPU logits {:.3} vs {:.3}, gap {:.2e} of the rms", want[g], want[w], gap / rms);
                if bits == KvBits::Half || gap > logits_bound * rms {
                    failed.push(format!("{bits:?} argmax {g} vs {w}, gap {:.2e} of the rms", gap / rms));
                }
            }
            failed
        }

        fn assert_passes(failed: &[String]) {
            assert!(failed.is_empty(), "over the bounds:\n{}", failed.join("\n"));
        }

        /// The prefill kernels compile (no dispatch, so no GPU work).
        #[test]
        fn kernels_compile() {
            let gpu = Gpu::new().unwrap();
            for name in [
                "prefill_gate_up_5bit",
                "prefill_gate_up_8bit",
                "prefill_silu_few",
                "attention_fa8",
                "attention_fa4",
                "prefill_ag_scores",
                "prefill_ag_scores_q8",
                "prefill_ag_scores_q4",
                "prefill_ag_softmax",
                "prefill_ag_pv",
                "prefill_ag_pv_q8",
                "prefill_ag_pv_q4",
                "prefill_ag_final",
            ] {
                gpu.pipeline(name);
            }
        }

        /// verify gives the same rows with `exact` false as with true.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn verify_is_exact_in_fast_mode() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let n = cfg.num_layers();
            let tokens = test_tokens(26);
            let (prompt, draft) = tokens[..25].split_at(20);
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 64).unwrap();
            prefill.enable_verify(&gpu, 5);
            let mut rows = Vec::new();
            for exact in [true, false] {
                let mut model = GpuModel::load(&gpu, &cfg, &weights, n, 64).unwrap();
                prefill.exact = true;
                prefill.run(&gpu, prompt, 0, &model.states().collect::<Vec<_>>());
                model.set_position(20);
                prefill.exact = exact;
                let v = prefill.verify(&gpu, draft, 20, &model.states().collect::<Vec<_>>(), None);
                rows.push(contents::<f32>(&v.logits.buffer, draft.len() * cfg.vocab_size).to_vec());
            }
            assert!(rows[0] == rows[1]);
        }

        /// Fast prefill of 1100 tokens in chunks of 1024 against exact
        /// prefill, which check_prefill holds to the CPU. A 1024-token chunk
        /// gives about 32 rows per expert, so this covers the large-m and
        /// gather kernels the short tests leave to qmm_few.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn fast_matches_exact_long() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let tokens = test_tokens(1100);
            let h = cfg.hidden_size;
            let capacity = tokens.len().next_multiple_of(KEY_BLOCK);
            let mut runs = Vec::new();
            for exact in [true, false] {
                let mut prefill = Prefill::new(&gpu, &cfg, &weights, 1024).unwrap();
                prefill.exact = exact;
                let owned = OwnedState::new_all(&gpu, &cfg, capacity);
                let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
                let (logits, x) = prefill.run_hidden(&gpu, &tokens, 0, &states);
                runs.push((logits, contents::<f32>(&x, tokens.len() * h).to_vec()));
            }
            let [(want_logits, want), (got_logits, got)] = [&runs[0], &runs[1]];
            let mut per_token: Vec<f32> = got
                .chunks_exact(h)
                .zip(want.chunks_exact(h))
                .map(|(g, w)| rel_rms(g, w))
                .collect();
            per_token.sort_by(f32::total_cmp);
            let median = per_token[per_token.len() / 2];
            let worst = per_token[per_token.len() - 1];
            let logits = rel_rms(got_logits, want_logits);
            println!(
                "fast vs exact, 1100 tokens: hidden median token {median:.2e}, worst {worst:.2e}, logits {logits:.2e}, argmax {} vs {}",
                argmax(got_logits),
                argmax(want_logits)
            );
            // fast_prefill_matches_token_by_token measures a median of 2e-4
            // to 2.5e-3 at layer 39 of a short prompt.
            assert!(median < 2e-2 && logits < 5e-2);
        }

        /// Prefill into a GpuModel's states, then decode with it: the logits
        /// match a GpuModel fed every token one at a time. Covers prefill
        /// after decode tokens too, in the same KV cache.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn decode_after_prefill_matches_gpu_model() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let capacity = 64;
            let n = cfg.num_layers();
            let mut by_token = GpuModel::load(&gpu, &cfg, &weights, n, capacity).unwrap();
            let mut with_prefill = GpuModel::load(&gpu, &cfg, &weights, n, capacity).unwrap();
            let prefill = Prefill::new(&gpu, &cfg, &weights, 64).unwrap();
            assert!(prefill.exact);
            let tokens = test_tokens(50);
            // prefill 30, decode 5, prefill 10, decode 5
            let plan = [(30, true), (5, false), (10, true), (5, false)];
            let mut at = 0;
            for (count, use_prefill) in plan {
                let part = &tokens[at..at + count];
                let mut want = Vec::new();
                for &t in part {
                    want = by_token.forward(t);
                }
                let got = if use_prefill {
                    let states: Vec<State> = with_prefill.states().collect();
                    let logits = prefill.run(&gpu, part, at, &states);
                    with_prefill.set_position(at + count);
                    logits
                } else {
                    let mut logits = Vec::new();
                    for &t in part {
                        logits = with_prefill.forward(t);
                    }
                    logits
                };
                at += count;
                let err = rel_rms(&got, &want);
                println!(
                    "{} {count} tokens to {at}: logits rel rms {err:.2e}, argmax {} vs {}",
                    if use_prefill { "prefill" } else { "decode" },
                    argmax(&got),
                    argmax(&want)
                );
                // Prefill and decode round slightly different keys to half,
                // which can flip a near tie of the router (2e-3 to 3e-2 on
                // these tokens; 1e-6 with an f32 cache). A bug is order 1.
                assert!(err < 5e-2);
                assert_eq!(argmax(&got), argmax(&want));
            }
        }

        /// verify gives each row the logits of decode fed one token at a
        /// time, and restore of a middle row leaves the states decode had
        /// there.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn verify_and_restore_match_decode() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let capacity = 64;
            let n = cfg.num_layers();
            let mut decode = GpuModel::load(&gpu, &cfg, &weights, n, capacity).unwrap();
            let mut spec = GpuModel::load(&gpu, &cfg, &weights, n, capacity).unwrap();
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 64).unwrap();
            prefill.enable_verify(&gpu, 5);
            let tokens = test_tokens(26);
            let (prompt, draft) = tokens[..25].split_at(20);

            prefill.run(&gpu, prompt, 0, &spec.states().collect::<Vec<_>>());
            spec.set_position(20);
            let v = prefill.verify(&gpu, draft, 20, &spec.states().collect::<Vec<_>>(), None);
            let logits = contents::<f32>(&v.logits.buffer, draft.len() * cfg.vocab_size);
            for &t in prompt {
                decode.forward(t);
            }
            let mut want = Vec::new();
            for (i, &t) in draft.iter().enumerate() {
                want.push(decode.forward(t));
                let got = &logits[i * cfg.vocab_size..(i + 1) * cfg.vocab_size];
                let err = rel_rms(got, &want[i]);
                println!(
                    "verify row {i}: logits rel rms {err:.2e}, argmax {} vs {}",
                    v.argmax[i],
                    argmax(&want[i])
                );
                // As in decode_after_prefill_matches_gpu_model.
                assert!(err < 5e-2);
                assert_eq!(v.argmax[i], argmax(&want[i]));
            }
            assert_eq!(v.argmax[0], argmax(got_row(logits, 0, cfg.vocab_size)));

            // Accept rows 0..=2, then decode the next token from there.
            prefill.restore(&gpu, 2, &spec.states().collect::<Vec<_>>());
            spec.set_position(23);
            let mut again = GpuModel::load(&gpu, &cfg, &weights, n, capacity).unwrap();
            for &t in &tokens[..23] {
                again.forward(t);
            }
            let want = again.forward(tokens[25]);
            let got = spec.forward(tokens[25]);
            let err = rel_rms(&got, &want);
            println!("after restore of row 2: logits rel rms {err:.2e}");
            assert!(err < 5e-2);
            assert_eq!(argmax(&got), argmax(&want));
        }

        fn got_row(x: &[f32], i: usize, n: usize) -> &[f32] {
            &x[i * n..(i + 1) * n]
        }

        fn read_f32(path: &std::path::Path) -> Vec<f32> {
            let b = std::fs::read(path).unwrap();
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }

        /// KL(p || q) of two logit vectors, in nats.
        fn kl(p: &[f32], q: &[f32]) -> f64 {
            let log_softmax = |x: &[f32]| -> Vec<f64> {
                let m = x.iter().copied().fold(f32::NEG_INFINITY, f32::max) as f64;
                let s: f64 = x.iter().map(|&v| (v as f64 - m).exp()).sum();
                x.iter().map(|&v| v as f64 - m - s.ln()).collect()
            };
            let (lp, lq) = (log_softmax(p), log_softmax(q));
            lp.iter().zip(&lq).map(|(a, b)| a.exp() * (a - b)).sum()
        }

        /// The 32k eval prompt of track F, prefilled in pieces that end at
        /// its checkpoints: KL against MLX f32 at each, and the largest |x|
        /// going into the half kernels.
        #[test]
        #[ignore = "needs the model and ~/saetta/eval on the Mac"]
        fn long_prompt_against_mlx_f32() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let eval = PathBuf::from(std::env::var("HOME").unwrap()).join("saetta/eval");
            let tokens: Vec<u32> = std::fs::read(eval.join("long_tokens.bin"))
                .unwrap()
                .as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect();
            let exact = exact_from_env();
            let capacity = tokens.len().next_multiple_of(KEY_BLOCK);
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 1024).unwrap();
            prefill.exact = exact;
            prefill.track_ranges(&gpu);
            let owned = OwnedState::new_all(&gpu, &cfg, capacity);
            let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
            let mut done = 0;
            for end in [1024, 4096, 8192, 16384, 32768] {
                let start = Instant::now();
                let logits = prefill.run(&gpu, &tokens[done..end], done, &states);
                let secs = start.elapsed().as_secs_f64();
                let want = read_f32(&eval.join(format!("f32_logits_{}.bin", end - 1)));
                let k = kl(&want, &logits);
                println!(
                    "exact {exact}, tokens {done}..{end} in {secs:.1} s: KL vs MLX f32 at {} {k:.2e}, argmax {} vs {}",
                    end - 1,
                    argmax(&logits),
                    argmax(&want),
                );
                done = end;
            }
            for (name, v) in RANGES.iter().zip(prefill.ranges()) {
                println!("max |x| {name}: {v}");
            }
        }

        /// Times short prefills, one layer and its parts, to find costs that
        /// do not shrink with the prompt.
        #[test]
        #[ignore = "profiling"]
        fn fixed_cost() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let t0 = Instant::now();
            let p = Prefill::new(&gpu, &cfg, &weights, 2048).unwrap();
            println!("new: {:.0} ms", t0.elapsed().as_secs_f64() * 1e3);
            let owned = OwnedState::new_all(&gpu, &cfg, 4096);
            let states: Vec<State> = owned.iter().map(|s| s.state()).collect();
            let mut pos = 0;
            for n in [1usize, 1, 1, 17, 17, 64, 64] {
                let toks = test_tokens(n);
                let t = Instant::now();
                p.run(&gpu, &toks, pos, &states);
                println!("run {n}: {:.1} ms", t.elapsed().as_secs_f64() * 1e3);
                pos += n;
            }
            let x = p.embed(&gpu, &test_tokens(17));
            for l in [0usize, 3] {
                for _ in 0..2 {
                    let t = Instant::now();
                    let d = gpu.run(|enc| p.encode_layer(&gpu, enc, l, &x, 17, pos, &states[l]));
                    println!(
                        "layer {l} 17 tokens: wall {:.1} ms, commit to done {:.1} ms",
                        t.elapsed().as_secs_f64() * 1e3,
                        d.as_secs_f64() * 1e3
                    );
                }
            }
            let s = &p.scratch;
            let layer = &p.layers[0];
            let normed = whole(&s.normed);
            let time = |name: &str, f: &dyn Fn(&ComputeCommandEncoderRef)| {
                gpu.run(f);
                let d = gpu.run(f);
                println!("  {name}: {:.2} ms", d.as_secs_f64() * 1e3);
            };
            let xt = at(&x, 0);
            time("moe 17", &|enc| {
                p.encode_moe(&gpu, enc, layer, &normed, &xt, 17)
            });
            if let (Mixer::Linear(w), State::Linear(st)) = (&layer.mixer, &states[0]) {
                time("deltanet 17", &|enc| {
                    p.encode_deltanet(&gpu, enc, w, st, &normed, &whole(&s.mixer_out), 17, None)
                });
                time("qkv qmm 17", &|enc| {
                    qmm(&gpu, enc, &w.qkv, &normed, &whole(&s.wide), 17, true)
                });
            }
            time("norm 17", &|enc| {
                norm(
                    &gpu,
                    enc,
                    &Norm::rows(17, 2048, 1e-6, &layer.input_norm),
                    &xt,
                    &normed,
                )
            });
            time("empty", &|_enc| {});
        }
    }
}
