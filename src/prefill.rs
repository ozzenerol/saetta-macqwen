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
    use crate::moe::Experts;
    use crate::qmv::{GpuQuant, View, set_views};
    use crate::weights::Weights;
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
    use std::time::Instant;

    const PREFIX: &str = "language_model.model";
    /// Rows and columns of a qmm tile; must match MM_BM and MM_BN.
    const TILE: usize = 64;
    /// Key block of attention_rows; the KV cache needs this much slack.
    const KEY_BLOCK: usize = 32;

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

    #[repr(C)]
    struct MmShape {
        m: u32,
        n: u32,
        k: u32,
        gather: u32,
    }

    /// `split` picks the kernel that splits x into two halves, for close
    /// to f32 results.
    fn qmm_pipeline(gpu: &Gpu, w: &GpuQuant, split: bool) -> metal::ComputePipelineState {
        let name = match w.bits {
            5 => "qmm_5bit",
            8 => "qmm_8bit",
            b => panic!("no qmm kernel for {b} bits"),
        };
        gpu.pipeline(&if split {
            format!("{name}_split")
        } else {
            name.to_string()
        })
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
        assert_eq!(w.experts, 1);
        enc.set_compute_pipeline_state(&qmm_pipeline(gpu, w, split));
        w.set_buffers(enc, 0);
        set_views(enc, 3, &[x, y, x, x, x]);
        let shape = MmShape {
            m: m as u32,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: 0,
        };
        params(enc, 5, &shape);
        groups(enc, (w.rows.div_ceil(TILE), m.div_ceil(TILE)), 128);
    }

    /// The (token, expert) pairs of a chunk sorted by expert, from moe_group.
    struct Grouped<'v> {
        tiles: &'v View,
        rows: &'v View,
        tile_count: &'v View,
        max_tiles: usize,
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
        w.set_buffers(enc, 0);
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

    /// What a layer keeps between tokens, on the GPU.
    pub enum State {
        Linear(GpuDeltaNetState),
        Full(KvState),
    }

    /// The KV cache, and the half copy of it that fast prefill reads.
    pub struct KvState {
        pub cache: GpuKvCache,
        half_keys: Buffer,
        half_values: Buffer,
        /// Tokens copied to the half buffers.
        half_len: usize,
    }

    impl KvState {
        fn new(gpu: &Gpu, cfg: &AttentionConfig, capacity: usize) -> KvState {
            let size = capacity * cfg.kv_heads * cfg.head_dim * 2;
            KvState {
                cache: GpuKvCache::new(gpu, cfg, capacity),
                half_keys: gpu.buffer(size),
                half_values: gpu.buffer(size),
                half_len: 0,
            }
        }
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
        /// conv output, `[chunk, 8192]`
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
            let max_tiles = pairs.div_ceil(TILE) + cfg.num_experts;
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
    }

    #[repr(C)]
    struct DeltaRowsParams {
        k_heads: u32,
        v_heads: u32,
        v_dim: u32,
        tokens: u32,
        channels: u32,
        eps: f32,
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
        /// Use the split qmm kernels: slower, close to f32.
        pub exact: bool,
        /// Largest |x| seen at each of `RANGES`, when tracked.
        ranges: Option<Buffer>,
    }

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
            Ok(Prefill {
                cfg,
                attention,
                deltanet,
                embed: matrix(&format!("{PREFIX}.embed_tokens"))?,
                layers,
                norm: vector(gpu, weights, &format!("{PREFIX}.norm.weight"))?,
                lm_head: matrix("language_model.lm_head")?,
                scratch: Scratch::new(gpu, cfg, chunk),
                chunk,
                exact: false,
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

        /// Empty states with room for `capacity` tokens in the KV caches.
        pub fn new_states(&self, gpu: &Gpu, capacity: usize) -> Vec<State> {
            let capacity = capacity.next_multiple_of(KEY_BLOCK);
            self.cfg
                .layer_types
                .iter()
                .map(|kind| match kind {
                    LayerType::LinearAttention => {
                        State::Linear(GpuDeltaNetState::new(gpu, &self.deltanet))
                    }
                    LayerType::FullAttention => {
                        State::Full(KvState::new(gpu, &self.attention, capacity))
                    }
                })
                .collect()
        }

        /// Feeds `tokens` after what `states` already hold and returns the
        /// logits of the last token.
        pub fn run(&self, gpu: &Gpu, tokens: &[u32], states: &mut [State]) -> Vec<f32> {
            let x = self.embed(gpu, tokens);
            for (l, state) in states.iter_mut().enumerate() {
                self.layer(gpu, l, &x, tokens.len(), state);
            }
            self.logits(gpu, &x, tokens.len())
        }

        /// The hidden states `[tokens, hidden]` after the embedding.
        pub fn embed(&self, gpu: &Gpu, tokens: &[u32]) -> Buffer {
            let h = self.cfg.hidden_size;
            let x = gpu.buffer(tokens.len() * h * 4);
            let ids = gpu.buffer_from(tokens);
            let name = match self.embed.bits {
                5 => "embed_rows_5bit",
                8 => "embed_rows_8bit",
                b => panic!("no embed kernel for {b} bits"),
            };
            gpu.run(|enc| {
                enc.set_compute_pipeline_state(&gpu.pipeline(name));
                self.embed.set_buffers(enc, 0);
                set_views(enc, 3, &[&whole(&ids), &whole(&x)]);
                params(enc, 5, &EmbedParams { cols: h as u32 });
                enc.dispatch_threads(
                    MTLSize::new(h as u64, tokens.len() as u64, 1),
                    MTLSize::new(256, 1, 1),
                );
            });
            x
        }

        /// Runs layer `l` over the `n` hidden states in `x`, in place, in
        /// one command buffer.
        pub fn layer(&self, gpu: &Gpu, l: usize, x: &Buffer, n: usize, state: &mut State) {
            let pos = match state {
                State::Full(kv) => {
                    assert!(
                        kv.cache.len + n.next_multiple_of(KEY_BLOCK) <= kv.cache.capacity,
                        "KV cache is full"
                    );
                    kv.cache.len
                }
                State::Linear(_) => 0,
            };
            gpu.run(|enc| {
                if let (State::Full(kv), false) = (&*state, self.exact) {
                    // Tokens added by decode since the last prefill.
                    let size = self.attention.kv_heads * self.attention.head_dim;
                    for (f32s, halves) in [
                        (&kv.cache.keys, &kv.half_keys),
                        (&kv.cache.values, &kv.half_values),
                    ] {
                        to_half(
                            gpu,
                            enc,
                            &at(f32s, kv.half_len * size),
                            &at_half(halves, kv.half_len * size),
                            (pos - kv.half_len) * size,
                        );
                    }
                }
                for start in (0..n).step_by(self.chunk) {
                    let t = self.chunk.min(n - start);
                    self.encode_chunk(gpu, enc, l, x, start, t, pos + start, state);
                }
            });
            if let State::Full(kv) = state {
                kv.cache.len += n;
                if !self.exact {
                    kv.half_len = kv.cache.len;
                }
            }
        }

        /// The logits of the last of the `n` hidden states in `x`.
        pub fn logits(&self, gpu: &Gpu, x: &Buffer, n: usize) -> Vec<f32> {
            let h = self.cfg.hidden_size;
            let last = gpu.buffer(h * 4);
            let logits = gpu.buffer(self.lm_head.rows * 4);
            gpu.run(|enc| {
                Norm::rows(1, h, self.cfg.rms_norm_eps, &self.norm).encode(
                    gpu,
                    enc,
                    &at(x, (n - 1) * h),
                    &whole(&last),
                );
                self.lm_head
                    .encode(gpu, enc, &whole(&last), &whole(&logits));
            });
            contents::<f32>(&logits, self.lm_head.rows).to_vec()
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

            Norm::rows(t, h, eps, &layer.input_norm).encode(gpu, enc, &xt, &normed);
            self.record(gpu, enc, 0, &normed, t * h);
            match (&layer.mixer, state) {
                (Mixer::Linear(w), State::Linear(st)) => {
                    self.encode_deltanet(gpu, enc, w, st, &normed, &mixer_out, t)
                }
                (Mixer::Full(w), State::Full(kv)) => {
                    self.encode_attention(gpu, enc, w, kv, &normed, &mixer_out, t, pos)
                }
                _ => panic!("layer {l}: state does not match the layer type"),
            }
            dispatch_1d(
                enc,
                &gpu.pipeline("add_in_place"),
                &[(&xt.buffer, xt.offset), (&s.mixer_out, 0)],
                t * h,
            );
            Norm::rows(t, h, eps, &layer.post_norm).encode(gpu, enc, &xt, &normed);
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
        ) {
            let c = &self.deltanet;
            let s = &self.scratch;
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let [wide, mixed, z, y, a, b] = [&s.wide, &s.mixed, &s.z, &s.y, &s.a, &s.b].map(whole);
            qmm(gpu, enc, &w.qkv, x, &wide, t, self.exact);
            qmm(gpu, enc, &w.z, x, &z, t, self.exact);
            qmm(gpu, enc, &w.b, x, &b, t, self.exact);
            qmm(gpu, enc, &w.a, x, &a, t, self.exact);

            enc.set_compute_pipeline_state(&gpu.pipeline("conv_rows"));
            set_views(enc, 0, &[&wide, &w.conv1d, &whole(&state.conv), &mixed]);
            let p = ConvRowsParams {
                channels: channels as u32,
                kernel_size: c.kernel as u32,
                tokens: t as u32,
            };
            params(enc, 4, &p);
            enc.dispatch_threads(MTLSize::new(channels as u64, 1, 1), MTLSize::new(256, 1, 1));

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
            let p = DeltaRowsParams {
                k_heads: c.k_heads as u32,
                v_heads: c.v_heads as u32,
                v_dim: c.v_dim as u32,
                tokens: t as u32,
                channels: channels as u32,
                eps: c.eps,
            };
            params(enc, 7, &p);
            // 2 simdgroups of 8 rows each per threadgroup.
            groups(enc, (c.v_dim.div_ceil(16), c.v_heads), 64);

            let mut gated = Norm::rows(t * c.v_heads, c.v_dim, c.eps, &w.norm);
            gated.gate = Some(&z);
            gated.encode(gpu, enc, &y, &y);
            self.record(gpu, enc, 1, &y, t * c.v_heads * c.v_dim);
            qmm(gpu, enc, &w.out, &y, out, t, self.exact);
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_attention(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            w: &AttentionWeights,
            kv: &KvState,
            x: &View,
            out: &View,
            t: usize,
            pos: usize,
        ) {
            let c = &self.attention;
            let s = &self.scratch;
            let d = c.head_dim;
            let kv_size = c.kv_heads * d;
            let cache = &kv.cache;
            let (q_and_gate, q, y) = (whole(&s.wide), whole(&s.z), whole(&s.y));
            let (keys, values) = (
                at(&cache.keys, pos * kv_size),
                at(&cache.values, pos * kv_size),
            );
            qmm(gpu, enc, &w.q, x, &q_and_gate, t, self.exact);
            qmm(gpu, enc, &w.k, x, &keys, t, self.exact);
            qmm(gpu, enc, &w.v, x, &values, t, self.exact);

            let mut norm = Norm::rows(t * c.heads, d, c.eps, &w.q_norm);
            norm.in_stride = 2 * d;
            norm.encode(gpu, enc, &q_and_gate, &q);
            Norm::rows(t * c.kv_heads, d, c.eps, &w.k_norm).encode(gpu, enc, &keys, &keys);
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

            if self.exact {
                enc.set_compute_pipeline_state(&gpu.pipeline("attention_rows"));
                set_views(enc, 0, &[&q, &whole(&cache.keys), &whole(&cache.values)]);
            } else {
                let q_half = whole(&s.q_half);
                to_half(gpu, enc, &q, &q_half, t * c.heads * d);
                let (hk, hv) = (
                    at_half(&kv.half_keys, pos * kv_size),
                    at_half(&kv.half_values, pos * kv_size),
                );
                to_half(gpu, enc, &keys, &hk, t * kv_size);
                to_half(gpu, enc, &values, &hv, t * kv_size);
                enc.set_compute_pipeline_state(&gpu.pipeline("attention_rows_half"));
                set_views(
                    enc,
                    0,
                    &[&q_half, &whole(&kv.half_keys), &whole(&kv.half_values)],
                );
            }
            set_views(enc, 3, &[&q_and_gate, &y]);
            let p = AttentionParams {
                heads: c.heads as u32,
                kv_heads: c.kv_heads as u32,
                pos: pos as u32,
                tokens: t as u32,
                scale: 1.0 / (d as f32).sqrt(),
            };
            params(enc, 5, &p);
            // 4 heads of one kv head per threadgroup.
            groups(enc, (t.div_ceil(8), c.heads / 4), 128);
            self.record(gpu, enc, 2, &y, t * c.heads * d);
            qmm(gpu, enc, &w.o, &y, out, t, self.exact);
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
            qmm(gpu, enc, &layer.router, normed, &router, t, self.exact);
            qmm(
                gpu,
                enc,
                &layer.shared_gate,
                normed,
                &shared_gate,
                t,
                self.exact,
            );
            qmm(gpu, enc, &layer.shared[0], normed, &sg, t, self.exact);
            qmm(gpu, enc, &layer.shared[1], normed, &su, t, self.exact);
            let silu_mul = gpu.pipeline("silu_mul");
            dispatch_1d(
                enc,
                &silu_mul,
                &[(&s.shared_g, 0), (&s.shared_u, 0)],
                t * layer.shared[0].rows,
            );
            self.record(gpu, enc, 3, &sg, t * layer.shared[0].rows);
            qmm(gpu, enc, &layer.shared[2], &sg, &sd, t, self.exact);

            let [ids, weights, order, rows, tiles, tile_count] = [
                &s.ids,
                &s.expert_weights,
                &s.order,
                &s.rows,
                &s.tiles,
                &s.tile_count,
            ]
            .map(whole);
            enc.set_compute_pipeline_state(&gpu.pipeline("moe_route"));
            set_views(enc, 0, &[&router, &ids, &weights]);
            let p = RouteParams {
                experts: cfg.num_experts as u32,
                top_k: top_k as u32,
            };
            params(enc, 3, &p);
            groups(enc, (t, 1), 32);

            enc.set_compute_pipeline_state(&gpu.pipeline("moe_group"));
            set_views(enc, 0, &[&ids, &order, &rows, &tiles, &tile_count]);
            let p = GroupParams {
                pairs: (t * top_k) as u32,
                top_k: top_k as u32,
                experts: cfg.num_experts as u32,
            };
            params(enc, 5, &p);
            groups(enc, (1, 1), 1024);

            let g = Grouped {
                tiles: &tiles,
                rows: &rows,
                tile_count: &tile_count,
                max_tiles: (t * top_k).div_ceil(TILE) + cfg.num_experts,
            };
            assert!(g.max_tiles <= s.max_tiles);
            let e = &layer.experts;
            let (eg, eu, eo) = (whole(&s.expert_g), whole(&s.expert_u), whole(&s.expert_out));
            qmm_grouped(gpu, enc, &e.gate, normed, &eg, &g, true, self.exact);
            qmm_grouped(gpu, enc, &e.up, normed, &eu, &g, true, self.exact);
            dispatch_1d(
                enc,
                &silu_mul,
                &[(&s.expert_g, 0), (&s.expert_u, 0)],
                t * top_k * e.size(),
            );
            self.record(gpu, enc, 4, &eg, t * top_k * e.size());
            qmm_grouped(gpu, enc, &e.down, &eg, &eo, &g, false, self.exact);

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

    /// Prompt tokens for tests and benchmarks: a fixed spread of ids.
    pub fn test_tokens(n: usize) -> Vec<u32> {
        (0..n).map(|i| ((i * 7919 + 13) % 150_000) as u32).collect()
    }

    /// Prefills `tokens` tokens `runs` times, after one warm-up run, and
    /// prints the best and median tokens per second.
    pub fn bench(
        dir: &std::path::Path,
        tokens: usize,
        chunk: usize,
        runs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let prefill = Prefill::new(&gpu, &cfg, &weights, chunk)?;
        let prompt = test_tokens(tokens);
        let mut times: Vec<f64> = (0..=runs)
            .map(|_| {
                let mut states = prefill.new_states(&gpu, tokens);
                let start = Instant::now();
                prefill.run(&gpu, &prompt, &mut states);
                start.elapsed().as_secs_f64()
            })
            .skip(1)
            .collect();
        times.sort_by(f64::total_cmp);
        println!(
            "prefill {tokens} tokens, chunk {}, {runs} runs: best {:.2} s ({:.0} tokens/s), median {:.2} s ({:.0} tokens/s)",
            prefill.chunk,
            times[0],
            tokens as f64 / times[0],
            times[runs / 2],
            tokens as f64 / times[runs / 2],
        );
        Ok(())
    }

    #[cfg(test)]
    mod tests {
        //! Prefill against the CPU model fed one token at a time:
        //!   scripts/mac.sh test --release --lib prefill -- --ignored --nocapture

        use super::*;
        use crate::cpu::matrix::Matrix;
        use crate::cpu::model::{LayerState, Model};
        use crate::cpu::ops::rms_norm;
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
                for (split, bound) in [(true, 1e-5), (false, 2e-3)] {
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

        /// With the split kernels, prefill is within f32 rounding of the CPU.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn exact_prefill_matches_token_by_token() {
            check_prefill(true, 2e-5, 1e-4);
        }

        /// With half inputs the error is that of rounding x to half, far
        /// below the bf16 MLX server's (1e-2 after a few layers, see the
        /// golden test in src/cpu/model.rs).
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn fast_prefill_matches_token_by_token() {
            check_prefill(false, 5e-3, 1e-2);
        }

        /// 45 tokens in two calls (29 then 16) with chunks of 64 rounded
        /// from 20: the hidden state after each layer and every layer state
        /// must be within `bound` of the CPU fed one token at a time, the
        /// logits within `logits_bound`, with the same argmax.
        fn check_prefill(exact: bool, bound: f32, logits_bound: f32) {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let model = Model::new(&cfg, &weights).unwrap();
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 20).unwrap();
            prefill.exact = exact;
            let tokens = test_tokens(45);
            let parts = [&tokens[..29], &tokens[29..]];
            let h = cfg.hidden_size;

            let mut cpu_states = model.new_states();
            let mut gpu_states = prefill.new_states(&gpu, tokens.len());
            let mut cpu_x: Vec<Vec<Vec<f32>>> = parts
                .iter()
                .map(|p| p.iter().map(|&t| model.embed(t)).collect())
                .collect();
            let gpu_x: Vec<Buffer> = parts.iter().map(|p| prefill.embed(&gpu, p)).collect();
            for l in 0..cfg.num_layers() {
                let mut err = 0.0f32;
                for (i, part) in parts.iter().enumerate() {
                    for x in cpu_x[i].iter_mut() {
                        *x = model.layer(l).step(x, &mut cpu_states[l]).unwrap().hidden;
                    }
                    prefill.layer(&gpu, l, &gpu_x[i], part.len(), &mut gpu_states[l]);
                    let got = contents::<f32>(&gpu_x[i], part.len() * h);
                    err = err.max(rel_rms(got, &cpu_x[i].concat()));
                }
                let state = match (&cpu_states[l], &gpu_states[l]) {
                    (LayerState::Linear(c), State::Linear(g)) => {
                        let conv = rel_rms(contents::<f32>(&g.conv, c.conv.len()), &c.conv);
                        let rec = rel_rms(
                            contents::<f32>(&g.recurrent, c.recurrent.len()),
                            &c.recurrent,
                        );
                        conv.max(rec)
                    }
                    (LayerState::Full(c), State::Full(kv)) => {
                        let g = &kv.cache;
                        assert_eq!(c.len(), g.len);
                        let n = c.keys().len();
                        let k = rel_rms(contents::<f32>(&g.keys, n), c.keys());
                        let v = rel_rms(contents::<f32>(&g.values, n), c.values());
                        k.max(v)
                    }
                    _ => panic!("layer {l}: state types differ"),
                };
                println!("layer {l}: hidden rel rms {err:.2e}, state {state:.2e}");
                assert!(err < bound && state < bound, "layer {l}");
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
            assert!(err < logits_bound);
            assert_eq!(
                crate::cpu::model::argmax(&got),
                crate::cpu::model::argmax(&want)
            );
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
            let exact = std::env::var("SAETTA_EXACT").is_ok();
            let mut prefill = Prefill::new(&gpu, &cfg, &weights, 1024).unwrap();
            prefill.exact = exact;
            prefill.track_ranges(&gpu);
            let mut states = prefill.new_states(&gpu, tokens.len());
            let mut done = 0;
            for end in [1024, 4096, 8192, 16384, 32768] {
                let start = Instant::now();
                let logits = prefill.run(&gpu, &tokens[done..end], &mut states);
                let secs = start.elapsed().as_secs_f64();
                let want = read_f32(&eval.join(format!("f32_logits_{}.bin", end - 1)));
                let k = kl(&want, &logits);
                println!(
                    "exact {exact}, tokens {done}..{end} in {secs:.1} s: KL vs MLX f32 at {} {k:.2e}, argmax {} vs {}",
                    end - 1,
                    crate::cpu::model::argmax(&logits),
                    crate::cpu::model::argmax(&want),
                );
                done = end;
            }
            for (name, v) in RANGES.iter().zip(prefill.ranges()) {
                println!("max |x| {name}: {v}");
            }
        }

        #[test]
        #[ignore = "profiling"]
        fn profile_parts() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let p = Prefill::new(&gpu, &cfg, &weights, 1024).unwrap();
            let t = 1024;
            let tokens = test_tokens(t);
            let x = p.embed(&gpu, &tokens);
            let mut states = p.new_states(&gpu, t);
            for l in [0usize, 3] {
                for _ in 0..3 {
                    let mut st = p.new_states(&gpu, t).remove(l);
                    let s = Instant::now();
                    p.layer(&gpu, l, &x, t, &mut st);
                    println!("layer {l}: {:.1} ms", s.elapsed().as_secs_f64() * 1e3);
                }
            }
            let layer = &p.layers[0];
            let s = &p.scratch;
            let normed = whole(&s.normed);
            let time = |name: &str, f: &dyn Fn(&ComputeCommandEncoderRef)| {
                gpu.run(f);
                let d = gpu.run(f);
                println!("{name}: {:.2} ms", d.as_secs_f64() * 1e3);
            };
            if let (Mixer::Linear(w), State::Linear(st)) = (&layer.mixer, &states[0]) {
                time("qkv qmm", &|enc| {
                    qmm(&gpu, enc, &w.qkv, &normed, &whole(&s.wide), t, false)
                });
                time("deltanet all", &|enc| {
                    p.encode_deltanet(&gpu, enc, w, st, &normed, &whole(&s.mixer_out), t)
                });
            }
            if let (Mixer::Full(w), State::Full(c)) = (&p.layers[3].mixer, &mut states[3]) {
                c.cache.len = 0;
                time("attention all", &|enc| {
                    p.encode_attention(&gpu, enc, w, c, &normed, &whole(&s.mixer_out), t, 0)
                });
                time("q qmm", &|enc| {
                    qmm(&gpu, enc, &w.q, &normed, &whole(&s.wide), t, false)
                });
            }
            let xt = at(&x, 0);
            time("moe all", &|enc| {
                p.encode_moe(&gpu, enc, layer, &normed, &xt, t)
            });
            time("shared gate qmm", &|enc| {
                qmm(
                    &gpu,
                    enc,
                    &layer.shared[0],
                    &normed,
                    &whole(&s.shared_g),
                    t,
                    false,
                )
            });
        }
    }
}
