//! The attention and Gated DeltaNet layers and RMSNorm on the GPU
//! (`kernels/layers.metal`), for one token. Each `record` adds its
//! dispatches to a `Plan` once; the plan then runs for every token, with the
//! position taken from the plan's `TokenParams`. Projections use `GpuQuant`
//! from `qmv.rs`.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::attention::DecodeAttention;
    use crate::config::Config;
    use crate::cpu::attention::AttentionConfig;
    use crate::cpu::deltanet::DeltaNetConfig;
    use crate::cpu::model::{attention_config, deltanet_config};
    use crate::gpu::Gpu;
    use crate::plan::Plan;
    use crate::qmv::{GpuQuant, View};
    use crate::rows;
    use crate::weights::Weights;
    use metal::{Buffer, MTLSize};

    fn whole(buffer: &Buffer) -> View {
        View::new(buffer.clone())
    }

    fn size(x: usize, y: usize) -> MTLSize {
        MTLSize::new(x as u64, y as u64, 1)
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct NormParams {
        dim: u32,
        in_stride: u32,
        out_stride: u32,
        eps: f32,
        scale: f32,
        weighted: u32,
        gated: u32,
    }

    /// Options of one `rms_norm` dispatch over `rows` rows of `dim` values.
    pub struct Norm<'v> {
        pub rows: usize,
        pub dim: usize,
        pub in_stride: usize,
        pub out_stride: usize,
        pub eps: f32,
        pub scale: f32,
        pub weight: Option<&'v View>,
        pub gate: Option<&'v View>,
    }

    impl Norm<'_> {
        /// Rows packed one after another, weighted, scale 1.
        pub fn rows<'v>(rows: usize, dim: usize, eps: f32, weight: &'v View) -> Norm<'v> {
            Norm {
                rows,
                dim,
                in_stride: dim,
                out_stride: dim,
                eps,
                scale: 1.0,
                weight: Some(weight),
                gate: None,
            }
        }

        /// `out = rms_norm(x)` as configured. `out` may be `x`.
        pub fn record(&self, plan: &mut Plan, x: &View, out: &View, parallel: bool) {
            let p = plan.constant(&[NormParams {
                dim: self.dim as u32,
                in_stride: self.in_stride as u32,
                out_stride: self.out_stride as u32,
                eps: self.eps,
                scale: self.scale,
                weighted: self.weight.is_some() as u32,
                gated: self.gate.is_some() as u32,
            }]);
            let buffers = [x, self.weight.unwrap_or(x), self.gate.unwrap_or(x), out, &p];
            let threads = self.dim.next_multiple_of(32).min(256);
            let (grid, group) = (size(self.rows, 1), size(threads, 1));
            if parallel {
                plan.op_parallel("rms_norm", &buffers, grid, group);
            } else {
                plan.op("rms_norm", &buffers, grid, group);
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct PrepParams {
        heads: u32,
        kv_heads: u32,
        dim: u32,
        half_dims: u32,
        log2_theta: f32,
        eps: f32,
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct ResidualParams {
        dim: u32,
        eps: f32,
        logit_stride: u32,
    }

    /// Records `x += y; out = rms_norm(x) * w` as one op, for buffers
    /// `[x, y, w, out]` of `dim` values.
    pub fn add_norm(plan: &mut Plan, buffers: [&View; 4], dim: usize, eps: f32) {
        residual(plan, "add_norm", &buffers, 1, dim, eps, 0);
    }

    /// `add_norm` over `rows` rows of x, y and out.
    pub fn add_norm_rows(plan: &mut Plan, buffers: [&View; 4], rows: usize, dim: usize, eps: f32) {
        residual(plan, "add_norm", &buffers, rows, dim, eps, 0);
    }

    /// Records `x += sigmoid(shared_logit[0]) * shared + routed;
    /// out = rms_norm(x) * w` as one op, for buffers
    /// `[x, shared_logit, shared, routed, w, out]`.
    pub fn mix_norm(plan: &mut Plan, buffers: [&View; 6], dim: usize, eps: f32) {
        residual(plan, "mix_norm", &buffers, 1, dim, eps, 0);
    }

    /// `mix_norm` over `rows` rows of x, shared, routed and out; row r's
    /// shared logit is `shared_logit[r * logit_stride]`.
    pub fn mix_norm_rows(
        plan: &mut Plan,
        buffers: [&View; 6],
        rows: usize,
        dim: usize,
        eps: f32,
        logit_stride: usize,
    ) {
        residual(plan, "mix_norm", &buffers, rows, dim, eps, logit_stride);
    }

    fn residual(
        plan: &mut Plan,
        kernel: &str,
        buffers: &[&View],
        rows: usize,
        dim: usize,
        eps: f32,
        logit_stride: usize,
    ) {
        let p = plan.constant(&[ResidualParams {
            dim: dim as u32,
            eps,
            logit_stride: logit_stride as u32,
        }]);
        let mut all = buffers.to_vec();
        all.push(&p);
        plan.op(kernel, &all, size(rows, 1), size(dim.min(1024), 1));
    }

    fn vector(gpu: &Gpu, weights: &Weights, name: &str) -> Result<View, String> {
        Ok(View::new(gpu.buffer_from(&weights.tensor(name)?.to_f32())))
    }

    fn matrix<'a>(
        gpu: &Gpu,
        weights: &'a Weights,
        cfg: &Config,
        module: &str,
    ) -> Result<GpuQuant<'a>, String> {
        Ok(GpuQuant::new(
            gpu,
            &weights.quantized(module, &cfg.quantization)?,
        ))
    }

    /// The keys (after norm and rope) and values on the GPU, each token
    /// `[kv_heads * head_dim]`, room for `capacity` tokens. Token `p` goes in
    /// slot `p`.
    pub struct GpuKvCache {
        pub keys: Buffer,
        pub values: Buffer,
        decode: DecodeAttention,
        pub capacity: usize,
    }

    impl GpuKvCache {
        pub fn new(gpu: &Gpu, cfg: &AttentionConfig, capacity: usize) -> GpuKvCache {
            let token = cfg.kv_heads * cfg.head_dim * 4;
            // attention_partial_rows reads whole chunks of 128 tokens.
            let room = capacity.next_multiple_of(128);
            GpuKvCache {
                keys: gpu.buffer(room * token),
                values: gpu.buffer(room * token),
                decode: DecodeAttention::new(gpu, cfg.heads, cfg.kv_heads, cfg.head_dim, capacity),
                capacity,
            }
        }
    }

    /// Full attention with q/k norm, partial RoPE, GQA and the output gate.
    pub struct GpuAttention<'a> {
        pub cfg: AttentionConfig,
        q_proj: GpuQuant<'a>,
        k_proj: GpuQuant<'a>,
        v_proj: GpuQuant<'a>,
        o_proj: GpuQuant<'a>,
        q_norm: View,
        k_norm: View,
        q_and_gate: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        out: Buffer,
    }

    impl<'a> GpuAttention<'a> {
        /// `prefix` is the layer, such as `language_model.model.layers.3`.
        pub fn load(
            gpu: &Gpu,
            cfg: &Config,
            weights: &'a Weights,
            prefix: &str,
        ) -> Result<GpuAttention<'a>, String> {
            let a = attention_config(cfg);
            if a.head_dim > 1024 || a.rope_dims > a.head_dim {
                return Err("attention_prep takes head_dim up to 1024".into());
            }
            let size = a.heads * a.head_dim;
            let kv = a.kv_heads * a.head_dim;
            let m = |n: &str| matrix(gpu, weights, cfg, &format!("{prefix}.self_attn.{n}"));
            Ok(GpuAttention {
                q_proj: m("q_proj")?,
                k_proj: m("k_proj")?,
                v_proj: m("v_proj")?,
                o_proj: m("o_proj")?,
                q_norm: vector(gpu, weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: vector(gpu, weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
                q_and_gate: gpu.buffer(2 * size * 4),
                q: gpu.buffer(size * 4),
                k: gpu.buffer(kv * 4),
                v: gpu.buffer(kv * 4),
                out: gpu.buffer(size * 4),
                cfg: a,
            })
        }

        /// Records one token: `x` is the normed hidden state, `out` gets the
        /// layer output, and the token's key and value go into `cache` at
        /// its position.
        pub fn record(&self, plan: &mut Plan, x: &View, out: &View, cache: &GpuKvCache) {
            let c = &self.cfg;
            let d = c.head_dim;
            let [q_and_gate, q, k, v, attn] =
                [&self.q_and_gate, &self.q, &self.k, &self.v, &self.out].map(whole);

            self.q_proj.record(plan, x, &q_and_gate, false);
            self.k_proj.record(plan, x, &k, true);
            self.v_proj.record(plan, x, &v, true);

            let (keys, values) = (whole(&cache.keys), whole(&cache.values));
            let p = plan.constant(&[PrepParams {
                heads: c.heads as u32,
                kv_heads: c.kv_heads as u32,
                dim: d as u32,
                half_dims: (c.rope_dims / 2) as u32,
                log2_theta: c.rope_theta.log2(),
                eps: c.eps,
            }]);
            let params = plan.params.clone();
            plan.op(
                "attention_prep",
                &[
                    &q_and_gate,
                    &k,
                    &v,
                    &self.q_norm,
                    &self.k_norm,
                    &q,
                    &keys,
                    &values,
                    &p,
                    &params,
                ],
                size(c.heads + c.kv_heads, 1),
                size(d, 1),
            );

            cache
                .decode
                .record(plan, &q, &keys, &values, &q_and_gate, &attn);

            self.o_proj.record(plan, &attn, out, false);
        }
    }

    /// The attention buffers of `MAX_ROWS` tokens, shared by all attention
    /// layers.
    pub struct AttentionRowsScratch {
        attention: rows::AttentionRows,
        q_and_gate: Buffer,
        q: Buffer,
        k: Buffer,
        v: Buffer,
        out: Buffer,
    }

    impl AttentionRowsScratch {
        /// For caches of `capacity` tokens and verifies of up to `rows`.
        pub fn new(gpu: &Gpu, cfg: &Config, capacity: usize, rows: usize) -> AttentionRowsScratch {
            let a = attention_config(cfg);
            let size = a.heads * a.head_dim * 4 * rows::MAX_ROWS;
            let kv = a.kv_heads * a.head_dim * 4 * rows::MAX_ROWS;
            AttentionRowsScratch {
                attention: rows::AttentionRows::new(gpu, a.heads, a.kv_heads, capacity, rows),
                q_and_gate: gpu.buffer(2 * size),
                q: gpu.buffer(size),
                k: gpu.buffer(kv),
                v: gpu.buffer(kv),
                out: gpu.buffer(size),
            }
        }
    }

    impl GpuAttention<'_> {
        /// Records `m` tokens at once: `x` and `out` are `[m, hidden]`, and
        /// row r takes its position from `params[r]` (TokenParams views),
        /// so it sees the keys of rows before it. Each weight and each chunk
        /// of the cache is read once for all rows.
        #[allow(clippy::too_many_arguments)]
        pub fn record_rows(
            &self,
            plan: &mut Plan,
            m: usize,
            x: &View,
            out: &View,
            cache: &GpuKvCache,
            scratch: &AttentionRowsScratch,
            params: &[View],
        ) {
            assert!(params.len() >= m);
            let c = &self.cfg;
            let d = c.head_dim;
            let s = scratch;
            let [q_and_gate, q, k, v, attn] = [&s.q_and_gate, &s.q, &s.k, &s.v, &s.out].map(whole);
            rows::record_proj(plan, &self.q_proj, m, x, &q_and_gate, false);
            rows::record_proj(plan, &self.k_proj, m, x, &k, true);
            rows::record_proj(plan, &self.v_proj, m, x, &v, true);

            let (keys, values) = (whole(&cache.keys), whole(&cache.values));
            let p = plan.constant(&[PrepParams {
                heads: c.heads as u32,
                kv_heads: c.kv_heads as u32,
                dim: d as u32,
                half_dims: (c.rope_dims / 2) as u32,
                log2_theta: c.rope_theta.log2(),
                eps: c.eps,
            }]);
            let (qs, kv) = (c.heads * d, c.kv_heads * d);
            let row = |b: &View, r: usize, floats: usize| View {
                buffer: b.buffer.clone(),
                offset: b.offset + (r * floats * 4) as u64,
            };
            // Each row's key and value go in at its own position, so the
            // rows' preps are independent of each other.
            for (r, row_params) in params.iter().enumerate().take(m) {
                let buffers = [
                    &row(&q_and_gate, r, 2 * qs),
                    &row(&k, r, kv),
                    &row(&v, r, kv),
                    &self.q_norm,
                    &self.k_norm,
                    &row(&q, r, qs),
                    &keys,
                    &values,
                    &p,
                    row_params,
                ];
                let (grid, group) = (size(c.heads + c.kv_heads, 1), size(d, 1));
                if r == 0 {
                    plan.op("attention_prep", &buffers, grid, group);
                } else {
                    plan.op_parallel("attention_prep", &buffers, grid, group);
                }
            }
            s.attention
                .record(plan, m, &q, &keys, &values, &q_and_gate, &attn, params);
            rows::record_proj(plan, &self.o_proj, m, &attn, out, false);
        }
    }

    /// The conv and recurrent state of one DeltaNet layer on the GPU.
    pub struct GpuDeltaNetState {
        /// `[kernel - 1, conv channels]`, oldest first.
        pub conv: Buffer,
        /// `[v_heads, v_dim, k_dim]`
        pub recurrent: Buffer,
    }

    impl GpuDeltaNetState {
        pub fn new(gpu: &Gpu, cfg: &DeltaNetConfig) -> GpuDeltaNetState {
            let channels = 2 * cfg.k_heads * cfg.k_dim + cfg.v_heads * cfg.v_dim;
            GpuDeltaNetState {
                conv: gpu.buffer((cfg.kernel - 1) * channels * 4),
                recurrent: gpu.buffer(cfg.v_heads * cfg.v_dim * cfg.k_dim * 4),
            }
        }
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct StepParams {
        k_heads: u32,
        v_heads: u32,
        v_dim: u32,
        kernel_size: u32,
        eps: f32,
        rows: u32,
        snapshots: u32,
    }

    /// Gated DeltaNet: projections, conv1d, q/k L2 norm, the delta rule,
    /// the gated RMSNorm and the output projection.
    pub struct GpuDeltaNet<'a> {
        pub cfg: DeltaNetConfig,
        in_proj_qkv: GpuQuant<'a>,
        in_proj_z: GpuQuant<'a>,
        in_proj_b: GpuQuant<'a>,
        in_proj_a: GpuQuant<'a>,
        out_proj: GpuQuant<'a>,
        conv1d: View,
        a_log: View,
        dt_bias: View,
        norm: View,
        qkv: Buffer,
        z: Buffer,
        a: Buffer,
        b: Buffer,
        y: Buffer,
    }

    impl<'a> GpuDeltaNet<'a> {
        pub fn load(
            gpu: &Gpu,
            cfg: &Config,
            weights: &'a Weights,
            prefix: &str,
        ) -> Result<GpuDeltaNet<'a>, String> {
            let c = deltanet_config(cfg);
            if c.k_dim != 128 || c.v_dim != 128 || c.v_heads != 2 * c.k_heads {
                return Err(
                    "deltanet_step is written for k_dim and v_dim 128, two value heads per key head"
                        .into(),
                );
            }
            let p = format!("{prefix}.linear_attn");
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let values = c.v_heads * c.v_dim;
            let m = |n: &str| matrix(gpu, weights, cfg, &format!("{p}.{n}"));
            let v = |n: &str| vector(gpu, weights, &format!("{p}.{n}"));
            Ok(GpuDeltaNet {
                in_proj_qkv: m("in_proj_qkv")?,
                in_proj_z: m("in_proj_z")?,
                in_proj_b: m("in_proj_b")?,
                in_proj_a: m("in_proj_a")?,
                out_proj: m("out_proj")?,
                conv1d: v("conv1d.weight")?,
                a_log: v("A_log")?,
                dt_bias: v("dt_bias")?,
                norm: v("norm.weight")?,
                qkv: gpu.buffer(channels * 4),
                z: gpu.buffer(values * 4),
                a: gpu.buffer(c.v_heads * 4),
                b: gpu.buffer(c.v_heads * 4),
                y: gpu.buffer(values * 4),
                cfg: c,
            })
        }

        /// Records one token: `x` is the normed hidden state, `out` gets the
        /// layer output, `state` is updated.
        pub fn record(&self, plan: &mut Plan, x: &View, out: &View, state: &GpuDeltaNetState) {
            let bufs = [&self.qkv, &self.z, &self.a, &self.b, &self.y].map(whole);
            self.record_with(plan, 1, x, out, state, &bufs, None);
        }

        /// Records `m` tokens one after the other, as `m` calls of `record`
        /// would: `x` and `out` are `[m, hidden]`. With `snaps`, the states
        /// after each row go there too (`GpuDeltaNetSnaps::restore`).
        #[allow(clippy::too_many_arguments)]
        pub fn record_rows(
            &self,
            plan: &mut Plan,
            m: usize,
            x: &View,
            out: &View,
            state: &GpuDeltaNetState,
            scratch: &DeltaNetRowsScratch,
            snaps: Option<&GpuDeltaNetSnaps>,
        ) {
            assert!(snaps.is_none_or(|s| m <= s.rows));
            let s = scratch;
            let bufs = [&s.qkv, &s.z, &s.a, &s.b, &s.y].map(|v| v.clone());
            self.record_with(plan, m, x, out, state, &bufs, snaps);
        }

        #[allow(clippy::too_many_arguments)]
        fn record_with(
            &self,
            plan: &mut Plan,
            m: usize,
            x: &View,
            out: &View,
            state: &GpuDeltaNetState,
            [qkv, z, a, b, y]: &[View; 5],
            snaps: Option<&GpuDeltaNetSnaps>,
        ) {
            let c = &self.cfg;
            let proj = |plan: &mut Plan, w: &GpuQuant, x: &View, y: &View, parallel| {
                if m == 1 {
                    w.record(plan, x, y, parallel);
                } else {
                    rows::record_proj(plan, w, m, x, y, parallel);
                }
            };
            proj(plan, &self.in_proj_qkv, x, qkv, false);
            proj(plan, &self.in_proj_z, x, z, true);
            proj(plan, &self.in_proj_b, x, b, true);
            proj(plan, &self.in_proj_a, x, a, true);

            let p = plan.constant(&[StepParams {
                k_heads: c.k_heads as u32,
                v_heads: c.v_heads as u32,
                v_dim: c.v_dim as u32,
                kernel_size: c.kernel as u32,
                eps: c.eps,
                rows: m as u32,
                snapshots: snaps.is_some() as u32,
            }]);
            let (conv, recurrent) = (whole(&state.conv), whole(&state.recurrent));
            // Without snapshots the last two buffers are never written.
            let (conv_snaps, state_snaps) = match snaps {
                Some(s) => (whole(&s.conv), whole(&s.recurrent)),
                None => (conv.clone(), recurrent.clone()),
            };
            plan.op(
                "deltanet_step",
                &[
                    qkv,
                    &self.conv1d,
                    &conv,
                    a,
                    b,
                    &self.a_log,
                    &self.dt_bias,
                    &recurrent,
                    z,
                    &self.norm,
                    y,
                    &p,
                    &conv_snaps,
                    &state_snaps,
                ],
                size(c.k_heads, 1),
                size(1024, 1),
            );

            proj(plan, &self.out_proj, y, out, false);
        }
    }

    /// The projections and step outputs of `MAX_ROWS` tokens, shared by
    /// all DeltaNet layers.
    pub struct DeltaNetRowsScratch {
        qkv: View,
        z: View,
        a: View,
        b: View,
        y: View,
    }

    impl DeltaNetRowsScratch {
        pub fn new(gpu: &Gpu, cfg: &Config) -> DeltaNetRowsScratch {
            let c = deltanet_config(cfg);
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let values = c.v_heads * c.v_dim;
            let v = |floats: usize| View::new(gpu.buffer(rows::MAX_ROWS * floats * 4));
            DeltaNetRowsScratch {
                qkv: v(channels),
                z: v(values),
                a: v(c.v_heads),
                b: v(c.v_heads),
                y: v(values),
            }
        }
    }

    /// The DeltaNet state after each of up to `rows` tokens of one layer.
    pub struct GpuDeltaNetSnaps {
        pub rows: usize,
        conv: Buffer,
        recurrent: Buffer,
    }

    impl GpuDeltaNetSnaps {
        /// About 2 MB per row.
        pub fn new(gpu: &Gpu, cfg: &DeltaNetConfig, rows: usize) -> GpuDeltaNetSnaps {
            let state = GpuDeltaNetState::new(gpu, cfg);
            let size = |b: &Buffer| rows * b.length() as usize;
            GpuDeltaNetSnaps {
                rows,
                conv: gpu.buffer(size(&state.conv)),
                recurrent: gpu.buffer(size(&state.recurrent)),
            }
        }

        /// Records copying the state after row `row` into `state`.
        pub fn record_restore(&self, plan: &mut Plan, row: usize, state: &GpuDeltaNetState) {
            for (from, to) in [
                (&self.conv, &state.conv),
                (&self.recurrent, &state.recurrent),
            ] {
                let n = to.length() as usize / 4;
                // copy_u32 has no bounds check.
                assert!(n.is_multiple_of(256));
                let src = View {
                    buffer: from.clone(),
                    offset: (row * n * 4) as u64,
                };
                crate::gpu_model::elementwise(plan, "copy_u32", &[&src, &whole(to)], n, true);
            }
        }
    }

    #[cfg(test)]
    mod tests {
        //! Each GPU layer against the CPU layer on real weights, run both
        //! from a serial encoder and replayed from the indirect command
        //! buffer, which must agree bit for bit:
        //!   scripts/mac.sh test --release --lib gpu_layers -- --ignored --nocapture

        use super::*;
        use crate::cpu::attention::{AttentionWeights, KvCache, attention};
        use crate::cpu::deltanet::{DeltaNetState, DeltaNetWeights, deltanet};
        use crate::cpu::matrix::Matrix;
        use crate::cpu::ops::rms_norm;
        use crate::gpu::contents;
        use crate::plan::TokenParams;
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";
        const TOKENS: usize = 6;
        /// Only the summation order differs from the CPU.
        const BOUND: f32 = 1e-5;

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        /// A hidden state of token `t` after input_layernorm of `prefix`.
        fn input(weights: &Weights, prefix: &str, t: usize, eps: f32) -> Vec<f32> {
            let mut x: Vec<f32> = (0..2048)
                .map(|i| ((i * 37 + t * 101) as f32 * 0.013).sin())
                .collect();
            let name = format!("{prefix}.input_layernorm.weight");
            rms_norm(&mut x, &weights.tensor(&name).unwrap().to_f32(), eps);
            x
        }

        fn write(buffer: &Buffer, data: &[f32]) {
            assert!(size_of_val(data) <= buffer.length() as usize);
            unsafe {
                std::ptr::copy_nonoverlapping(
                    data.as_ptr(),
                    buffer.contents() as *mut f32,
                    data.len(),
                )
            };
        }

        fn read(buffer: &Buffer, n: usize) -> Vec<f32> {
            contents::<f32>(buffer, n).to_vec()
        }

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            assert_eq!(got.len(), want.len());
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        fn bits(x: &[f32]) -> Vec<u32> {
            x.iter().map(|v| v.to_bits()).collect()
        }

        fn quant<'a>(weights: &'a Weights, cfg: &Config, module: &str) -> Matrix<'a> {
            Matrix::quant(weights.quantized(module, &cfg.quantization).unwrap())
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn rms_norm_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let name = "language_model.model.layers.0.input_layernorm.weight";
            let w = weights.tensor(name).unwrap().to_f32();
            let x: Vec<f32> = (0..2 * 2048)
                .map(|i| (i as f32 * 0.07).cos() * 3.0)
                .collect();
            let mut want = x.clone();
            for row in want.as_chunks_mut::<2048>().0 {
                rms_norm(row, &w, cfg.rms_norm_eps);
            }
            let (xv, wv, out) = (
                View::new(gpu.buffer_from(&x)),
                View::new(gpu.buffer_from(&w)),
                View::new(gpu.buffer(x.len() * 4)),
            );
            let mut plan = Plan::new(&gpu);
            Norm::rows(2, 2048, cfg.rms_norm_eps, &wv).record(&mut plan, &xv, &out, false);
            plan.run_serial();
            let serial = read(&out.buffer, x.len());
            plan.build().run(&gpu);
            assert_eq!(bits(&serial), bits(&read(&out.buffer, x.len())));
            let err = rel_rms(&serial, &want);
            println!("rms_norm: rel rms {err:.2e}");
            assert!(err < BOUND);
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn deltanet_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let prefix = "language_model.model.layers.0";
            let layer = GpuDeltaNet::load(&gpu, &cfg, &weights, prefix).unwrap();
            let x = View::new(gpu.buffer(cfg.hidden_size * 4));
            let mut runs = Vec::new();
            for _ in 0..2 {
                let state = GpuDeltaNetState::new(&gpu, &layer.cfg);
                let out = View::new(gpu.buffer(cfg.hidden_size * 4));
                let mut plan = Plan::new(&gpu);
                layer.record(&mut plan, &x, &out, &state);
                runs.push((plan, state, out));
            }
            let built = runs[1].0.build();
            println!("deltanet: {} ops", runs[0].0.len());

            let p = format!("{prefix}.linear_attn");
            let v = |n: &str| weights.tensor(&format!("{p}.{n}")).unwrap().to_f32();
            let (conv1d, a_log, dt_bias, norm) = (
                v("conv1d.weight"),
                v("A_log"),
                v("dt_bias"),
                v("norm.weight"),
            );
            let m = |n: &str| quant(&weights, &cfg, &format!("{p}.{n}"));
            let w = DeltaNetWeights {
                in_proj_qkv: m("in_proj_qkv"),
                in_proj_z: m("in_proj_z"),
                in_proj_b: m("in_proj_b"),
                in_proj_a: m("in_proj_a"),
                out_proj: m("out_proj"),
                conv1d: &conv1d,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm: &norm,
            };
            let c = deltanet_config(&cfg);
            let mut cpu_state = DeltaNetState::new(&c);

            let n = cfg.hidden_size;
            for t in 0..TOKENS {
                let input = input(&weights, prefix, t, cfg.rms_norm_eps);
                let want = deltanet(&c, &w, &input, &mut cpu_state);
                write(&x.buffer, &input);
                runs[0].0.set_params(TokenParams::new(0, t as u32));
                runs[0].0.run_serial();
                runs[1].0.set_params(TokenParams::new(0, t as u32));
                built.run(&gpu);
                let serial = read(&runs[0].2.buffer, n);
                assert_eq!(
                    bits(&serial),
                    bits(&read(&runs[1].2.buffer, n)),
                    "token {t}: replay"
                );
                let err = rel_rms(&serial, &want);
                println!("deltanet token {t}: rel rms {err:.2e}");
                assert!(err < BOUND, "token {t}");
            }
            let (conv_n, rec_n) = (cpu_state.conv.len(), cpu_state.recurrent.len());
            let state = &runs[0].1;
            let conv = rel_rms(&read(&state.conv, conv_n), &cpu_state.conv);
            let rec = rel_rms(&read(&state.recurrent, rec_n), &cpu_state.recurrent);
            println!("deltanet state: conv {conv:.2e}, recurrent {rec:.2e}");
            assert!(conv < BOUND && rec < BOUND);
            assert_eq!(
                bits(&read(&state.recurrent, rec_n)),
                bits(&read(&runs[1].1.recurrent, rec_n))
            );
        }

        /// TOKENS tokens as one `record_rows` against one `record` per
        /// token: outputs, the final state and every snapshot bit for bit,
        /// and a restore to each row.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn deltanet_rows_match_tokens() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let prefix = "language_model.model.layers.0";
            let layer = GpuDeltaNet::load(&gpu, &cfg, &weights, prefix).unwrap();
            let n = cfg.hidden_size;
            let inputs: Vec<Vec<f32>> = (0..TOKENS)
                .map(|t| input(&weights, prefix, t, cfg.rms_norm_eps))
                .collect();

            let state = GpuDeltaNetState::new(&gpu, &layer.cfg);
            let x = View::new(gpu.buffer(n * 4));
            let out = View::new(gpu.buffer(n * 4));
            let mut plan = Plan::new(&gpu);
            layer.record(&mut plan, &x, &out, &state);
            let built = plan.build();
            let mut want = Vec::new();
            let mut states = Vec::new();
            for input in &inputs {
                write(&x.buffer, input);
                built.run(&gpu);
                want.extend(read(&out.buffer, n));
                states.push((
                    bits(&read(&state.conv, state.conv.length() as usize / 4)),
                    bits(&read(
                        &state.recurrent,
                        state.recurrent.length() as usize / 4,
                    )),
                ));
            }

            let rows_state = GpuDeltaNetState::new(&gpu, &layer.cfg);
            let scratch = DeltaNetRowsScratch::new(&gpu, &cfg);
            let snaps = GpuDeltaNetSnaps::new(&gpu, &layer.cfg, TOKENS);
            let xs = View::new(gpu.buffer_from(&inputs.concat()));
            let outs = View::new(gpu.buffer(TOKENS * n * 4));
            let mut plan = Plan::new(&gpu);
            layer.record_rows(
                &mut plan,
                TOKENS,
                &xs,
                &outs,
                &rows_state,
                &scratch,
                Some(&snaps),
            );
            plan.build().run(&gpu);
            assert_eq!(
                bits(&read(&outs.buffer, TOKENS * n)),
                bits(&want),
                "outputs"
            );
            let last = &states[TOKENS - 1];
            let conv_n = rows_state.conv.length() as usize / 4;
            let rec_n = rows_state.recurrent.length() as usize / 4;
            assert_eq!(bits(&read(&rows_state.conv, conv_n)), last.0, "final conv");
            assert_eq!(
                bits(&read(&rows_state.recurrent, rec_n)),
                last.1,
                "final recurrent"
            );
            for (row, (conv, rec)) in states.iter().enumerate() {
                let mut plan = Plan::new(&gpu);
                snaps.record_restore(&mut plan, row, &rows_state);
                plan.build().run(&gpu);
                assert_eq!(
                    &bits(&read(&rows_state.conv, conv_n)),
                    conv,
                    "row {row} conv"
                );
                assert_eq!(
                    &bits(&read(&rows_state.recurrent, rec_n)),
                    rec,
                    "row {row} recurrent"
                );
            }
            println!("deltanet rows: {TOKENS} rows equal to {TOKENS} tokens, restores equal");
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn attention_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let prefix = "language_model.model.layers.3";
            let layer = GpuAttention::load(&gpu, &cfg, &weights, prefix).unwrap();
            let x = View::new(gpu.buffer(cfg.hidden_size * 4));
            let mut runs = Vec::new();
            for _ in 0..2 {
                let cache = GpuKvCache::new(&gpu, &layer.cfg, 64);
                let out = View::new(gpu.buffer(cfg.hidden_size * 4));
                let mut plan = Plan::new(&gpu);
                layer.record(&mut plan, &x, &out, &cache);
                runs.push((plan, cache, out));
            }
            let built = runs[1].0.build();
            println!("attention: {} ops", runs[0].0.len());

            let p = format!("{prefix}.self_attn");
            let v = |n: &str| weights.tensor(&format!("{p}.{n}")).unwrap().to_f32();
            let (q_norm, k_norm) = (v("q_norm.weight"), v("k_norm.weight"));
            let m = |n: &str| quant(&weights, &cfg, &format!("{p}.{n}"));
            let w = AttentionWeights {
                q_proj: m("q_proj"),
                k_proj: m("k_proj"),
                v_proj: m("v_proj"),
                o_proj: m("o_proj"),
                q_norm: &q_norm,
                k_norm: &k_norm,
            };
            let c = attention_config(&cfg);
            let mut cpu_cache = KvCache::default();

            let n = cfg.hidden_size;
            for t in 0..TOKENS {
                let input = input(&weights, prefix, t, cfg.rms_norm_eps);
                let want = attention(&c, &w, &input, &mut cpu_cache);
                write(&x.buffer, &input);
                runs[0].0.set_params(TokenParams::new(0, t as u32));
                runs[0].0.run_serial();
                runs[1].0.set_params(TokenParams::new(0, t as u32));
                built.run(&gpu);
                let serial = read(&runs[0].2.buffer, n);
                assert_eq!(
                    bits(&serial),
                    bits(&read(&runs[1].2.buffer, n)),
                    "token {t}: replay"
                );
                let err = rel_rms(&serial, &want);
                println!("attention token {t}: rel rms {err:.2e}");
                assert!(err < BOUND, "token {t}");
            }
            let kv_n = cpu_cache.keys().len();
            let cache = &runs[0].1;
            let keys = rel_rms(&read(&cache.keys, kv_n), cpu_cache.keys());
            let values = rel_rms(&read(&cache.values, kv_n), cpu_cache.values());
            println!("attention cache: keys {keys:.2e}, values {values:.2e}");
            assert!(keys < BOUND && values < BOUND);
        }
    }
}
