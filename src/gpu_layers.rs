//! The attention and Gated DeltaNet layers and RMSNorm on the GPU
//! (`kernels/layers.metal`), for one token. Each `encode` only adds
//! dispatches to the caller's encoder, so a whole token can go in one
//! command buffer. Projections use `GpuQuant` from `qmv.rs`.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::Config;
    use crate::cpu::attention::AttentionConfig;
    use crate::cpu::deltanet::DeltaNetConfig;
    use crate::cpu::model::{attention_config, deltanet_config};
    use crate::gpu::Gpu;
    use crate::qmv::{GpuQuant, View};
    use crate::weights::Weights;
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

    /// A view `floats` values into `buffer`.
    fn at(buffer: &Buffer, floats: usize) -> View {
        View {
            buffer: buffer.clone(),
            offset: (floats * 4) as u64,
        }
    }

    fn bind(enc: &ComputeCommandEncoderRef, views: &[&View]) {
        for (i, v) in views.iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
    }

    fn params<T>(enc: &ComputeCommandEncoderRef, index: u64, p: &T) {
        enc.set_bytes(index, size_of::<T>() as u64, (p as *const T).cast());
    }

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
        pub fn encode(&self, gpu: &Gpu, enc: &ComputeCommandEncoderRef, x: &View, out: &View) {
            enc.set_compute_pipeline_state(&gpu.pipeline("rms_norm"));
            bind(
                enc,
                &[x, self.weight.unwrap_or(x), self.gate.unwrap_or(x), out],
            );
            let p = NormParams {
                dim: self.dim as u32,
                in_stride: self.in_stride as u32,
                out_stride: self.out_stride as u32,
                eps: self.eps,
                scale: self.scale,
                weighted: self.weight.is_some() as u32,
                gated: self.gate.is_some() as u32,
            };
            params(enc, 4, &p);
            let threads = self.dim.next_multiple_of(32).min(256) as u64;
            enc.dispatch_thread_groups(
                MTLSize::new(self.rows as u64, 1, 1),
                MTLSize::new(threads, 1, 1),
            );
        }
    }

    #[repr(C)]
    struct RopeParams {
        stride: u32,
        half_dims: u32,
        pos: u32,
        log2_theta: f32,
    }

    fn rope(
        gpu: &Gpu,
        enc: &ComputeCommandEncoderRef,
        x: &View,
        rows: usize,
        cfg: &AttentionConfig,
        pos: usize,
    ) {
        enc.set_compute_pipeline_state(&gpu.pipeline("rope"));
        bind(enc, &[x]);
        let half = cfg.rope_dims / 2;
        let p = RopeParams {
            stride: cfg.head_dim as u32,
            half_dims: half as u32,
            pos: pos as u32,
            log2_theta: cfg.rope_theta.log2(),
        };
        params(enc, 1, &p);
        enc.dispatch_threads(
            MTLSize::new(half as u64, rows as u64, 1),
            MTLSize::new(half.min(64) as u64, 1, 1),
        );
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
    /// `[kv_heads * head_dim]`, room for `capacity` tokens.
    pub struct GpuKvCache {
        pub keys: Buffer,
        pub values: Buffer,
        scores: Buffer,
        pub capacity: usize,
        pub len: usize,
    }

    impl GpuKvCache {
        pub fn new(gpu: &Gpu, cfg: &AttentionConfig, capacity: usize) -> GpuKvCache {
            let token = cfg.kv_heads * cfg.head_dim * 4;
            GpuKvCache {
                keys: gpu.buffer(capacity * token),
                values: gpu.buffer(capacity * token),
                scores: gpu.buffer(cfg.heads * capacity * 4),
                capacity,
                len: 0,
            }
        }
    }

    #[repr(C)]
    struct AttentionParams {
        heads: u32,
        kv_heads: u32,
        dim: u32,
        len: u32,
        capacity: u32,
        scale: f32,
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
            let size = a.heads * a.head_dim;
            Ok(GpuAttention {
                q_proj: matrix(gpu, weights, cfg, &format!("{prefix}.self_attn.q_proj"))?,
                k_proj: matrix(gpu, weights, cfg, &format!("{prefix}.self_attn.k_proj"))?,
                v_proj: matrix(gpu, weights, cfg, &format!("{prefix}.self_attn.v_proj"))?,
                o_proj: matrix(gpu, weights, cfg, &format!("{prefix}.self_attn.o_proj"))?,
                q_norm: vector(gpu, weights, &format!("{prefix}.self_attn.q_norm.weight"))?,
                k_norm: vector(gpu, weights, &format!("{prefix}.self_attn.k_norm.weight"))?,
                q_and_gate: gpu.buffer(2 * size * 4),
                q: gpu.buffer(size * 4),
                out: gpu.buffer(size * 4),
                cfg: a,
            })
        }

        /// Encodes one token: `x` is the normed hidden state, `out` gets the
        /// layer output. Appends the token to `cache`.
        pub fn encode(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &View,
            out: &View,
            cache: &mut GpuKvCache,
        ) {
            let c = &self.cfg;
            let (d, pos) = (c.head_dim, cache.len);
            assert!(pos < cache.capacity, "KV cache is full");
            let slot = pos * c.kv_heads * d;
            let q_and_gate = View::new(self.q_and_gate.clone());
            let q = View::new(self.q.clone());
            let key = at(&cache.keys, slot);

            self.q_proj.encode(gpu, enc, x, &q_and_gate);
            self.k_proj.encode(gpu, enc, x, &key);
            self.v_proj.encode(gpu, enc, x, &at(&cache.values, slot));

            let mut norm = Norm::rows(c.heads, d, c.eps, &self.q_norm);
            norm.in_stride = 2 * d;
            norm.encode(gpu, enc, &q_and_gate, &q);
            rope(gpu, enc, &q, c.heads, c, pos);
            Norm::rows(c.kv_heads, d, c.eps, &self.k_norm).encode(gpu, enc, &key, &key);
            rope(gpu, enc, &key, c.kv_heads, c, pos);

            enc.set_compute_pipeline_state(&gpu.pipeline("attention"));
            let attn = View::new(self.out.clone());
            bind(
                enc,
                &[
                    &q,
                    &View::new(cache.keys.clone()),
                    &View::new(cache.values.clone()),
                    &q_and_gate,
                    &attn,
                    &View::new(cache.scores.clone()),
                ],
            );
            let p = AttentionParams {
                heads: c.heads as u32,
                kv_heads: c.kv_heads as u32,
                dim: d as u32,
                len: (pos + 1) as u32,
                capacity: cache.capacity as u32,
                scale: 1.0 / (d as f32).sqrt(),
            };
            params(enc, 6, &p);
            enc.dispatch_thread_groups(MTLSize::new(c.heads as u64, 1, 1), MTLSize::new(256, 1, 1));

            self.o_proj.encode(gpu, enc, &attn, out);
            cache.len += 1;
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
    struct ConvParams {
        channels: u32,
        kernel_size: u32,
    }

    #[repr(C)]
    struct DeltaParams {
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
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
        mixed: Buffer,
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
            let p = format!("{prefix}.linear_attn");
            let channels = 2 * c.k_heads * c.k_dim + c.v_heads * c.v_dim;
            let values = c.v_heads * c.v_dim;
            Ok(GpuDeltaNet {
                in_proj_qkv: matrix(gpu, weights, cfg, &format!("{p}.in_proj_qkv"))?,
                in_proj_z: matrix(gpu, weights, cfg, &format!("{p}.in_proj_z"))?,
                in_proj_b: matrix(gpu, weights, cfg, &format!("{p}.in_proj_b"))?,
                in_proj_a: matrix(gpu, weights, cfg, &format!("{p}.in_proj_a"))?,
                out_proj: matrix(gpu, weights, cfg, &format!("{p}.out_proj"))?,
                conv1d: vector(gpu, weights, &format!("{p}.conv1d.weight"))?,
                a_log: vector(gpu, weights, &format!("{p}.A_log"))?,
                dt_bias: vector(gpu, weights, &format!("{p}.dt_bias"))?,
                norm: vector(gpu, weights, &format!("{p}.norm.weight"))?,
                qkv: gpu.buffer(channels * 4),
                mixed: gpu.buffer(channels * 4),
                z: gpu.buffer(values * 4),
                a: gpu.buffer(c.v_heads * 4),
                b: gpu.buffer(c.v_heads * 4),
                y: gpu.buffer(values * 4),
                cfg: c,
            })
        }

        /// Encodes one token: `x` is the normed hidden state, `out` gets the
        /// layer output, `state` is updated.
        pub fn encode(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &View,
            out: &View,
            state: &GpuDeltaNetState,
        ) {
            let c = &self.cfg;
            let key_size = c.k_heads * c.k_dim;
            let channels = 2 * key_size + c.v_heads * c.v_dim;
            let [qkv, mixed, z, a, b, y] =
                [&self.qkv, &self.mixed, &self.z, &self.a, &self.b, &self.y]
                    .map(|buf| View::new(buf.clone()));

            self.in_proj_qkv.encode(gpu, enc, x, &qkv);
            self.in_proj_z.encode(gpu, enc, x, &z);
            self.in_proj_b.encode(gpu, enc, x, &b);
            self.in_proj_a.encode(gpu, enc, x, &a);

            enc.set_compute_pipeline_state(&gpu.pipeline("deltanet_conv"));
            bind(
                enc,
                &[&qkv, &self.conv1d, &View::new(state.conv.clone()), &mixed],
            );
            let p = ConvParams {
                channels: channels as u32,
                kernel_size: c.kernel as u32,
            };
            params(enc, 4, &p);
            enc.dispatch_threads(MTLSize::new(channels as u64, 1, 1), MTLSize::new(256, 1, 1));

            // normalize_qk in MLX: rms_norm with eps / k_dim, then q scaled
            // by 1 / k_dim and k by 1 / sqrt(k_dim).
            let (q, k, v) = (
                at(&self.mixed, 0),
                at(&self.mixed, key_size),
                at(&self.mixed, 2 * key_size),
            );
            let l2 = |scale: f32| Norm {
                rows: c.k_heads,
                dim: c.k_dim,
                in_stride: c.k_dim,
                out_stride: c.k_dim,
                eps: c.eps / c.k_dim as f32,
                scale,
                weight: None,
                gate: None,
            };
            l2(1.0 / c.k_dim as f32).encode(gpu, enc, &q, &q);
            l2(1.0 / (c.k_dim as f32).sqrt()).encode(gpu, enc, &k, &k);

            enc.set_compute_pipeline_state(&gpu.pipeline("delta_rule"));
            bind(
                enc,
                &[
                    &q,
                    &k,
                    &v,
                    &a,
                    &b,
                    &self.a_log,
                    &self.dt_bias,
                    &View::new(state.recurrent.clone()),
                    &y,
                ],
            );
            let p = DeltaParams {
                k_heads: c.k_heads as u32,
                v_heads: c.v_heads as u32,
                k_dim: c.k_dim as u32,
                v_dim: c.v_dim as u32,
            };
            params(enc, 9, &p);
            enc.dispatch_threads(
                MTLSize::new((c.v_heads * c.v_dim) as u64, 1, 1),
                MTLSize::new(128, 1, 1),
            );

            let mut gated = Norm::rows(c.v_heads, c.v_dim, c.eps, &self.norm);
            gated.gate = Some(&z);
            gated.encode(gpu, enc, &y, &y);

            self.out_proj.encode(gpu, enc, &y, out);
        }
    }

    #[cfg(test)]
    mod tests {
        //! Each GPU layer against the CPU layer on real weights:
        //!   scripts/mac.sh test --release --lib gpu_layers -- --ignored --nocapture

        use super::*;
        use crate::cpu::attention::{AttentionWeights, KvCache, attention};
        use crate::cpu::deltanet::{DeltaNetState, DeltaNetWeights, deltanet};
        use crate::cpu::matrix::Matrix;
        use crate::cpu::ops::rms_norm;
        use crate::gpu::contents;
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
            let w = weights
                .tensor(&format!("{prefix}.input_layernorm.weight"))
                .unwrap()
                .to_f32();
            rms_norm(&mut x, &w, eps);
            x
        }

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            assert_eq!(got.len(), want.len());
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
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
            gpu.run(|enc| Norm::rows(2, 2048, cfg.rms_norm_eps, &wv).encode(&gpu, enc, &xv, &out));
            let err = rel_rms(contents::<f32>(&out.buffer, x.len()), &want);
            println!("rms_norm: rel rms {err:.2e}");
            assert!(err < BOUND);
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn deltanet_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let prefix = "language_model.model.layers.0";
            let gpu_layer = GpuDeltaNet::load(&gpu, &cfg, &weights, prefix).unwrap();
            let gpu_state = GpuDeltaNetState::new(&gpu, &gpu_layer.cfg);

            let p = format!("{prefix}.linear_attn");
            let v = |n: &str| weights.tensor(&format!("{p}.{n}")).unwrap().to_f32();
            let (conv1d, a_log, dt_bias, norm) = (
                v("conv1d.weight"),
                v("A_log"),
                v("dt_bias"),
                v("norm.weight"),
            );
            let w = DeltaNetWeights {
                in_proj_qkv: quant(&weights, &cfg, &format!("{p}.in_proj_qkv")),
                in_proj_z: quant(&weights, &cfg, &format!("{p}.in_proj_z")),
                in_proj_b: quant(&weights, &cfg, &format!("{p}.in_proj_b")),
                in_proj_a: quant(&weights, &cfg, &format!("{p}.in_proj_a")),
                out_proj: quant(&weights, &cfg, &format!("{p}.out_proj")),
                conv1d: &conv1d,
                a_log: &a_log,
                dt_bias: &dt_bias,
                norm: &norm,
            };
            let c = deltanet_config(&cfg);
            let mut cpu_state = DeltaNetState::new(&c);

            let out = View::new(gpu.buffer(cfg.hidden_size * 4));
            for t in 0..TOKENS {
                let x = input(&weights, prefix, t, cfg.rms_norm_eps);
                let want = deltanet(&c, &w, &x, &mut cpu_state);
                let xv = View::new(gpu.buffer_from(&x));
                gpu.run(|enc| gpu_layer.encode(&gpu, enc, &xv, &out, &gpu_state));
                let err = rel_rms(contents::<f32>(&out.buffer, want.len()), &want);
                println!("deltanet token {t}: rel rms {err:.2e}");
                assert!(err < BOUND, "token {t}");
            }
            let conv = rel_rms(
                contents::<f32>(&gpu_state.conv, cpu_state.conv.len()),
                &cpu_state.conv,
            );
            let rec = rel_rms(
                contents::<f32>(&gpu_state.recurrent, cpu_state.recurrent.len()),
                &cpu_state.recurrent,
            );
            println!("deltanet state: conv {conv:.2e}, recurrent {rec:.2e}");
            assert!(conv < BOUND && rec < BOUND);
        }

        #[test]
        #[ignore = "needs the model on the Mac"]
        fn attention_matches_cpu() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let prefix = "language_model.model.layers.3";
            let gpu_layer = GpuAttention::load(&gpu, &cfg, &weights, prefix).unwrap();
            let mut gpu_cache = GpuKvCache::new(&gpu, &gpu_layer.cfg, 64);

            let p = format!("{prefix}.self_attn");
            let v = |n: &str| weights.tensor(&format!("{p}.{n}")).unwrap().to_f32();
            let (q_norm, k_norm) = (v("q_norm.weight"), v("k_norm.weight"));
            let w = AttentionWeights {
                q_proj: quant(&weights, &cfg, &format!("{p}.q_proj")),
                k_proj: quant(&weights, &cfg, &format!("{p}.k_proj")),
                v_proj: quant(&weights, &cfg, &format!("{p}.v_proj")),
                o_proj: quant(&weights, &cfg, &format!("{p}.o_proj")),
                q_norm: &q_norm,
                k_norm: &k_norm,
            };
            let c = attention_config(&cfg);
            let mut cpu_cache = KvCache::default();

            let out = View::new(gpu.buffer(cfg.hidden_size * 4));
            for t in 0..TOKENS {
                let x = input(&weights, prefix, t, cfg.rms_norm_eps);
                let want = attention(&c, &w, &x, &mut cpu_cache);
                let xv = View::new(gpu.buffer_from(&x));
                gpu.run(|enc| gpu_layer.encode(&gpu, enc, &xv, &out, &mut gpu_cache));
                let err = rel_rms(contents::<f32>(&out.buffer, want.len()), &want);
                println!("attention token {t}: rel rms {err:.2e}");
                assert!(err < BOUND, "token {t}");
            }
            let n = cpu_cache.keys().len();
            let keys = rel_rms(contents::<f32>(&gpu_cache.keys, n), cpu_cache.keys());
            let values = rel_rms(contents::<f32>(&gpu_cache.values, n), cpu_cache.values());
            println!("attention cache: keys {keys:.2e}, values {values:.2e}");
            assert!(keys < BOUND && values < BOUND);
        }
    }
}
