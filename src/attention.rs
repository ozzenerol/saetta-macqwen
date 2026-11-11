//! Split-K attention for one decode token (`kernels/attention.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::Gpu;
    use crate::plan::Plan;
    use crate::qmv::View;
    use metal::{Buffer, MTLSize};

    /// Must match DEC_CHUNK, DEC_GROUP and DEC_DIM in the kernel.
    const CHUNK: usize = 128;
    const GROUP: usize = 8;
    const DIM: usize = 256;

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DecodeParams {
        heads: u32,
        kv_heads: u32,
        capacity: u32,
        scale: f32,
    }

    /// The scratch for the per-chunk partial results of one cache size.
    /// Layers that run one after the other can share it.
    pub struct DecodeAttention {
        heads: usize,
        kv_heads: usize,
        capacity: usize,
        part_o: Buffer,
        part_ml: Buffer,
        /// attention_partial or attention_partial_mma.
        pub kernel: &'static str,
    }

    impl DecodeAttention {
        pub fn new(
            gpu: &Gpu,
            heads: usize,
            kv_heads: usize,
            head_dim: usize,
            capacity: usize,
        ) -> DecodeAttention {
            assert_eq!(
                head_dim, DIM,
                "decode attention is written for head dim 256"
            );
            assert_eq!(
                heads,
                kv_heads * GROUP,
                "decode attention needs 8 q heads per kv head"
            );
            let chunks = capacity.div_ceil(CHUNK);
            DecodeAttention {
                heads,
                kv_heads,
                capacity,
                part_o: gpu.buffer(chunks * heads * DIM * 4),
                part_ml: gpu.buffer(chunks * heads * 2 * 4),
                kernel: "attention_partial_mma",
            }
        }

        /// Records out = softmax(q K^T / sqrt(dim)) V * sigmoid(gate) over the
        /// first kv_len tokens (from the plan's TokenParams). `q` is
        /// `[heads, 256]`, `keys` and `values` the cache, `q_and_gate` the
        /// q_proj output whose gate of head h starts at h * 512 + 256.
        pub fn record(
            &self,
            plan: &mut Plan,
            q: &View,
            keys: &View,
            values: &View,
            q_and_gate: &View,
            out: &View,
        ) {
            let p = plan.constant(&[DecodeParams {
                heads: self.heads as u32,
                kv_heads: self.kv_heads as u32,
                capacity: self.capacity as u32,
                scale: 1.0 / (DIM as f32).sqrt(),
            }]);
            let params = plan.params.clone();
            let (o, ml) = (
                View::new(self.part_o.clone()),
                View::new(self.part_ml.clone()),
            );
            plan.op(
                self.kernel,
                &[q, keys, values, &o, &ml, &p, &params],
                MTLSize::new(
                    self.capacity.div_ceil(CHUNK) as u64,
                    self.kv_heads as u64,
                    1,
                ),
                MTLSize::new(256, 1, 1),
            );
            plan.op(
                "attention_reduce",
                &[&o, &ml, q_and_gate, out, &p, &params],
                MTLSize::new(self.heads as u64, (DIM / 64) as u64, 1),
                MTLSize::new(1024, 1, 1),
            );
        }
    }

    /// Times the split-K attention of `layers` full-attention layers for one
    /// token at each context length. Each layer has its own cache, so
    /// nothing stays in the cache.
    pub fn bench(lengths: &[usize], layers: usize, runs: usize) -> Result<(), String> {
        let gpu = Gpu::new()?;
        let (heads, kv) = (16, 2);
        let capacity = *lengths.iter().max().unwrap();
        let token = kv * DIM;
        let fill: Vec<half::f16> = (0..capacity * token)
            .map(|i| half::f16::from_f32(((i * 7919) % 2000) as f32 / 1000.0 - 1.0))
            .collect();
        let caches: Vec<(View, View)> = (0..layers)
            .map(|_| {
                (
                    View::new(gpu.buffer_from(&fill)),
                    View::new(gpu.buffer_from(&fill)),
                )
            })
            .collect();
        let q = View::new(gpu.buffer_from(&vec![0.01f32; heads * DIM]));
        let qg = View::new(gpu.buffer_from(&vec![0.5f32; heads * 2 * DIM]));
        let out = View::new(gpu.buffer(heads * DIM * 4));
        let mut att = DecodeAttention::new(&gpu, heads, kv, DIM, capacity);
        println!(
            "attention for one token, {layers} layers, 16 q / 2 kv heads of 256, half cache; median of {runs} runs after 1 warm-up:"
        );
        for kernel in ["attention_partial", "attention_partial_mma"] {
            att.kernel = kernel;
            let mut plan = Plan::new(&gpu);
            for (k, v) in &caches {
                att.record(&mut plan, &q, k, v, &qg, &out);
            }
            let built = plan.build();
            let reduce = plan.build_where(|k| k == "attention_reduce", true);
            println!("  {kernel}:");
            for &len in lengths {
                plan.set_params(crate::plan::TokenParams::new(0, len as u32 - 1));
                let mut t: Vec<f64> = (0..=runs)
                    .map(|_| built.run(&gpu).as_secs_f64())
                    .skip(1)
                    .collect();
                t.sort_by(f64::total_cmp);
                let time = t[runs / 2];
                let mut r: Vec<f64> = (0..=runs)
                    .map(|_| reduce.run(&gpu).as_secs_f64())
                    .skip(1)
                    .collect();
                r.sort_by(f64::total_cmp);
                let bytes = (layers * len * token * 2 * 2) as f64;
                println!(
                    "    {len:>6} tokens: {:.3} ms ({:.0} GB/s), reduce alone {:.3} ms",
                    time * 1e3,
                    bytes / time / 1e9,
                    r[runs / 2] * 1e3
                );
            }
        }
        Ok(())
    }

    /// Times whole decode tokens of the real model at each context length.
    /// The cache holds whatever is in it; the work per token only depends on
    /// the length.
    pub fn bench_decode(
        dir: &std::path::Path,
        lengths: &[usize],
        capacity: usize,
        runs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let cfg = crate::config::Config::load(dir)?;
        let weights = crate::weights::Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let mut model =
            crate::gpu_model::GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), capacity)?;
        println!(
            "decode, whole model, room for {capacity} tokens, logits read back; median of {runs} tokens after 1 warm-up:"
        );
        for &len in lengths {
            let mut t: Vec<f64> = (0..=runs)
                .map(|_| {
                    model.set_position(len - 1);
                    let start = std::time::Instant::now();
                    model.forward(220);
                    start.elapsed().as_secs_f64()
                })
                .skip(1)
                .collect();
            t.sort_by(f64::total_cmp);
            println!(
                "  {len:>6} tokens of context: {:.2} ms/token ({:.1} tok/s)",
                t[runs / 2] * 1e3,
                1.0 / t[runs / 2]
            );
        }
        Ok(())
    }
}
