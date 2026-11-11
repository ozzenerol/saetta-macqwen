//! Split-K attention for one decode token (`kernels/attention.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::Gpu;
    use crate::plan::Plan;
    use crate::qmv::View;
    use metal::{Buffer, MTLSize};

    /// Must match ATT_CHUNK, ATT_GROUP and ATT_DIM in the kernel.
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
                "attention_partial",
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
                MTLSize::new(self.heads as u64, 1, 1),
                MTLSize::new(DIM as u64, 1, 1),
            );
        }
    }

    /// Must match AttentionParams in kernels/layers.metal (the old kernel).
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct OldParams {
        heads: u32,
        kv_heads: u32,
        dim: u32,
        capacity: u32,
        scale: f32,
    }

    /// Times the attention of `layers` full-attention layers for one token
    /// at each context length, split-K against the old one-group-per-head
    /// kernel. Each layer has its own cache, so nothing stays in the cache.
    pub fn bench(lengths: &[usize], layers: usize, runs: usize) -> Result<(), String> {
        let gpu = Gpu::new()?;
        let (heads, kv) = (16, 2);
        let capacity = *lengths.iter().max().unwrap();
        let token = kv * DIM;
        let fill: Vec<f32> = (0..capacity * token)
            .map(|i| ((i * 7919) % 2000) as f32 / 1000.0 - 1.0)
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
        let scores = View::new(gpu.buffer(heads * capacity * 4));
        let att = DecodeAttention::new(&gpu, heads, kv, DIM, capacity);

        let mut new_plan = Plan::new(&gpu);
        let mut old_plan = Plan::new(&gpu);
        let old_p = old_plan.constant(&[OldParams {
            heads: heads as u32,
            kv_heads: kv as u32,
            dim: DIM as u32,
            capacity: capacity as u32,
            scale: 1.0 / (DIM as f32).sqrt(),
        }]);
        let old_params = old_plan.params.clone();
        for (k, v) in &caches {
            att.record(&mut new_plan, &q, k, v, &qg, &out);
            old_plan.op(
                "attention",
                &[&q, k, v, &qg, &out, &scores, &old_p, &old_params],
                MTLSize::new(heads as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }
        let (new_built, old_built) = (new_plan.build(), old_plan.build());
        println!(
            "attention for one token, {layers} layers, 16 q / 2 kv heads of 256, f32 cache; median of {runs} runs after 1 warm-up:"
        );
        for &len in lengths {
            let time = |plan: &Plan, built: &crate::plan::Built| {
                plan.set_params(crate::plan::TokenParams::new(0, len as u32 - 1));
                let mut t: Vec<f64> = (0..=runs)
                    .map(|_| built.run(&gpu).as_secs_f64())
                    .skip(1)
                    .collect();
                t.sort_by(f64::total_cmp);
                t[runs / 2]
            };
            let new = time(&new_plan, &new_built);
            let old = time(&old_plan, &old_built);
            let bytes = (layers * len * token * 4 * 2) as f64;
            println!(
                "  {len:>6} tokens: split-K {:.3} ms ({:.0} GB/s), old {:.3} ms, {:.1}x",
                new * 1e3,
                bytes / new / 1e9,
                old * 1e3,
                old / new
            );
        }
        Ok(())
    }
}
