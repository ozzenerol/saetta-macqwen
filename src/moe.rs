//! The routed experts of an MoE block on the GPU (`kernels/moe.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::Quantization;
    use crate::gpu::Gpu;
    use crate::qmv::{GpuQuant, View, set_views};
    use crate::weights::Weights;
    use metal::{ComputeCommandEncoderRef, MTLSize};

    /// Must match MOE_ROWS and MOE_SIMDS in the kernel.
    const ROWS_PER_SIMD: usize = 1;
    const SIMDS: usize = 8;

    /// The `switch_mlp` gate, up and down stacks of one layer.
    pub struct Experts<'a> {
        pub gate: GpuQuant<'a>,
        pub up: GpuQuant<'a>,
        pub down: GpuQuant<'a>,
    }

    impl<'a> Experts<'a> {
        /// `layer` is a module prefix such as `language_model.model.layers.0`.
        pub fn new(
            gpu: &Gpu,
            weights: &'a Weights,
            quant: &Quantization,
            layer: &str,
        ) -> Result<Experts<'a>, String> {
            let load = |name: &str| -> Result<GpuQuant<'a>, String> {
                let q = weights.quantized(&format!("{layer}.mlp.switch_mlp.{name}"), quant)?;
                Ok(GpuQuant::new(gpu, &q))
            };
            let e = Experts {
                gate: load("gate_proj")?,
                up: load("up_proj")?,
                down: load("down_proj")?,
            };
            let block = ROWS_PER_SIMD * SIMDS;
            assert_eq!(e.gate.bits, 5, "moe kernels are 5-bit");
            assert!(e.gate.rows.is_multiple_of(block) && e.down.rows.is_multiple_of(block));
            assert_eq!((e.up.rows, e.up.cols), (e.gate.rows, e.gate.cols));
            assert_eq!((e.down.rows, e.down.cols), (e.gate.cols, e.gate.rows));
            Ok(e)
        }

        /// The hidden size of one expert (rows of gate and up).
        pub fn size(&self) -> usize {
            self.gate.rows
        }

        /// The bytes read for `count` experts.
        pub fn bytes(&self, count: usize) -> usize {
            count * (self.gate.bytes() + self.up.bytes() + self.down.bytes())
        }

        /// Encodes y = sum of weights[i] * expert(ids[i], x) for i < count, in
        /// two dispatches. `ids` holds u32, `weights` f32, and `h` is scratch
        /// for count * size() f32 values. `x` and `y` have the model width.
        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &View,
            ids: &View,
            weights: &View,
            count: usize,
            h: &View,
            y: &View,
        ) {
            self.encode_gate_up(gpu, enc, x, ids, count, h);
            self.encode_down(gpu, enc, h, ids, weights, count, y);
        }

        /// h[i] = silu(gate[ids[i]] x) * up[ids[i]] x for i < count.
        pub fn encode_gate_up(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &View,
            ids: &View,
            count: usize,
            h: &View,
        ) {
            enc.set_compute_pipeline_state(&gpu.pipeline("moe_gate_up_5bit"));
            self.gate.set_buffers(enc, 0);
            self.up.set_buffers(enc, 3);
            set_views(enc, 6, &[x, h]);
            let shape = [self.gate.rows as u32, self.gate.cols as u32, count as u32];
            enc.set_bytes(8, 12, shape.as_ptr().cast());
            set_views(enc, 9, &[ids]);
            enc.dispatch_thread_groups(
                MTLSize::new(row_groups(self.gate.rows), count as u64, 1),
                MTLSize::new(32 * SIMDS as u64, 1, 1),
            );
        }

        /// y = sum of weights[i] * down[ids[i]] h[i] for i < count.
        #[allow(clippy::too_many_arguments)]
        pub fn encode_down(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            h: &View,
            ids: &View,
            weights: &View,
            count: usize,
            y: &View,
        ) {
            enc.set_compute_pipeline_state(&gpu.pipeline("moe_down_5bit"));
            self.down.set_buffers(enc, 0);
            set_views(enc, 3, &[h, y]);
            let shape = [self.down.rows as u32, self.down.cols as u32, count as u32];
            enc.set_bytes(5, 12, shape.as_ptr().cast());
            set_views(enc, 6, &[ids, weights]);
            enc.dispatch_thread_groups(
                MTLSize::new(row_groups(self.down.rows), 1, 1),
                MTLSize::new(32 * SIMDS as u64, 1, 1),
            );
        }
    }

    fn row_groups(rows: usize) -> u64 {
        rows.div_ceil(ROWS_PER_SIMD * SIMDS) as u64
    }

    /// Expert ids for layer `l` of the benchmark: 8 distinct, spread out.
    pub fn bench_ids(l: usize) -> [u32; 8] {
        std::array::from_fn(|i| ((l * 37 + i * 29 + 11) % 256) as u32)
    }

    /// Runs the routed experts of every layer once per command buffer, as one
    /// token of decode would: `runs` timed runs after one warm-up. Prints
    /// best and median microseconds per layer and GB/s.
    pub fn bench(dir: &std::path::Path, runs: usize) -> Result<(), Box<dyn std::error::Error>> {
        let config = crate::config::Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let layers: Vec<Experts> = (0..config.num_layers())
            .map(|l| {
                Experts::new(
                    &gpu,
                    &weights,
                    &config.quantization,
                    &format!("language_model.model.layers.{l}"),
                )
            })
            .collect::<Result<_, _>>()?;
        let hidden = layers[0].gate.cols;
        let x = View::new(gpu.buffer_from(&vec![0.01f32; hidden]));
        let ids: Vec<View> = (0..layers.len())
            .map(|l| View::new(gpu.buffer_from(&bench_ids(l))))
            .collect();
        let w = View::new(gpu.buffer_from(&[0.125f32; 8]));
        let h = View::new(gpu.buffer(8 * layers[0].size() * 4));
        let y = View::new(gpu.buffer(hidden * 4));

        let n = layers.len() as f64;
        let per_layer = layers[0].bytes(8) as f64;
        let parts = [
            ("gate+up+down", per_layer),
            (
                "gate+up only",
                (layers[0].gate.bytes() + layers[0].up.bytes()) as f64 * 8.0,
            ),
            ("down only", layers[0].down.bytes() as f64 * 8.0),
        ];
        println!(
            "routed experts, {} layers x 8 experts, {:.1} MB per layer, {runs} runs each:",
            layers.len(),
            per_layer / 1e6
        );
        for (part, (label, bytes)) in parts.iter().enumerate() {
            let mut times: Vec<f64> = (0..=runs)
                .map(|_| {
                    gpu.run(|enc| {
                        for (layer, ids) in layers.iter().zip(&ids) {
                            if part != 2 {
                                layer.encode_gate_up(&gpu, enc, &x, ids, 8, &h);
                            }
                            if part != 1 {
                                layer.encode_down(&gpu, enc, &h, ids, &w, 8, &y);
                            }
                        }
                    })
                    .as_secs_f64()
                })
                .skip(1)
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "  {label}: best {:.1} us per layer ({:.0} GB/s), median {:.1} us ({:.0} GB/s)",
                times[0] / n * 1e6,
                bytes * n / times[0] / 1e9,
                times[runs / 2] / n * 1e6,
                bytes * n / times[runs / 2] / 1e9,
            );
        }
        Ok(())
    }
}
