//! The routed experts of an MoE block on the GPU (`kernels/moe.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::Quantization;
    use crate::gpu::Gpu;
    use crate::plan::Plan;
    use crate::qmv::{GpuQuant, View};
    use crate::weights::Weights;
    use metal::MTLSize;

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
            Experts::load(weights, quant, layer, |q| GpuQuant::new(gpu, q))
        }

        fn load(
            weights: &'a Weights,
            quant: &Quantization,
            layer: &str,
            make: impl Fn(&crate::weights::QuantTensor<'a>) -> GpuQuant<'a>,
        ) -> Result<Experts<'a>, String> {
            let load = |name: &str| -> Result<GpuQuant<'a>, String> {
                let q = weights.quantized(&format!("{layer}.mlp.switch_mlp.{name}"), quant)?;
                Ok(make(&q))
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
    }

    impl Experts<'static> {
        /// Copies of experts `ids` only, numbered 0.. in that order. See
        /// `GpuQuant::subset`.
        pub fn subset(
            gpu: &Gpu,
            weights: &Weights,
            quant: &Quantization,
            layer: &str,
            ids: &[usize],
        ) -> Result<Experts<'static>, String> {
            let load = |name: &str| -> Result<GpuQuant<'static>, String> {
                let q = weights.quantized(&format!("{layer}.mlp.switch_mlp.{name}"), quant)?;
                Ok(GpuQuant::subset(gpu, &q, ids))
            };
            Ok(Experts {
                gate: load("gate_proj")?,
                up: load("up_proj")?,
                down: load("down_proj")?,
            })
        }
    }

    impl Experts<'_> {
        /// The hidden size of one expert (rows of gate and up).
        pub fn size(&self) -> usize {
            self.gate.rows
        }

        /// The bytes read for `count` experts.
        pub fn bytes(&self, count: usize) -> usize {
            count * (self.gate.bytes() + self.up.bytes() + self.down.bytes())
        }

        /// Records y = sum of weights[i] * expert(ids[i], x) for i < count,
        /// as two ops. `ids` holds u32, `weights` f32, and `h` is scratch for
        /// count * size() f32 values. `x` and `y` have the model width.
        #[allow(clippy::too_many_arguments)]
        pub fn record(
            &self,
            plan: &mut Plan,
            x: &View,
            ids: &View,
            weights: &View,
            count: usize,
            h: &View,
            y: &View,
        ) {
            self.record_gate_up(plan, x, ids, count, h, false);
            self.record_down(plan, h, ids, weights, count, y);
        }

        /// h[i] = silu(gate[ids[i]] x) * up[ids[i]] x for i < count.
        /// `parallel` marks it independent of the op recorded before it.
        pub fn record_gate_up(
            &self,
            plan: &mut Plan,
            x: &View,
            ids: &View,
            count: usize,
            h: &View,
            parallel: bool,
        ) {
            let (g, u) = (&self.gate, &self.up);
            let shape = plan.constant(&[g.rows as u32, g.cols as u32, count as u32]);
            let buffers = [
                &g.words, &g.scales, &g.biases, &u.words, &u.scales, &u.biases, x, h, &shape, ids,
            ];
            let grid = MTLSize::new(row_groups(g.rows), count as u64, 1);
            let threads = MTLSize::new(32 * SIMDS as u64, 1, 1);
            if parallel {
                plan.op_parallel("moe_gate_up_5bit", &buffers, grid, threads);
            } else {
                plan.op("moe_gate_up_5bit", &buffers, grid, threads);
            }
        }

        /// y = sum of weights[i] * down[ids[i]] h[i] for i < count.
        pub fn record_down(
            &self,
            plan: &mut Plan,
            h: &View,
            ids: &View,
            weights: &View,
            count: usize,
            y: &View,
        ) {
            let d = &self.down;
            let shape = plan.constant(&[d.rows as u32, d.cols as u32, count as u32]);
            plan.op(
                "moe_down_5bit",
                &[&d.words, &d.scales, &d.biases, h, y, &shape, ids, weights],
                MTLSize::new(row_groups(d.rows), 1, 1),
                MTLSize::new(32 * SIMDS as u64, 1, 1),
            );
        }
    }

    fn row_groups(rows: usize) -> u64 {
        rows.div_ceil(ROWS_PER_SIMD * SIMDS) as u64
    }

    /// The router and shared-expert gate of one layer, both 8-bit.
    pub struct Router<'a> {
        pub gate: GpuQuant<'a>,
        pub shared_gate: GpuQuant<'a>,
    }

    impl<'a> Router<'a> {
        /// `layer` is a module prefix such as `language_model.model.layers.0`.
        pub fn new(
            gpu: &Gpu,
            weights: &'a Weights,
            quant: &Quantization,
            layer: &str,
        ) -> Result<Router<'a>, String> {
            let load = |name: &str| -> Result<GpuQuant<'a>, String> {
                let q = weights.quantized(&format!("{layer}.mlp.{name}"), quant)?;
                Ok(GpuQuant::new(gpu, &q))
            };
            let r = Router {
                gate: load("gate")?,
                shared_gate: load("shared_expert_gate")?,
            };
            assert_eq!(
                (r.gate.bits, r.shared_gate.bits),
                (8, 8),
                "router kernel is 8-bit"
            );
            assert_eq!(r.shared_gate.rows, 1);
            assert_eq!(r.gate.cols, r.shared_gate.cols);
            Ok(r)
        }

        /// Number of floats `record` writes: one logit per expert, then the
        /// shared expert's gate logit.
        pub fn outputs(&self) -> usize {
            self.gate.rows + 1
        }

        /// Records out[..experts] = gate x and out[experts] = shared_gate x,
        /// as one op.
        pub fn record(&self, plan: &mut Plan, x: &View, out: &View, parallel: bool) {
            let (g, s) = (&self.gate, &self.shared_gate);
            let shape = plan.constant(&[g.rows as u32, g.cols as u32]);
            let buffers = [
                &g.words, &g.scales, &g.biases, &s.words, &s.scales, &s.biases, x, out, &shape,
            ];
            let grid = MTLSize::new(self.outputs().div_ceil(ROUTER_SIMDS) as u64, 1, 1);
            let threads = MTLSize::new(32 * ROUTER_SIMDS as u64, 1, 1);
            if parallel {
                plan.op_parallel("router_8bit", &buffers, grid, threads);
            } else {
                plan.op("router_8bit", &buffers, grid, threads);
            }
        }
    }

    /// Must match ROUTER_SIMDS in kernels/router.metal.
    const ROUTER_SIMDS: usize = 8;

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
        // Only the 8 used experts of each layer, copied: about 690 MB in all,
        // the same bytes one token reads.
        let layers: Vec<Experts> = (0..config.num_layers())
            .map(|l| {
                let used = bench_ids(l).map(|e| e as usize);
                Experts::subset(
                    &gpu,
                    &weights,
                    &config.quantization,
                    &format!("language_model.model.layers.{l}"),
                    &used,
                )
            })
            .collect::<Result<_, _>>()?;
        let hidden = layers[0].gate.cols;
        let x = View::new(gpu.buffer_from(&vec![0.01f32; hidden]));
        let slots = View::new(gpu.buffer_from(&[0u32, 1, 2, 3, 4, 5, 6, 7]));
        let ids: Vec<&View> = vec![&slots; layers.len()];
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
            let mut plan = Plan::new(&gpu);
            for (layer, ids) in layers.iter().zip(&ids) {
                if part != 2 {
                    layer.record_gate_up(&mut plan, &x, ids, 8, &h, false);
                }
                if part != 1 {
                    layer.record_down(&mut plan, &h, ids, &w, 8, &y);
                }
            }
            let built = plan.build();
            let mut times: Vec<f64> = (0..=runs)
                .map(|_| built.run(&gpu).as_secs_f64())
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

        // The router: one router_8bit op per layer against two qmv ops.
        let routers: Vec<Router> = (0..config.num_layers())
            .map(|l| {
                Router::new(
                    &gpu,
                    &weights,
                    &config.quantization,
                    &format!("language_model.model.layers.{l}"),
                )
            })
            .collect::<Result<_, _>>()?;
        let logits = View::new(gpu.buffer(routers[0].outputs() * 4));
        let shared = View::new(gpu.buffer(4));
        for (label, fused) in [("router_8bit", true), ("2 x qmv_8bit", false)] {
            let mut plan = Plan::new(&gpu);
            for r in &routers {
                if fused {
                    r.record(&mut plan, &x, &logits, false);
                } else {
                    r.gate.record(&mut plan, &x, &logits, false);
                    r.shared_gate.record(&mut plan, &x, &shared, true);
                }
            }
            let built = plan.build();
            let mut times: Vec<f64> = (0..=runs)
                .map(|_| built.run(&gpu).as_secs_f64())
                .skip(1)
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "  router, {label}: best {:.1} us per layer, median {:.1} us",
                times[0] / n * 1e6,
                times[runs / 2] / n * 1e6,
            );
        }
        Ok(())
    }
}
