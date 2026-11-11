//! Decode for 2 to 8 rows at once (`kernels/rows.metal`), for the
//! speculative verify: each weight and each KV chunk is read once for all
//! rows, so a verify of a few rows costs about one decode token.

#[cfg(target_os = "macos")]
pub use mac::*;

/// Most rows one verify takes.
pub const MAX_ROWS: usize = 8;

#[cfg(target_os = "macos")]
mod mac {
    use super::MAX_ROWS;
    use crate::gpu::Gpu;
    use crate::moe::{Experts, Router};
    use crate::plan::Plan;
    use crate::qmv::{GpuQuant, View};
    use metal::MTLSize;

    /// Must match QMV_ROWS and QMV_SIMDS in qmv.metal.
    const ROWS_PER_SIMD: usize = 4;
    const SIMDS: usize = 2;

    /// Records y = W x for `m` rows: `x` is `[m, cols]`, `y` gets
    /// `[m, rows]`. Row j matches `GpuQuant::record` on row j of `x`.
    pub fn record_qmv(plan: &mut Plan, w: &GpuQuant, m: usize, x: &View, y: &View, parallel: bool) {
        assert_eq!(w.experts, 1, "stacked experts go through moe");
        assert!((1..=MAX_ROWS).contains(&m), "{m} rows");
        assert!(
            matches!(w.bits, 5 | 8),
            "no qmv_rows kernel for {} bits",
            w.bits
        );
        let kernel = format!("qmv_rows_{}bit_{m}", w.bits);
        let shape = plan.constant(&[w.rows as u32, w.cols as u32]);
        let buffers = [&w.words, &w.scales, &w.biases, x, y, &shape];
        let grid = MTLSize::new(w.rows.div_ceil(ROWS_PER_SIMD * SIMDS) as u64, 1, 1);
        let threads = MTLSize::new(32 * SIMDS as u64, 1, 1);
        if parallel {
            plan.op_parallel(&kernel, &buffers, grid, threads);
        } else {
            plan.op(&kernel, &buffers, grid, threads);
        }
    }

    fn op(
        plan: &mut Plan,
        kernel: &str,
        buffers: &[&View],
        grid: MTLSize,
        threads: MTLSize,
        parallel: bool,
    ) {
        if parallel {
            plan.op_parallel(kernel, buffers, grid, threads);
        } else {
            plan.op(kernel, buffers, grid, threads);
        }
    }

    /// Must match ROUTER_SIMDS in kernels/router.metal and MOE_SIMDS in
    /// kernels/moe.metal.
    const ROUTER_SIMDS: usize = 8;
    const MOE_SIMDS: usize = 8;

    /// `Router::record` for `m` rows: `out` gets `[m, experts + 1]`.
    pub fn record_router(
        plan: &mut Plan,
        r: &Router,
        m: usize,
        x: &View,
        out: &View,
        parallel: bool,
    ) {
        let (g, s) = (&r.gate, &r.shared_gate);
        let shape = plan.constant(&[g.rows as u32, g.cols as u32]);
        let buffers = [
            &g.words, &g.scales, &g.biases, &s.words, &s.scales, &s.biases, x, out, &shape,
        ];
        let grid = MTLSize::new(r.outputs().div_ceil(ROUTER_SIMDS) as u64, 1, 1);
        let threads = MTLSize::new(32 * ROUTER_SIMDS as u64, 1, 1);
        op(
            plan,
            &format!("router_rows_{m}"),
            &buffers,
            grid,
            threads,
            parallel,
        );
    }

    /// The distinct experts of `m` rows, for `MoeRows::record`.
    pub struct MoeRows {
        list: View,
        slots: View,
    }

    impl MoeRows {
        pub fn new(gpu: &Gpu, top_k: usize) -> MoeRows {
            let most = MAX_ROWS * top_k;
            MoeRows {
                list: View::new(gpu.buffer((1 + most) * 4)),
                slots: View::new(gpu.buffer(most * MAX_ROWS * 4)),
            }
        }

        /// `Experts::record` for `m` rows: `x` and `y` are `[m, hidden]`,
        /// `ids` and `weights` `[m, top_k]`, `h` scratch for `[m, top_k,
        /// size]`. Each distinct expert is read once. `parallel` applies to
        /// the first op.
        #[allow(clippy::too_many_arguments)]
        pub fn record(
            &self,
            plan: &mut Plan,
            e: &Experts,
            m: usize,
            top_k: usize,
            x: &View,
            ids: &View,
            weights: &View,
            h: &View,
            y: &View,
            parallel: bool,
        ) {
            let u = plan.constant(&[m as u32, top_k as u32]);
            op(
                plan,
                "moe_union",
                &[ids, &self.list, &self.slots, &u],
                MTLSize::new(1, 1, 1),
                MTLSize::new(32, 1, 1),
                parallel,
            );
            let (g, up, d) = (&e.gate, &e.up, &e.down);
            let threads = MTLSize::new(32 * MOE_SIMDS as u64, 1, 1);
            let shape = plan.constant(&[g.rows as u32, g.cols as u32, top_k as u32]);
            plan.op(
                &format!("moe_gate_up_rows_{m}"),
                &[
                    &g.words,
                    &g.scales,
                    &g.biases,
                    &up.words,
                    &up.scales,
                    &up.biases,
                    x,
                    h,
                    &shape,
                    &self.list,
                    &self.slots,
                ],
                MTLSize::new((g.rows / MOE_SIMDS) as u64, (m * top_k) as u64, 1),
                threads,
            );
            let shape = plan.constant(&[d.rows as u32, d.cols as u32, top_k as u32]);
            plan.op(
                &format!("moe_down_rows_{m}"),
                &[
                    &d.words,
                    &d.scales,
                    &d.biases,
                    h,
                    y,
                    &shape,
                    &self.list,
                    &self.slots,
                    weights,
                ],
                MTLSize::new((d.rows / MOE_SIMDS) as u64, 1, 1),
                threads,
            );
        }
    }

    #[cfg(test)]
    mod tests {
        //!   scripts/mac.sh test --release --lib rows -- --ignored --nocapture
        use super::*;
        use crate::config::Config;
        use crate::gpu::{Gpu, contents};
        use crate::weights::Weights;
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";
        const MODULES: [&str; 3] = [
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "language_model.model.layers.0.mlp.shared_expert.down_proj",
            "language_model.model.layers.0.mlp.gate",
        ];

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        fn input(m: usize, cols: usize) -> Vec<f32> {
            (0..m * cols)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) * 0.013)
                .collect()
        }

        /// Every row of qmv_rows equals qmv on that row, bit for bit.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn qmv_rows_match_qmv() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            for module in MODULES {
                let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization).unwrap());
                for m in 1..=MAX_ROWS {
                    let xs = input(m, w.cols);
                    let x = View::new(gpu.buffer_from(&xs));
                    let y = View::new(gpu.buffer(m * w.rows * 4));
                    let mut plan = Plan::new(&gpu);
                    record_qmv(&mut plan, &w, m, &x, &y, false);
                    plan.build().run(&gpu);
                    let got = contents::<f32>(&y.buffer, m * w.rows).to_vec();
                    for j in 0..m {
                        let xj = View::new(gpu.buffer_from(&xs[j * w.cols..(j + 1) * w.cols]));
                        let yj = View::new(gpu.buffer(w.rows * 4));
                        let mut plan = Plan::new(&gpu);
                        w.record(&mut plan, &xj, &yj, false);
                        plan.build().run(&gpu);
                        let want = contents::<f32>(&yj.buffer, w.rows);
                        let bad = (0..w.rows)
                            .filter(|&r| got[j * w.rows + r] != want[r])
                            .count();
                        assert_eq!(bad, 0, "{module} m {m} row {j}: {bad} values differ");
                    }
                }
                println!("{module}: rows 1 to {MAX_ROWS} match qmv");
            }
        }

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        /// router_rows and the MoE rows against router_8bit and
        /// `Experts::record` on each row. Fast math compiles each kernel's
        /// sums its own way, so they differ in the last bits.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn moe_rows_match_decode() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let layer = "language_model.model.layers.0";
            let router = Router::new(&gpu, &weights, &cfg.quantization, layer).unwrap();
            let experts = Experts::new(&gpu, &weights, &cfg.quantization, layer).unwrap();
            let (hidden, k, size) = (cfg.hidden_size, cfg.num_experts_per_tok, experts.size());
            let rows = MoeRows::new(&gpu, k);
            for m in 1..=MAX_ROWS {
                let xs = input(m, hidden);
                let x = View::new(gpu.buffer_from(&xs));
                // Rows share some experts: row j picks 3j, 3j + 1, ... mod 40.
                let ids: Vec<u32> = (0..m * k)
                    .map(|i| ((i / k * 3 + i % k * 5) % 40) as u32)
                    .collect();
                let wts: Vec<f32> = (0..m * k).map(|i| 0.05 + (i % 7) as f32 * 0.03).collect();
                let (idv, wv) = (
                    View::new(gpu.buffer_from(&ids)),
                    View::new(gpu.buffer_from(&wts)),
                );
                let out = View::new(gpu.buffer(m * router.outputs() * 4));
                let h = View::new(gpu.buffer(m * k * size * 4));
                let y = View::new(gpu.buffer(m * hidden * 4));
                let mut plan = Plan::new(&gpu);
                record_router(&mut plan, &router, m, &x, &out, false);
                rows.record(&mut plan, &experts, m, k, &x, &idv, &wv, &h, &y, true);
                plan.build().run(&gpu);
                let got_r = contents::<f32>(&out.buffer, m * router.outputs()).to_vec();
                let got_y = contents::<f32>(&y.buffer, m * hidden).to_vec();

                for j in 0..m {
                    let xj = View::new(gpu.buffer_from(&xs[j * hidden..(j + 1) * hidden]));
                    let idj = View::new(gpu.buffer_from(&ids[j * k..(j + 1) * k]));
                    let wj = View::new(gpu.buffer_from(&wts[j * k..(j + 1) * k]));
                    let rj = View::new(gpu.buffer(router.outputs() * 4));
                    let hj = View::new(gpu.buffer(k * size * 4));
                    let yj = View::new(gpu.buffer(hidden * 4));
                    let mut plan = Plan::new(&gpu);
                    router.record(&mut plan, &xj, &rj, false);
                    experts.record(&mut plan, &xj, &idj, &wj, k, &hj, &yj);
                    plan.build().run(&gpu);
                    let n = router.outputs();
                    let rel = rel_rms(&got_r[j * n..(j + 1) * n], contents::<f32>(&rj.buffer, n));
                    assert!(rel < 1e-6, "router m {m} row {j}: rel rms {rel}");
                    let want = contents::<f32>(&yj.buffer, hidden);
                    let rel = rel_rms(&got_y[j * hidden..(j + 1) * hidden], want);
                    assert!(rel < 1e-6, "moe m {m} row {j}: rel rms {rel}");
                }
                println!("m {m}: router and moe within 1e-6");
            }
        }

        /// Microseconds per qmv_rows call on the qkv projection, against
        /// qmv. Take the bench lock (timed.sh).
        #[test]
        #[ignore = "a timing run"]
        fn qmv_rows_time() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            for module in MODULES {
                let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization).unwrap());
                let time = |m: usize| {
                    let x = View::new(gpu.buffer_from(&input(m.max(1), w.cols)));
                    let y = View::new(gpu.buffer(m.max(1) * w.rows * 4));
                    let mut plan = Plan::new(&gpu);
                    for _ in 0..50 {
                        if m == 0 {
                            w.record(&mut plan, &x, &y, false);
                        } else {
                            record_qmv(&mut plan, &w, m, &x, &y, false);
                        }
                    }
                    let built = plan.build();
                    let mut t: Vec<f64> = (0..11)
                        .map(|_| built.run_timed(&gpu).1 * 1e6 / 50.0)
                        .collect();
                    t.sort_by(f64::total_cmp);
                    t[5]
                };
                let base = time(0);
                let rows: Vec<String> = (1..=MAX_ROWS)
                    .map(|m| format!("m{m} {:.1}", time(m)))
                    .collect();
                println!(
                    "{module} {}x{}: qmv {base:.1} us; {}",
                    w.rows,
                    w.cols,
                    rows.join(", ")
                );
            }
        }
    }
}
