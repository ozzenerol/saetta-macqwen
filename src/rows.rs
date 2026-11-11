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

    /// The m-row projection the verify uses: `record_qmv_reg`.
    pub fn record_proj(
        plan: &mut Plan,
        w: &GpuQuant,
        m: usize,
        x: &View,
        y: &View,
        parallel: bool,
    ) {
        record_qmv_reg(plan, w, m, x, y, parallel);
    }

    /// Must match QR and QR_SIMDS in rows.metal.
    const QR: usize = 8;
    const QR_SIMDS: usize = 2;
    /// Rows of x one qmv_reg dispatch takes.
    const REG_ROWS: usize = 4;

    /// `record_qmv` with x held in registers: up to 4 rows per dispatch,
    /// more as a second dispatch. Each weight is read once per 4 rows.
    pub fn record_qmv_reg(
        plan: &mut Plan,
        w: &GpuQuant,
        m: usize,
        x: &View,
        y: &View,
        parallel: bool,
    ) {
        record_qmv_reg_strided(plan, w, m, x, y, w.rows, parallel);
    }

    /// `record_qmv_reg` with rows of y `y_stride` floats apart.
    pub fn record_qmv_reg_strided(
        plan: &mut Plan,
        w: &GpuQuant,
        m: usize,
        x: &View,
        y: &View,
        y_stride: usize,
        parallel: bool,
    ) {
        assert_eq!(w.experts, 1, "stacked experts go through moe");
        assert!((1..=MAX_ROWS).contains(&m), "{m} rows");
        assert!(w.cols.is_multiple_of(64));
        assert!(
            matches!(w.bits, 5 | 8),
            "no qmv_reg kernel for {} bits",
            w.bits
        );
        let shape = plan.constant(&[w.rows as u32, w.cols as u32, y_stride as u32]);
        let grid = MTLSize::new(w.rows.div_ceil(QR * QR_SIMDS) as u64, 1, 1);
        let threads = MTLSize::new(32 * QR_SIMDS as u64, 1, 1);
        for (pass, r0) in (0..m).step_by(REG_ROWS).enumerate() {
            let rows = REG_ROWS.min(m - r0);
            let at = |v: &View, floats: usize| View {
                buffer: v.buffer.clone(),
                offset: v.offset + (r0 * floats * 4) as u64,
            };
            let kernel = format!("qmv_reg_{}bit_{rows}", w.bits);
            let buffers = [
                &w.words,
                &w.scales,
                &w.biases,
                &at(x, w.cols),
                &at(y, y_stride),
                &shape,
            ];
            op(plan, &kernel, &buffers, grid, threads, parallel || pass > 0);
        }
    }

    /// Must match MMA_SIMDS in rows.metal: simdgroups per 8-row tile.
    const MMA_SIMDS: usize = 8;

    /// `record_qmv` with simdgroup matrices. `x` must have `MAX_ROWS`
    /// readable rows; only the first `m` rows of `y` are written.
    pub fn record_qmv_mma(
        plan: &mut Plan,
        w: &GpuQuant,
        m: usize,
        x: &View,
        y: &View,
        parallel: bool,
    ) {
        assert_eq!(w.experts, 1, "stacked experts go through moe");
        assert!((1..=MAX_ROWS).contains(&m), "{m} rows");
        assert!(w.cols.is_multiple_of(32 * MMA_SIMDS));
        let readable = x.buffer.length() - x.offset;
        assert!(
            readable as usize >= MAX_ROWS * w.cols * 4,
            "x needs {MAX_ROWS} rows"
        );
        let kernel = match w.bits {
            5 => "qmv_mma_5bit",
            8 => "qmv_mma_8bit",
            b => panic!("no qmv_mma kernel for {b} bits"),
        };
        let shape = plan.constant(&[w.rows as u32, w.cols as u32, m as u32]);
        let buffers = [&w.words, &w.scales, &w.biases, x, y, &shape];
        let grid = MTLSize::new(w.rows.div_ceil(8) as u64, 1, 1);
        let threads = MTLSize::new(32 * MMA_SIMDS as u64, 1, 1);
        op(plan, kernel, &buffers, grid, threads, parallel);
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

    /// Must match MOE_SIMDS in kernels/moe.metal.
    const MOE_SIMDS: usize = 8;

    /// `Router::record` for `m` rows: `out` gets `[m, experts + 1]`, as
    /// two qmv_reg ops.
    pub fn record_router(
        plan: &mut Plan,
        r: &Router,
        m: usize,
        x: &View,
        out: &View,
        parallel: bool,
    ) {
        let n = r.outputs();
        record_qmv_reg_strided(plan, &r.gate, m, x, out, n, parallel);
        let shared = View {
            buffer: out.buffer.clone(),
            offset: out.offset + (r.gate.rows * 4) as u64,
        };
        record_qmv_reg_strided(plan, &r.shared_gate, m, x, &shared, n, true);
    }

    /// Must match RowsAttentionParams in kernels/attention_rows.metal.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct RowsAttentionParams {
        heads: u32,
        kv_heads: u32,
        capacity: u32,
        scale: f32,
        rows: u32,
        chunks: u32,
    }

    /// Must match DecodeParams in kernels/attention.metal.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct DecodeParams {
        heads: u32,
        kv_heads: u32,
        capacity: u32,
        scale: f32,
    }

    /// Must match DEC_CHUNK and DEC_DIM in kernels/attention.metal.
    const CHUNK: usize = 128;
    const DIM: usize = 256;

    /// Decode attention for `m` rows (`attention_partial_rows`, then
    /// `attention_reduce` per row), with the per-row partial results of
    /// up to `rows` rows. Layers that run one after the other share it.
    pub struct AttentionRows {
        heads: usize,
        kv_heads: usize,
        capacity: usize,
        rows: usize,
        part_o: View,
        part_ml: View,
    }

    impl AttentionRows {
        /// About `rows * capacity / 128 * heads` KB.
        pub fn new(
            gpu: &Gpu,
            heads: usize,
            kv_heads: usize,
            capacity: usize,
            rows: usize,
        ) -> AttentionRows {
            let chunks = capacity.div_ceil(CHUNK);
            AttentionRows {
                heads,
                kv_heads,
                capacity,
                rows,
                part_o: View::new(gpu.buffer(rows * chunks * heads * DIM * 4)),
                part_ml: View::new(gpu.buffer(rows * chunks * heads * 2 * 4)),
            }
        }

        /// Records `DecodeAttention::record` for `m` rows: `q` is `[m,
        /// heads, 256]`, `q_and_gate` `[m, heads, 512]`, `out` `[m, heads,
        /// 256]`, and row r takes its kv_len from `params[r]`.
        #[allow(clippy::too_many_arguments)]
        pub fn record(
            &self,
            plan: &mut Plan,
            m: usize,
            q: &View,
            keys: &View,
            values: &View,
            q_and_gate: &View,
            out: &View,
            params: &[View],
        ) {
            assert!(m <= self.rows && params.len() >= m);
            let chunks = self.capacity.div_ceil(CHUNK);
            let scale = 1.0 / (DIM as f32).sqrt();
            let p = plan.constant(&[RowsAttentionParams {
                heads: self.heads as u32,
                kv_heads: self.kv_heads as u32,
                capacity: self.capacity as u32,
                scale,
                rows: m as u32,
                chunks: chunks as u32,
            }]);
            plan.op(
                "attention_partial_rows",
                &[q, keys, values, &self.part_o, &self.part_ml, &p, &params[0]],
                MTLSize::new(chunks as u64, self.kv_heads as u64, 1),
                MTLSize::new(256, 1, 1),
            );
            let p = plan.constant(&[DecodeParams {
                heads: self.heads as u32,
                kv_heads: self.kv_heads as u32,
                capacity: self.capacity as u32,
                scale,
            }]);
            let at = |v: &View, floats: usize| View {
                buffer: v.buffer.clone(),
                offset: v.offset + (floats * 4) as u64,
            };
            let part = chunks * self.heads;
            let qs = self.heads * DIM;
            for (r, row_params) in params.iter().enumerate().take(m) {
                let buffers = [
                    &at(&self.part_o, r * part * DIM),
                    &at(&self.part_ml, r * part * 2),
                    &at(q_and_gate, r * 2 * qs),
                    &at(out, r * qs),
                    &p,
                    row_params,
                ];
                let (grid, group) = (
                    MTLSize::new(self.heads as u64, 1, 1),
                    MTLSize::new(DIM as u64, 1, 1),
                );
                if r == 0 {
                    plan.op("attention_reduce", &buffers, grid, group);
                } else {
                    plan.op_parallel("attention_reduce", &buffers, grid, group);
                }
            }
        }
    }

    /// The distinct experts of `m` rows, for `MoeRows::record`.
    pub struct MoeRows {
        list: View,
        slots: View,
        /// Use the simdgroup-matrix kernels.
        pub mma: bool,
    }

    impl MoeRows {
        pub fn new(gpu: &Gpu, top_k: usize) -> MoeRows {
            let most = MAX_ROWS * top_k;
            MoeRows {
                list: View::new(gpu.buffer((1 + most) * 4)),
                slots: View::new(gpu.buffer(most * MAX_ROWS * 4)),
                mma: false,
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
            if self.mma {
                let readable = (x.buffer.length() - x.offset) as usize;
                assert!(readable >= MAX_ROWS * g.cols * 4, "x needs {MAX_ROWS} rows");
                let threads = MTLSize::new(32 * MMA_SIMDS as u64, 1, 1);
                let rows = plan.constant(&[m as u32]);
                let shape = plan.constant(&[g.rows as u32, g.cols as u32, top_k as u32]);
                plan.op(
                    "moe_gate_up_mma",
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
                        &rows,
                    ],
                    MTLSize::new((g.rows / 8) as u64, (m * top_k) as u64, 1),
                    threads,
                );
                let shape = plan.constant(&[d.rows as u32, d.cols as u32, top_k as u32]);
                plan.op(
                    "moe_down_mma",
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
                        &rows,
                    ],
                    MTLSize::new((d.rows / 8) as u64, 1, 1),
                    threads,
                );
                return;
            }
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
                let xs = input(MAX_ROWS, w.cols);
                let x = View::new(gpu.buffer_from(&xs));
                for (mma, m) in [false, true]
                    .into_iter()
                    .flat_map(|a| (1..=MAX_ROWS).map(move |m| (a, m)))
                {
                    let (y, want) = (
                        View::new(gpu.buffer(m * w.rows * 4)),
                        View::new(gpu.buffer(m * w.rows * 4)),
                    );
                    let mut plan = Plan::new(&gpu);
                    if mma {
                        record_qmv_mma(&mut plan, &w, m, &x, &y, false);
                    } else {
                        record_qmv_reg(&mut plan, &w, m, &x, &y, false);
                    }
                    record_qmv(&mut plan, &w, m, &x, &want, true);
                    plan.build().run(&gpu);
                    let (got, want) = (
                        contents::<f32>(&y.buffer, m * w.rows),
                        contents::<f32>(&want.buffer, m * w.rows),
                    );
                    let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
                    let norm: f32 = want.iter().map(|w| w * w).sum();
                    let rel = (err / norm).sqrt();
                    assert!(rel < 1e-6, "{module} mma {mma} m {m}: rel rms {rel}");
                }
                println!("{module}: reg and mma rows 1 to {MAX_ROWS} within 1e-6");
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
            let mut rows = MoeRows::new(&gpu, k);
            for (mma, m) in [false, true]
                .into_iter()
                .flat_map(|a| (1..=MAX_ROWS).map(move |m| (a, m)))
            {
                rows.mma = mma;
                let xs = input(MAX_ROWS, hidden);
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
                    assert!(rel < 1e-5, "moe m {m} row {j}: rel rms {rel}");
                }
                println!("mma {mma} m {m}: router within 1e-6, moe within 1e-5");
            }
        }

        /// Microseconds per MoE block (routed experts only) of layer 0, decode
        /// against the rows kernels, with all m * 8 experts distinct (the
        /// worst case). Take the bench lock (timed.sh).
        #[test]
        #[ignore = "a timing run"]
        fn moe_rows_time() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let layer = "language_model.model.layers.0";
            let experts = Experts::new(&gpu, &weights, &cfg.quantization, layer).unwrap();
            let (hidden, k, size) = (cfg.hidden_size, cfg.num_experts_per_tok, experts.size());
            let mut rows = MoeRows::new(&gpu, k);
            let x = View::new(gpu.buffer_from(&input(MAX_ROWS, hidden)));
            let ids: Vec<u32> = (0..MAX_ROWS * k).map(|i| (i * 37 % 256) as u32).collect();
            let wts = vec![0.125f32; MAX_ROWS * k];
            let (idv, wv) = (
                View::new(gpu.buffer_from(&ids)),
                View::new(gpu.buffer_from(&wts)),
            );
            let h = View::new(gpu.buffer(MAX_ROWS * k * size * 4));
            let y = View::new(gpu.buffer(MAX_ROWS * hidden * 4));
            let mut time = |m: usize, mma: bool| {
                rows.mma = mma;
                let mut plan = Plan::new(&gpu);
                for _ in 0..20 {
                    if m == 0 {
                        experts.record(&mut plan, &x, &idv, &wv, k, &h, &y);
                    } else {
                        rows.record(&mut plan, &experts, m, k, &x, &idv, &wv, &h, &y, false);
                    }
                }
                let built = plan.build();
                let mut t: Vec<f64> = (0..11)
                    .map(|_| built.run_timed(&gpu).1 * 1e6 / 20.0)
                    .collect();
                t.sort_by(f64::total_cmp);
                t[5]
            };
            let base = time(0, false);
            let all: Vec<String> = (1..=MAX_ROWS)
                .map(|m| format!("m{m} {:.1}/{:.1}", time(m, false), time(m, true)))
                .collect();
            println!("moe: decode {base:.1} us; rows/mma {}", all.join(", "));
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
                    let x = View::new(gpu.buffer_from(&input(MAX_ROWS, w.cols)));
                    let y = View::new(gpu.buffer(MAX_ROWS * w.rows * 4));
                    let mut plan = Plan::new(&gpu);
                    for _ in 0..50 {
                        match m {
                            0 => w.record(&mut plan, &x, &y, false),
                            1..=8 => record_qmv(&mut plan, &w, m, &x, &y, false),
                            9..=16 => record_qmv_reg(&mut plan, &w, m - 8, &x, &y, false),
                            _ => record_qmv_mma(&mut plan, &w, m - 16, &x, &y, false),
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
                    .map(|m| format!("m{m} {:.1}/{:.1}/{:.1}", time(m), time(m + 8), time(m + 16)))
                    .collect();
                println!(
                    "{module} {}x{}: qmv {base:.1} us; rows/reg/mma {}",
                    w.rows,
                    w.cols,
                    rows.join(", ")
                );
            }
        }
    }
}
