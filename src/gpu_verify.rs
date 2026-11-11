//! The speculative verify on the decode path: `m` tokens (1 to `rows`)
//! through the whole model as one recorded plan, so the weights are read
//! once for all of them (`rows.rs`). Each row sees the rows before it, as
//! if the tokens had been decoded one by one, and the DeltaNet state after
//! every row is kept so `accept` can roll back to any of them.
//!
//! All buffers are made once by `GpuModel::enable_verify`.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::Config;
    use crate::gpu::{Gpu, contents};
    use crate::gpu_layers::{
        AttentionRowsScratch, DeltaNetRowsScratch, GpuDeltaNetSnaps, Norm, add_norm_rows,
        mix_norm_rows,
    };
    use crate::gpu_model::{
        GpuLayer, GpuModel, Mixer, elementwise, record_embed, record_route_rows, view,
    };
    use crate::plan::{Built, Plan, TokenParams};
    use crate::qmv::View;
    use crate::rows::{self, MAX_ROWS, MoeRows};
    use metal::{Buffer, MTLSize};

    /// Row `r` of a buffer of rows of `floats` values.
    fn row(v: &View, r: usize, floats: usize) -> View {
        View {
            buffer: v.buffer.clone(),
            offset: v.offset + (r * floats * 4) as u64,
        }
    }

    /// The buffers of `MAX_ROWS` rows of one layer step, shared by all
    /// layers.
    struct RowsScratch {
        normed: View,
        mixed: View,
        moe_normed: View,
        router: View,
        ids: View,
        weights: View,
        shared_gate: View,
        shared_up: View,
        shared_h: View,
        shared: View,
        experts_h: View,
        routed: View,
        moe: MoeRows,
        deltanet: DeltaNetRowsScratch,
        attention: AttentionRowsScratch,
    }

    impl RowsScratch {
        fn new(gpu: &Gpu, cfg: &Config, capacity: usize, rows: usize) -> RowsScratch {
            let m = MAX_ROWS;
            let (h, k) = (cfg.hidden_size, cfg.num_experts_per_tok);
            let shared = cfg.shared_expert_intermediate_size;
            RowsScratch {
                normed: view(gpu, m * h),
                mixed: view(gpu, m * h),
                moe_normed: view(gpu, m * h),
                router: view(gpu, m * (cfg.num_experts + 1)),
                ids: view(gpu, m * k),
                weights: view(gpu, m * k),
                shared_gate: view(gpu, m * shared),
                shared_up: view(gpu, m * shared),
                shared_h: view(gpu, m * shared),
                shared: view(gpu, m * h),
                experts_h: view(gpu, m * k * cfg.moe_intermediate_size),
                routed: view(gpu, m * h),
                moe: MoeRows::new(gpu, k),
                deltanet: DeltaNetRowsScratch::new(gpu, cfg),
                attention: AttentionRowsScratch::new(gpu, cfg, capacity, rows),
            }
        }
    }

    pub struct GpuVerify<'a> {
        rows: usize,
        /// `plans[m - 1]` runs `m` rows.
        plans: Vec<(Plan<'a>, Built)>,
        /// `restores[r]` sets the DeltaNet states to the ones after row r.
        restores: Vec<Built>,
        /// One TokenParams per row.
        params: Buffer,
        final_hidden: View,
        logits: View,
        argmax: View,
        /// Rows and position of the last `verify`.
        last: Option<(usize, usize)>,
    }

    impl GpuLayer<'_> {
        /// `GpuLayer::record` for `m` rows of `x`. `s.normed` must hold
        /// `input_norm(x)`.
        #[allow(clippy::too_many_arguments)]
        fn record_rows(
            &self,
            plan: &mut Plan,
            cfg: &Config,
            m: usize,
            x: &View,
            s: &RowsScratch,
            snaps: Option<&GpuDeltaNetSnaps>,
            params: &[View],
            next_norm: &View,
            next_out: &View,
        ) {
            let (h, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
            match &self.mixer {
                Mixer::Linear(mx, state) => {
                    mx.record_rows(plan, m, &s.normed, &s.mixed, state, &s.deltanet, snaps)
                }
                Mixer::Full(mx, cache) => {
                    mx.record_rows(plan, m, &s.normed, &s.mixed, cache, &s.attention, params)
                }
            }
            let moe = &self.moe;
            add_norm_rows(
                plan,
                [x, &s.mixed, &moe.post_norm, &s.moe_normed],
                m,
                h,
                eps,
            );

            let logits = moe.router.outputs();
            let k = cfg.num_experts_per_tok;
            rows::record_router(plan, &moe.router, m, &s.moe_normed, &s.router, false);
            rows::record_proj(
                plan,
                &moe.shared_gate,
                m,
                &s.moe_normed,
                &s.shared_gate,
                true,
            );
            rows::record_proj(plan, &moe.shared_up, m, &s.moe_normed, &s.shared_up, true);
            record_route_rows(
                plan,
                m,
                &s.router,
                logits,
                cfg.num_experts,
                k,
                &s.ids,
                &s.weights,
            );
            elementwise(
                plan,
                "silu_mul",
                &[&s.shared_gate, &s.shared_up, &s.shared_h],
                m * moe.shared_gate.rows,
                true,
            );
            rows::record_proj(plan, &moe.shared_down, m, &s.shared_h, &s.shared, false);
            s.moe.record(
                plan,
                &moe.experts,
                m,
                k,
                &s.moe_normed,
                &s.ids,
                &s.weights,
                &s.experts_h,
                &s.routed,
                true,
            );
            // Each row's shared expert gate logit follows its expert logits.
            let shared_logit = View {
                buffer: s.router.buffer.clone(),
                offset: s.router.offset + (cfg.num_experts * 4) as u64,
            };
            mix_norm_rows(
                plan,
                [x, &shared_logit, &s.shared, &s.routed, next_norm, next_out],
                m,
                h,
                eps,
                logits,
            );
        }
    }

    impl<'a> GpuModel<'a> {
        /// Makes the verify for up to `rows` tokens (at most `MAX_ROWS`):
        /// one plan per row count, and about 2 MB per row per DeltaNet
        /// layer for the states after each row.
        pub fn enable_verify(&mut self, rows: usize) {
            assert!(
                (1..=MAX_ROWS).contains(&rows),
                "verify takes 1 to {MAX_ROWS} rows"
            );
            let (gpu, cfg) = (self.gpu, self.cfg);
            let h = cfg.hidden_size;
            let vocab = self.lm_head.rows;
            let s = RowsScratch::new(gpu, cfg, self.capacity, rows);
            let x = view(gpu, MAX_ROWS * h);
            let final_hidden = view(gpu, MAX_ROWS * h);
            let logits = view(gpu, MAX_ROWS * vocab);
            let argmax = view(gpu, MAX_ROWS);
            let params = gpu.buffer(MAX_ROWS * size_of::<TokenParams>());
            let param_views: Vec<View> = (0..MAX_ROWS)
                .map(|r| View {
                    buffer: params.clone(),
                    offset: (r * size_of::<TokenParams>()) as u64,
                })
                .collect();
            let snaps: Vec<Option<GpuDeltaNetSnaps>> = self
                .layers
                .iter()
                .map(|l| match &l.mixer {
                    Mixer::Linear(m, _) => Some(GpuDeltaNetSnaps::new(gpu, &m.cfg, rows)),
                    Mixer::Full(..) => None,
                })
                .collect();

            let plans = (1..=rows)
                .map(|m| {
                    let mut plan = Plan::new(gpu);
                    let saved = plan.params.clone();
                    for (r, p) in param_views.iter().enumerate().take(m) {
                        plan.params = p.clone();
                        record_embed(&mut plan, &self.embed, &row(&x, r, h));
                    }
                    plan.params = saved.clone();
                    Norm::rows(m, h, cfg.rms_norm_eps, &self.layers[0].input_norm)
                        .record(&mut plan, &x, &s.normed, false);
                    for (l, layer) in self.layers.iter().enumerate() {
                        let (next_norm, next_out) = match self.layers.get(l + 1) {
                            Some(next) => (&next.input_norm, &s.normed),
                            None => (&self.final_norm, &final_hidden),
                        };
                        layer.record_rows(
                            &mut plan,
                            cfg,
                            m,
                            &x,
                            &s,
                            snaps[l].as_ref(),
                            &param_views,
                            next_norm,
                            next_out,
                        );
                    }
                    rows::record_proj(&mut plan, &self.lm_head, m, &final_hidden, &logits, false);
                    let n = plan.constant(&[vocab as u32]);
                    plan.op(
                        "argmax_rows",
                        &[&logits, &argmax, &n],
                        MTLSize::new(m as u64, 1, 1),
                        MTLSize::new(1024, 1, 1),
                    );
                    // The model's sampler on every row, with that row's
                    // params, as Prefill::verify does.
                    for (r, p) in param_views.iter().enumerate().take(m) {
                        plan.params = p.clone();
                        self.sampler.record(&mut plan, &row(&logits, r, vocab));
                    }
                    plan.params = saved;
                    let built = plan.build();
                    (plan, built)
                })
                .collect();

            let restores = (0..rows)
                .map(|r| {
                    let mut plan = Plan::new(gpu);
                    for (layer, snap) in self.layers.iter().zip(&snaps) {
                        if let (Mixer::Linear(_, state), Some(snap)) = (&layer.mixer, snap) {
                            snap.record_restore(&mut plan, r, state);
                        }
                    }
                    plan.build()
                })
                .collect();

            self.verify = Some(GpuVerify {
                rows,
                plans,
                restores,
                params,
                final_hidden,
                logits,
                argmax,
                last: None,
            });
        }

        /// Runs `tokens` at the current position and the ones after it, and
        /// returns the argmax of the logits after each. The states move on
        /// by all of them; `accept` then keeps the first n.
        pub fn verify(&mut self, tokens: &[u32]) -> Vec<u32> {
            let pos = self.position;
            let v = self.verify.as_mut().expect("enable_verify first");
            let m = tokens.len();
            assert!(m >= 1 && m <= v.rows, "verify takes 1 to {} tokens", v.rows);
            assert!(pos + m <= self.capacity, "context is full");
            let p = v.params.contents() as *mut TokenParams;
            for (r, &t) in tokens.iter().enumerate() {
                unsafe { *p.add(r) = TokenParams::new(t, (pos + r) as u32) };
            }
            v.plans[m - 1].1.run(self.gpu);
            v.last = Some((m, pos));
            self.position = pos + m;
            contents::<u32>(&v.argmax.buffer, m).to_vec()
        }

        /// Keeps the first `n` tokens of the last `verify` (1 to its rows)
        /// and drops the rest: the DeltaNet states go back to the ones
        /// after token n, and the position to the verify position + n.
        pub fn accept(&mut self, n: usize) {
            let v = self.verify.as_mut().expect("enable_verify first");
            let (m, pos) = v.last.take().expect("accept follows verify");
            assert!(n >= 1 && n <= m, "accept 1 to {m} tokens");
            if n < m {
                v.restores[n - 1].run(self.gpu);
            }
            self.position = pos + n;
        }

        /// The final-norm hidden state after row `r` of the last verify,
        /// which the MTP layer takes.
        pub fn verify_hidden(&self, r: usize) -> View {
            let v = self.verify.as_ref().expect("enable_verify first");
            row(&v.final_hidden, r, self.cfg.hidden_size)
        }

        /// The logits after row `r` of the last verify.
        pub fn verify_logits(&self, r: usize) -> Vec<f32> {
            let v = self.verify.as_ref().expect("enable_verify first");
            let n = self.lm_head.rows;
            let all = contents::<f32>(&v.logits.buffer, (r + 1) * n);
            all[r * n..].to_vec()
        }

        /// The recorded verify of `m` rows, for profiling.
        pub fn verify_plan(&self, m: usize) -> &Plan<'a> {
            &self.verify.as_ref().expect("enable_verify first").plans[m - 1].0
        }

        /// GPU milliseconds of one verify of `m` rows at the current
        /// position, which it leaves unchanged.
        pub fn time_verify(&self, m: usize) -> f64 {
            let v = self.verify.as_ref().expect("enable_verify first");
            let p = v.params.contents() as *mut TokenParams;
            for r in 0..m {
                unsafe { *p.add(r) = TokenParams::new(0, (self.position + r) as u32) };
            }
            v.plans[m - 1].1.run_timed(self.gpu).1 * 1e3
        }
    }

    #[cfg(test)]
    mod tests {
        //! The verify against decoding the same tokens one by one:
        //!   scripts/mac.sh test --release --lib gpu_verify -- --ignored --nocapture
        use super::*;
        use crate::config::Config;
        use crate::plan::TokenParams;
        use crate::weights::Weights;
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";
        const PROMPT: [u32; 5] = [9707, 11, 1879, 0, 198];
        /// Six rows take two passes of attention_partial_rows.
        const DRAFT: [u32; 6] = [3838, 374, 279, 6722, 315, 1879];

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        /// Layer 0 on one row against the decode recording, stage by stage.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn layer_rows_match_layer() {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let (cfg, weights) = (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap());
            let gpu = Gpu::new().unwrap();
            let model = GpuModel::load(&gpu, &cfg, &weights, 1, 64).unwrap();
            let layer = &model.layers[0];
            let h = cfg.hidden_size;

            let ds = crate::gpu_model::Scratch::new(&gpu, &cfg);
            let dx = view(&gpu, h);
            let mut plan = Plan::new(&gpu);
            record_embed(&mut plan, &model.embed, &dx);
            layer.record_input_norm(&mut plan, &cfg, &dx, &ds);
            let normed0 = view(&gpu, h);
            crate::gpu_model::elementwise(&mut plan, "copy_u32", &[&ds.normed, &normed0], h, false);
            let out = view(&gpu, h);
            layer.record(&mut plan, &cfg, &dx, &ds, &model.final_norm, &out);
            plan.set_params(TokenParams::new(DRAFT[0], 0));
            plan.build().run(&gpu);

            let s = RowsScratch::new(&gpu, &cfg, 64, 1);
            let x = view(&gpu, MAX_ROWS * h);
            let params = View::new(gpu.buffer_from(&[TokenParams::new(DRAFT[0], 0)]));
            let rout = view(&gpu, MAX_ROWS * h);
            let mut plan = Plan::new(&gpu);
            plan.params = params.clone();
            record_embed(&mut plan, &model.embed, &x);
            Norm::rows(1, h, cfg.rms_norm_eps, &layer.input_norm)
                .record(&mut plan, &x, &s.normed, false);
            let rnormed0 = view(&gpu, h);
            crate::gpu_model::elementwise(&mut plan, "copy_u32", &[&s.normed, &rnormed0], h, false);
            layer.record_rows(
                &mut plan,
                &cfg,
                1,
                &x,
                &s,
                None,
                &[params],
                &model.final_norm,
                &rout,
            );
            layer.reset();
            plan.build().run(&gpu);

            let r = |v: &View, n: usize| contents::<f32>(&v.buffer, n).to_vec();
            let k = cfg.num_experts_per_tok;
            for (name, a, b, n) in [
                ("normed0", &normed0, &rnormed0, h),
                ("mixed", &ds.mixed, &s.mixed, h),
                ("moe normed", &ds.moe.normed, &s.moe_normed, h),
                ("router", &ds.moe.router, &s.router, cfg.num_experts + 1),
                ("weights", &ds.moe.weights, &s.weights, k),
                ("shared", &ds.moe.shared, &s.shared, h),
                ("routed", &ds.moe.routed, &s.routed, h),
                ("x", &dx, &x, h),
                ("out", &out, &rout, h),
            ] {
                let err = rel_rms(&r(b, n), &r(a, n));
                println!("{name}: rel rms {err:.2e}");
                assert!(err < 1e-5, "{name}");
            }
            assert_eq!(
                contents::<u32>(&ds.moe.ids.buffer, k),
                contents::<u32>(&s.ids.buffer, k)
            );
        }

        /// SAETTA_LAYERS layers (default 4; 40 needs timed.sh): after a
        /// prompt, a verify of 6 tokens gives the logits of decoding them
        /// one by one, and `accept` of each count leaves the model where
        /// decoding that many would.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn verify_matches_decode() {
            let layers: usize = std::env::var("SAETTA_LAYERS")
                .map(|v| v.parse().unwrap())
                .unwrap_or(4);
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let (cfg, weights) = (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap());
            let gpu = Gpu::new().unwrap();
            let mut model = GpuModel::load(&gpu, &cfg, &weights, layers, 512).unwrap();
            model.enable_verify(DRAFT.len());

            // 300 tokens, so the rows' keys cross a chunk boundary.
            let long: Vec<u32> = (0..300)
                .map(|i| PROMPT[i % PROMPT.len()] + i as u32)
                .collect();
            let prompt = |model: &mut GpuModel| {
                model.reset();
                for &t in &long {
                    model.run(t);
                }
            };
            prompt(&mut model);
            let want: Vec<Vec<f32>> = DRAFT.iter().map(|&t| model.forward(t)).collect();
            let after: Vec<f32> = model.forward(13);

            for n in 1..=DRAFT.len() {
                prompt(&mut model);
                let argmax = model.verify(&DRAFT);
                for (r, w) in want.iter().enumerate() {
                    let got = model.verify_logits(r);
                    let err = rel_rms(&got, w);
                    assert!(err < 1e-5, "row {r}: rel rms {err}");
                    assert_eq!(argmax[r], crate::cpu::model::argmax(w), "row {r}");
                    if n == 1 {
                        println!("row {r}: rel rms {err:.2e}");
                    }
                }
                model.accept(n);
                assert_eq!(model.position(), long.len() + n);
                // Decoding on from the accepted rows matches decoding them.
                for &t in &DRAFT[n..] {
                    model.run(t);
                }
                let got = model.forward(13);
                let err = rel_rms(&got, &after);
                println!("accept {n}: next token rel rms {err:.2e}");
                assert!(err < 1e-5, "accept {n}: rel rms {err}");
            }
        }

        /// GPU ms of a verify of 1 to 8 rows on all 40 layers, next to one
        /// decode token, at 40 tokens of context and at 32k. The cache
        /// holds zeros, which costs the same. Run it under timed.sh.
        #[test]
        #[ignore = "a timing run"]
        fn verify_time() {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let (cfg, weights) = (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap());
            let gpu = Gpu::new().unwrap();
            let mut model =
                GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), 33 * 1024).unwrap();
            model.enable_verify(MAX_ROWS);
            fn median(f: &mut dyn FnMut() -> f64) -> f64 {
                f();
                let mut t: Vec<f64> = (0..9).map(|_| f()).collect();
                t.sort_by(f64::total_cmp);
                t[4]
            }
            for ctx in [40, 32 * 1024] {
                model.set_position(ctx);
                let decode = median(&mut || {
                    model.plan.set_params(TokenParams::new(0, ctx as u32));
                    model.built.run_timed(&gpu).1 * 1e3
                });
                let rows: Vec<String> = (1..=MAX_ROWS)
                    .map(|m| format!("m{m} {:.2}", median(&mut || model.time_verify(m))))
                    .collect();
                println!(
                    "ctx {ctx}: decode {decode:.2} ms; verify {}",
                    rows.join(", ")
                );
                // Each kernel's ops of a 4-row verify alone, with barriers.
                let plan = model.verify_plan(4);
                for (name, ops) in plan.kernels() {
                    let built = plan.build_where(|k| k == name, true);
                    let ms = median(&mut || built.run_timed(&gpu).1 * 1e3);
                    if ms > 0.05 {
                        println!("  m4 {name}: {ops} ops {ms:.2} ms");
                    }
                }
            }
        }
    }
}
