//! The MTP layer on the GPU, as `mtp.rs` on the CPU: from a normed hidden
//! state at position `t` and the token at `t + 1`, it drafts the token at
//! `t + 2`. One step is one command buffer, recorded once into a `Plan`, and
//! the drafted token is picked on the GPU so only its id is read back.
//!
//! RoPE runs at the MTP cache length (`MtpPosition::Hidden`); the cache
//! holds one entry per step, as on the CPU.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::{Config, LayerType};
    use crate::gpu::{Gpu, contents};
    use crate::gpu_layers::Norm;
    use crate::gpu_model::{GpuLayer, Scratch, load_embed, matrix, record_embed, vector, view};
    use crate::plan::{Built, Plan, TokenParams};
    use crate::qmv::View;
    use crate::weights::Weights;
    use metal::MTLSize;

    const PREFIX: &str = "language_model.mtp";

    /// Records `out[0] = argmax(x[0..n])` as one threadgroup.
    pub fn record_argmax(plan: &mut Plan, x: &View, n: usize, out: &View) {
        let len = plan.constant(&[n as u32]);
        plan.op(
            "argmax",
            &[x, out, &len],
            MTLSize::new(1, 1, 1),
            MTLSize::new(1024, 1, 1),
        );
    }

    pub struct GpuMtp<'a> {
        gpu: &'a Gpu,
        cfg: &'a Config,
        layer: GpuLayer<'a>,
        plan: Plan<'a>,
        built: Built,
        /// The same step without lm_head and the argmax, for `record`.
        record_plan: Plan<'a>,
        record_built: Built,
        input: View,
        output: View,
        token: View,
        capacity: usize,
        position: usize,
    }

    impl<'a> GpuMtp<'a> {
        /// Loads the layer with room for `capacity` steps in its cache.
        pub fn load(
            gpu: &'a Gpu,
            cfg: &'a Config,
            weights: &'a Weights,
            capacity: usize,
        ) -> Result<GpuMtp<'a>, String> {
            let hidden = cfg.hidden_size;
            let eps = cfg.rms_norm_eps;
            let embed = load_embed(gpu, cfg, weights)?;
            let v = |n: &str| vector(gpu, weights, &format!("{PREFIX}.{n}.weight"));
            let pre_embedding = v("pre_fc_norm_embedding")?;
            let pre_hidden = v("pre_fc_norm_hidden")?;
            let norm = v("norm")?;
            let fc = matrix(gpu, weights, cfg, &format!("{PREFIX}.fc"))?;
            let layer = GpuLayer::load(
                gpu,
                cfg,
                weights,
                &format!("{PREFIX}.layers.0"),
                LayerType::FullAttention,
                capacity,
            )?;
            let lm_head = matrix(gpu, weights, cfg, "language_model.lm_head")?;

            let scratch = Scratch::new(gpu, cfg);
            let input = view(gpu, hidden);
            // fc's input: the normed embedding, then the normed hidden state.
            let cat = view(gpu, 2 * hidden);
            let cat_hidden = View {
                buffer: cat.buffer.clone(),
                offset: (hidden * 4) as u64,
            };
            let x = view(gpu, hidden);
            let output = view(gpu, hidden);
            let logits = view(gpu, lm_head.rows);
            let token = View::new(gpu.buffer(4));

            // Everything up to the output hidden state; `step` adds lm_head
            // and the argmax.
            let layer_plan = || {
                let mut plan = Plan::new(gpu);
                record_embed(&mut plan, &embed, &cat);
                Norm::rows(1, hidden, eps, &pre_hidden).record(
                    &mut plan,
                    &input,
                    &cat_hidden,
                    true,
                );
                Norm::rows(1, hidden, eps, &pre_embedding).record(&mut plan, &cat, &cat, false);
                fc.record(&mut plan, &cat, &x, false);
                layer.record(&mut plan, cfg, &x, &scratch);
                Norm::rows(1, hidden, eps, &norm).record(&mut plan, &x, &output, false);
                plan
            };
            let mut plan = layer_plan();
            lm_head.record(&mut plan, &output, &logits, false);
            record_argmax(&mut plan, &logits, lm_head.rows, &token);
            let built = plan.build();
            let record_plan = layer_plan();
            let record_built = record_plan.build();

            Ok(GpuMtp {
                gpu,
                cfg,
                layer,
                plan,
                built,
                record_plan,
                record_built,
                input,
                output,
                token,
                capacity,
                position: 0,
            })
        }

        /// One step at position `len()`: `hidden` is a normed hidden state,
        /// the main model's (`GpuModel::final_hidden`) or this layer's own
        /// `output()`, and `token` the token after it. Returns the drafted
        /// token; `output()` then holds the new hidden state.
        pub fn step(&mut self, hidden: &View, token: u32) -> u32 {
            self.run(hidden, token, true);
            contents::<u32>(&self.token.buffer, 1)[0]
        }

        /// `step` without the draft: adds the cache entry and sets
        /// `output()`, but skips lm_head, which is most of a step.
        pub fn record(&mut self, hidden: &View, token: u32) {
            self.run(hidden, token, false);
        }

        fn run(&mut self, hidden: &View, token: u32, draft: bool) {
            assert!(
                self.position < self.capacity,
                "MTP cache is full ({} steps)",
                self.capacity
            );
            copy(hidden, &self.input, self.cfg.hidden_size);
            let params = TokenParams::new(token, self.position as u32);
            if draft {
                self.plan.set_params(params);
                self.built.run(self.gpu);
            } else {
                self.record_plan.set_params(params);
                self.record_built.run(self.gpu);
            }
            self.position += 1;
        }

        /// The normed hidden state of the last step.
        pub fn output(&self) -> &View {
            &self.output
        }

        /// Steps in the cache.
        pub fn len(&self) -> usize {
            self.position
        }

        pub fn is_empty(&self) -> bool {
            self.position == 0
        }

        /// Keeps the first `len` steps of the cache.
        pub fn truncate(&mut self, len: usize) {
            assert!(len <= self.position);
            self.position = len;
        }

        pub fn reset(&mut self) {
            self.layer.reset();
            self.position = 0;
        }
    }

    /// Copies `n` floats between shared buffers while the GPU is idle.
    fn copy(from: &View, to: &View, n: usize) {
        unsafe {
            let src = (from.buffer.contents() as *const u8).add(from.offset as usize);
            let dst = (to.buffer.contents() as *mut u8).add(to.offset as usize);
            std::ptr::copy(src, dst, n * 4);
        }
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cpu::model::{LayerState, Model, argmax};
        use crate::mtp::{Mtp, MtpPosition};
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            assert_eq!(got.len(), want.len());
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        #[test]
        fn argmax_matches_cpu() {
            let gpu = Gpu::new().unwrap();
            for n in [1, 31, 1000, 248320] {
                let mut x: Vec<f32> = (0..n).map(|i| ((i * 7919) as f32 * 0.37).sin()).collect();
                if n > 900 {
                    x[700] = 5.0;
                    x[n - 1] = 5.0;
                    x[300] = 5.0;
                }
                let input = View::new(gpu.buffer_from(&x));
                let out = View::new(gpu.buffer(4));
                let mut plan = Plan::new(&gpu);
                record_argmax(&mut plan, &input, n, &out);
                plan.build().run(&gpu);
                assert_eq!(contents::<u32>(&out.buffer, 1)[0], argmax(&x), "n {n}");
            }
        }

        /// A made-up normed hidden state, standing in for the main model's.
        fn hidden(t: usize) -> Vec<f32> {
            let x: Vec<f32> = (0..2048)
                .map(|i| ((i * 37 + t * 101) as f32 * 0.013).sin())
                .collect();
            let rms = (x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32).sqrt();
            x.iter().map(|v| v / rms).collect()
        }

        /// The CPU MTP step and the drafted token, as the speculative loop
        /// takes it.
        fn cpu_step(
            mtp: &Mtp,
            model: &Model,
            h: &[f32],
            token: u32,
            state: &mut LayerState,
        ) -> (Vec<f32>, u32) {
            let out = mtp.step(model, h, token, state).unwrap();
            let t = argmax(&model.logits(&out));
            (out, t)
        }

        /// A round as in `speculative::generate`: steps from main-model
        /// hidden states, 3 drafts from the layer's own output, then going
        /// back and stepping from a main-model state again.
        #[test]
        #[ignore = "needs the model and mtp.safetensors on the Mac"]
        fn matches_cpu_mtp() {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let cfg = Config::load(&dir).unwrap();
            let weights = Weights::open(&dir).unwrap();
            let model = Model::new(&cfg, &weights).unwrap();
            let mtp = Mtp::new(&cfg, &weights, MtpPosition::Hidden).unwrap();
            let gpu = Gpu::new().unwrap();
            let mut g = GpuMtp::load(&gpu, &cfg, &weights, 64).unwrap();
            let mut state = mtp.new_state();
            let main = |t: usize| View::new(gpu.buffer_from(&hidden(t)));
            let check = |what: &str, g: &GpuMtp, got: u32, want: &(Vec<f32>, u32)| {
                let out = contents::<f32>(&g.output().buffer, 2048);
                let err = rel_rms(out, &want.0);
                println!("{what}: token {got}, rel rms {err:.2e}");
                assert!(err < 1e-5, "{what}: rel rms {err}");
                assert_eq!(got, want.1, "{what}");
            };

            let tokens = [9707u32, 11, 1879, 0, 198];
            for (t, &token) in tokens.iter().enumerate() {
                let want = cpu_step(&mtp, &model, &hidden(t), token, &mut state);
                let got = g.step(&main(t), token);
                check(&format!("main {t}"), &g, got, &want);
            }
            let start = g.len();
            let (mut cpu_h, mut token) = (hidden(9), 3838);
            let mut drafted = g.step(&main(9), token);
            for d in 0..3 {
                let want = cpu_step(&mtp, &model, &cpu_h, token, &mut state);
                if d > 0 {
                    let own = g.output().clone();
                    drafted = g.step(&own, token);
                }
                check(&format!("draft {d}"), &g, drafted, &want);
                (cpu_h, token) = (want.0, want.1);
            }

            g.truncate(start + 1);
            if let LayerState::Full(kv) = &mut state {
                kv.truncate(start + 1);
            }
            let want = cpu_step(&mtp, &model, &hidden(20), 42, &mut state);
            let got = g.step(&main(20), 42);
            check("after truncate", &g, got, &want);

            let want = cpu_step(&mtp, &model, &hidden(21), 7, &mut state);
            g.record(&main(21), 7);
            check("record", &g, want.1, &want);
            let want = cpu_step(&mtp, &model, &hidden(22), 8, &mut state);
            let got = g.step(&main(22), 8);
            check("step after record", &g, got, &want);
            let got = want.1;

            let mut times = Vec::new();
            let mut token = got;
            for _ in 0..20 {
                let own = g.output().clone();
                let start = std::time::Instant::now();
                token = g.step(&own, token);
                times.push(start.elapsed());
            }
            times.sort();
            println!("step: best {:.2?}, median {:.2?}", times[0], times[10]);
        }
    }
}
