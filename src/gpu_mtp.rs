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
    use crate::gpu_model::{
        GpuLayer, Scratch, kv_grow, kv_start, load_embed, matrix, record_embed, vector, view,
    };
    use crate::plan::{Built, Plan, TokenParams};
    use crate::qmv::{GpuQuant, View};
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

    /// What the step plans read and write, kept to record them again when
    /// the cache grows.
    struct Parts<'a> {
        embed: GpuQuant<'a>,
        pre_embedding: View,
        pre_hidden: View,
        norm: View,
        fc: GpuQuant<'a>,
        /// lm_head, or the parts of it the drafts pick from (`draft_heads`),
        /// each with its first token id. Their logits sit one after another.
        heads: Vec<(usize, GpuQuant<'a>)>,
        lm_head: GpuQuant<'a>,
        scratch: Scratch,
        /// fc's input: the normed embedding, then the normed hidden state.
        cat: View,
        x: View,
        logits: View,
    }

    pub struct GpuMtp<'a> {
        gpu: &'a Gpu,
        cfg: &'a Config,
        layer: GpuLayer<'a>,
        parts: Parts<'a>,
        plan: Plan<'a>,
        built: Built,
        /// The same step without lm_head and the argmax, for `record`.
        record_plan: Plan<'a>,
        record_built: Built,
        input: View,
        output: View,
        token: View,
        capacity: usize,
        /// The most the cache may grow to.
        limit: usize,
        position: usize,
    }

    impl<'a> GpuMtp<'a> {
        /// Loads the layer with room for up to `limit` steps in its cache,
        /// which starts at `kv_start(limit)` and grows by `reserve`.
        pub fn load(
            gpu: &'a Gpu,
            cfg: &'a Config,
            weights: &'a Weights,
            limit: usize,
        ) -> Result<GpuMtp<'a>, String> {
            let hidden = cfg.hidden_size;
            let v = |n: &str| vector(gpu, weights, &format!("{PREFIX}.{n}.weight"));
            let capacity = kv_start(limit);
            let layer = GpuLayer::load(
                gpu,
                cfg,
                weights,
                &format!("{PREFIX}.layers.0"),
                LayerType::FullAttention,
                capacity,
            )?;
            let lm_head = matrix(gpu, weights, cfg, "language_model.lm_head")?;
            let heads = draft_heads(&lm_head, cfg, vocab_from_env());
            let logits = view(gpu, heads.iter().map(|(_, h)| h.rows).sum());
            let parts = Parts {
                embed: load_embed(gpu, cfg, weights)?,
                pre_embedding: v("pre_fc_norm_embedding")?,
                pre_hidden: v("pre_fc_norm_hidden")?,
                norm: v("norm")?,
                fc: matrix(gpu, weights, cfg, &format!("{PREFIX}.fc"))?,
                heads,
                lm_head,
                scratch: Scratch::new(gpu, cfg),
                cat: view(gpu, 2 * hidden),
                x: view(gpu, hidden),
                logits,
            };
            let input = view(gpu, hidden);
            let output = view(gpu, hidden);
            let token = View::new(gpu.buffer(4));
            let (plan, record_plan) =
                record_plans(gpu, cfg, &layer, &parts, &input, &output, &token);
            let (built, record_built) = (plan.build(), record_plan.build());

            Ok(GpuMtp {
                gpu,
                cfg,
                layer,
                parts,
                plan,
                built,
                record_plan,
                record_built,
                input,
                output,
                token,
                capacity,
                limit,
                position: 0,
            })
        }

        /// Makes room for `need` steps, growing the cache if it is smaller,
        /// up to the limit.
        pub fn reserve(&mut self, need: usize) {
            if need <= self.capacity {
                return;
            }
            let capacity = kv_grow(self.capacity, need, self.limit);
            self.resize(capacity);
        }

        /// Shrinks the cache to the smallest step that holds `need` steps,
        /// when that is at most half of what it holds now, as
        /// `GpuModel::shrink` does.
        pub fn shrink(&mut self, need: usize) {
            let fit = kv_grow(kv_start(self.limit), need.clamp(1, self.limit), self.limit);
            if fit * 2 <= self.capacity {
                self.resize(fit);
            }
        }

        fn resize(&mut self, capacity: usize) {
            self.layer.resize(self.gpu, capacity);
            self.capacity = capacity;
            self.rebuild();
        }

        /// Drafts from the first `vocab` ids and the special tokens only, as
        /// SAETTA_MTP_VOCAB does; 0 drafts from all of them.
        pub fn set_draft_vocab(&mut self, vocab: usize) {
            let p = &mut self.parts;
            p.heads = draft_heads(&p.lm_head, self.cfg, vocab);
            p.logits = view(self.gpu, p.heads.iter().map(|(_, h)| h.rows).sum());
            self.rebuild();
        }

        fn rebuild(&mut self) {
            let (plan, record_plan) = record_plans(
                self.gpu,
                self.cfg,
                &self.layer,
                &self.parts,
                &self.input,
                &self.output,
                &self.token,
            );
            self.built = plan.build();
            self.record_built = record_plan.build();
            self.plan = plan;
            self.record_plan = record_plan;
        }

        /// The KV cache's buffer bytes as allocated now.
        pub fn kv_bytes(&self) -> u64 {
            self.layer.kv_bytes()
        }

        /// One step at position `len()`: `hidden` is a normed hidden state,
        /// the main model's (`GpuModel::final_hidden`) or this layer's own
        /// `output()`, and `token` the token after it. Returns the drafted
        /// token; `output()` then holds the new hidden state.
        pub fn step(&mut self, hidden: &View, token: u32) -> u32 {
            self.run(hidden, token, true);
            let mut i = contents::<u32>(&self.token.buffer, 1)[0] as usize;
            for (first, head) in &self.parts.heads {
                if i < head.rows {
                    return (first + i) as u32;
                }
                i -= head.rows;
            }
            unreachable!("argmax past the logits")
        }

        /// `step` without the draft: adds the cache entry and sets
        /// `output()`, but skips lm_head, which is most of a step.
        pub fn record(&mut self, hidden: &View, token: u32) {
            self.run(hidden, token, false);
        }

        fn run(&mut self, hidden: &View, token: u32, draft: bool) {
            assert!(
                self.position < self.limit,
                "MTP cache is full ({} steps)",
                self.limit
            );
            self.reserve(self.position + 1);
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

    /// Token ids the drafts pick from by default, besides the special
    /// tokens. Measured with spec_bench (batch 6): 65536 keeps the
    /// acceptance closer to the full vocabulary than 32768 (0.81, 0.66,
    /// 0.72 vs 0.84, 0.55, 0.58 by draft) at the same speed, code 163.7 vs
    /// 152.6 tok/s and prose 132.9 vs 122.0; 16384 gains nothing.
    pub const DRAFT_VOCAB: usize = 65536;

    /// SAETTA_MTP_VOCAB=N drafts only from the first N token ids and the
    /// special tokens; 0 drafts from the whole vocabulary. Unset is
    /// `DRAFT_VOCAB`.
    fn vocab_from_env() -> usize {
        std::env::var("SAETTA_MTP_VOCAB")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DRAFT_VOCAB)
    }

    /// The parts of lm_head the drafts pick from. With `vocab` > 0, the
    /// first `vocab` ids (byte-level BPE gives the earlier merges, the
    /// common tokens, the lower ids) and the special tokens from the
    /// lowest end-of-text id on, which tool calls and the chat template
    /// need. lm_head is most of a draft step, so 64k of 248k ids make a
    /// step several times cheaper; a draft outside them is a miss, never a
    /// wrong output.
    fn draft_heads<'a>(
        lm_head: &GpuQuant<'a>,
        cfg: &Config,
        vocab: usize,
    ) -> Vec<(usize, GpuQuant<'a>)> {
        let rows = lm_head.rows;
        // qmv writes 8 rows per threadgroup; keep both parts whole.
        let special = cfg
            .eos_token_ids
            .iter()
            .copied()
            .min()
            .map_or(rows, |t| t as usize)
            / 8
            * 8;
        let vocab = vocab.next_multiple_of(8);
        if vocab == 0 || vocab >= special {
            return vec![(0, lm_head.row_range(0, rows))];
        }
        vec![
            (0, lm_head.row_range(0, vocab)),
            (special, lm_head.row_range(special, rows - special)),
        ]
    }

    /// The step plan (with lm_head and the argmax) and the record plan
    /// (without), both up to the output hidden state.
    fn record_plans<'a>(
        gpu: &'a Gpu,
        cfg: &Config,
        layer: &GpuLayer,
        p: &Parts,
        input: &View,
        output: &View,
        token: &View,
    ) -> (Plan<'a>, Plan<'a>) {
        let (hidden, eps) = (cfg.hidden_size, cfg.rms_norm_eps);
        let cat_hidden = View {
            buffer: p.cat.buffer.clone(),
            offset: p.cat.offset + (hidden * 4) as u64,
        };
        let layer_plan = || {
            let mut plan = Plan::new(gpu);
            record_embed(&mut plan, &p.embed, &p.cat);
            Norm::rows(1, hidden, eps, &p.pre_hidden).record(&mut plan, input, &cat_hidden, true);
            Norm::rows(1, hidden, eps, &p.pre_embedding).record(&mut plan, &p.cat, &p.cat, false);
            p.fc.record(&mut plan, &p.cat, &p.x, false);
            layer.record_input_norm(&mut plan, cfg, &p.x, &p.scratch);
            layer.record(&mut plan, cfg, &p.x, &p.scratch, &p.norm, output);
            plan
        };
        let mut plan = layer_plan();
        let mut n = 0;
        for (_, head) in &p.heads {
            let at = View {
                buffer: p.logits.buffer.clone(),
                offset: p.logits.offset + (n * 4) as u64,
            };
            head.record(&mut plan, output, &at, false);
            n += head.rows;
        }
        record_argmax(&mut plan, &p.logits, n, token);
        (plan, layer_plan())
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

        /// With a small draft vocabulary the draft is the argmax of the full
        /// logits over the ids kept: the first ones and the special tokens.
        #[test]
        #[ignore = "needs the model and mtp.safetensors on the Mac"]
        fn small_draft_vocab_takes_the_argmax_of_its_ids() {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            let cfg = Config::load(&dir).unwrap();
            let weights = Weights::open(&dir).unwrap();
            let gpu = Gpu::new().unwrap();
            let mut g = GpuMtp::load(&gpu, &cfg, &weights, 64).unwrap();
            let vocab = 4096;
            let special = *cfg.eos_token_ids.iter().min().unwrap() as usize / 8 * 8;
            let tokens = [9707u32, 11, 1879, 0, 198, 248046];
            let (mut hits, mut special_hits) = (0, 0);
            for (t, &token) in tokens.iter().enumerate() {
                let h = View::new(gpu.buffer_from(&hidden(t)));
                g.set_draft_vocab(0);
                let full = g.step(&h, token);
                let logits = contents::<f32>(&g.parts.logits.buffer, cfg.vocab_size).to_vec();
                assert_eq!(full, argmax(&logits));
                let mut kept = logits.clone();
                for (i, l) in kept.iter_mut().enumerate() {
                    if (vocab..special).contains(&i) {
                        *l = f32::NEG_INFINITY;
                    }
                }
                let want = argmax(&kept) as usize;
                g.truncate(g.len() - 1);
                g.set_draft_vocab(vocab);
                let got = g.step(&h, token) as usize;
                println!("step {t}: full {full}, small {got}, want {want}");
                assert_eq!(got, want);
                hits += usize::from(got as u32 == full);
                special_hits += usize::from(got >= special);
            }
            println!(
                "{hits} of {} drafts equal, {special_hits} special",
                tokens.len()
            );
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
            // The CPU drafts from the whole vocabulary.
            g.set_draft_vocab(0);
            let mut state = mtp.new_state();
            // The GPU stores the MTP keys and values as half; so does the CPU.
            if let LayerState::Full(c) = &mut state {
                c.round_to_half();
            }
            let main = |t: usize| View::new(gpu.buffer_from(&hidden(t)));
            let check = |what: &str, g: &GpuMtp, got: u32, want: &(Vec<f32>, u32)| {
                let out = contents::<f32>(&g.output().buffer, 2048);
                let err = rel_rms(out, &want.0);
                println!("{what}: token {got}, rel rms {err:.2e}");
                // As whole_model_matches_cpu: both round the keys to half, but
                // a key now and then lands on the next half value (1.6e-5).
                assert!(err < 1e-4, "{what}: rel rms {err}");
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

            let mut times = Vec::new();
            for _ in 0..20 {
                let own = g.output().clone();
                let start = std::time::Instant::now();
                g.record(&own, token);
                times.push(start.elapsed());
            }
            times.sort();
            println!("record: best {:.2?}, median {:.2?}", times[0], times[10]);
        }
    }
}
