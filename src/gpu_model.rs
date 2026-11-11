//! The whole model on the GPU for one token at a time (`kernels/model.metal`
//! and the kernels of the layers). Every op of a token is recorded once
//! into a `Plan` and replayed from an indirect command buffer, so a token is
//! one command buffer. The weights are wrapped from the mmap without a copy.
//!
//! Each layer follows `cpu::model::Layer::step`: `h = x + mixer(norm(x))`,
//! then `h + moe(norm(h))`, where the MoE is the router (softmax, top 8,
//! renormalized), the routed experts and the gated shared expert.

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::config::{Config, LayerType};
    use crate::gpu::{Gpu, contents};
    use crate::gpu_layers::{
        GpuAttention, GpuDeltaNet, GpuDeltaNetState, GpuKvCache, Norm, add_norm, mix_norm,
    };
    use crate::moe::{Experts, Router};
    use crate::plan::{Built, Plan, TokenParams};
    use crate::qmv::{GpuQuant, View};
    use crate::sample::{Params, Sampler};
    use crate::sample_gpu::GpuSampler;
    use crate::weights::Weights;
    use metal::{Buffer, MTLSize};

    const PREFIX: &str = "language_model.model";
    /// Threads per threadgroup of the element-wise kernels.
    const GROUP: usize = 256;

    pub(crate) fn view(gpu: &Gpu, floats: usize) -> View {
        View::new(gpu.buffer(floats * 4))
    }

    pub(crate) fn vector(gpu: &Gpu, weights: &Weights, name: &str) -> Result<View, String> {
        Ok(View::new(gpu.buffer_from(&weights.tensor(name)?.to_f32())))
    }

    pub(crate) fn matrix<'a>(
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

    /// An element-wise op over `n` values; `n` goes in as the last buffer.
    pub(crate) fn elementwise(
        plan: &mut Plan,
        kernel: &str,
        buffers: &[&View],
        n: usize,
        parallel: bool,
    ) {
        let len = plan.constant(&[n as u32]);
        let mut all = buffers.to_vec();
        all.push(&len);
        let grid = MTLSize::new(n.div_ceil(GROUP) as u64, 1, 1);
        let group = MTLSize::new(GROUP as u64, 1, 1);
        if parallel {
            plan.op_parallel(kernel, &all, grid, group);
        } else {
            plan.op(kernel, &all, grid, group);
        }
    }

    /// Records the router: `ids` and `weights` get the `top_k` best of the
    /// softmax over `logits`, renormalized. One simdgroup.
    pub fn record_route(
        plan: &mut Plan,
        logits: &View,
        experts: usize,
        top_k: usize,
        ids: &View,
        weights: &View,
    ) {
        // ROUTE_PER_LANE values in each of 32 lanes.
        assert!(experts <= 256 && top_k <= 16);
        let shape = plan.constant(&[experts as u32, top_k as u32]);
        plan.op(
            "route_top_k",
            &[logits, ids, weights, &shape],
            MTLSize::new(1, 1, 1),
            MTLSize::new(32, 1, 1),
        );
    }

    /// The scratch buffers of the MoE block, shared by all layers.
    struct MoeScratch {
        normed: View,
        /// The router's output: `num_experts` logits, then the shared
        /// expert gate logit, which `shared_logit` points at.
        router: View,
        ids: View,
        weights: View,
        shared_logit: View,
        shared_gate: View,
        shared_up: View,
        shared_h: View,
        shared: View,
        experts_h: View,
        routed: View,
    }

    /// The scratch buffers of one layer step, shared by all layers.
    pub(crate) struct Scratch {
        normed: View,
        mixed: View,
        moe: MoeScratch,
    }

    impl Scratch {
        pub(crate) fn new(gpu: &Gpu, cfg: &Config) -> Scratch {
            let hidden = cfg.hidden_size;
            let shared_size = cfg.shared_expert_intermediate_size;
            let router = view(gpu, cfg.num_experts + 1);
            Scratch {
                normed: view(gpu, hidden),
                mixed: view(gpu, hidden),
                moe: MoeScratch {
                    normed: view(gpu, hidden),
                    router: router.clone(),
                    ids: view(gpu, cfg.num_experts_per_tok),
                    weights: view(gpu, cfg.num_experts_per_tok),
                    shared_logit: View {
                        buffer: router.buffer,
                        offset: (cfg.num_experts * 4) as u64,
                    },
                    shared_gate: view(gpu, shared_size),
                    shared_up: view(gpu, shared_size),
                    shared_h: view(gpu, shared_size),
                    shared: view(gpu, hidden),
                    experts_h: view(gpu, cfg.num_experts_per_tok * cfg.moe_intermediate_size),
                    routed: view(gpu, hidden),
                },
            }
        }
    }

    struct GpuMoe<'a> {
        post_norm: View,
        /// Router logits then the shared expert gate, 257 values.
        router: Router<'a>,
        shared_gate: GpuQuant<'a>,
        shared_up: GpuQuant<'a>,
        shared_down: GpuQuant<'a>,
        experts: Experts<'a>,
    }

    impl<'a> GpuMoe<'a> {
        fn load(
            gpu: &Gpu,
            cfg: &Config,
            weights: &'a Weights,
            prefix: &str,
        ) -> Result<GpuMoe<'a>, String> {
            let m = |n: &str| matrix(gpu, weights, cfg, &format!("{prefix}.mlp.{n}"));
            Ok(GpuMoe {
                post_norm: vector(
                    gpu,
                    weights,
                    &format!("{prefix}.post_attention_layernorm.weight"),
                )?,
                router: Router::new(gpu, weights, &cfg.quantization, prefix)?,
                shared_gate: m("shared_expert.gate_proj")?,
                shared_up: m("shared_expert.up_proj")?,
                shared_down: m("shared_expert.down_proj")?,
                experts: Experts::new(gpu, weights, &cfg.quantization, prefix)?,
            })
        }

        /// `x += moe(normed)`, where `s.normed` already holds
        /// `post_norm(x)`; then `next_out = rms_norm(x) * next_norm`.
        fn record(
            &self,
            plan: &mut Plan,
            cfg: &Config,
            x: &View,
            s: &MoeScratch,
            next_norm: &View,
            next_out: &View,
        ) {
            let hidden = cfg.hidden_size;
            self.router.record(plan, &s.normed, &s.router, false);
            self.shared_gate
                .record(plan, &s.normed, &s.shared_gate, true);
            self.shared_up.record(plan, &s.normed, &s.shared_up, true);
            record_route(
                plan,
                &s.router,
                cfg.num_experts,
                cfg.num_experts_per_tok,
                &s.ids,
                &s.weights,
            );
            elementwise(
                plan,
                "silu_mul",
                &[&s.shared_gate, &s.shared_up, &s.shared_h],
                self.shared_gate.rows,
                true,
            );
            // The barrier before shared_down also covers route_top_k, so the
            // routed experts' first op can run alongside it.
            self.shared_down.record(plan, &s.shared_h, &s.shared, false);
            let k = cfg.num_experts_per_tok;
            self.experts
                .record_gate_up(plan, &s.normed, &s.ids, k, &s.experts_h, true);
            self.experts
                .record_down(plan, &s.experts_h, &s.ids, &s.weights, k, &s.routed);
            mix_norm(
                plan,
                [
                    x,
                    &s.shared_logit,
                    &s.shared,
                    &s.routed,
                    next_norm,
                    next_out,
                ],
                hidden,
                cfg.rms_norm_eps,
            );
        }
    }

    enum Mixer<'a> {
        Linear(GpuDeltaNet<'a>, GpuDeltaNetState),
        Full(GpuAttention<'a>, GpuKvCache),
    }

    pub(crate) struct GpuLayer<'a> {
        input_norm: View,
        mixer: Mixer<'a>,
        moe: GpuMoe<'a>,
    }

    impl<'a> GpuLayer<'a> {
        /// The layer under `prefix`, such as `language_model.model.layers.3`.
        /// An attention layer gets a cache of `capacity` tokens.
        pub(crate) fn load(
            gpu: &Gpu,
            cfg: &Config,
            weights: &'a Weights,
            prefix: &str,
            kind: LayerType,
            capacity: usize,
        ) -> Result<GpuLayer<'a>, String> {
            let mixer = match kind {
                LayerType::LinearAttention => {
                    let m = GpuDeltaNet::load(gpu, cfg, weights, prefix)?;
                    let state = GpuDeltaNetState::new(gpu, &m.cfg);
                    Mixer::Linear(m, state)
                }
                LayerType::FullAttention => {
                    let m = GpuAttention::load(gpu, cfg, weights, prefix)?;
                    let cache = GpuKvCache::new(gpu, &m.cfg, capacity);
                    Mixer::Full(m, cache)
                }
            };
            Ok(GpuLayer {
                input_norm: vector(gpu, weights, &format!("{prefix}.input_layernorm.weight"))?,
                mixer,
                moe: GpuMoe::load(gpu, cfg, weights, prefix)?,
            })
        }

        /// Records `s.normed = input_norm(x)`, which `record` expects. Only
        /// the first layer needs it; each layer writes the next one's.
        pub(crate) fn record_input_norm(
            &self,
            plan: &mut Plan,
            cfg: &Config,
            x: &View,
            s: &Scratch,
        ) {
            Norm::rows(1, cfg.hidden_size, cfg.rms_norm_eps, &self.input_norm)
                .record(plan, x, &s.normed, false);
        }

        pub(crate) fn input_norm(&self) -> &View {
            &self.input_norm
        }

        /// `x += mixer(s.normed)`, then `x += moe(post_norm(x))`, then
        /// `next_out = rms_norm(x) * next_norm`: the next layer's input norm
        /// into `s.normed`, or the final norm. `s.normed` must hold
        /// `input_norm(x)` (`record_input_norm`, or the layer before).
        pub(crate) fn record(
            &self,
            plan: &mut Plan,
            cfg: &Config,
            x: &View,
            s: &Scratch,
            next_norm: &View,
            next_out: &View,
        ) {
            match &self.mixer {
                Mixer::Linear(m, state) => m.record(plan, &s.normed, &s.mixed, state),
                Mixer::Full(m, cache) => m.record(plan, &s.normed, &s.mixed, cache),
            }
            add_norm(
                plan,
                [x, &s.mixed, &self.moe.post_norm, &s.moe.normed],
                cfg.hidden_size,
                cfg.rms_norm_eps,
            );
            self.moe.record(plan, cfg, x, &s.moe, next_norm, next_out);
        }

        /// Grows an attention layer's cache to `capacity` tokens; a DeltaNet
        /// layer has nothing to grow.
        pub(crate) fn grow(&mut self, gpu: &Gpu, capacity: usize) {
            if let Mixer::Full(_, cache) = &mut self.mixer {
                cache.grow(gpu, capacity);
            }
        }

        /// Bytes of the attention cache as allocated now, 0 for DeltaNet.
        pub(crate) fn kv_bytes(&self) -> u64 {
            match &self.mixer {
                Mixer::Full(_, c) => c.keys.length() + c.values.length(),
                Mixer::Linear(..) => 0,
            }
        }

        /// Clears a DeltaNet state; an attention cache needs no clearing.
        pub(crate) fn reset(&self) {
            if let Mixer::Linear(_, state) = &self.mixer {
                zero(&state.conv);
                zero(&state.recurrent);
            }
        }
    }

    /// The 5-bit embedding table, checked for the kernel.
    pub(crate) fn load_embed<'a>(
        gpu: &Gpu,
        cfg: &Config,
        weights: &'a Weights,
    ) -> Result<GpuQuant<'a>, String> {
        let embed = matrix(gpu, weights, cfg, &format!("{PREFIX}.embed_tokens"))?;
        if embed.bits != 5 {
            return Err(format!(
                "embedding is {}-bit, the kernel is 5-bit",
                embed.bits
            ));
        }
        Ok(embed)
    }

    /// Tokens the attention caches start with: Metal makes a whole buffer
    /// resident once a kernel uses it, so a short chat must not pay for the
    /// full context.
    pub const KV_START: usize = 32768;

    /// The first capacity for a context of `limit` tokens.
    pub fn kv_start(limit: usize) -> usize {
        KV_START.min(limit)
    }

    /// The capacity to grow to from `capacity` for `need` tokens: doubled
    /// until it fits, at most `limit`.
    pub fn kv_grow(capacity: usize, need: usize, limit: usize) -> usize {
        assert!(
            need <= limit,
            "{need} tokens do not fit in {limit} of context"
        );
        let mut c = capacity.max(1);
        while c < need {
            c *= 2;
        }
        c.min(limit)
    }

    /// `x = embed[params.token]`.
    pub(crate) fn record_embed(plan: &mut Plan, embed: &GpuQuant, x: &View) {
        let cols = embed.cols;
        let shape = plan.constant(&[cols as u32]);
        let params = plan.params.clone();
        plan.op(
            "embed_5bit",
            &[
                &embed.words,
                &embed.scales,
                &embed.biases,
                &params,
                x,
                &shape,
            ],
            MTLSize::new(cols.div_ceil(GROUP) as u64, 1, 1),
            MTLSize::new(GROUP as u64, 1, 1),
        );
    }

    /// What a token reads back: the logits, and the hidden state before the
    /// final norm (for tests that run only some layers).
    pub struct GpuModel<'a> {
        gpu: &'a Gpu,
        cfg: &'a Config,
        layers: Vec<GpuLayer<'a>>,
        embed: GpuQuant<'a>,
        final_norm: View,
        lm_head: GpuQuant<'a>,
        plan: Plan<'a>,
        built: Built,
        hidden: View,
        final_hidden: View,
        logits: View,
        scratch: Scratch,
        /// Samples the next token from `logits` in the same command buffer.
        sampler: GpuSampler,
        /// Tokens the attention caches hold now.
        allocated: usize,
        /// Tokens they may grow to: the context.
        capacity: usize,
        position: usize,
    }

    /// The MoE router of the last layer that ran, for debugging.
    pub struct RouteDebug {
        /// The router's input, after the post-attention norm.
        pub normed: Vec<f32>,
        /// One logit per expert.
        pub logits: Vec<f32>,
        pub ids: Vec<u32>,
        pub weights: Vec<f32>,
    }

    impl<'a> GpuModel<'a> {
        /// Loads the first `layers` layers (all of them for the real model)
        /// with room for `capacity` tokens in the attention caches.
        pub fn load(
            gpu: &'a Gpu,
            cfg: &'a Config,
            weights: &'a Weights,
            layers: usize,
            capacity: usize,
        ) -> Result<GpuModel<'a>, String> {
            Self::load_from(gpu, cfg, weights, layers, capacity, kv_start(capacity))
        }

        /// `load` with caches that start at `start` tokens.
        pub fn load_from(
            gpu: &'a Gpu,
            cfg: &'a Config,
            weights: &'a Weights,
            layers: usize,
            capacity: usize,
            start: usize,
        ) -> Result<GpuModel<'a>, String> {
            assert!(layers <= cfg.num_layers());
            let embed = load_embed(gpu, cfg, weights)?;
            let allocated = start.min(capacity);
            let loaded = (0..layers)
                .map(|l| {
                    let prefix = format!("{PREFIX}.layers.{l}");
                    GpuLayer::load(gpu, cfg, weights, &prefix, cfg.layer_types[l], allocated)
                })
                .collect::<Result<Vec<_>, _>>()?;
            let final_norm = vector(gpu, weights, &format!("{PREFIX}.norm.weight"))?;
            let lm_head = matrix(gpu, weights, cfg, "language_model.lm_head")?;

            let hidden = cfg.hidden_size;
            let scratch = Scratch::new(gpu, cfg);
            let x = view(gpu, hidden);
            let final_hidden = view(gpu, hidden);
            let logits = view(gpu, lm_head.rows);
            let sampler = GpuSampler::new(gpu);
            let greedy = Params {
                temperature: 0.0,
                top_p: 1.0,
                top_k: 1,
            };
            sampler.set(lm_head.rows, &greedy, false);
            let plan = Plan::new(gpu);
            let built = plan.build();

            let mut model = GpuModel {
                gpu,
                cfg,
                layers: loaded,
                embed,
                final_norm,
                lm_head,
                plan,
                built,
                hidden: x,
                final_hidden,
                logits,
                scratch,
                sampler,
                allocated,
                capacity,
                position: 0,
            };
            model.record_plan();
            Ok(model)
        }

        /// Records one token's run into `plan` and builds it.
        fn record_plan(&mut self) {
            let (cfg, s) = (self.cfg, &self.scratch);
            let mut plan = Plan::new(self.gpu);
            record_embed(&mut plan, &self.embed, &self.hidden);
            self.layers[0].record_input_norm(&mut plan, cfg, &self.hidden, s);
            for (l, layer) in self.layers.iter().enumerate() {
                let (next_norm, next_out) = match self.layers.get(l + 1) {
                    Some(next) => (next.input_norm(), &s.normed),
                    None => (&self.final_norm, &self.final_hidden),
                };
                layer.record(&mut plan, cfg, &self.hidden, s, next_norm, next_out);
            }
            self.lm_head
                .record(&mut plan, &self.final_hidden, &self.logits, false);
            self.sampler.record(&mut plan, &self.logits);
            self.built = plan.build();
            self.plan = plan;
        }

        /// Makes room for `need` tokens in the attention caches, growing them
        /// (and recording the plan again) if they are smaller. At most the
        /// capacity.
        pub fn reserve(&mut self, need: usize) {
            if need <= self.allocated {
                return;
            }
            let allocated = kv_grow(self.allocated, need, self.capacity);
            let before = self.kv_bytes();
            for layer in &mut self.layers {
                layer.grow(self.gpu, allocated);
            }
            eprintln!(
                "KV grows from {} to {} tokens: {} MB now, {} MB more for the copy",
                self.allocated,
                allocated,
                self.kv_bytes() >> 20,
                before >> 20
            );
            self.allocated = allocated;
            self.record_plan();
        }

        /// Tokens the attention caches hold now; `capacity` is the most.
        pub fn allocated(&self) -> usize {
            self.allocated
        }

        /// Bytes of the attention caches as allocated now.
        pub fn kv_bytes(&self) -> u64 {
            self.layers.iter().map(GpuLayer::kv_bytes).sum()
        }

        /// The recorded token, for profiling.
        pub fn plan(&self) -> &Plan<'a> {
            &self.plan
        }

        pub fn gpu(&self) -> &'a Gpu {
            self.gpu
        }

        /// Tokens in the state.
        pub fn position(&self) -> usize {
            self.position
        }

        /// Sets the number of tokens in the state, after the states were
        /// restored to that length. The next token runs at this position.
        pub fn set_position(&mut self, position: usize) {
            assert!(position <= self.capacity);
            self.position = position;
        }

        /// The DeltaNet state of every linear layer, in layer order.
        pub fn deltanet_states(&self) -> Vec<&GpuDeltaNetState> {
            self.states()
                .filter_map(|s| match s {
                    State::Linear(s) => Some(s),
                    State::Full(_) => None,
                })
                .collect()
        }

        pub fn capacity(&self) -> usize {
            self.capacity
        }

        /// Runs one token and returns the logits after it.
        pub fn forward(&mut self, token: u32) -> Vec<f32> {
            self.run(token);
            contents::<f32>(&self.logits.buffer, self.cfg.vocab_size).to_vec()
        }

        /// The sampler after the lm_head, whose ring speculative decoding
        /// shares.
        pub fn sampler(&self) -> &GpuSampler {
            &self.sampler
        }

        /// Runs one token and samples the next on the GPU, the token
        /// `sampler.sample(logits, p)` would draw; `sampler` goes on after
        /// that draw. `p` needs `sample_gpu::supports`.
        pub fn run_sampled(&mut self, token: u32, p: &Params, sampler: &mut Sampler) -> u32 {
            self.sampler.set(self.cfg.vocab_size, p, false);
            self.sampler.seed(sampler.state());
            let position = self.position;
            self.run(token);
            *sampler = Sampler::new(self.sampler.rng());
            self.sampler.token(position + 1)
        }

        /// Runs one token and leaves the results on the GPU.
        pub fn run(&mut self, token: u32) {
            assert!(
                self.position < self.capacity,
                "context is full ({} tokens)",
                self.capacity
            );
            self.reserve(self.position + 1);
            self.plan
                .set_params(TokenParams::new(token, self.position as u32));
            self.built.run(self.gpu);
            self.position += 1;
        }

        /// The hidden state after the final norm, which the MTP layer takes.
        /// It holds the last token's until the next run.
        pub fn final_hidden(&self) -> &View {
            &self.final_hidden
        }

        /// The router of the last loaded layer at the last token.
        pub fn route_debug(&self) -> RouteDebug {
            let m = &self.scratch.moe;
            let k = self.cfg.num_experts_per_tok;
            RouteDebug {
                normed: contents::<f32>(&m.normed.buffer, self.cfg.hidden_size).to_vec(),
                logits: contents::<f32>(&m.router.buffer, self.cfg.num_experts).to_vec(),
                ids: contents::<u32>(&m.ids.buffer, k).to_vec(),
                weights: contents::<f32>(&m.weights.buffer, k).to_vec(),
            }
        }

        /// The hidden state after the last layer, before the final norm.
        pub fn hidden(&self) -> Vec<f32> {
            contents::<f32>(&self.hidden.buffer, self.cfg.hidden_size).to_vec()
        }

        /// Clears the DeltaNet states and the position. The attention caches
        /// are only read up to the position, so they need no clearing.
        pub fn reset(&mut self) {
            self.layers.iter().for_each(GpuLayer::reset);
            self.position = 0;
        }

        /// The DeltaNet state of each linear layer and the cache of each
        /// attention layer, in layer order.
        pub fn states(&self) -> impl Iterator<Item = State<'_>> {
            self.layers.iter().map(|l| match &l.mixer {
                Mixer::Linear(_, s) => State::Linear(s),
                Mixer::Full(_, c) => State::Full(c),
            })
        }
    }

    pub enum State<'m> {
        Linear(&'m GpuDeltaNetState),
        Full(&'m GpuKvCache),
    }

    fn zero(buffer: &Buffer) {
        unsafe { std::ptr::write_bytes(buffer.contents() as *mut u8, 0, buffer.length() as usize) };
    }

    #[cfg(test)]
    mod tests {
        use super::*;
        use crate::cpu::matrix::Matrix;
        use crate::cpu::model::{Model, argmax};
        use crate::cpu::moe::route;
        use std::path::PathBuf;

        const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";
        const TOKENS: [u32; 6] = [9707, 11, 1879, 0, 198, 3838];

        fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
            assert_eq!(got.len(), want.len());
            let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
            let norm: f32 = want.iter().map(|w| w * w).sum();
            (err / norm).sqrt()
        }

        /// The router kernel against `cpu::moe::route` on made-up logits,
        /// including ties. Runs on any Mac, no model needed.
        #[test]
        fn route_matches_cpu() {
            let gpu = Gpu::new().unwrap();
            let (experts, top_k) = (256, 8);
            for case in 0..4 {
                let mut logits: Vec<f32> = (0..experts)
                    .map(|i| ((i * 37 + case * 11) as f32 * 0.71).sin() * 4.0)
                    .collect();
                if case == 3 {
                    logits[5] = 9.0;
                    logits[200] = 9.0;
                    logits[17] = 9.0;
                }
                let identity: Vec<f32> = (0..experts * experts)
                    .map(|i| (i % (experts + 1) == 0) as u8 as f32)
                    .collect();
                let want = route(&Matrix::Dense(&identity), &logits, top_k);

                let mut plan = Plan::new(&gpu);
                let input = View::new(gpu.buffer_from(&logits));
                let (ids, weights) = (view(&gpu, top_k), view(&gpu, top_k));
                record_route(&mut plan, &input, experts, top_k, &ids, &weights);
                plan.build().run(&gpu);
                let got_ids = contents::<u32>(&ids.buffer, top_k);
                let got_w = contents::<f32>(&weights.buffer, top_k);
                for (k, &(e, w)) in want.iter().enumerate() {
                    assert_eq!(got_ids[k] as usize, e, "case {case} slot {k}");
                    assert!((got_w[k] - w).abs() < 1e-6, "case {case} slot {k}");
                }
            }
        }

        /// Microseconds per route_top_k, as 40 dependent ops in one plan,
        /// the way a token runs them. Take the bench lock (timed.sh).
        #[test]
        #[ignore = "a timing run"]
        fn route_time() {
            let gpu = Gpu::new().unwrap();
            let logits: Vec<f32> = (0..256).map(|i| (i as f32 * 0.71).sin()).collect();
            let input = View::new(gpu.buffer_from(&logits));
            let (ids, weights) = (view(&gpu, 8), view(&gpu, 8));
            let mut plan = Plan::new(&gpu);
            for _ in 0..40 {
                record_route(&mut plan, &input, 256, 8, &ids, &weights);
            }
            let built = plan.build();
            let mut times: Vec<f64> = (0..21)
                .map(|_| built.run(&gpu).as_secs_f64() * 1e6 / 40.0)
                .collect();
            times.sort_by(f64::total_cmp);
            println!(
                "route_top_k: best {:.2} us, median {:.2} us per op",
                times[0], times[10]
            );
        }

        fn open() -> (Config, Weights) {
            let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
            (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
        }

        /// The first SAETTA_LAYERS layers (default 4) against the CPU model,
        /// token by token. Only the first layers fit next to mlx_lm.server.
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn first_layers_match_cpu() {
            let layers: usize = std::env::var("SAETTA_LAYERS")
                .map(|v| v.parse().unwrap())
                .unwrap_or(4);
            let (cfg, weights) = open();
            let cpu = Model::new(&cfg, &weights).unwrap();
            let gpu = Gpu::new().unwrap();
            let mut model = GpuModel::load(&gpu, &cfg, &weights, layers, 64).unwrap();
            let mut states: Vec<_> = (0..layers).map(|l| cpu.new_state(l)).collect();
            for (t, &token) in TOKENS.iter().enumerate() {
                let mut x = cpu.embed(token);
                for (l, state) in states.iter_mut().enumerate() {
                    x = cpu.layer(l).step(&x, state).unwrap().hidden;
                }
                let start = std::time::Instant::now();
                model.run(token);
                let took = start.elapsed();
                let err = rel_rms(&model.hidden(), &x);
                println!("token {t}: rel rms {err:.2e}, {took:.2?}");
                assert!(err < 1e-4, "token {t}: rel rms {err}");
            }

            model.reset();
            let mut x = cpu.embed(TOKENS[0]);
            let mut fresh: Vec<_> = (0..layers).map(|l| cpu.new_state(l)).collect();
            for (l, state) in fresh.iter_mut().enumerate() {
                x = cpu.layer(l).step(&x, state).unwrap().hidden;
            }
            model.run(TOKENS[0]);
            assert!(rel_rms(&model.hidden(), &x) < 1e-4, "after reset");
        }

        /// A cache that grows from 128 tokens gives the same hidden states as
        /// one allocated whole, over the first four layers (one attention).
        #[test]
        #[ignore = "needs the model on the Mac"]
        fn growing_cache_matches_whole() {
            let (cfg, weights) = open();
            let gpu = Gpu::new().unwrap();
            let mut grows = GpuModel::load_from(&gpu, &cfg, &weights, 4, 1024, 128).unwrap();
            let mut whole = GpuModel::load(&gpu, &cfg, &weights, 4, 1024).unwrap();
            assert_eq!((grows.allocated(), whole.allocated()), (128, 1024));
            let mut worst = 0f32;
            for t in 0..600u32 {
                let token = (t * 7919) % 50000;
                grows.run(token);
                whole.run(token);
                let err = rel_rms(&grows.hidden(), &whole.hidden());
                worst = worst.max(err);
                assert!(err < 1e-6, "token {t}: rel rms {err}");
            }
            println!("worst rel rms {worst:.2e}, allocated {}", grows.allocated());
            assert_eq!(grows.allocated(), 1024);
        }

        /// All 40 layers: logits against the CPU model and the same greedy
        /// tokens. Needs about 25 GB, so mlx_lm.server must not be running.
        #[test]
        #[ignore = "needs the whole model on the GPU"]
        fn whole_model_matches_cpu() {
            let (cfg, weights) = open();
            let cpu = Model::new(&cfg, &weights).unwrap();
            let gpu = Gpu::new().unwrap();
            let mut model = GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), 256).unwrap();
            let mut states = cpu.new_states();
            // The GPU cache is half; the CPU rounds its keys and values the
            // same way, so what is left is the f32 arithmetic.
            for s in &mut states {
                if let crate::cpu::model::LayerState::Full(c) = s {
                    c.round_to_half();
                }
            }
            let mut cpu_logits = Vec::new();
            let mut gpu_logits = Vec::new();
            for &t in &TOKENS {
                cpu_logits = cpu.forward(t, &mut states).unwrap();
                gpu_logits = model.forward(t);
                println!("rel rms {:.2e}", rel_rms(&gpu_logits, &cpu_logits));
            }
            for i in 0..16 {
                let (c, g) = (argmax(&cpu_logits), argmax(&gpu_logits));
                let err = rel_rms(&gpu_logits, &cpu_logits);
                println!("token {i}: cpu {c}, gpu {g}, rel rms {err:.2e}");
                assert!(err < 1e-4);
                assert_eq!(c, g, "token {i}");
                cpu_logits = cpu.forward(c, &mut states).unwrap();
                gpu_logits = model.forward(c);
            }
        }
    }
}
