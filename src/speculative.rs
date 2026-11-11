//! Speculative decoding with the MTP layer as the drafter.
//!
//! Each round the MTP layer drafts `k` tokens after the pending token, the
//! main model runs over the pending token and the drafts, and the longest
//! accepted prefix is kept. The main model's state then goes back to the
//! last accepted token: full attention layers cut their KV cache, Gated
//! DeltaNet layers take a copy of their state saved during the verify pass.
//! The verify pass runs over all `k + 1` tokens, as a batched GPU pass will,
//! so the rollback is exercised on every rejection.

use crate::cpu::deltanet::DeltaNetState;
use crate::cpu::model::{LayerState, Model};
use crate::mtp::Mtp;
use crate::sample::{Params, Sampler, argmax};

/// Counts from one generation.
#[derive(Debug, Default, Clone)]
pub struct Stats {
    pub rounds: usize,
    pub drafted: usize,
    pub accepted: usize,
    /// `accepted_at[i]` counts the rounds where draft `i` was accepted.
    pub accepted_at: Vec<usize>,
}

impl Stats {
    /// Tokens produced per main model pass.
    pub fn tokens_per_round(&self) -> f32 {
        (self.rounds + self.accepted) as f32 / self.rounds.max(1) as f32
    }
}

/// How tokens are picked and drafts checked. Drafts are always the MTP
/// layer's argmax, so a GPU drafter only has to return a token id.
pub trait Decider {
    /// A token from the main model's logits.
    fn pick(&mut self, logits: &[f32]) -> u32;
    /// Checks `draft` against the main model's `logits`. `None` accepts it;
    /// `Some(t)` rejects it and puts `t` in its place.
    fn check(&mut self, logits: &[f32], draft: u32) -> Option<u32>;
    /// True when `pick` is the argmax, so a GPU argmax can stand in for it.
    fn greedy(&self) -> bool {
        false
    }
    /// The params and random stream `pick` samples with, so a GPU sampler
    /// can draw the same token in its place.
    fn sampling(&mut self) -> Option<(Params, &mut Sampler)> {
        None
    }
}

/// Greedy decoding: a draft is accepted when it is the main model's argmax,
/// so the output is the same as plain greedy decoding.
pub struct Greedy;

impl Decider for Greedy {
    fn pick(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }

    fn check(&mut self, logits: &[f32], draft: u32) -> Option<u32> {
        let best = argmax(logits);
        (best != draft).then_some(best)
    }

    fn greedy(&self) -> bool {
        true
    }
}

/// Sampling with the rejection rule for a drafter that always proposes one
/// token `d` (its distribution `q` is one-hot): `d` is kept with probability
/// `p(d)`, and a rejected `d` is replaced by a draw from `p` without `d`,
/// which is `max(0, p - q)` normalized. The output then has the same
/// distribution as sampling from the main model. `p` is the distribution
/// `Sampler::sample` draws from: top_k and top_p cut at temperature 1, then
/// the temperature.
pub struct Sampled {
    pub params: Params,
    pub sampler: Sampler,
}

impl Sampled {
    /// The final distribution, best first.
    fn distribution(&self, logits: &[f32]) -> Vec<(u32, f32)> {
        let c = self.sampler.candidates(logits, &self.params);
        let best = c[0].1;
        let t = self.params.temperature;
        let w: Vec<f32> = c.iter().map(|&(_, l)| ((l - best) / t).exp()).collect();
        let sum: f32 = w.iter().sum();
        c.iter()
            .zip(&w)
            .map(|(&(id, _), w)| (id, w / sum))
            .collect()
    }

    fn draw(&mut self, dist: &[(u32, f32)]) -> u32 {
        let total: f32 = dist.iter().map(|d| d.1).sum();
        let mut r = self.sampler.next_f32() * total;
        for &(id, p) in dist {
            r -= p;
            if r < 0.0 {
                return id;
            }
        }
        dist.iter().rev().find(|d| d.1 > 0.0).unwrap().0
    }
}

impl Decider for Sampled {
    fn pick(&mut self, logits: &[f32]) -> u32 {
        self.sampler.sample(logits, &self.params)
    }

    fn check(&mut self, logits: &[f32], draft: u32) -> Option<u32> {
        if self.params.temperature <= 0.0 {
            return Greedy.check(logits, draft);
        }
        let mut p = self.distribution(logits);
        let pd = p.iter().find(|d| d.0 == draft).map_or(0.0, |d| d.1);
        if self.sampler.next_f32() < pd {
            return None;
        }
        p.retain(|d| d.0 != draft);
        if p.is_empty() {
            return Some(draft);
        }
        Some(self.draw(&p))
    }

    fn greedy(&self) -> bool {
        self.params.temperature <= 0.0
    }

    fn sampling(&mut self) -> Option<(Params, &mut Sampler)> {
        Some((self.params, &mut self.sampler))
    }
}

/// What a layer needs to go back to an earlier token.
enum Saved {
    Linear(DeltaNetState),
    Full(usize),
}

fn save(states: &[LayerState]) -> Vec<Saved> {
    states
        .iter()
        .map(|s| match s {
            LayerState::Linear(d) => Saved::Linear(d.clone()),
            LayerState::Full(kv) => Saved::Full(kv.len()),
        })
        .collect()
}

fn restore(states: &mut [LayerState], saved: Vec<Saved>) {
    for (state, saved) in states.iter_mut().zip(saved) {
        match (state, saved) {
            (LayerState::Linear(d), Saved::Linear(s)) => *d = s,
            (LayerState::Full(kv), Saved::Full(len)) => kv.truncate(len),
            _ => unreachable!("saved state of another layer type"),
        }
    }
}

/// Generates `count` tokens after `prompt`, drafting `k` tokens per round.
pub fn generate(
    model: &Model,
    mtp: &Mtp,
    prompt: &[u32],
    count: usize,
    k: usize,
    decider: &mut impl Decider,
    stats: &mut Stats,
) -> Result<Vec<u32>, String> {
    assert!(!prompt.is_empty() && k > 0);
    stats.accepted_at.resize(k, 0);
    let mut states = model.new_states();
    let mut mtp_state = mtp.new_state();

    // The MTP cache gets one entry per position: hidden state t with token t+1.
    let mut hidden = model.forward_hidden(prompt[0], &mut states)?;
    for &t in &prompt[1..] {
        mtp.step(model, &hidden, t, &mut mtp_state)?;
        hidden = model.forward_hidden(t, &mut states)?;
    }
    let mut pending = decider.pick(&model.logits(&hidden));

    let mut out = Vec::with_capacity(count + k + 1);
    while out.len() < count {
        out.push(pending);
        if out.len() == count {
            break;
        }

        // Draft k tokens after `pending`.
        let mtp_len = mtp_len(&mtp_state);
        let (mut g, mut token) = (hidden.clone(), pending);
        let mut drafts = Vec::with_capacity(k);
        for _ in 0..k {
            g = mtp.step(model, &g, token, &mut mtp_state)?;
            token = argmax(&model.logits(&g));
            drafts.push(token);
        }

        // Verify: run `pending` and all drafts, saving the state after each.
        let mut hiddens = Vec::with_capacity(k + 1);
        let mut saved = Vec::with_capacity(k);
        for t in std::iter::once(pending).chain(drafts.iter().copied()) {
            hiddens.push(model.forward_hidden(t, &mut states)?);
            if saved.len() < k {
                saved.push(save(&states));
            }
        }

        let mut accepted = 0;
        let mut next = None;
        for (i, &draft) in drafts.iter().enumerate() {
            let logits = model.logits(&hiddens[i]);
            match decider.check(&logits, draft) {
                None => accepted += 1,
                Some(t) => {
                    next = Some(t);
                    break;
                }
            }
        }
        let next = match next {
            Some(t) => t,
            None => decider.pick(&model.logits(&hiddens[k])),
        };

        stats.rounds += 1;
        stats.drafted += k;
        stats.accepted += accepted;
        for a in &mut stats.accepted_at[..accepted] {
            *a += 1;
        }

        // Go back to the last accepted token.
        if accepted < k {
            restore(&mut states, saved.swap_remove(accepted));
        }
        // Keep the MTP entry made from the true hidden state, then replace
        // the drafted entries with ones made from the main model's hidden
        // states of the accepted tokens.
        truncate_mtp(&mut mtp_state, mtp_len + 1);
        for i in 0..accepted {
            mtp.step(model, &hiddens[i], drafts[i], &mut mtp_state)?;
        }

        out.extend(&drafts[..accepted]);
        hidden = hiddens.swap_remove(accepted);
        pending = next;
    }
    out.truncate(count);
    Ok(out)
}

/// Greedy decoding of up to `count` tokens, stopping after a token in
/// `stop`, with the MTP layer drafting `k` tokens at every position. Returns the tokens and, per position, how many
/// leading drafts match the tokens that follow. The MTP cache holds the same
/// entries as in `generate` at the start of a round, so `lengths[t]` is the
/// number of drafts a round starting at `t` would accept.
pub fn draft_lengths(
    model: &Model,
    mtp: &Mtp,
    prompt: &[u32],
    count: usize,
    k: usize,
    stop: &[u32],
) -> Result<(Vec<u32>, Vec<usize>), String> {
    let mut states = model.new_states();
    let mut mtp_state = mtp.new_state();
    let mut hidden = model.forward_hidden(prompt[0], &mut states)?;
    for &t in &prompt[1..] {
        mtp.step(model, &hidden, t, &mut mtp_state)?;
        hidden = model.forward_hidden(t, &mut states)?;
    }

    let mut tokens = Vec::with_capacity(count);
    let mut drafts = Vec::with_capacity(count);
    for _ in 0..count {
        let token = argmax(&model.logits(&hidden));
        tokens.push(token);
        let len = mtp_len(&mtp_state);
        let (mut g, mut t) = (hidden.clone(), token);
        let mut chain = Vec::with_capacity(k);
        for _ in 0..k {
            g = mtp.step(model, &g, t, &mut mtp_state)?;
            t = argmax(&model.logits(&g));
            chain.push(t);
        }
        truncate_mtp(&mut mtp_state, len + 1);
        drafts.push(chain);
        if stop.contains(&token) {
            break;
        }
        hidden = model.forward_hidden(token, &mut states)?;
    }

    let lengths = drafts
        .iter()
        .enumerate()
        .map(|(i, chain)| {
            let after = &tokens[i + 1..];
            chain.iter().zip(after).take_while(|(d, t)| d == t).count()
        })
        .collect();
    Ok((tokens, lengths))
}

/// Chooses the number of drafts per round, 0 to `max`, for the most tokens
/// per second: the expected tokens of a round with `k` drafts, from the
/// recent acceptance, over the measured time of such a round at this
/// context length. 0 is a plain decode step. A longer context makes every
/// main pass dearer, which favours longer drafts.
pub struct DraftLength {
    max: usize,
    /// Seconds per round, by context bucket and then `k`; 0 is unmeasured.
    time: Vec<Vec<f64>>,
    /// `accept[i]`: the chance draft `i` is kept when the drafts before it
    /// were.
    accept: Vec<f64>,
    /// The round each length last ran in.
    tried: Vec<usize>,
    /// Rounds run per length. The first of each is a warm-up (pipelines,
    /// buffers) and its time is thrown away by the second.
    runs: Vec<usize>,
    rounds: usize,
}

/// Rounds between tries of the length that ran least recently, so every
/// time and the acceptance stay current.
const EXPLORE: usize = 64;
/// Weight of a new round in the running averages.
const TIME_WEIGHT: f64 = 0.1;
const ACCEPT_WEIGHT: f64 = 0.05;

/// Context buckets by power of two: under 128, 128 to 255, and so on.
fn bucket(context: usize) -> usize {
    context.max(64).ilog2() as usize - 6
}

impl DraftLength {
    pub fn new(max: usize) -> DraftLength {
        DraftLength {
            max,
            time: Vec::new(),
            accept: vec![0.8; max],
            tried: vec![0; max + 1],
            runs: vec![0; max + 1],
            rounds: 0,
        }
    }

    /// Tokens a round with `k` drafts yields on average.
    pub fn expected(&self, k: usize) -> f64 {
        let mut chance = 1.0;
        let mut tokens = 1.0;
        for a in &self.accept[..k] {
            chance *= a;
            tokens += chance;
        }
        tokens
    }

    fn times(&mut self, context: usize) -> &mut Vec<f64> {
        let b = bucket(context);
        while self.time.len() <= b {
            // A new bucket starts from the one below it.
            let below = self.time.last().cloned();
            self.time
                .push(below.unwrap_or_else(|| vec![0.0; self.max + 1]));
        }
        &mut self.time[b]
    }

    /// The draft length for the next round at `context` tokens.
    pub fn next(&mut self, context: usize) -> usize {
        self.rounds += 1;
        let max = self.max;
        let time = self.times(context).clone();
        let unmeasured = |k: usize| time[k] == 0.0 || self.runs[k] < 2;
        let k = if let Some(k) = (0..=max).rev().find(|&k| unmeasured(k)) {
            k
        } else if self.rounds.is_multiple_of(EXPLORE) {
            (0..=max).min_by_key(|&k| self.tried[k]).unwrap()
        } else {
            let rate = |k: usize| self.expected(k) / time[k];
            (0..=max)
                .max_by(|&a, &b| rate(a).total_cmp(&rate(b)))
                .unwrap()
        };
        self.tried[k] = self.rounds;
        k
    }

    /// The round times in ms per length at `context`, for the log.
    pub fn times_ms(&self, context: usize) -> Vec<f64> {
        let b = bucket(context).min(self.time.len().saturating_sub(1));
        self.time.get(b).map_or(Vec::new(), |t| {
            t.iter().map(|s| (s * 1e4).round() / 10.0).collect()
        })
    }

    /// A round with `k` drafts at `context` kept `accepted` of them and
    /// took `seconds`.
    pub fn record(&mut self, context: usize, k: usize, accepted: usize, seconds: f64) {
        self.runs[k] += 1;
        let first = self.runs[k] <= 2;
        let t = &mut self.times(context)[k];
        *t = if *t == 0.0 || first {
            seconds
        } else {
            // Quick to come down, slow to go up as a longer context does,
            // and a stalled round (the GPU sometimes waits 250 ms or more)
            // counts as at most 3 times the estimate.
            let w = if seconds < *t { 0.5 } else { TIME_WEIGHT };
            *t + (seconds.min(3.0 * *t) - *t) * w
        };
        for i in 0..k.min(accepted + 1) {
            let kept = if i < accepted { 1.0 } else { 0.0 };
            self.accept[i] += (kept - self.accept[i]) * ACCEPT_WEIGHT;
        }
    }
}

/// The rounds `generate` makes over tokens whose draft lengths are `lengths`.
pub fn simulate(lengths: &[usize], k: usize) -> Stats {
    let mut stats = Stats {
        accepted_at: vec![0; k],
        ..Stats::default()
    };
    let mut t = 0;
    while t < lengths.len() {
        // Drafts past the last known token cannot be checked.
        let a = lengths[t].min(k).min(lengths.len() - t - 1);
        stats.rounds += 1;
        stats.drafted += k;
        stats.accepted += a;
        for x in &mut stats.accepted_at[..a] {
            *x += 1;
        }
        t += a + 1;
    }
    stats
}

fn mtp_len(state: &LayerState) -> usize {
    match state {
        LayerState::Full(kv) => kv.len(),
        LayerState::Linear(_) => unreachable!("the MTP layer has a KV cache"),
    }
}

fn truncate_mtp(state: &mut LayerState, len: usize) {
    if let LayerState::Full(kv) = state {
        kv.truncate(len);
    }
}

#[cfg(target_os = "macos")]
pub use gpu::GpuSpec;

#[cfg(target_os = "macos")]
mod gpu {
    //! The same rounds as `generate`, on the GPU model with the GPU MTP
    //! layer.

    use super::{Decider, DraftLength, Stats};
    use crate::config::Config;
    use crate::gpu::{Gpu, contents};
    use crate::gpu_model::GpuModel;
    use crate::gpu_mtp::GpuMtp;
    use crate::prefill::Prefill;
    use crate::qmv::View;
    use crate::sample::Sampler;
    use crate::sample_gpu::{NO_DRAFT, slot, supports};
    use std::time::Instant;

    /// The drafter and what a round needs to go back. The main model is
    /// passed in, so the same model also serves without drafts.
    pub struct GpuSpec<'a> {
        pub mtp: GpuMtp<'a>,
        gpu: &'a Gpu,
        k: usize,
        hidden: usize,
        vocab: usize,
        /// The normed hidden state after the last token the model ran.
        last: View,
        /// For each MTP cache entry, the model position of its hidden state.
        mtp_positions: Vec<usize>,
        length: DraftLength,
        /// Rounds per draft length since the last report.
        lengths: Vec<usize>,
        pub stats: Stats,
    }

    impl<'a> GpuSpec<'a> {
        /// Drafts up to `k` tokens per round, as many as pays. `prefill`
        /// checks them and needs `enable_verify` for `k + 1` rows.
        pub fn new(model: &GpuModel<'a>, mtp: GpuMtp<'a>, cfg: &Config, k: usize) -> GpuSpec<'a> {
            let gpu = model.gpu();
            let hidden = cfg.hidden_size;
            GpuSpec {
                last: View::new(gpu.buffer(hidden * 4)),
                mtp_positions: Vec::new(),
                length: DraftLength::new(k),
                lengths: vec![0; k + 1],
                stats: Stats {
                    accepted_at: vec![0; k],
                    ..Stats::default()
                },
                mtp,
                gpu,
                k,
                hidden,
                vocab: cfg.vocab_size,
            }
        }

        /// One MTP step from `h`, the normed hidden state at `position`,
        /// with the token after it.
        fn mtp_step(&mut self, h: &View, position: usize, token: u32) -> u32 {
            self.mtp_positions.push(position);
            self.mtp.step(h, token)
        }

        /// `mtp_step` without the draft, for entries nothing drafts from.
        fn mtp_record(&mut self, h: &View, position: usize, token: u32) {
            self.mtp_positions.push(position);
            self.mtp.record(h, token);
        }

        /// Records the prompt's last tokens in the MTP cache. `rows` holds
        /// the normed hidden states of `tokens`, which ran at positions
        /// `pos ..`; `last` must hold the state at `pos - 1` when `pos > 0`.
        /// Only the last `window` tokens get an entry: an entry costs about
        /// as much as a decode step, and older ones matter less.
        pub fn after_prefill(&mut self, tokens: &[u32], rows: &[f32], pos: usize, window: usize) {
            let n = tokens.len();
            let h = self.hidden;
            let start = n.saturating_sub(window).max(usize::from(pos == 0));
            let staged = View::new(self.gpu.buffer_from(rows));
            for r in start..n {
                if r == 0 {
                    let last = self.last.clone();
                    self.mtp_record(&last, pos - 1, tokens[0]);
                } else {
                    let at = View {
                        buffer: staged.buffer.clone(),
                        offset: ((r - 1) * h * 4) as u64,
                    };
                    self.mtp_record(&at, pos + r - 1, tokens[r]);
                }
            }
            write(&self.last, &rows[(n - 1) * h..n * h]);
        }

        /// After going back to `len` tokens: keeps the MTP entries made from
        /// states before `len - 1` and sets the last hidden state.
        pub fn after_restore(&mut self, len: usize, last: &[f32]) {
            let keep = self
                .mtp_positions
                .iter()
                .take_while(|&&p| p + 1 < len)
                .count();
            // Entries kept from another conversation with the same prefix
            // would be stale; within one conversation they are exact.
            eprintln!(
                "MTP cache: kept {keep} entries going back to {len} tokens ({} dropped)",
                self.mtp_positions.len() - keep
            );
            self.mtp.truncate(keep);
            self.mtp_positions.truncate(keep);
            write(&self.last, last);
        }

        /// Prints the drafting counts since the last call and clears them.
        pub fn report(&mut self, context: usize) {
            let s = &self.stats;
            if s.rounds > 0 {
                eprintln!(
                    "draft rounds: ms per length {:?}, accept {:?}",
                    self.length.times_ms(context),
                    self.length.accept
                );
                eprintln!(
                    "drafts: {} rounds, {} of {} accepted, {:.2} tokens per round, accepted at {:?}, rounds per length {:?}",
                    s.rounds,
                    s.accepted,
                    s.drafted,
                    s.tokens_per_round(),
                    s.accepted_at,
                    self.lengths
                );
            }
            self.lengths.fill(0);
            self.stats = Stats {
                accepted_at: vec![0; self.k],
                ..Stats::default()
            };
        }

        /// A plain decode step that keeps the MTP cache and the last hidden
        /// state up to date, for a token run outside `speculate` (the
        /// prompt cache runs a prompt's last token this way).
        pub fn step(&mut self, model: &mut GpuModel<'a>, token: u32) -> Vec<f32> {
            let pos = model.position();
            if pos > 0 {
                let last = self.last.clone();
                self.mtp_record(&last, pos - 1, token);
            }
            let logits = model.forward(token);
            copy(model.final_hidden(), &self.last, self.hidden);
            logits
        }

        /// `step` without the MTP entry, for prompt tokens before the MTP
        /// window: the drafts only look back that far.
        pub fn step_plain(&mut self, model: &mut GpuModel<'a>, token: u32) -> Vec<f32> {
            let logits = model.forward(token);
            copy(model.final_hidden(), &self.last, self.hidden);
            logits
        }

        pub fn last_hidden(&self) -> Vec<f32> {
            read(&self.last, self.hidden)
        }

        pub fn reset(&mut self) {
            self.mtp.reset();
            self.mtp_positions.clear();
        }

        /// One round: runs `pending` and the drafts, keeps the accepted
        /// drafts and returns them with the next token, which is not run.
        /// The number of drafts is chosen per round.
        pub fn speculate(
            &mut self,
            model: &mut GpuModel<'a>,
            prefill: &Prefill,
            pending: u32,
            decide: &mut dyn Decider,
        ) -> (Vec<u32>, u32) {
            let context = model.position();
            let k = self.length.next(context);
            let start = Instant::now();
            let (drafts, next) = if k == 0 {
                self.plain(model, pending, decide)
            } else {
                self.round(model, prefill, pending, decide, k)
            };
            let seconds = start.elapsed().as_secs_f64();
            eprintln!(
                "ROUND k {k} ctx {context} accepted {} {:.1} ms",
                drafts.len(),
                seconds * 1e3
            );
            self.length.record(context, k, drafts.len(), seconds);
            self.lengths[k] += 1;
            (drafts, next)
        }

        /// A round without drafts: a decode step that keeps the MTP cache.
        fn plain(
            &mut self,
            model: &mut GpuModel<'a>,
            pending: u32,
            decide: &mut dyn Decider,
        ) -> (Vec<u32>, u32) {
            let last = self.last.clone();
            self.mtp_record(&last, model.position() - 1, pending);
            let next = match decide.sampling() {
                Some((p, sampler)) if supports(&p) => model.run_sampled(pending, &p, sampler),
                _ => {
                    let logits = model.forward(pending);
                    decide.pick(&logits)
                }
            };
            copy(model.final_hidden(), &self.last, self.hidden);
            (Vec::new(), next)
        }

        fn round(
            &mut self,
            model: &mut GpuModel<'a>,
            prefill: &Prefill,
            pending: u32,
            decide: &mut dyn Decider,
            k: usize,
        ) -> (Vec<u32>, u32) {
            let vocab = self.vocab;
            let base = model.position();
            let mark = self.mtp.len();

            let last = self.last.clone();
            let mut drafts = vec![self.mtp_step(&last, base - 1, pending)];
            for i in 1..k {
                let own = self.mtp.output().clone();
                let d = self.mtp_step(&own, base + i - 1, drafts[i - 1]);
                drafts.push(d);
            }

            // One pass over `pending` and the drafts, keeping each row.
            let tokens: Vec<u32> = std::iter::once(pending)
                .chain(drafts.iter().copied())
                .collect();
            // At temperature > 0 the GPU sampler applies the rejection rule
            // to each row, in the verify's command buffer.
            let on_gpu = match decide.sampling() {
                Some((p, sampler)) if p.temperature > 0.0 && supports(&p) => {
                    let s = model.sampler();
                    s.set(vocab, &p, false);
                    s.seed(sampler.state());
                    for (i, &d) in drafts.iter().enumerate() {
                        s.set_draft(base + i + 1, d);
                    }
                    s.set_draft(base + k + 1, NO_DRAFT);
                    true
                }
                _ => false,
            };
            // The verify writes a KV row for every token it runs.
            model.reserve(base + tokens.len());
            let states: Vec<_> = model.states().collect();
            let sampler = on_gpu.then(|| model.sampler());
            let v = prefill.verify(self.gpu, &tokens, base, &states, sampler);
            let row = |i: usize| {
                let at = v.logits.offset as usize / 4 + i * vocab;
                &contents::<f32>(&v.logits.buffer, at + vocab)[at..]
            };

            let (accepted, next) = if on_gpu {
                let s = model.sampler();
                let ring = s.read();
                s.clear_drafts();
                if let Some((_, sampler)) = decide.sampling() {
                    *sampler = Sampler::new(s.rng());
                }
                let accepted = (0..k)
                    .take_while(|&i| ring.accepted[slot(base + i + 1)] == 1)
                    .count();
                (accepted, ring.ids[slot(base + accepted + 1)])
            } else {
                let mut accepted = 0;
                let mut next = None;
                for (i, &d) in drafts.iter().enumerate() {
                    let verdict = if decide.greedy() {
                        (v.argmax[i] != d).then_some(v.argmax[i])
                    } else {
                        decide.check(row(i), d)
                    };
                    match verdict {
                        None => accepted += 1,
                        Some(t) => {
                            next = Some(t);
                            break;
                        }
                    }
                }
                let next = next.unwrap_or_else(|| {
                    if decide.greedy() {
                        v.argmax[k]
                    } else {
                        decide.pick(row(k))
                    }
                });
                (accepted, next)
            };

            if accepted < k {
                prefill.restore(self.gpu, accepted, &states);
            }
            drop(states);
            model.set_position(base + accepted + 1);
            self.mtp.truncate(mark + 1);
            self.mtp_positions.truncate(mark + 1);
            for (i, &d) in drafts[..accepted].iter().enumerate() {
                self.mtp_record(&v.hidden_row(i), base + i, d);
            }
            copy(&v.hidden_row(accepted), &self.last, self.hidden);

            self.stats.rounds += 1;
            self.stats.drafted += k;
            self.stats.accepted += accepted;
            for a in &mut self.stats.accepted_at[..accepted] {
                *a += 1;
            }
            drafts.truncate(accepted);
            (drafts, next)
        }
    }

    fn ptr(v: &View) -> *mut f32 {
        unsafe { (v.buffer.contents() as *mut u8).add(v.offset as usize) as *mut f32 }
    }

    /// Copies `n` floats between shared buffers while the GPU is idle.
    fn copy(from: &View, to: &View, n: usize) {
        unsafe { std::ptr::copy(ptr(from), ptr(to), n) };
    }

    fn write(to: &View, values: &[f32]) {
        unsafe { std::ptr::copy(values.as_ptr(), ptr(to), values.len()) };
    }

    fn read(from: &View, n: usize) -> Vec<f32> {
        unsafe { std::slice::from_raw_parts(ptr(from), n).to_vec() }
    }
}

#[cfg(all(test, target_os = "macos"))]
mod gpu_real {
    //! With the real model on the Mac, through the bench lock:
    //!   ~/saetta/timed.sh E cargo test --release --lib speculative::gpu_real -- --ignored --nocapture

    use super::*;
    use crate::chat::{self, Message};
    use crate::config::Config;
    use crate::gpu::Gpu;
    use crate::gpu_model::GpuModel;
    use crate::gpu_mtp::GpuMtp;
    use crate::tokenizer::Tokenizer;
    use crate::weights::Weights;
    use std::path::PathBuf;
    use std::time::Instant;

    const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

    /// The two best logits' gap.
    fn top2_gap(logits: &[f32]) -> f32 {
        let (mut a, mut b) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
        for &l in logits {
            if l > a {
                (a, b) = (l, a);
            } else if l > b {
                b = l;
            }
        }
        a - b
    }

    fn contents(v: &crate::qmv::View, n: usize) -> Vec<f32> {
        crate::gpu::contents::<f32>(&v.buffer, n + v.offset as usize / 4)[v.offset as usize / 4..]
            .to_vec()
    }

    #[test]
    #[ignore = "needs the model on the Mac"]
    fn greedy_matches_plain_gpu_greedy() {
        let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
        let cfg = Config::load(&dir).unwrap();
        let weights = Weights::open(&dir).unwrap();
        let tok = Tokenizer::load(&dir).unwrap();
        let gpu = Gpu::new().unwrap();
        let h = cfg.hidden_size;
        let count = 48;
        let opts = chat::Options {
            enable_thinking: false,
            ..chat::Options::default()
        };
        let prompts = [
            "Write a Rust function that parses a version string like 1.2.3 into a struct with major, minor and patch fields.",
            "Explain in one short paragraph why the sky is blue.",
        ];

        for text in prompts {
            let message = Message {
                role: "user".into(),
                content: text.into(),
                ..Message::default()
            };
            let prompt = tok.encode(&chat::render(&[message], &[], &opts).unwrap());

            // Plain greedy, keeping the gap of each step.
            let mut plain = GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), 1024).unwrap();
            let mut logits = Vec::new();
            for &t in &prompt {
                logits = plain.forward(t);
            }
            let (mut want, mut gaps) = (Vec::new(), Vec::new());
            let start = Instant::now();
            for _ in 0..count {
                let t = argmax(&logits);
                want.push(t);
                gaps.push(top2_gap(&logits));
                logits = plain.forward(t);
            }
            let plain_time = start.elapsed();
            drop(plain);

            let mut model = GpuModel::load(&gpu, &cfg, &weights, cfg.num_layers(), 1024).unwrap();
            let mtp = GpuMtp::load(&gpu, &cfg, &weights, 1024).unwrap();
            let mut spec = GpuSpec::new(&model, mtp, &cfg, 3);
            let mut prefill = crate::prefill::Prefill::new(&gpu, &cfg, &weights, 64).unwrap();
            prefill.exact = true;
            prefill.enable_verify(&gpu, 4);
            let mut rows = Vec::with_capacity(prompt.len() * h);
            let mut logits = Vec::new();
            for &t in &prompt {
                logits = model.forward(t);
                rows.extend(contents(model.final_hidden(), h));
            }
            spec.after_prefill(&prompt, &rows, 0, 32);
            let mut got = Vec::new();
            let mut pending = argmax(&logits);
            let start = Instant::now();
            while got.len() < count {
                got.push(pending);
                let (accepted, next) = spec.speculate(&mut model, &prefill, pending, &mut Greedy);
                got.extend(accepted);
                pending = next;
            }
            let spec_time = start.elapsed();
            got.truncate(count);

            println!(
                "{:?}\n  plain {:.1} tok/s, speculative {:.1} tok/s, {:.2} tokens per round, {:?}",
                tok.decode(&got),
                count as f64 / plain_time.as_secs_f64(),
                count as f64 / spec_time.as_secs_f64(),
                spec.stats.tokens_per_round(),
                spec.stats
            );
            if let Some(i) = (0..count).find(|&i| got[i] != want[i]) {
                println!("first difference at {i}: top-2 gap {:.2e}", gaps[i]);
                assert!(gaps[i] < 1e-4, "differs at {i} with gap {}", gaps[i]);
            }
        }
    }
}

#[cfg(test)]
mod real {
    //! With the real model on the Mac:
    //!   scripts/mac.sh test --release --lib speculative::real -- --ignored --nocapture --test-threads 1

    use super::*;
    use crate::chat::{self, Message};
    use crate::config::Config;
    use crate::cpu::model::Model;
    use crate::mtp::{Mtp, MtpPosition};
    use crate::tokenizer::Tokenizer;
    use crate::weights::Weights;
    use serde_json::json;
    use std::path::PathBuf;
    use std::time::Instant;

    const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

    fn model_dir() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR)
    }

    #[test]
    #[ignore = "needs the model on the Mac"]
    fn greedy_speculative_matches_plain_greedy() {
        let dir = model_dir();
        let cfg = Config::load(&dir).unwrap();
        let weights = Weights::open(&dir).unwrap();
        let model = Model::new(&cfg, &weights).unwrap();
        let tok = Tokenizer::load(&dir).unwrap();
        let prompt = tok.encode("The capital of Romania is");
        let want = model.generate(&prompt, 16, |_, _| {}).unwrap();
        for position in [MtpPosition::Hidden, MtpPosition::Token] {
            let mtp = Mtp::new(&cfg, &weights, position).unwrap();
            let mut stats = Stats::default();
            let got = generate(&model, &mtp, &prompt, 16, 3, &mut Greedy, &mut stats).unwrap();
            println!("{position:?}: {:?}, {stats:?}", tok.decode(&got));
            assert_eq!(got, want, "{position:?}");
        }
    }

    fn user(text: &str) -> Message {
        Message {
            role: "user".into(),
            content: text.into(),
            ..Message::default()
        }
    }

    #[test]
    #[ignore = "needs the model on the Mac; takes minutes"]
    fn acceptance_rate() {
        let dir = model_dir();
        let cfg = Config::load(&dir).unwrap();
        let weights = Weights::open(&dir).unwrap();
        let model = Model::new(&cfg, &weights).unwrap();
        let tok = Tokenizer::load(&dir).unwrap();
        let opts = chat::Options {
            enable_thinking: false,
            ..chat::Options::default()
        };
        let read_file = json!({"type": "function", "function": {
            "name": "read_file",
            "description": "Read a file from the workspace",
            "parameters": {"type": "object",
                "properties": {"path": {"type": "string", "description": "Path of the file"}},
                "required": ["path"]}}});
        let cases = [
            (
                "code",
                vec![user(
                    "Write a Rust function that parses a version string like 1.2.3 into a struct with major, minor and patch fields, and returns an error for bad input.",
                )],
                vec![],
            ),
            (
                "tool call",
                vec![user(
                    "Open src/main.rs and tell me what the main function does.",
                )],
                vec![read_file],
            ),
            (
                "json",
                vec![user(
                    "Reply with only a JSON object describing a user with fields id (a number), name, email and roles (a list of strings).",
                )],
                vec![],
            ),
            (
                "prose",
                vec![user("Explain in one short paragraph why the sky is blue.")],
                vec![],
            ),
        ];
        // RoPE is relative, so both MTP positions give the same drafts up to
        // rounding (the greedy test runs both); one is enough here.
        let (count, k) = (96, 4);
        for position in [MtpPosition::Hidden] {
            let mtp = Mtp::new(&cfg, &weights, position).unwrap();
            let mut all = Vec::new();
            for (name, messages, tools) in &cases {
                let prompt = tok.encode(&chat::render(messages, tools, &opts).unwrap());
                let start = Instant::now();
                let (tokens, lengths) =
                    draft_lengths(&model, &mtp, &prompt, count, k, &cfg.eos_token_ids).unwrap();
                let elapsed = start.elapsed();
                let per_k: Vec<String> = (2..=k)
                    .map(|k| format!("k={k} {:.2}", simulate(&lengths, k).tokens_per_round()))
                    .collect();
                println!(
                    "{name}: prompt {} tokens, {} generated, {:.2?} per token, first draft accepted {:.0}%, tokens per round {}",
                    prompt.len(),
                    tokens.len(),
                    elapsed / (prompt.len() + tokens.len()) as u32,
                    100.0 * lengths.iter().filter(|&&l| l > 0).count() as f32
                        / lengths.len() as f32,
                    per_k.join(", ")
                );
                println!("  {:?}", tok.decode(&tokens));
                all.extend(lengths);
            }
            for k in 1..=4 {
                println!(
                    "{position:?} all prompts, k={k}: {:.2} tokens per round",
                    simulate(&all, k).tokens_per_round()
                );
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Fake rounds: a main pass costs `pass` seconds plus `per_row` per row,
    /// and draft i is kept with chance `accept`.
    fn settle(pass: f64, per_row: f64, accept: f64) -> usize {
        let mut d = DraftLength::new(7);
        let mut r = Sampler::new(5);
        let mut counts = [0; 8];
        for _ in 0..3000 {
            let k = d.next(1000);
            let accepted = (0..k).take_while(|_| r.next_f32() < accept as f32).count();
            d.record(1000, k, accepted, pass + per_row * (k + 1) as f64);
            counts[k] += 1;
        }
        (0..8).max_by_key(|&k| counts[k]).unwrap()
    }

    #[test]
    fn draft_length_follows_cost_and_acceptance() {
        // A first plain round far too slow is measured again later.
        let mut d = DraftLength::new(3);
        for k in [3, 3, 2, 2, 1, 1, 0] {
            assert_eq!(d.next(100), k);
            d.record(100, k, k, 0.04);
        }
        // The second plain round stalls, and so does one round in 50.
        assert_eq!(d.next(100), 0);
        d.record(100, 0, 0, 1.0);
        for i in 0..2000 {
            let k = d.next(100);
            let t = if k == 0 { 0.01 } else { 0.04 };
            d.record(100, k, k.min(1), if i % 50 == 0 { 0.5 } else { t });
        }
        assert_eq!(d.next(101), 0);

        // Rows nearly free and drafts usually kept: draft the most.
        assert_eq!(settle(0.010, 0.0005, 0.9), 7);
        // Rows as dear as a pass: drafting never pays.
        assert_eq!(settle(0.0, 0.010, 0.9), 0);
        // In between, a short draft.
        let k = settle(0.010, 0.003, 0.6);
        assert!((1..=3).contains(&k), "{k}");
    }

    #[test]
    fn new_context_bucket_starts_from_the_one_below() {
        let mut d = DraftLength::new(2);
        for k in [2, 2, 1, 1, 0, 0] {
            assert_eq!(d.next(100), k);
            d.record(100, k, k, 0.01);
        }
        // Measured below, so no unmeasured lengths at 5000 tokens.
        let k = d.next(5000);
        assert!(k <= 2);
        assert_eq!(d.time.len(), bucket(5000) + 1);
    }

    /// A one-hot drafter plus the rule must give tokens distributed as p,
    /// whichever token it drafts.
    #[test]
    fn rejection_rule_keeps_the_target_distribution() {
        let target = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let params = Params {
            temperature: 0.8,
            top_p: 1.0,
            top_k: 5,
        };
        let mut d = Sampled {
            params,
            sampler: Sampler::new(42),
        };
        let p = d.distribution(&target);

        for draft in [0, 3, 5] {
            let n = 200_000;
            let mut counts = [0usize; 6];
            let mut kept = 0;
            for _ in 0..n {
                let out = match d.check(&target, draft) {
                    None => {
                        kept += 1;
                        draft
                    }
                    Some(t) => t,
                };
                counts[out as usize] += 1;
            }
            for (id, want) in &p {
                let got = counts[*id as usize] as f32 / n as f32;
                assert!(
                    (got - want).abs() < 0.005,
                    "draft {draft}, token {id}: {got} vs {want}"
                );
            }
            assert_eq!(counts[5], 0, "token 5 is cut by top_k");
            let want_kept = p.iter().find(|x| x.0 == draft).map_or(0.0, |x| x.1);
            let got_kept = kept as f32 / n as f32;
            assert!(
                (got_kept - want_kept).abs() < 0.005,
                "draft {draft} kept {got_kept}"
            );
        }
    }

    #[test]
    fn greedy_check_takes_the_argmax() {
        let logits = [0.0, 3.0, 1.0];
        assert_eq!(Greedy.check(&logits, 1), None);
        assert_eq!(Greedy.check(&logits, 2), Some(1));
    }
}
