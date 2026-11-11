//! Speculative decoding with the MTP layer as the drafter.
//!
//! Each round the MTP layer drafts `k` tokens after the pending token, the
//! main model runs over the pending token and the drafts, and the longest
//! accepted prefix is kept. The main model's state then goes back to the
//! last accepted token: full attention layers cut their KV cache, Gated
//! DeltaNet layers take a copy of their state saved during the verify pass.
//! The verify pass runs over all `k + 1` tokens, as a batched GPU pass will,
//! so the rollback is exercised on every rejection.

use std::collections::HashMap;

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

/// How tokens are picked and drafts checked.
pub trait Decider {
    /// A token from the main model's logits.
    fn pick(&mut self, logits: &[f32]) -> u32;
    /// A draft token from the MTP layer's logits.
    fn draft(&mut self, logits: &[f32]) -> u32;
    /// Checks `draft`, drafted from `draft_logits`, against the main model's
    /// `logits`. `None` accepts it; `Some(t)` rejects it and puts `t` in its
    /// place.
    fn check(&mut self, logits: &[f32], draft_logits: &[f32], draft: u32) -> Option<u32>;
}

/// Greedy decoding: a draft is accepted when it is the main model's argmax,
/// so the output is the same as plain greedy decoding.
pub struct Greedy;

impl Decider for Greedy {
    fn pick(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }

    fn draft(&mut self, logits: &[f32]) -> u32 {
        argmax(logits)
    }

    fn check(&mut self, logits: &[f32], _: &[f32], draft: u32) -> Option<u32> {
        let best = argmax(logits);
        (best != draft).then_some(best)
    }
}

/// Sampling with the standard rejection rule: draft `d` drawn from the
/// drafter's distribution `q` is kept with probability `min(1, p(d) / q(d))`,
/// and a rejected draft is replaced by a draw from `max(0, p - q)`. The
/// output then has the same distribution as sampling from the main model.
/// `p` and `q` are the distributions `Sampler::sample` draws from: top_k
/// and top_p cut at temperature 1, then the temperature.
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

    fn draft(&mut self, logits: &[f32]) -> u32 {
        self.sampler.sample(logits, &self.params)
    }

    fn check(&mut self, logits: &[f32], draft_logits: &[f32], draft: u32) -> Option<u32> {
        if self.params.temperature <= 0.0 {
            return Greedy.check(logits, draft_logits, draft);
        }
        let p = self.distribution(logits);
        let q: HashMap<u32, f32> = self.distribution(draft_logits).into_iter().collect();
        let q_of = |id: u32| q.get(&id).copied().unwrap_or(0.0);
        let pd = p.iter().find(|d| d.0 == draft).map_or(0.0, |d| d.1);
        if self.sampler.next_f32() * q_of(draft) < pd {
            return None;
        }
        let residual: Vec<(u32, f32)> = p
            .iter()
            .map(|&(id, pi)| (id, (pi - q_of(id)).max(0.0)))
            .collect();
        if residual.iter().all(|r| r.1 == 0.0) {
            return Some(self.draw(&p));
        }
        Some(self.draw(&residual))
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
            let logits = model.logits(&g);
            token = decider.draft(&logits);
            drafts.push((token, logits));
        }

        // Verify: run `pending` and all drafts, saving the state after each.
        let mut hiddens = Vec::with_capacity(k + 1);
        let mut saved = Vec::with_capacity(k);
        for t in std::iter::once(pending).chain(drafts.iter().map(|d| d.0)) {
            hiddens.push(model.forward_hidden(t, &mut states)?);
            if saved.len() < k {
                saved.push(save(&states));
            }
        }

        let mut accepted = 0;
        let mut next = None;
        for (i, (draft, draft_logits)) in drafts.iter().enumerate() {
            let logits = model.logits(&hiddens[i]);
            match decider.check(&logits, draft_logits, *draft) {
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
            mtp.step(model, &hiddens[i], drafts[i].0, &mut mtp_state)?;
        }

        out.extend(drafts.iter().take(accepted).map(|d| d.0));
        hidden = hiddens.swap_remove(accepted);
        pending = next;
    }
    out.truncate(count);
    Ok(out)
}

/// Greedy decoding of `count` tokens with the MTP layer drafting `k`
/// tokens at every position. Returns the tokens and, per position, how many
/// leading drafts match the tokens that follow. The MTP cache holds the same
/// entries as in `generate` at the start of a round, so `lengths[t]` is the
/// number of drafts a round starting at `t` would accept.
pub fn draft_lengths(
    model: &Model,
    mtp: &Mtp,
    prompt: &[u32],
    count: usize,
    k: usize,
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
                "prose",
                vec![user("Explain in one short paragraph why the sky is blue.")],
                vec![],
            ),
        ];
        let (count, k) = (64, 3);
        for position in [MtpPosition::Hidden, MtpPosition::Token] {
            let mtp = Mtp::new(&cfg, &weights, position).unwrap();
            let mut all = Vec::new();
            for (name, messages, tools) in &cases {
                let prompt = tok.encode(&chat::render(messages, tools, &opts).unwrap());
                let start = Instant::now();
                let (tokens, lengths) = draft_lengths(&model, &mtp, &prompt, count, k).unwrap();
                let elapsed = start.elapsed();
                let s = simulate(&lengths, k);
                println!(
                    "{position:?} {name}: prompt {} tokens, {:.2?} per token, first draft accepted {:.0}%, accepted_at {:?} of {} rounds, {:.2} tokens per round",
                    prompt.len(),
                    elapsed / (prompt.len() + count) as u32,
                    100.0 * lengths.iter().filter(|&&l| l > 0).count() as f32
                        / lengths.len() as f32,
                    s.accepted_at,
                    s.rounds,
                    s.tokens_per_round()
                );
                println!("  {:?}", tok.decode(&tokens));
                all.extend(lengths);
            }
            for k in 1..=3 {
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

    /// Drafting from q and applying the rule must give tokens distributed
    /// as p, whatever q is.
    #[test]
    fn rejection_rule_keeps_the_target_distribution() {
        let target = [2.0f32, 1.0, 0.5, 0.0, -1.0, -3.0];
        let draft = [0.0f32, 2.0, 0.0, 1.0, 0.5, -1.0];
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

        let n = 200_000;
        let mut counts = [0usize; 6];
        let mut kept = 0;
        for _ in 0..n {
            let token = d.draft(&draft);
            let out = match d.check(&target, &draft, token) {
                None => {
                    kept += 1;
                    token
                }
                Some(t) => t,
            };
            counts[out as usize] += 1;
        }
        for (id, want) in &p {
            let got = counts[*id as usize] as f32 / n as f32;
            assert!((got - want).abs() < 0.005, "token {id}: {got} vs {want}");
        }
        assert_eq!(counts[5], 0, "token 5 is cut by top_k");
        assert!(kept > n / 4 && kept < n, "kept {kept} of {n}");
    }

    #[test]
    fn greedy_check_takes_the_argmax() {
        let logits = [0.0, 3.0, 1.0];
        assert_eq!(Greedy.check(&logits, &[], 1), None);
        assert_eq!(Greedy.check(&logits, &[], 2), Some(1));
    }
}
