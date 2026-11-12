#![cfg(target_os = "macos")]
//! Sampling with the real model keeps the model's distribution: the GPU
//! sampler and its draft rule on real logits, the verify rows against plain
//! decode, and whole sampled decodes with and without drafts.
//!
//!   cargo test --release --test sample_real -- --ignored --nocapture --test-threads 1
//!
//! SAETTA_SAMPLES sets the decodes per mode in sampled_decode_matches_plain
//! (default 200).

use saetta::chat::{self, Message};
use saetta::config::Config;
use saetta::gpu::Gpu;
use saetta::gpu_model::GpuModel;
use saetta::gpu_mtp::GpuMtp;
use saetta::plan::{Plan, TokenParams};
use saetta::qmv::View;
use saetta::sample::{Params, Sampler};
use saetta::sample_gpu::{GpuSampler, NO_DRAFT, RING, slot};
use saetta::speculative::{Decider, GpuSpec, Sampled};
use saetta::tokenizer::Tokenizer;
use saetta::weights::Weights;
use std::collections::HashMap;
use std::path::PathBuf;
use std::time::Instant;

const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";

/// The serve defaults qh uses.
const PARAMS: Params = Params {
    temperature: 1.0,
    top_p: 0.95,
    top_k: 20,
};

struct Loaded {
    cfg: Config,
    weights: Weights,
    tok: Tokenizer,
    gpu: Gpu,
}

fn load() -> Loaded {
    let dir = PathBuf::from(std::env::var("HOME").unwrap()).join(MODEL_DIR);
    Loaded {
        cfg: Config::load(&dir).unwrap(),
        weights: Weights::open(&dir).unwrap(),
        tok: Tokenizer::load(&dir).unwrap(),
        gpu: Gpu::new().unwrap(),
    }
}

fn prompt(tok: &Tokenizer, text: &str, thinking: bool) -> Vec<u32> {
    let message = Message {
        role: "user".into(),
        content: text.into(),
        ..Message::default()
    };
    let opts = chat::Options {
        enable_thinking: thinking,
        ..chat::Options::default()
    };
    tok.encode(&chat::render(&[message], &[], &opts).unwrap())
}

/// The distribution the sampler should draw from, best first: top_k and
/// top_p cut at temperature 1, then the temperature, in f64.
fn exact(logits: &[f32], p: &Params) -> Vec<(u32, f64)> {
    let c = Sampler::new(0).candidates(logits, p);
    let best = c[0].1 as f64;
    let w: Vec<f64> = c
        .iter()
        .map(|&(_, l)| ((l as f64 - best) / p.temperature as f64).exp())
        .collect();
    let sum: f64 = w.iter().sum();
    c.iter()
        .zip(&w)
        .map(|(&(id, _), w)| (id, w / sum))
        .collect()
}

fn tv(counts: &HashMap<u32, usize>, n: usize, p: &[(u32, f64)]) -> f64 {
    let mut d: f64 = p
        .iter()
        .map(|&(id, q)| (counts.get(&id).copied().unwrap_or(0) as f64 / n as f64 - q).abs())
        .sum();
    d += counts
        .iter()
        .filter(|(id, _)| !p.iter().any(|q| q.0 == **id))
        .map(|(_, &c)| c as f64 / n as f64)
        .sum::<f64>();
    d / 2.0
}

/// The GPU sampler recorded once per ring slot over the same logits, so one
/// run draws RING tokens, each with the draft set in its slot.
struct Draws<'g> {
    gpu: &'g Gpu,
    sampler: GpuSampler,
    _plan: Plan<'g>,
    built: saetta::plan::Built,
}

impl<'g> Draws<'g> {
    fn new(gpu: &'g Gpu, logits: &[f32], p: &Params, seed: u64) -> Draws<'g> {
        let x = View::new(gpu.buffer_from(logits));
        let sampler = GpuSampler::new(gpu);
        sampler.set(logits.len(), p, false);
        sampler.seed(seed);
        let params: Vec<TokenParams> = (0..RING as u32).map(|r| TokenParams::new(0, r)).collect();
        let params = gpu.buffer_from(&params);
        let mut plan = Plan::new(gpu);
        for r in 0..RING {
            plan.params = View {
                buffer: params.clone(),
                offset: (r * size_of::<TokenParams>()) as u64,
            };
            sampler.record(&mut plan, &x);
        }
        let built = plan.build();
        Draws {
            gpu,
            sampler,
            _plan: plan,
            built,
        }
    }

    /// Token counts and how many were the accepted draft, over `runs * RING`
    /// draws.
    fn run(&self, draft: u32, runs: usize) -> (HashMap<u32, usize>, usize) {
        let mut counts = HashMap::new();
        let mut kept = 0;
        for _ in 0..runs {
            for pos in 1..=RING {
                self.sampler.set_draft(pos, draft);
            }
            self.built.run(self.gpu);
            let r = self.sampler.read();
            for pos in 1..=RING {
                let s = slot(pos);
                *counts.entry(r.ids[s]).or_insert(0) += 1;
                kept += r.accepted[s] as usize;
                assert_eq!(
                    r.accepted[s] == 1,
                    draft != NO_DRAFT && r.ids[s] == draft,
                    "accepted flag"
                );
            }
        }
        (counts, kept)
    }
}

fn top10(tok: &Tokenizer, counts: &HashMap<u32, usize>, n: usize, p: &[(u32, f64)]) -> String {
    let mut ids: Vec<u32> = p.iter().map(|q| q.0).take(10).collect();
    let mut extra: Vec<(&u32, &usize)> =
        counts.iter().filter(|(id, _)| !ids.contains(id)).collect();
    extra.sort_by(|a, b| b.1.cmp(a.1));
    ids.extend(extra.iter().take(3).map(|e| *e.0));
    ids.iter()
        .map(|&id| {
            let want = p.iter().find(|q| q.0 == id).map_or(0.0, |q| q.1);
            let got = counts.get(&id).copied().unwrap_or(0) as f64 / n as f64;
            format!("{:?} {got:.4}/{want:.4}", tok.decode(&[id]))
        })
        .collect::<Vec<_>>()
        .join(", ")
}

/// Plain decode logits along a greedy continuation of `prompt` and the
/// verify rows' logits for the same tokens, rows of 4.
fn plain_and_verify(
    model: &mut GpuModel,
    prompt: &[u32],
    steps: usize,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    model.reset();
    let mut plain = Vec::new();
    for &t in prompt {
        plain = vec![model.forward(t)];
    }
    let mut generated = Vec::new();
    for _ in 0..steps {
        let t = saetta::sample::argmax(plain.last().unwrap());
        generated.push(t);
        plain.push(model.forward(t));
    }
    model.reset();
    let mut verified = Vec::new();
    for &t in prompt {
        verified = vec![model.forward(t)];
    }
    for group in generated.chunks(4) {
        model.verify(group);
        model.accept(group.len());
        verified.extend((0..group.len()).map(|r| model.verify_logits(r)));
    }
    (plain, verified)
}

/// On real logits from a code and a prose answer: GPU draws without a
/// draft, with the argmax, a mid-probability token and a token outside the
/// cut as the draft, each against the exact distribution; and the cut
/// distribution of each verify row against plain decode's.
#[test]
#[ignore = "needs the model on the Mac"]
fn gpu_sampler_matches_real_distributions() {
    let m = load();
    let mut model = GpuModel::load(&m.gpu, &m.cfg, &m.weights, m.cfg.num_layers(), 1024).unwrap();
    model.enable_verify(4);
    let texts = [
        (
            "code",
            "Write a Rust function that parses a version string like 1.2.3 into a struct with major, minor and patch fields.",
        ),
        (
            "prose",
            "Write a short story about a lighthouse keeper who finds a message in a bottle.",
        ),
    ];
    let runs = 512; // 32768 draws per case
    let n = runs * RING;
    let mut worst_tv = 0.0f64;
    let mut worst_verify = 0.0f64;
    for (name, text) in texts {
        let prompt = prompt(&m.tok, text, false);
        let (plain, verified) = plain_and_verify(&mut model, &prompt, 32);
        for i in 0..plain.len() {
            let (a, b) = (exact(&plain[i], &PARAMS), exact(&verified[i], &PARAMS));
            let mut d = 0.0;
            for &(id, q) in &a {
                d += (q - b.iter().find(|x| x.0 == id).map_or(0.0, |x| x.1)).abs();
            }
            for &(id, q) in &b {
                if !a.iter().any(|x| x.0 == id) {
                    d += q;
                }
            }
            worst_verify = worst_verify.max(d / 2.0);
        }

        for step in [0, 5, 13, 21, 29] {
            for (source, logits) in [("plain", &plain[step]), ("verify", &verified[step])] {
                let p = exact(logits, &PARAMS);
                let mut order: Vec<u32> = (0..logits.len() as u32).collect();
                order.select_nth_unstable_by(p.len(), |&a, &b| {
                    logits[b as usize].total_cmp(&logits[a as usize])
                });
                order.truncate(p.len() + 1);
                order.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
                let outside = *order
                    .iter()
                    .find(|id| !p.iter().any(|q| q.0 == **id))
                    .unwrap();
                let mid = p
                    .iter()
                    .skip(1)
                    .min_by(|a, b| (a.1 - 0.25).abs().total_cmp(&(b.1 - 0.25).abs()))
                    .map_or(p[0].0, |q| q.0);
                let draws = Draws::new(&m.gpu, logits, &PARAMS, 1 + step as u64);
                println!(
                    "{name} step {step} ({source}): {} candidates, top p {:.3}",
                    p.len(),
                    p[0].1
                );
                for (case, draft) in [
                    ("none", NO_DRAFT),
                    ("argmax", p[0].0),
                    ("mid", mid),
                    ("outside", outside),
                ] {
                    let (counts, kept) = draws.run(draft, runs);
                    let d = tv(&counts, n, &p);
                    let bound = 1.5 * (p.len() as f64 / n as f64).sqrt();
                    let out: usize = counts
                        .iter()
                        .filter(|(id, _)| !p.iter().any(|q| q.0 == **id))
                        .map(|(_, &c)| c)
                        .sum();
                    let pd = p.iter().find(|q| q.0 == draft).map_or(0.0, |q| q.1);
                    let acc = kept as f64 / n as f64;
                    println!(
                        "  draft {case}: TV {d:.4} (bound {bound:.4}), outside the cut {out}, kept {acc:.4} vs p(draft) {pd:.4}\n    {}",
                        top10(&m.tok, &counts, n, &p)
                    );
                    worst_tv = worst_tv.max(d / bound);
                    assert_eq!(
                        out, 0,
                        "{name} step {step} {source} draft {case}: draws outside the cut"
                    );
                    assert!(
                        d < bound,
                        "{name} step {step} {source} draft {case}: TV {d} over {bound}"
                    );
                    let sd = (pd * (1.0 - pd) / n as f64).sqrt();
                    assert!(
                        (acc - pd).abs() < 5.0 * sd + 1e-3,
                        "{name} step {step} {source} draft {case}: kept {acc} vs {pd}"
                    );
                }
            }
        }
    }
    println!(
        "worst TV over its bound {worst_tv:.2}; worst TV between verify and plain cut distributions {worst_verify:.4}"
    );
    assert!(
        worst_verify < 0.05,
        "verify rows sample another distribution: TV {worst_verify}"
    );
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Mode {
    Off,
    Mtp,
    Lookup,
}

/// Chi-square homogeneity of two samples, as a z score (Wilson-Hilferty).
/// Categories with fewer than 10 draws in both together are pooled.
fn chi2_z(a: &HashMap<u32, usize>, b: &HashMap<u32, usize>) -> f64 {
    let (na, nb) = (
        a.values().sum::<usize>() as f64,
        b.values().sum::<usize>() as f64,
    );
    let mut cats: Vec<(f64, f64)> = Vec::new();
    let mut rest = (0.0, 0.0);
    let ids: std::collections::HashSet<&u32> = a.keys().chain(b.keys()).collect();
    for id in ids {
        let x = a.get(id).copied().unwrap_or(0) as f64;
        let y = b.get(id).copied().unwrap_or(0) as f64;
        if x + y < 10.0 {
            rest = (rest.0 + x, rest.1 + y);
        } else {
            cats.push((x, y));
        }
    }
    if rest.0 + rest.1 > 0.0 {
        cats.push(rest);
    }
    if cats.len() < 2 {
        return 0.0;
    }
    let n = na + nb;
    let mut chi2 = 0.0;
    for &(x, y) in &cats {
        let col = x + y;
        let (ea, eb) = (na * col / n, nb * col / n);
        chi2 += (x - ea).powi(2) / ea + (y - eb).powi(2) / eb;
    }
    let df = (cats.len() - 1) as f64;
    ((chi2 / df).powf(1.0 / 3.0) - (1.0 - 2.0 / (9.0 * df))) / (2.0 / (9.0 * df)).sqrt()
}

fn mean_sd(x: &[f64]) -> (f64, f64) {
    let m = x.iter().sum::<f64>() / x.len() as f64;
    let v = x.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (x.len() - 1).max(1) as f64;
    (m, v.sqrt())
}

/// Whole sampled decodes (thinking on, the qh defaults) without drafts, with
/// MTP drafts and with MTP and lookup drafts, interleaved by sample: the
/// tokens at positions 0-4 and the length of the thinking must not tell the
/// modes apart.
#[test]
#[ignore = "needs the model on the Mac; about 20 minutes"]
fn sampled_decode_matches_plain() {
    let m = load();
    let samples: usize = std::env::var("SAETTA_SAMPLES").map_or(200, |s| s.parse().unwrap());
    let max = 512;
    let prompt = prompt(&m.tok, "What is 17 + 25? Reply with the number only.", true);
    let think_end = m.tok.encode("</think>");
    assert_eq!(think_end.len(), 1, "</think> is one token");
    let think_end = think_end[0];
    let stop = |t: u32| t == think_end || m.cfg.eos_token_ids.contains(&t);

    let capacity = (prompt.len() + max + 64).next_multiple_of(128);
    let mut model =
        GpuModel::load(&m.gpu, &m.cfg, &m.weights, m.cfg.num_layers(), capacity).unwrap();
    model.enable_verify(8);
    let mtp = GpuMtp::load(&m.gpu, &m.cfg, &m.weights, capacity).unwrap();
    let mut spec = GpuSpec::new(&model, mtp, &m.cfg, 3).with_lookup(7);

    let modes = [Mode::Off, Mode::Mtp, Mode::Lookup];
    let mut firsts: Vec<Vec<HashMap<u32, usize>>> = vec![vec![HashMap::new(); 5]; modes.len()];
    let mut thinking: Vec<Vec<f64>> = vec![Vec::new(); modes.len()];
    let mut cut: Vec<usize> = vec![0; modes.len()];
    let mut tokens: Vec<usize> = vec![0; modes.len()];
    let mut seconds: Vec<f64> = vec![0.0; modes.len()];
    let start = Instant::now();
    for i in 0..samples {
        for (k, &mode) in modes.iter().enumerate() {
            let mut decide = Sampled {
                params: PARAMS,
                sampler: Sampler::new(1000 + (i * modes.len() + k) as u64),
            };
            model.reset();
            spec.reset();
            spec.set_lookup(mode == Mode::Lookup);
            let mut logits = Vec::new();
            for &t in &prompt {
                logits = spec.step(&mut model, t);
            }
            let began = Instant::now();
            let mut out = vec![decide.pick(&logits)];
            while out.len() < max && !stop(*out.last().unwrap()) {
                let pending = *out.last().unwrap();
                if mode == Mode::Off {
                    let next = model.run_sampled(pending, &PARAMS, &mut decide.sampler);
                    out.push(next);
                } else {
                    let (accepted, next) = spec.speculate(&mut model, pending, &mut decide);
                    for t in accepted.into_iter().chain([next]) {
                        let done = stop(t);
                        out.push(t);
                        if done {
                            break;
                        }
                    }
                }
            }
            out.truncate(max);
            seconds[k] += began.elapsed().as_secs_f64();
            tokens[k] += out.len();
            for (p, &t) in out.iter().take(5).enumerate() {
                *firsts[k][p].entry(t).or_insert(0) += 1;
            }
            match out.iter().position(|&t| t == think_end) {
                Some(at) => thinking[k].push(at as f64),
                None => {
                    cut[k] += 1;
                    thinking[k].push(out.len() as f64);
                }
            }
        }
        if (i + 1) % 20 == 0 {
            println!(
                "{} of {samples} samples, {:.0} s",
                i + 1,
                start.elapsed().as_secs_f64()
            );
        }
    }
    spec.report(prompt.len());

    let mut worst = 0.0f64;
    for (k, mode) in modes.iter().enumerate() {
        let (mean, sd) = mean_sd(&thinking[k]);
        println!(
            "{mode:?}: thinking {mean:.1} tokens (sd {sd:.1}, {} without </think> in {max}), {:.1} tok/s",
            cut[k],
            tokens[k] as f64 / seconds[k]
        );
        for (p, c) in firsts[k].iter().enumerate() {
            let n: usize = c.values().sum();
            let mut top: Vec<(&u32, &usize)> = c.iter().collect();
            top.sort_by(|a, b| b.1.cmp(a.1).then(a.0.cmp(b.0)));
            let line: Vec<String> = top
                .iter()
                .take(10)
                .map(|&(&id, &c)| format!("{:?} {:.3}", m.tok.decode(&[id]), c as f64 / n as f64))
                .collect();
            println!("  position {p}: {}", line.join(", "));
        }
        if k > 0 {
            let (m0, s0) = mean_sd(&thinking[0]);
            let se =
                (s0 * s0 / thinking[0].len() as f64 + sd * sd / thinking[k].len() as f64).sqrt();
            let z = (mean - m0) / se.max(1e-9);
            println!("  thinking vs Off: {:+.1} tokens, z {z:.2}", mean - m0);
            worst = worst.max(z.abs());
            for (p, (a, b)) in firsts[0].iter().zip(&firsts[k]).enumerate().skip(1) {
                let z = chi2_z(a, b);
                println!("  position {p} vs Off: chi-square z {z:.2}");
                worst = worst.max(z);
            }
        }
    }
    println!("worst z {worst:.2}");
    assert!(
        worst < 4.0,
        "a drafting mode samples another distribution (z {worst:.2})"
    );
}
