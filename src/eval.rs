//! The quality gate: a model behind `server::Model` against MLX reference
//! data from `tools/eval_ref.py`.
//!
//! - ppl: perplexity on about 2k tokens of prose, code and JSON, against MLX
//!   in float32. The bf16 MLX run (what mlx_lm.server runs) is printed next
//!   to it to show how far a lower precision moves the number.
//! - long: the first N tokens of a 32768-token prompt, fed through
//!   `prefill` in segments that end at each checkpoint position below N, as
//!   the server feeds a long prompt, or one token at a time with `decode`,
//!   which also gives the perplexity over the whole prompt. The logits at
//!   each checkpoint are compared with MLX float32 by KL divergence and the
//!   top token.
//!
//! The long check passes on the median KL over all checkpoints, not on each
//! one. Some positions are ill-conditioned: at position 4095 of the 32k
//! prompt the GPU model's KL went from 3.4e-2 to 6.3e-2 between builds with
//! small kernel changes, while the other checkpoints stayed below 6e-4 and
//! the perplexity over the first 4096 tokens stayed within 0.042% of MLX
//! float32. MLX's own bf16 run is at 4.9e-3 there. Rounding moves expert
//! choices at near-ties (layer 1 at position 4051 has two experts within
//! 9e-7), and the later layers carry that forward. Forcing the CPU's choice
//! at 4051 did not bring 4095 down, so no single tie explains it. A real bug
//! moves most checkpoints and the perplexity; an ill-conditioned position
//! moves one.

use crate::server::Model;
use serde_json::Value;
use std::error::Error;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

/// Largest accepted relative perplexity difference from MLX float32.
pub const MAX_PPL_REL: f64 = 0.005;
/// Largest accepted median KL divergence, in nats, over the long-context
/// checkpoints.
pub const MAX_MEDIAN_KL: f64 = 1e-3;
/// Largest accepted KL divergence at any one checkpoint.
pub const MAX_KL: f64 = 0.1;
/// Smallest accepted share of checkpoints with the same top token as MLX.
pub const MIN_TOP1_SAME: f64 = 0.9;

pub struct Reference {
    dir: PathBuf,
    pub ppl_tokens: Vec<u32>,
    pub long_tokens: Vec<u32>,
    pub checkpoints: Vec<usize>,
}

impl Reference {
    pub fn load(dir: &Path) -> Result<Reference, Box<dyn Error>> {
        let index: Value = serde_json::from_str(&fs::read_to_string(dir.join("index.json"))?)?;
        let checkpoints = index["checkpoints"]
            .as_array()
            .ok_or("index.json has no checkpoints")?
            .iter()
            .map(|v| v.as_u64().map(|p| p as usize))
            .collect::<Option<Vec<_>>>()
            .ok_or("bad checkpoint")?;
        Ok(Reference {
            dir: dir.to_path_buf(),
            // A long-only reference has no ppl text.
            ppl_tokens: read_u32(&dir.join("ppl_tokens.bin")).unwrap_or_default(),
            long_tokens: read_u32(&dir.join("long_tokens.bin"))?,
            checkpoints,
        })
    }

    /// Per-position next-token NLL of a run, `run` is "f32" or "bf16".
    pub fn nll(&self, run: &str, seq: &str) -> Result<Vec<f32>, Box<dyn Error>> {
        read_f32(&self.dir.join(format!("{run}_{seq}_nll.bin")))
    }

    pub fn argmax(&self, run: &str, seq: &str) -> Result<Vec<u32>, Box<dyn Error>> {
        read_u32(&self.dir.join(format!("{run}_{seq}_argmax.bin")))
    }

    pub fn logits(&self, run: &str, pos: usize) -> Result<Vec<f32>, Box<dyn Error>> {
        read_f32(&self.dir.join(format!("{run}_logits_{pos}.bin")))
    }

    /// The bf16 data, which long references may leave out.
    fn bf16_nll(&self, seq: &str) -> Option<Vec<f32>> {
        self.nll("bf16", seq).ok()
    }

    fn bf16_logits(&self, pos: usize) -> Option<Vec<f32>> {
        self.logits("bf16", pos).ok()
    }
}

fn read_f32(path: &Path) -> Result<Vec<f32>, Box<dyn Error>> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| f32::from_le_bytes(b))
        .collect())
}

fn read_u32(path: &Path) -> Result<Vec<u32>, Box<dyn Error>> {
    let bytes = fs::read(path).map_err(|e| format!("{}: {e}", path.display()))?;
    Ok(bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect())
}

/// `-log softmax(logits)[target]`, in f64.
pub fn nll(logits: &[f32], target: u32) -> f64 {
    log_sum_exp(logits) - logits[target as usize] as f64
}

fn log_sum_exp(logits: &[f32]) -> f64 {
    let max = logits.iter().fold(f32::NEG_INFINITY, |m, &x| m.max(x)) as f64;
    let sum: f64 = logits.iter().map(|&x| (x as f64 - max).exp()).sum();
    max + sum.ln()
}

/// KL(P || Q) in nats, where P is the softmax of `reference`.
pub fn kl_divergence(reference: &[f32], ours: &[f32]) -> f64 {
    let (lp, lq) = (log_sum_exp(reference), log_sum_exp(ours));
    reference
        .iter()
        .zip(ours)
        .map(|(&p, &q)| {
            let log_p = p as f64 - lp;
            log_p.exp() * (log_p - (q as f64 - lq))
        })
        .sum()
}

/// Ids of the `k` largest logits, largest first.
pub fn top_k(logits: &[f32], k: usize) -> Vec<u32> {
    let mut ids: Vec<u32> = (0..logits.len() as u32).collect();
    ids.sort_by(|&a, &b| logits[b as usize].total_cmp(&logits[a as usize]));
    ids.truncate(k);
    ids
}

pub fn perplexity(nll: &[f32]) -> f64 {
    let mean = nll.iter().map(|&x| x as f64).sum::<f64>() / nll.len() as f64;
    mean.exp()
}

/// What a model gives for one token sequence.
pub struct Run {
    /// NLL of token i+1 from the logits after token i.
    pub nll: Vec<f32>,
    pub argmax: Vec<u32>,
    /// Full logits at the requested positions.
    pub logits: Vec<(usize, Vec<f32>)>,
}

/// Feeds the tokens one at a time from a fresh state.
pub fn run<M: Model>(model: &mut M, tokens: &[u32], keep: &[usize]) -> Run {
    model.reset();
    let start = Instant::now();
    let mut out = Run {
        nll: Vec::new(),
        argmax: Vec::new(),
        logits: Vec::new(),
    };
    for (i, &token) in tokens.iter().enumerate() {
        let logits = model.step(token);
        if let Some(&next) = tokens.get(i + 1) {
            out.nll.push(nll(&logits, next) as f32);
            out.argmax.push(crate::sample::argmax(&logits));
        }
        if keep.contains(&i) {
            out.logits.push((i, logits));
        }
        if (i + 1) % 256 == 0 {
            let ms = start.elapsed().as_secs_f64() * 1e3 / (i + 1) as f64;
            eprintln!("  {}/{} tokens, {ms:.0} ms/token", i + 1, tokens.len());
        }
    }
    out
}

pub struct PplReport {
    pub ours: f64,
    pub mlx_f32: f64,
    pub mlx_bf16: Option<f64>,
    pub mean_abs_diff: f64,
    pub max_abs_diff: f64,
    pub top1_agree: f64,
}

impl PplReport {
    pub fn new(
        ours: &Run,
        f32_nll: &[f32],
        f32_argmax: &[u32],
        bf16_nll: Option<&[f32]>,
    ) -> PplReport {
        let n = ours.nll.len();
        let diffs: Vec<f64> = ours
            .nll
            .iter()
            .zip(f32_nll)
            .map(|(&a, &b)| (a as f64 - b as f64).abs())
            .collect();
        let agree = ours
            .argmax
            .iter()
            .zip(f32_argmax)
            .filter(|(a, b)| a == b)
            .count();
        PplReport {
            ours: perplexity(&ours.nll),
            mlx_f32: perplexity(&f32_nll[..n]),
            mlx_bf16: bf16_nll.map(|b| perplexity(&b[..n])),
            mean_abs_diff: diffs.iter().sum::<f64>() / n as f64,
            max_abs_diff: diffs.iter().fold(0.0, |m: f64, &d| m.max(d)),
            top1_agree: agree as f64 / n as f64,
        }
    }

    pub fn rel(&self) -> f64 {
        (self.ours / self.mlx_f32 - 1.0).abs()
    }

    pub fn pass(&self) -> bool {
        self.rel() <= MAX_PPL_REL
    }

    pub fn print(&self, name: &str, tokens: usize) {
        let bf16 = match self.mlx_bf16 {
            Some(b) => format!(
                ", MLX bf16 {b:.4} ({:+.3}%)",
                (b / self.mlx_f32 - 1.0) * 100.0
            ),
            None => String::new(),
        };
        println!(
            "{name}: {tokens} tokens, ppl {:.4} vs MLX f32 {:.4} ({:+.3}%){bf16}",
            self.ours,
            self.mlx_f32,
            (self.ours / self.mlx_f32 - 1.0) * 100.0,
        );
        println!(
            "  per token |NLL diff| mean {:.5} max {:.4}, top-1 agrees {:.2}%  {}",
            self.mean_abs_diff,
            self.max_abs_diff,
            self.top1_agree * 100.0,
            verdict(self.pass())
        );
    }
}

pub struct LogitsReport {
    pub pos: usize,
    pub kl: f64,
    pub kl_bf16: Option<f64>,
    pub top1_same: bool,
    pub top10_overlap: usize,
}

impl LogitsReport {
    pub fn new(
        pos: usize,
        f32_ref: &[f32],
        bf16_ref: Option<&[f32]>,
        ours: &[f32],
    ) -> LogitsReport {
        let ref_top = top_k(f32_ref, 10);
        let our_top = top_k(ours, 10);
        LogitsReport {
            pos,
            kl: kl_divergence(f32_ref, ours),
            kl_bf16: bf16_ref.map(|b| kl_divergence(f32_ref, b)),
            top1_same: ref_top[0] == our_top[0],
            top10_overlap: our_top.iter().filter(|id| ref_top.contains(id)).count(),
        }
    }

    pub fn print(&self) {
        let bf16 = match self.kl_bf16 {
            Some(k) => format!(" (MLX bf16 {k:.2e})"),
            None => String::new(),
        };
        println!(
            "  position {:6}: KL {:.2e}{bf16}, top-1 {}, top-10 overlap {}/10  {}",
            self.pos,
            self.kl,
            if self.top1_same { "same" } else { "DIFFERENT" },
            self.top10_overlap,
            if self.kl > MAX_MEDIAN_KL { "high" } else { "" }
        );
    }
}

/// Prints the summary over the checkpoints and returns whether it passes.
pub fn checkpoints_pass(reports: &[LogitsReport]) -> bool {
    let mut kls: Vec<f64> = reports.iter().map(|r| r.kl).collect();
    kls.sort_by(f64::total_cmp);
    let n = kls.len();
    let median = if n % 2 == 1 {
        kls[n / 2]
    } else {
        (kls[n / 2 - 1] + kls[n / 2]) / 2.0
    };
    let max = kls[n - 1];
    let same = reports.iter().filter(|r| r.top1_same).count() as f64 / n as f64;
    let pass = median <= MAX_MEDIAN_KL && max <= MAX_KL && same >= MIN_TOP1_SAME;
    println!(
        "  {n} checkpoints: median KL {median:.2e} (max {MAX_MEDIAN_KL:.0e}), max KL {max:.2e} (max {MAX_KL}), top-1 same {:.0}% (min {:.0}%)  {}",
        same * 100.0,
        MIN_TOP1_SAME * 100.0,
        verdict(pass)
    );
    if n < 8 {
        println!("  only {n} checkpoints below this length; the median says little");
    }
    pass
}

fn verdict(pass: bool) -> &'static str {
    if pass { "ok" } else { "FAIL" }
}

pub struct Options {
    pub ppl: bool,
    /// How many tokens of the long prompt to run; 0 skips it.
    pub long: usize,
    /// Feed the long prompt one token at a time through `step` instead of
    /// `prefill`, to check the decode path at long positions.
    pub decode: bool,
}

/// Mean and max |NLL diff| from MLX float32 per 1024 positions, to show
/// where along the prompt a model starts to drift.
fn print_blocks(ours: &[f32], reference: &[f32]) {
    for (i, (a, b)) in ours.chunks(1024).zip(reference.chunks(1024)).enumerate() {
        let diffs: Vec<f64> = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| (x as f64 - y as f64).abs())
            .collect();
        let max = diffs.iter().fold(0.0, |m: f64, &d| m.max(d));
        println!(
            "  positions {:5}..{:5}: |NLL diff| mean {:.5} max {:.4}",
            i * 1024,
            i * 1024 + a.len(),
            diffs.iter().sum::<f64>() / diffs.len() as f64,
            max
        );
    }
}

/// Prefills from a fresh state in segments that end at each checkpoint, and
/// keeps the logits there.
fn prefill_to<M: Model>(
    model: &mut M,
    tokens: &[u32],
    checkpoints: &[usize],
) -> Vec<(usize, Vec<f32>)> {
    model.reset();
    let mut done = 0;
    let mut out = Vec::new();
    for &pos in checkpoints {
        out.push((pos, model.prefill(&tokens[done..=pos])));
        done = pos + 1;
    }
    out
}

/// Runs the gate and prints a report. Returns whether every check passed.
pub fn gate<M: Model>(
    model: &mut M,
    reference: &Reference,
    options: &Options,
) -> Result<bool, Box<dyn Error>> {
    let mut pass = true;
    let long = options.long;
    if options.ppl {
        let tokens = &reference.ppl_tokens;
        if tokens.is_empty() {
            return Err("this reference has no ppl text; use --no-ppl".into());
        }
        eprintln!("ppl: {} tokens", tokens.len());
        let ours = run(model, tokens, &[]);
        let report = PplReport::new(
            &ours,
            &reference.nll("f32", "ppl")?,
            &reference.argmax("f32", "ppl")?,
            reference.bf16_nll("ppl").as_deref(),
        );
        report.print("ppl", tokens.len());
        pass &= report.pass();
    }
    if long > 0 {
        let long = long.min(reference.long_tokens.len());
        let tokens = &reference.long_tokens[..long];
        let checkpoints: Vec<usize> = reference
            .checkpoints
            .iter()
            .copied()
            .filter(|&p| p < long)
            .collect();
        if checkpoints.is_empty() {
            return Err(format!(
                "no checkpoint below {long} tokens; the first is {:?}",
                reference.checkpoints.first()
            )
            .into());
        }
        let path = if options.decode { "step" } else { "prefill" };
        println!("long: {long} tokens through {path}");
        let start = Instant::now();
        let ours = if options.decode {
            let ours = run(model, tokens, &checkpoints);
            let f32_nll = reference.nll("f32", "long")?;
            let report = PplReport::new(
                &ours,
                &f32_nll,
                &reference.argmax("f32", "long")?,
                reference.bf16_nll("long").as_deref(),
            );
            report.print("long", long);
            pass &= report.pass();
            print_blocks(&ours.nll, &f32_nll);
            ours.logits
        } else {
            prefill_to(model, tokens, &checkpoints)
        };
        let mut reports = Vec::new();
        for (pos, logits) in ours {
            let r = LogitsReport::new(
                pos,
                &reference.logits("f32", pos)?,
                reference.bf16_logits(pos).as_deref(),
                &logits,
            );
            r.print();
            reports.push(r);
        }
        pass &= checkpoints_pass(&reports);
        eprintln!("  {long} tokens in {:.1} s", start.elapsed().as_secs_f64());
    }
    println!("{}", if pass { "eval passed" } else { "eval FAILED" });
    Ok(pass)
}
