use saetta::eval::{Options, Reference, gate, kl_divergence, nll, perplexity, top_k};
use saetta::server::Model;
use std::fs;
use std::path::{Path, PathBuf};

#[test]
fn math_matches_numpy() {
    // numpy: logsumexp([1, 2, 3]) - 1
    assert!((nll(&[1.0, 2.0, 3.0], 0) - 2.4076059644443806).abs() < 1e-12);
    // softmax is shift invariant
    assert!((nll(&[101.0, 102.0, 103.0], 2) - 0.40760596444438013).abs() < 1e-9);
    // sum p * log(p / q) for p = softmax([0, 1]), q = softmax([1, 0])
    assert!((kl_divergence(&[0.0, 1.0], &[1.0, 0.0]) - 0.46211715726000974).abs() < 1e-12);
    assert_eq!(kl_divergence(&[0.5, -2.0, 3.0], &[0.5, -2.0, 3.0]), 0.0);
    assert_eq!(top_k(&[0.1, 5.0, -1.0, 2.0], 2), vec![1, 3]);
    assert!((perplexity(&[1.0, 2.0, 3.0]) - 2f64.exp()).abs() < 1e-12);
}

/// Logits that depend on the token and how many tokens came before it,
/// so a model that loses its state gives different numbers.
struct Fake {
    pos: usize,
    noise: f32,
}

const VOCAB: usize = 16;

fn fake_logits(token: u32, pos: usize, noise: f32) -> Vec<f32> {
    (0..VOCAB)
        .map(|i| ((i * 7 + token as usize * 3 + pos) % 11) as f32 * 0.5 + noise * (i % 3) as f32)
        .collect()
}

impl Model for Fake {
    fn prefill(&mut self, tokens: &[u32]) -> Vec<f32> {
        let mut logits = Vec::new();
        for &t in tokens {
            logits = self.step(t);
        }
        logits
    }
    fn step(&mut self, token: u32) -> Vec<f32> {
        self.pos += 1;
        fake_logits(token, self.pos - 1, self.noise)
    }
    fn truncate(&mut self, _len: usize) -> usize {
        0
    }
    fn reset(&mut self) {
        self.pos = 0;
    }
}

fn write(path: PathBuf, values: impl IntoIterator<Item = [u8; 4]>) {
    fs::write(path, values.into_iter().flatten().collect::<Vec<u8>>()).unwrap();
}

/// A reference dir in the format tools/eval_ref.py writes, from the fake
/// model without noise.
fn write_reference(dir: &Path, ppl: &[u32], long: &[u32], checkpoints: &[usize]) {
    fs::create_dir_all(dir).unwrap();
    fs::write(
        dir.join("index.json"),
        format!("{{\"checkpoints\": {checkpoints:?}}}"),
    )
    .unwrap();
    write(
        dir.join("ppl_tokens.bin"),
        ppl.iter().map(|t| t.to_le_bytes()),
    );
    write(
        dir.join("long_tokens.bin"),
        long.iter().map(|t| t.to_le_bytes()),
    );
    for run in ["f32", "bf16"] {
        for (name, tokens) in [("ppl", ppl), ("long", long)] {
            let logits: Vec<Vec<f32>> = tokens
                .iter()
                .enumerate()
                .map(|(i, &t)| fake_logits(t, i, 0.0))
                .collect();
            let nlls = (0..tokens.len() - 1)
                .map(|i| (nll(&logits[i], tokens[i + 1]) as f32).to_le_bytes());
            write(dir.join(format!("{run}_{name}_nll.bin")), nlls);
            let argmax = logits[..tokens.len() - 1]
                .iter()
                .map(|l| top_k(l, 1)[0].to_le_bytes());
            write(dir.join(format!("{run}_{name}_argmax.bin")), argmax);
        }
        for &p in checkpoints {
            let logits = fake_logits(long[p], p, 0.0);
            write(
                dir.join(format!("{run}_logits_{p}.bin")),
                logits.iter().map(|x| x.to_le_bytes()),
            );
        }
    }
}

fn opts(ppl: bool, long: usize, decode: bool) -> Options {
    Options { ppl, long, decode }
}

#[test]
fn gate_passes_same_model_and_fails_a_different_one() {
    let dir = std::env::temp_dir().join(format!("saetta-eval-{}", std::process::id()));
    let ppl: Vec<u32> = (0..40).map(|i| (i * 5 % VOCAB) as u32).collect();
    let long: Vec<u32> = (0..64).map(|i| (i * 3 % VOCAB) as u32).collect();
    write_reference(&dir, &ppl, &long, &[15, 63]);
    let reference = Reference::load(&dir).unwrap();

    let mut same = Fake { pos: 0, noise: 0.0 };
    assert!(gate(&mut same, &reference, &opts(true, 64, false)).unwrap());
    // Only the checkpoint at 15 is below 32 tokens.
    assert!(gate(&mut same, &reference, &opts(false, 32, false)).unwrap());
    assert!(gate(&mut same, &reference, &opts(false, 8, false)).is_err());

    let mut off = Fake { pos: 0, noise: 0.3 };
    assert!(!gate(&mut off, &reference, &opts(true, 0, false)).unwrap());
    assert!(!gate(&mut off, &reference, &opts(false, 64, false)).unwrap());
    assert!(gate(&mut same, &reference, &opts(false, 64, true)).unwrap());
    assert!(!gate(&mut off, &reference, &opts(false, 64, true)).unwrap());
    fs::remove_dir_all(&dir).unwrap();
}
