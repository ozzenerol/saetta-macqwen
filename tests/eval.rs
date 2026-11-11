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

/// GPU against MLX float32 at a few long positions, after the first k
/// layers, for the k in SAETTA_BISECT_LAYERS (for example "40,20,10").
/// Needs the data from tools/eval_bisect.py. Run under ~/saetta/timed.sh:
///   cargo test --release --test eval gpu_bisect -- --ignored --nocapture
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn gpu_bisect() {
    use saetta::config::Config;
    use saetta::eval::kl_divergence;
    use saetta::gpu::Gpu;
    use saetta::gpu_model::GpuModel;
    use saetta::weights::Weights;

    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let dir = home.join("llm/saetta-models/Qwen3.6-35B-A3B-5bit");
    let eval = home.join("saetta/eval");
    let bisect = eval.join("bisect");
    let index: serde_json::Value =
        serde_json::from_str(&fs::read_to_string(bisect.join("index.json")).unwrap()).unwrap();
    let positions: Vec<usize> = index["positions"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_u64().unwrap() as usize)
        .collect();
    let (length, shift) = (
        index["length"].as_u64().unwrap() as usize,
        index["shift"].as_u64().unwrap() as usize,
    );
    let read = |path: PathBuf| -> Vec<f32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect()
    };
    let long: Vec<u32> = fs::read(eval.join("long_tokens.bin"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect();
    let layers: Vec<usize> = std::env::var("SAETTA_BISECT_LAYERS")
        .unwrap_or("40".into())
        .split(',')
        .map(|k| k.trim().parse().unwrap())
        .collect();

    let cfg = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let hidden = cfg.hidden_size;
    for &k in &layers {
        let mut model = GpuModel::load(&gpu, &cfg, &weights, k, length + 1).unwrap();
        for (name, start) in [("a", 0), ("b", shift)] {
            let tokens = &long[start..start + length];
            let mlx_hidden = read(bisect.join(format!("{name}_hidden.bin")));
            let mlx_logits = read(bisect.join(format!("{name}_logits.bin")));
            model.reset();
            for (pos, &token) in tokens.iter().enumerate() {
                let Some(slot) = positions.iter().position(|&p| p == pos) else {
                    model.run(token);
                    continue;
                };
                let logits = model.forward(token);
                let ours = model.hidden();
                let at = ((k - 1) * positions.len() + slot) * hidden;
                let theirs = &mlx_hidden[at..at + hidden];
                let err: f64 = ours
                    .iter()
                    .zip(theirs)
                    .map(|(&a, &b)| ((a - b) as f64).powi(2))
                    .sum();
                let norm: f64 = theirs.iter().map(|&b| (b as f64).powi(2)).sum();
                let mut line = format!(
                    "layers {k:2} seq {name} position {pos} (text {}): hidden rel err {:.2e}",
                    pos + start,
                    (err / norm).sqrt()
                );
                if k == cfg.num_layers() {
                    let vocab = cfg.vocab_size;
                    let kl = kl_divergence(&mlx_logits[slot * vocab..(slot + 1) * vocab], &logits);
                    line += &format!(", logits KL {kl:.2e}");
                }
                println!("{line}");
            }
        }
    }
}

/// The first k layers on the GPU and on the CPU, token by token over the
/// first 4097 long tokens, with the hidden states compared at every position
/// from 3900 on, and the CPU against MLX at the bisect positions. k is
/// SAETTA_BISECT_LAYERS (one number). Run under ~/saetta/timed.sh:
///   cargo test --release --test eval gpu_vs_cpu -- --ignored --nocapture
#[cfg(target_os = "macos")]
#[test]
#[ignore]
fn gpu_vs_cpu() {
    use saetta::config::Config;
    use saetta::cpu::model::Model as CpuModel;
    use saetta::gpu::Gpu;
    use saetta::gpu_model::GpuModel;
    use saetta::weights::Weights;

    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let dir = home.join("llm/saetta-models/Qwen3.6-35B-A3B-5bit");
    let eval = home.join("saetta/eval");
    let read = |path: PathBuf| -> Vec<f32> {
        let bytes = fs::read(path).unwrap();
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|&b| f32::from_le_bytes(b))
            .collect()
    };
    let long: Vec<u32> = fs::read(eval.join("long_tokens.bin"))
        .unwrap()
        .as_chunks::<4>()
        .0
        .iter()
        .map(|&b| u32::from_le_bytes(b))
        .collect();
    let positions = [3994, 3995, 3996, 4094, 4095, 4096];
    let mlx = read(eval.join("bisect/a_hidden.bin"));
    let k: usize = std::env::var("SAETTA_BISECT_LAYERS")
        .unwrap_or("2".into())
        .parse()
        .unwrap();
    let rel = |a: &[f32], b: &[f32]| -> f64 {
        let err: f64 = a
            .iter()
            .zip(b)
            .map(|(&x, &y)| ((x - y) as f64).powi(2))
            .sum();
        let norm: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
        (err / norm).sqrt()
    };

    let cfg = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let mut on_gpu = GpuModel::load(&gpu, &cfg, &weights, k, 4100).unwrap();
    let cpu = CpuModel::new(&cfg, &weights).unwrap();
    let mut states: Vec<_> = (0..k).map(|i| cpu.new_state(i)).collect();
    let hidden = cfg.hidden_size;
    let mut worst = 0.0f64;
    for (pos, &token) in long[..4097].iter().enumerate() {
        on_gpu.run(token);
        let mut x = cpu.embed(token);
        let mut experts = Vec::new();
        for (i, state) in states.iter_mut().enumerate() {
            let out = cpu.layer(i).step(&x, state).unwrap();
            x = out.hidden;
            experts = out.experts;
        }
        let g = on_gpu.hidden();
        let d = rel(&g, &x);
        worst = worst.max(d);
        if let Some(slot) = positions.iter().position(|&p| p == pos) {
            let at = ((k - 1) * positions.len() + slot) * hidden;
            let m = &mlx[at..at + hidden];
            println!(
                "position {pos}: GPU vs CPU {d:.2e}, CPU vs MLX {:.2e}, GPU vs MLX {:.2e}",
                rel(&x, m),
                rel(&g, m)
            );
        } else if pos >= 3900 && d > 1e-5 {
            println!("position {pos}: GPU vs CPU {d:.2e}");
        }
        if d > 1e-3 {
            println!("  CPU experts of layer {} at {pos}: {experts:?}", k - 1);
            let r = on_gpu.route_debug();
            let cpu_logits = cpu.layer(k - 1).router().matvec(&r.normed);
            let mut rank: Vec<usize> = (0..cpu_logits.len()).collect();
            rank.sort_by(|&a, &b| cpu_logits[b].total_cmp(&cpu_logits[a]));
            println!("  GPU ids {:?} weights {:?}", r.ids, r.weights);
            println!("  CPU matvec on the GPU's router input, best 10 (id, CPU logit, GPU logit):");
            for &e in &rank[..10] {
                println!("    {e:3} {:.7} {:.7}", cpu_logits[e], r.logits[e]);
            }
            let worst = cpu_logits
                .iter()
                .zip(&r.logits)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!("  router logits, GPU vs CPU on the same input: worst abs diff {worst:.2e}");
        }
    }
    println!("worst GPU vs CPU over all positions: {worst:.2e}");
}
