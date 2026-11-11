#![cfg(target_os = "macos")]
//! Checks the two MoE kernels against `quant::dequantize` and an f64 CPU
//! version of the routed experts, on real layer 0 weights.

use half::bf16;
use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::moe::Experts;
use saetta::plan::Plan;
use saetta::qmv::View;
use saetta::quant;
use saetta::weights::{QuantTensor, Weights};
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").ok()?);
    ["llm/saetta-models", "llm/mlx-models"]
        .iter()
        .map(|d| home.join(d).join("Qwen3.6-35B-A3B-5bit"))
        .find(|d| d.exists())
}

/// Expert `e` of a stacked tensor, dequantized to `rows x cols`.
fn expert(q: &QuantTensor, e: usize) -> Vec<f32> {
    let (rows, cols) = (q.out_features(), q.in_features());
    let words = rows * quant::words_per_row(cols, q.params.bits);
    let groups = rows * cols / 64;
    let bf16s = |b: &[u8]| -> Vec<bf16> {
        b.as_chunks::<2>()
            .0
            .iter()
            .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
            .collect()
    };
    let w: Vec<u32> = q.weight.bytes[e * words * 4..(e + 1) * words * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect();
    let s = bf16s(&q.scales.bytes[e * groups * 2..(e + 1) * groups * 2]);
    let b = bf16s(&q.biases.bytes[e * groups * 2..(e + 1) * groups * 2]);
    quant::dequantize(&w, &s, &b, q.params.bits, rows, cols)
}

fn matvec(w: &[f32], x: &[f64]) -> Vec<f64> {
    w.chunks(x.len())
        .map(|row| row.iter().zip(x).map(|(&a, &b)| a as f64 * b).sum())
        .collect()
}

#[test]
fn matches_cpu_on_layer_0() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no model");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let layer = "language_model.model.layers.0";
    let experts = Experts::new(&gpu, &weights, &config.quantization, layer).unwrap();

    let hidden = experts.gate.cols;
    let x: Vec<f32> = (0..hidden)
        .map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let ids: [u32; 8] = [0, 3, 17, 42, 100, 128, 200, 255];
    let w: [f32; 8] = [0.3, 0.2, 0.15, 0.1, 0.1, 0.07, 0.05, 0.03];

    let xb = View::new(gpu.buffer_from(&x));
    let idb = View::new(gpu.buffer_from(&ids));
    let wb = View::new(gpu.buffer_from(&w));
    let hb = View::new(gpu.buffer(8 * experts.size() * 4));
    let yb = View::new(gpu.buffer(hidden * 4));
    let mut plan = Plan::new(&gpu);
    experts.record(&mut plan, &xb, &idb, &wb, 8, &hb, &yb);
    plan.run_serial();
    let y = contents::<f32>(&yb.buffer, hidden);

    let module = |name: &str| {
        weights
            .quantized(
                &format!("{layer}.mlp.switch_mlp.{name}"),
                &config.quantization,
            )
            .unwrap()
    };
    let (gate, up, down) = (module("gate_proj"), module("up_proj"), module("down_proj"));
    let x64: Vec<f64> = x.iter().map(|&v| v as f64).collect();
    let mut want = vec![0f64; hidden];
    for (&e, &weight) in ids.iter().zip(&w) {
        let g = matvec(&expert(&gate, e as usize), &x64);
        let u = matvec(&expert(&up, e as usize), &x64);
        let h: Vec<f64> = g
            .iter()
            .zip(&u)
            .map(|(&g, &u)| g / (1.0 + (-g).exp()) * u)
            .collect();
        for (o, d) in want.iter_mut().zip(matvec(&expert(&down, e as usize), &h)) {
            *o += weight as f64 * d;
        }
    }

    let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
    let worst = y
        .iter()
        .zip(&want)
        .map(|(&a, &b)| (a as f64 - b).abs())
        .fold(0.0, f64::max);
    eprintln!("moe layer 0: max |y| {scale:e}, worst error {worst:e}");
    assert!(worst < 1e-5 * scale, "worst {worst} for max |y| {scale}");
}
