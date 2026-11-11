#![cfg(target_os = "macos")]
//! Checks the qmv kernel against `quant::dequantize` and a CPU dot product on
//! real layer 0 matrices. Skipped when the model is not there.

use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::qmv::{GpuQuant, View};
use saetta::quant;
use saetta::weights::Weights;
use std::path::PathBuf;

fn model_dir() -> Option<PathBuf> {
    let dir =
        PathBuf::from(std::env::var("HOME").ok()?).join("llm/mlx-models/Qwen3.6-35B-A3B-5bit");
    dir.exists().then_some(dir)
}

/// A fixed input with mixed signs and sizes.
fn input(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 % 1000) as f32 / 500.0 - 1.0) * (1.0 + (i % 13) as f32))
        .collect()
}

#[test]
fn matches_cpu_on_real_matrices() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no model");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();

    let layer = "language_model.model.layers.0.";
    for module in [
        "linear_attn.in_proj_qkv",
        "linear_attn.out_proj",
        "mlp.shared_expert.down_proj",
        "mlp.gate",
        "mlp.shared_expert_gate",
    ] {
        let name = format!("{layer}{module}");
        let q = weights.quantized(&name, &config.quantization).unwrap();
        let m = GpuQuant::new(&gpu, &q);

        let x = input(m.cols);
        let xb = View::new(gpu.buffer_from(&x));
        let yb = View::new(gpu.buffer(m.rows * 4));
        gpu.run(|enc| m.encode(&gpu, enc, &xb, &yb));
        let y = contents::<f32>(&yb.buffer, m.rows);

        let w = quant::dequantize(
            &q.weight.to_u32(),
            &bf16s(q.scales.bytes),
            &bf16s(q.biases.bytes),
            m.bits,
            m.rows,
            m.cols,
        );
        let mut worst = 0f64;
        for r in 0..m.rows {
            let row = &w[r * m.cols..(r + 1) * m.cols];
            let exact: f64 = row.iter().zip(&x).map(|(&a, &b)| a as f64 * b as f64).sum();
            let scale: f64 = row
                .iter()
                .zip(&x)
                .map(|(&a, &b)| (a as f64 * b as f64).abs())
                .sum();
            worst = worst.max((y[r] as f64 - exact).abs() / (scale + 1e-12));
        }
        eprintln!(
            "{module}: {}x{} {}-bit, worst error / sum|w x| {worst:e}, {} parts copied",
            m.rows, m.cols, m.bits, m.copies
        );
        assert!(worst < 1e-5, "{module}: {worst}");
    }
}

fn bf16s(bytes: &[u8]) -> Vec<half::bf16> {
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)))
        .collect()
}
