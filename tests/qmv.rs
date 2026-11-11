#![cfg(target_os = "macos")]
//! Checks the qmv kernel against `quant::dequantize` and a CPU dot product on
//! real layer 0 matrices. Skipped when the model is not there.

use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::plan::Plan;
use saetta::qmv::{GpuQuant, View};
use saetta::quant;
use saetta::weights::Weights;
use std::path::PathBuf;

/// The aligned copy from `saetta convert` if it is there, else the MLX dir.
fn model_dir() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").ok()?);
    ["llm/saetta-models", "llm/mlx-models"]
        .iter()
        .map(|d| home.join(d).join("Qwen3.6-35B-A3B-5bit"))
        .find(|d| d.exists())
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
        let mut plan = Plan::new(&gpu);
        m.record(&mut plan, &xb, &yb, false);
        plan.run_serial();
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

#[test]
fn staged_matches_plain() {
    use saetta::qmv::Variant;
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no model");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    for module in [
        "linear_attn.in_proj_qkv",
        "linear_attn.out_proj",
        "mlp.shared_expert.down_proj",
        "mlp.gate",
    ] {
        let q = weights
            .quantized(
                &format!("language_model.model.layers.0.{module}"),
                &config.quantization,
            )
            .unwrap();
        let m = GpuQuant::new(&gpu, &q);
        let x = input(m.cols);
        let nw: Vec<f32> = (0..m.cols).map(|i| 0.5 + (i % 5) as f32 * 0.25).collect();
        let eps = 1e-6f32;
        let mean = x.iter().map(|v| v * v).sum::<f32>() / m.cols as f32;
        let inv = 1.0 / (mean + eps).sqrt();
        let normed: Vec<f32> = x.iter().zip(&nw).map(|(a, w)| a * w * inv).collect();
        let (xb, nb, wb) = (
            View::new(gpu.buffer_from(&x)),
            View::new(gpu.buffer_from(&normed)),
            View::new(gpu.buffer_from(&nw)),
        );
        let read = |v: &View| contents::<f32>(&v.buffer, m.rows).to_vec();
        let top = |v: &[f32]| v.iter().fold(0f32, |a, b| a.max(b.abs()));
        let want = View::new(gpu.buffer(m.rows * 4));
        let want_n = View::new(gpu.buffer(m.rows * 4));
        let mut plan = Plan::new(&gpu);
        m.record(&mut plan, &xb, &want, false);
        m.record(&mut plan, &nb, &want_n, false);
        plan.run_serial();
        let (w, wn) = (read(&want), read(&want_n));
        let mut variants = vec![(Variant::Staged(4), false), (Variant::Staged(8), false)];
        if m.bits == 5 {
            variants.push((Variant::StagedNorm(&wb, eps), true));
        }
        for (variant, norm) in variants {
            let got = View::new(gpu.buffer(m.rows * 4));
            let mut plan = Plan::new(&gpu);
            m.record_variant(&mut plan, variant, &xb, &got, false);
            plan.run_serial();
            let g = read(&got);
            let r = if norm { &wn } else { &w };
            let err = g.iter().zip(r).fold(0f32, |a, (x, y)| a.max((x - y).abs())) / top(r);
            eprintln!("{module} {variant:?}: {err:e}");
            assert!(
                err < if norm { 1e-5 } else { 1e-6 },
                "{module} {variant:?}: {err}"
            );
        }
    }
}
