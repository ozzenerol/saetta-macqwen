#![cfg(target_os = "macos")]
//! Checks the router kernel against qmv (itself checked against the CPU) on
//! real weights, and times both.

use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::moe::Router;
use saetta::plan::Plan;
use saetta::qmv::View;
use saetta::quant;
use saetta::weights::Weights;
use std::path::PathBuf;

#[test]
fn matches_cpu_on_layers_0_to_3() {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let Some(dir) = ["llm/saetta-models", "llm/mlx-models"]
        .iter()
        .map(|d| home.join(d).join("Qwen3.6-35B-A3B-5bit"))
        .find(|d| d.exists())
    else {
        eprintln!("skipped: no model");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let x: Vec<f32> = (0..2048)
        .map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let xb = View::new(gpu.buffer_from(&x));

    for l in 0..4 {
        let layer = format!("language_model.model.layers.{l}");
        let router = Router::new(&gpu, &weights, &config.quantization, &layer).unwrap();
        let out = View::new(gpu.buffer(router.outputs() * 4));
        let mut plan = Plan::new(&gpu);
        router.record(&mut plan, &xb, &out, false);
        plan.run_serial();
        let got = contents::<f32>(&out.buffer, router.outputs()).to_vec();

        let mut want = Vec::new();
        for name in ["gate", "shared_expert_gate"] {
            let q = weights
                .quantized(&format!("{layer}.mlp.{name}"), &config.quantization)
                .unwrap();
            let bf16s = |b: &[u8]| -> Vec<half::bf16> {
                b.as_chunks::<2>()
                    .0
                    .iter()
                    .map(|c| half::bf16::from_bits(u16::from_le_bytes(*c)))
                    .collect()
            };
            let w = quant::dequantize(
                &q.weight.to_u32(),
                &bf16s(q.scales.bytes),
                &bf16s(q.biases.bytes),
                8,
                q.out_features(),
                q.in_features(),
            );
            for row in w.chunks(2048) {
                want.push(
                    row.iter()
                        .zip(&x)
                        .map(|(&a, &b)| a as f64 * b as f64)
                        .sum::<f64>(),
                );
            }
        }
        let scale = want.iter().map(|v| v.abs()).fold(0.0, f64::max);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0, f64::max);
        eprintln!(
            "layer {l}: {} outputs, max |logit| {scale:.3}, worst error {worst:e}",
            got.len()
        );
        assert!(worst < 1e-5 * scale.max(1.0), "layer {l}: {worst}");
    }
}
