#![cfg(target_os = "macos")]
//! Checks the kernels with a folded RMS norm (qmv_norm, router_norm,
//! moe_gate_up_norm) against the plain kernels on a CPU-normed input, and
//! qmv_acc against qmv plus the add, on real layer 0 weights.

use saetta::config::Config;
use saetta::cpu::ops::rms_norm;
use saetta::gpu::{Gpu, contents};
use saetta::moe::{Experts, Router};
use saetta::plan::Plan;
use saetta::qmv::{GpuQuant, Mode, Norm, View};
use saetta::weights::Weights;
use std::path::PathBuf;

const EPS: f32 = 1e-6;

fn model_dir() -> Option<PathBuf> {
    let home = PathBuf::from(std::env::var("HOME").ok()?);
    ["llm/saetta-models", "llm/mlx-models"]
        .iter()
        .map(|d| home.join(d).join("Qwen3.6-35B-A3B-5bit"))
        .find(|d| d.exists())
}

fn input(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| ((i * 7919 + seed) % 1000) as f32 / 500.0 - 1.0)
        .map(|v| v * 3.0)
        .collect()
}

/// Worst |a - b| over max |b|.
fn rel(a: &[f32], b: &[f32]) -> f32 {
    let top = b.iter().fold(0f32, |m, v| m.max(v.abs()));
    a.iter().zip(b).fold(0f32, |m, (x, y)| m.max((x - y).abs())) / top
}

#[test]
fn folded_norm_matches_plain() {
    let Some(dir) = model_dir() else {
        eprintln!("skipped: no model");
        return;
    };
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();
    let layer = "language_model.model.layers.0";
    let read = |v: &View, n: usize| contents::<f32>(&v.buffer, n).to_vec();

    let x = input(2048, 0);
    let nw = input(2048, 17);
    let mut normed = x.clone();
    rms_norm(&mut normed, &nw, EPS);
    let xb = View::new(gpu.buffer_from(&x));
    let nb = View::new(gpu.buffer_from(&normed));
    let wb = View::new(gpu.buffer_from(&nw));
    let norm = Norm {
        weight: &wb,
        eps: EPS,
    };

    for module in [
        "linear_attn.in_proj_qkv",
        "mlp.shared_expert.gate_proj",
        "mlp.gate",
    ] {
        let q = weights
            .quantized(&format!("{layer}.{module}"), &config.quantization)
            .unwrap();
        let m = GpuQuant::new(&gpu, &q);
        let y0 = View::new(gpu.buffer(m.rows * 4));
        let y1 = View::new(gpu.buffer(m.rows * 4));
        let mut plan = Plan::new(&gpu);
        m.record(&mut plan, &nb, &y0, false);
        m.record_mode(&mut plan, &xb, &y1, false, Mode::Norm(norm));
        plan.run_serial();
        let e = rel(&read(&y1, m.rows), &read(&y0, m.rows));
        eprintln!("qmv_norm {module} {}-bit: {e:e}", m.bits);
        assert!(e < 1e-5, "{module}: {e}");

        let base = input(m.rows, 5);
        let ya = View::new(gpu.buffer_from(&base));
        let mut plan = Plan::new(&gpu);
        m.record_mode(&mut plan, &nb, &ya, false, Mode::Acc);
        plan.run_serial();
        let want: Vec<f32> = read(&y0, m.rows)
            .iter()
            .zip(&base)
            .map(|(a, b)| a + b)
            .collect();
        let e = rel(&read(&ya, m.rows), &want);
        eprintln!("qmv_acc {module}: {e:e}");
        assert!(e < 1e-6, "{module} acc: {e}");
    }

    let router = Router::new(&gpu, &weights, &config.quantization, layer).unwrap();
    let n = router.outputs();
    let (r0, r1) = (View::new(gpu.buffer(n * 4)), View::new(gpu.buffer(n * 4)));
    let mut plan = Plan::new(&gpu);
    router.record(&mut plan, &nb, &r0, false);
    router.record_norm(&mut plan, &xb, Some(norm), &r1, false);
    plan.run_serial();
    let e = rel(&read(&r1, n), &read(&r0, n));
    eprintln!("router_norm: {e:e}");
    assert!(e < 1e-5, "router: {e}");

    let experts = Experts::new(&gpu, &weights, &config.quantization, layer).unwrap();
    let ids: [u32; 8] = [0, 3, 17, 42, 100, 128, 200, 255];
    let idb = View::new(gpu.buffer_from(&ids));
    let len = 8 * experts.size();
    let (h0, h1) = (
        View::new(gpu.buffer(len * 4)),
        View::new(gpu.buffer(len * 4)),
    );
    let mut plan = Plan::new(&gpu);
    experts.record_gate_up(&mut plan, &nb, &idb, 8, &h0, false);
    experts.record_gate_up_norm(&mut plan, &xb, Some(norm), &idb, 8, &h1, false);
    plan.run_serial();
    let e = rel(&read(&h1, len), &read(&h0, len));
    eprintln!("moe_gate_up_norm: {e:e}");
    assert!(e < 1e-5, "moe: {e}");
}
