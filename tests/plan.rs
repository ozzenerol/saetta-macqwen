#![cfg(target_os = "macos")]
//! A plan must give the same result run from a serial encoder and replayed
//! from its indirect command buffer.

use metal::MTLSize;
use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::moe::{Experts, bench_ids};
use saetta::plan::{Plan, TokenParams};
use saetta::qmv::{GpuQuant, View};
use saetta::weights::Weights;
use std::path::PathBuf;

const N: usize = 1024;

/// 100 dependent ops each add 1, then one adds the token position.
fn chain(plan: &mut Plan, data: &View) {
    let grid = MTLSize::new(N as u64 / 256, 1, 1);
    let tg = MTLSize::new(256, 1, 1);
    for _ in 0..100 {
        plan.op("add_one", &[data], grid, tg);
    }
    let params = plan.params.clone();
    plan.op("add_position", &[data, &params], grid, tg);
}

#[test]
fn serial_and_replay_agree_on_a_chain() {
    let gpu = Gpu::new().unwrap();
    let data = View::new(gpu.buffer(N * 4));
    let mut plan = Plan::new(&gpu);
    chain(&mut plan, &data);
    let built = plan.build();

    plan.set_params(TokenParams::new(0, 7));
    plan.run_serial();
    assert!(contents::<f32>(&data.buffer, N).iter().all(|&v| v == 107.0));
    plan.set_params(TokenParams::new(0, 1000));
    built.run(&gpu);
    assert!(
        contents::<f32>(&data.buffer, N)
            .iter()
            .all(|&v| v == 1207.0)
    );
}

#[test]
fn serial_and_replay_agree_on_real_layers() {
    let home = PathBuf::from(std::env::var("HOME").unwrap());
    let dir = home.join("llm/saetta-models/Qwen3.6-35B-A3B-5bit");
    if !dir.exists() {
        eprintln!("skipped: no aligned model");
        return;
    }
    let config = Config::load(&dir).unwrap();
    let weights = Weights::open(&dir).unwrap();
    let gpu = Gpu::new().unwrap();

    // Layers 0 to 3. Each runs the routed experts on the same input into
    // its own output; in the Gated DeltaNet layers an out_proj (2048 x 4096)
    // first reads the previous layer's expert scratch `h`, so data flows
    // between ops. (Chaining the experts directly underflows to 0 without
    // norms.)
    let hidden = 2048;
    let x: Vec<f32> = (0..hidden)
        .map(|i| (i * 7919 % 1000) as f32 / 500.0 - 1.0)
        .collect();
    let xb = View::new(gpu.buffer_from(&x));
    let h = View::new(gpu.buffer(8 * 512 * 4));
    let w = View::new(gpu.buffer_from(&[0.125f32; 8]));
    let outs: Vec<View> = (0..8).map(|_| View::new(gpu.buffer(hidden * 4))).collect();
    let mut plan = Plan::new(&gpu);
    for l in 0..4 {
        let layer = format!("language_model.model.layers.{l}");
        let out_proj = format!("{layer}.linear_attn.out_proj");
        if let Ok(q) = weights.quantized(&out_proj, &config.quantization) {
            GpuQuant::new(&gpu, &q).record(&mut plan, &h, &outs[2 * l + 1], false);
        }
        let experts = Experts::new(&gpu, &weights, &config.quantization, &layer).unwrap();
        let ids = View::new(gpu.buffer_from(&bench_ids(l)));
        experts.record(&mut plan, &xb, &ids, &w, 8, &h, &outs[2 * l]);
    }

    let read = || -> Vec<f32> {
        outs.iter()
            .flat_map(|o| contents::<f32>(&o.buffer, hidden).to_vec())
            .collect()
    };
    let reset = || {
        for b in outs.iter().chain([&h]) {
            unsafe {
                std::ptr::write_bytes(
                    b.buffer.contents() as *mut u8,
                    0,
                    b.buffer.length() as usize,
                )
            };
        }
    };
    reset();
    plan.run_serial();
    let serial = read();
    reset();
    plan.build().run(&gpu);
    let replay = read();
    let max = serial.iter().map(|v| v.abs()).fold(0f32, f32::max);
    eprintln!("{} ops, max |y| {max:e}", plan.len());
    assert!(serial.iter().all(|v| v.is_finite()) && max > 0.0);
    assert_eq!(serial, replay);
}
