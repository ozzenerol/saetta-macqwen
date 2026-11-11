#![cfg(target_os = "macos")]
//! Checks the split-K decode attention against an f64 CPU version on random
//! data, at lengths around the 128-token chunk size and longer.

use saetta::attention::DecodeAttention;
use saetta::gpu::{Gpu, contents};
use saetta::plan::{Plan, TokenParams};
use saetta::qmv::View;

const HEADS: usize = 16;
const KV: usize = 2;
const DIM: usize = 256;

/// Deterministic values in about [-1, 1].
fn noise(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
        })
        .collect()
}

fn reference(q: &[f32], k: &[f32], v: &[f32], gate: &[f32], len: usize) -> Vec<f64> {
    let mut out = vec![0f64; HEADS * DIM];
    for h in 0..HEADS {
        let kv = h / (HEADS / KV);
        let scores: Vec<f64> = (0..len)
            .map(|t| {
                (0..DIM)
                    .map(|i| q[h * DIM + i] as f64 * k[(t * KV + kv) * DIM + i] as f64)
                    .sum::<f64>()
                    / (DIM as f64).sqrt()
            })
            .collect();
        let m = scores.iter().cloned().fold(f64::MIN, f64::max);
        let e: Vec<f64> = scores.iter().map(|s| (s - m).exp()).collect();
        let sum: f64 = e.iter().sum();
        for i in 0..DIM {
            let a: f64 = (0..len)
                .map(|t| e[t] * v[(t * KV + kv) * DIM + i] as f64)
                .sum();
            let g = gate[h * 2 * DIM + DIM + i] as f64;
            out[h * DIM + i] = a / sum / (1.0 + (-g).exp());
        }
    }
    out
}

#[test]
fn matches_cpu_at_several_lengths() {
    let gpu = Gpu::new().unwrap();
    let capacity = 4096;
    let att = DecodeAttention::new(&gpu, HEADS, KV, DIM, capacity);
    // Big keys make the softmax peaky, which tests the max handling.
    let keys: Vec<f32> = noise(capacity * KV * DIM, 1)
        .iter()
        .map(|x| x * 4.0)
        .collect();
    let values = noise(capacity * KV * DIM, 2);
    let q = noise(HEADS * DIM, 3);
    let q_and_gate = noise(HEADS * 2 * DIM, 4);
    let [kb, vb, qb, gb] = [&keys, &values, &q, &q_and_gate].map(|d| View::new(gpu.buffer_from(d)));
    let out = View::new(gpu.buffer(HEADS * DIM * 4));

    let mut plan = Plan::new(&gpu);
    att.record(&mut plan, &qb, &kb, &vb, &gb, &out);
    let built = plan.build();
    for len in [1, 2, 127, 128, 129, 1000, 4096] {
        plan.set_params(TokenParams::new(0, len as u32 - 1));
        built.run(&gpu);
        let got = contents::<f32>(&out.buffer, HEADS * DIM);
        let want = reference(&q, &keys, &values, &q_and_gate, len);
        let worst = got
            .iter()
            .zip(&want)
            .map(|(&a, &b)| (a as f64 - b).abs())
            .fold(0.0, f64::max);
        eprintln!("len {len}: worst error {worst:e}");
        assert!(worst < 2e-5, "len {len}: {worst}");
    }
}
