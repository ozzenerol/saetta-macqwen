#![cfg(target_os = "macos")]
//! Checks the split-K decode attention against an f64 CPU version on random
//! data, at lengths around the 128-token chunk size and longer.

use half::f16;
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
    let mut att = DecodeAttention::new(&gpu, HEADS, KV, DIM, capacity);
    // Big keys make the softmax peaky, which tests the max handling.
    let keys: Vec<f32> = noise(capacity * KV * DIM, 1)
        .iter()
        .map(|x| x * 4.0)
        .collect();
    let values = noise(capacity * KV * DIM, 2);
    // The cache is half; the reference uses the same rounded values.
    let round = |x: &[f32]| -> Vec<f32> { x.iter().map(|&v| f16::from_f32(v).to_f32()).collect() };
    let (keys, values) = (round(&keys), round(&values));
    let to_half = |x: &[f32]| -> Vec<f16> { x.iter().map(|&v| f16::from_f32(v)).collect() };
    let q = noise(HEADS * DIM, 3);
    let q_and_gate = noise(HEADS * 2 * DIM, 4);
    let [kb, vb] = [&keys, &values].map(|d| View::new(gpu.buffer_from(&to_half(d))));
    let [qb, gb] = [&q, &q_and_gate].map(|d| View::new(gpu.buffer_from(d)));
    let out = View::new(gpu.buffer(HEADS * DIM * 4));

    for kernel in ["attention_partial", "attention_partial_mma"] {
        att.kernel = kernel;
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
            eprintln!("{kernel} len {len}: worst error {worst:e}");
            assert!(worst < 2e-5, "{kernel} len {len}: {worst}");
        }
    }
}

/// attention_partial_mma_q8/q4 on a quantized cache against the f64
/// reference on the values the codes stand for.
#[test]
fn quantized_matches_cpu() {
    use metal::MTLSize;
    use saetta::kv_quant::{KvBits, dequantize, quantize};
    let gpu = Gpu::new().unwrap();
    let capacity = 1024;
    let chunks = capacity / 128;
    let q = noise(HEADS * DIM, 3);
    let q_and_gate = noise(HEADS * 2 * DIM, 4);
    let [qb, gb] = [&q, &q_and_gate].map(|d| View::new(gpu.buffer_from(d)));
    let part_o = View::new(gpu.buffer(chunks * HEADS * DIM * 4));
    let part_ml = View::new(gpu.buffer(chunks * HEADS * 2 * 4));
    let out = View::new(gpu.buffer(HEADS * DIM * 4));
    #[repr(C)]
    struct DecodeParams {
        heads: u32,
        kv_heads: u32,
        capacity: u32,
        scale: f32,
    }
    let dp = DecodeParams { heads: HEADS as u32, kv_heads: KV as u32, capacity: capacity as u32, scale: 1.0 / (DIM as f32).sqrt() };
    for (bits, kernel) in [(KvBits::Eight, "attention_partial_mma_q8"), (KvBits::Four, "attention_partial_mma_q4")] {
        let keys: Vec<f32> = noise(capacity * KV * DIM, 1).iter().map(|x| x * 4.0).collect();
        let values = noise(capacity * KV * DIM, 2);
        let (kc, ks, kz) = quantize(&keys, bits);
        let (vc, vs, vz) = quantize(&values, bits);
        let (keys, values) = (dequantize(&kc, &ks, &kz, bits), dequantize(&vc, &vs, &vz, bits));
        let bufs: Vec<View> = vec![
            View::new(gpu.buffer_from(&kc)),
            View::new(gpu.buffer_from(&vc)),
            View::new(gpu.buffer_from(&ks)),
            View::new(gpu.buffer_from(&kz)),
            View::new(gpu.buffer_from(&vs)),
            View::new(gpu.buffer_from(&vz)),
        ];
        for len in [1, 127, 129, 1000, 1024] {
            let tp = TokenParams::new(0, len as u32 - 1);
            gpu.run(|enc| {
                enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
                enc.set_buffer(0, Some(&qb.buffer), 0);
                for (i, b) in bufs.iter().enumerate() {
                    enc.set_buffer(1 + i as u64, Some(&b.buffer), 0);
                }
                enc.set_buffer(7, Some(&part_o.buffer), 0);
                enc.set_buffer(8, Some(&part_ml.buffer), 0);
                enc.set_bytes(9, size_of::<DecodeParams>() as u64, (&dp as *const DecodeParams).cast());
                enc.set_bytes(10, size_of::<TokenParams>() as u64, (&tp as *const TokenParams).cast());
                enc.dispatch_thread_groups(MTLSize::new(chunks as u64, KV as u64, 1), MTLSize::new(256, 1, 1));
                enc.set_compute_pipeline_state(&gpu.pipeline("attention_reduce"));
                for (i, b) in [&part_o, &part_ml, &gb, &out].iter().enumerate() {
                    enc.set_buffer(i as u64, Some(&b.buffer), 0);
                }
                enc.set_bytes(4, size_of::<DecodeParams>() as u64, (&dp as *const DecodeParams).cast());
                enc.set_bytes(5, size_of::<TokenParams>() as u64, (&tp as *const TokenParams).cast());
                enc.dispatch_thread_groups(MTLSize::new(HEADS as u64, (DIM / 64) as u64, 1), MTLSize::new(1024, 1, 1));
            });
            let got = contents::<f32>(&out.buffer, HEADS * DIM);
            let want = reference(&q, &keys, &values, &q_and_gate, len);
            let worst = got.iter().zip(&want).map(|(&a, &b)| (a as f64 - b).abs()).fold(0.0, f64::max);
            eprintln!("{kernel} len {len}: worst error {worst:e}");
            assert!(worst < 2e-5, "{kernel} len {len}: {worst}");
        }
    }
}
