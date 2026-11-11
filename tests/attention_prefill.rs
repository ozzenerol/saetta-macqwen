#![cfg(target_os = "macos")]
//! attention_flash against prefill's attention_rows2_half on random data.

use metal::MTLSize;
use saetta::attention_prefill::{FlashParams, encode_kernel};
use saetta::gpu::{Gpu, contents};
use saetta::qmv::View;

const HEADS: usize = 16;
const KV: usize = 2;
const DIM: usize = 256;

fn noise(n: usize, seed: u64, amp: f32) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            (((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0) * amp
        })
        .collect()
}

fn halves(x: &[f32]) -> Vec<half::f16> {
    x.iter().map(|&v| half::f16::from_f32(v)).collect()
}

#[test]
fn matches_rows2() {
    let gpu = Gpu::new().unwrap();
    for (pos, tokens) in [(0usize, 1usize), (0, 37), (5, 8), (70, 100), (1000, 64), (0, 300)] {
        let capacity = (pos + tokens).next_multiple_of(64) + 64;
        let q = View::new(gpu.buffer_from(&halves(&noise(tokens * HEADS * DIM, 1, 1.0))));
        let k = View::new(gpu.buffer_from(&halves(&noise(capacity * KV * DIM, 2, 3.0))));
        let v = View::new(gpu.buffer_from(&halves(&noise(capacity * KV * DIM, 3, 1.0))));
        let g = View::new(gpu.buffer_from(&noise(tokens * HEADS * 2 * DIM, 4, 1.0)));
        let want = View::new(gpu.buffer(tokens * HEADS * DIM * 4));
        let got = View::new(gpu.buffer(tokens * HEADS * DIM * 4));
        let p = FlashParams {
            heads: HEADS as u32,
            kv_heads: KV as u32,
            pos: pos as u32,
            tokens: tokens as u32,
            scale: 1.0 / (DIM as f32).sqrt(),
        };
        gpu.run(|enc| {
            enc.set_compute_pipeline_state(&gpu.pipeline("attention_rows2_half"));
            for (i, b) in [&q, &k, &v, &g, &want].iter().enumerate() {
                enc.set_buffer(i as u64, Some(&b.buffer), b.offset);
            }
            enc.set_bytes(5, size_of::<FlashParams>() as u64, (&p as *const FlashParams).cast());
            enc.dispatch_thread_groups(
                MTLSize::new(tokens.div_ceil(8) as u64, (HEADS / 4) as u64, 1),
                MTLSize::new(128, 1, 1),
            );
        });
        for kernel in ["attention_flash", "attention_flash_pre", "attention_flash_reg_64", "attention_g4_64", "attention_g8_64", "attention_g4_32", "attention_g4_128", "attention_gq4_64", "attention_gq8_64", "attention_gq4_128", "attention_g2_128", "attention_g2_256", "attention_gq2_128", "attention_gq2_256", "attention_gq1_256"] {
            gpu.run(|enc| encode_kernel(enc, &gpu, kernel, &q, &k, &v, &g, &got, p));
            let n = tokens * HEADS * DIM;
            let (a, b) = (contents::<f32>(&got.buffer, n), contents::<f32>(&want.buffer, n));
            let worst = a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0f32, f32::max);
            let nan = a.iter().any(|x| !x.is_finite());
            eprintln!("{kernel} pos {pos} tokens {tokens}: worst abs diff {worst:.2e}");
            assert!(!nan && worst < 2e-3, "{kernel} pos {pos} tokens {tokens}: {worst}");
        }
    }
}
