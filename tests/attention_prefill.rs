#![cfg(target_os = "macos")]
//! Checks the flash prefill attention kernels against prefill's
//! attention_rows2_half on spread data, at chunk shapes that leave partial
//! threadgroups and key blocks.

use saetta::attention_prefill::{FlashParams, KERNELS, encode, encode_rows2, spread};
use saetta::gpu::{Gpu, contents};
use saetta::qmv::View;

#[test]
fn matches_rows2() {
    let gpu = Gpu::new().unwrap();
    let (heads, kv, dim) = (16usize, 2usize, 256usize);
    for (pos, tokens) in [
        (0usize, 1usize),
        (0, 7),
        (5, 8),
        (70, 100),
        (1000, 64),
        (0, 300),
        (4093, 37),
    ] {
        let capacity = (pos + tokens).next_multiple_of(64) + 64;
        let h = |n: usize, seed| View::new(gpu.buffer_from(&spread(n, seed)));
        let (q, k, v) = (
            h(tokens * heads * dim, 1),
            h(capacity * kv * dim, 2),
            h(capacity * kv * dim, 3),
        );
        let gate: Vec<f32> = (0..tokens * heads * 2 * dim)
            .map(|i| (i % 7) as f32 - 3.0)
            .collect();
        let g = View::new(gpu.buffer_from(&gate));
        let n = tokens * heads * dim;
        let want = View::new(gpu.buffer(n * 4));
        let got = View::new(gpu.buffer(n * 4));
        let p = FlashParams {
            heads: heads as u32,
            kv_heads: kv as u32,
            pos: pos as u32,
            tokens: tokens as u32,
            scale: 1.0 / (dim as f32).sqrt(),
        };
        gpu.run(|enc| encode_rows2(enc, &gpu, &q, &k, &v, &g, &want, p));
        let b = contents::<f32>(&want.buffer, n).to_vec();
        for kernel in KERNELS {
            gpu.run(|enc| encode(enc, &gpu, kernel, &q, &k, &v, &g, &got, p));
            let a = contents::<f32>(&got.buffer, n);
            let worst = a
                .iter()
                .zip(&b)
                .map(|(x, y)| (x - y).abs())
                .fold(0.0f32, f32::max);
            let finite = a.iter().all(|x| x.is_finite());
            eprintln!("{kernel} pos {pos} tokens {tokens}: worst abs diff {worst:.2e}");
            assert!(
                finite && worst < 2e-3,
                "{kernel} pos {pos} tokens {tokens}: {worst}"
            );
        }
    }
}
