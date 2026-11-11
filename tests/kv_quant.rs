#![cfg(target_os = "macos")]
//! The kv_quantize kernel against the CPU version in src/kv_quant.rs.

use half::f16;
use saetta::gpu::{Gpu, contents};
use saetta::kv_quant::{GROUP, KvBits, QuantRows, dequantize, quantize};
use saetta::plan::{Plan, TokenParams};
use saetta::qmv::View;

const VALUES: usize = 2 * 256;

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

/// Rows of keys-like data: some large groups, some constant ones.
fn rows(tokens: usize) -> Vec<f32> {
    let mut x: Vec<f32> = noise(tokens * VALUES, 7)
        .iter()
        .enumerate()
        .map(|(i, v)| {
            v * if (i / GROUP).is_multiple_of(3) {
                20.0
            } else {
                1.0
            }
        })
        .collect();
    x[GROUP..2 * GROUP].fill(-1.5);
    x
}

/// The GPU codes may differ from the CPU ones by one step where the fast
/// math reciprocal rounds the other way; scales and biases are exact.
fn compare(rows: &QuantRows, x: &[f32], from_token: usize, bits: KvBits) {
    let groups = x.len() / GROUP;
    let first = from_token * rows.per_token;
    let words = bits.words_per_group();
    let data = &contents::<u32>(&rows.data, (first + groups) * words)[first * words..];
    let scales = &contents::<f16>(&rows.scales, first + groups)[first..];
    let biases = &contents::<f16>(&rows.biases, first + groups)[first..];
    let (want_data, want_scales, want_biases) = quantize(x, bits);
    assert_eq!(scales, &want_scales[..], "{bits:?} scales");
    assert_eq!(biases, &want_biases[..], "{bits:?} biases");
    let got = dequantize(data, scales, biases, bits);
    let want = dequantize(&want_data, &want_scales, &want_biases, bits);
    let mut off = 0;
    for (i, (a, b)) in got.iter().zip(&want).enumerate() {
        if a != b {
            off += 1;
            let step = scales[i / GROUP].to_f32();
            assert!(
                (a - b).abs() <= step * 1.001,
                "{bits:?} value {i}: {a} vs {b}"
            );
        }
    }
    println!("{bits:?}: {off} of {} codes one step off", got.len());
    assert!(off * 1000 <= got.len(), "{bits:?}: {off} codes off");
}

#[test]
fn prefill_rows_match_cpu() {
    let gpu = Gpu::new().unwrap();
    for bits in [KvBits::Eight, KvBits::Four] {
        let tokens = 37;
        let cache = QuantRows::new(&gpu, bits, VALUES, 64);
        let x = rows(tokens);
        let xb = View::new(gpu.buffer_from(&x));
        gpu.run(|enc| cache.encode(&gpu, enc, &xb, 5, tokens));
        compare(&cache, &x, 5, bits);
    }
}

#[test]
fn decode_row_matches_cpu() {
    let gpu = Gpu::new().unwrap();
    for bits in [KvBits::Eight, KvBits::Four] {
        let cache = QuantRows::new(&gpu, bits, VALUES, 64);
        let x = rows(1);
        let xb = View::new(gpu.buffer_from(&x));
        let mut plan = Plan::new(&gpu);
        cache.record(&mut plan, &xb);
        let built = plan.build();
        plan.set_params(TokenParams::new(0, 41));
        built.run(&gpu);
        compare(&cache, &x, 41, bits);
    }
}
