#![cfg(target_os = "macos")]
//! qmm2 against prefill's qmm (split 1) on real weights, plain and in gather
//! mode. The inputs are exact in half, so the only difference is W rounded
//! to half in qmm2. Needs the model on the Mac:
//!   scripts/mac.sh test --release --test qmm -- --ignored

use metal::MTLSize;
use saetta::config::Config;
use saetta::gpu::{Gpu, contents};
use saetta::qmm::{QmmShape, VARIANTS, encode, encode_gather};
use saetta::qmv::{GpuQuant, View};
use saetta::weights::Weights;
use std::path::PathBuf;

fn open() -> (Config, Weights) {
    let dir = PathBuf::from(std::env::var("HOME").unwrap())
        .join("llm/saetta-models/Qwen3.6-35B-A3B-5bit");
    (Config::load(&dir).unwrap(), Weights::open(&dir).unwrap())
}

/// Values in about [-2, 2], exact in half.
fn input(n: usize) -> (Vec<f32>, Vec<half::f16>) {
    let mut s = 12345u64;
    let f: Vec<f32> = (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            let v = ((s >> 40) as f32 / (1u64 << 24) as f32) * 4.0 - 2.0;
            half::f16::from_f32(v).to_f32()
        })
        .collect();
    let h = f.iter().map(|&v| half::f16::from_f32(v)).collect();
    (f, h)
}

#[allow(clippy::too_many_arguments)]
fn old_qmm(
    gpu: &Gpu,
    w: &GpuQuant,
    x: &View,
    y: &View,
    shape: QmmShape,
    extra: [&View; 3],
    grid_y: usize,
) {
    let name = if w.bits == 5 { "qmm_5bit" } else { "qmm_8bit" };
    gpu.run(|enc| {
        enc.set_compute_pipeline_state(&gpu.pipeline(name));
        for (i, v) in [
            &w.words, &w.scales, &w.biases, x, y, extra[0], extra[1], extra[2],
        ]
        .iter()
        .enumerate()
        {
            let at = if i < 5 { i } else { i + 1 };
            enc.set_buffer(at as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(
            5,
            size_of::<QmmShape>() as u64,
            (&shape as *const QmmShape).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new(w.rows.div_ceil(64) as u64, grid_y as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    });
}

fn rel_rms(a: &[f32], b: &[f32]) -> f64 {
    let e: f64 = a
        .iter()
        .zip(b)
        .map(|(&x, &y)| ((x - y) as f64).powi(2))
        .sum();
    let n: f64 = b.iter().map(|&y| (y as f64).powi(2)).sum();
    (e / n).sqrt()
}

#[test]
#[ignore = "needs the model on the Mac"]
fn plain_matches_qmm() {
    let (cfg, weights) = open();
    let gpu = Gpu::new().unwrap();
    // 5-bit, and the 8-bit router.
    for module in [
        "language_model.model.layers.0.linear_attn.in_proj_z",
        "language_model.model.layers.0.mlp.gate",
    ] {
        let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization).unwrap());
        for m in [1, 37, 64, 100, 200] {
            let (xf, xh) = input(m * w.cols);
            let (xf, xh) = (
                View::new(gpu.buffer_from(&xf)),
                View::new(gpu.buffer_from(&xh)),
            );
            let want = View::new(gpu.buffer(m * w.rows * 4));
            let shape = QmmShape {
                m: m as u32,
                n: w.rows as u32,
                k: w.cols as u32,
                gather: 0,
            };
            old_qmm(&gpu, &w, &xf, &want, shape, [&xf, &xf, &xf], m.div_ceil(64));
            let want = contents::<f32>(&want.buffer, m * w.rows).to_vec();
            for v in VARIANTS {
                let y = View::new(gpu.buffer(m * w.rows * 4));
                gpu.run(|enc| encode(enc, &gpu, v, &w, &xh, &y, m));
                let got = contents::<f32>(&y.buffer, m * w.rows);
                let e = rel_rms(got, &want);
                eprintln!(
                    "{} {}-bit {}x{} m {m}: rel rms {e:.2e}",
                    v.name, w.bits, w.rows, w.cols
                );
                assert!(e < 1e-3, "{} m {m}: {e}", v.name);
            }
        }
    }
}

#[test]
#[ignore = "needs the model on the Mac"]
fn gather_matches_qmm() {
    let (cfg, weights) = open();
    let gpu = Gpu::new().unwrap();
    let module = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
    let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization).unwrap());
    let m = 300;
    let (xf, xh) = input(m * w.cols);
    let (xf, xh) = (
        View::new(gpu.buffer_from(&xf)),
        View::new(gpu.buffer_from(&xh)),
    );
    for v in VARIANTS {
        // Tiles of 9 to v.rows rows over a few experts; y rows are packed.
        let mut tiles = Vec::new();
        let mut first = 0u32;
        for (i, e) in [3u32, 17, 100, 255, 0].iter().enumerate() {
            let n = [9, 13, v.rows, v.rows - 1, 40.min(v.rows)][i] as u32;
            tiles.push([*e, first, n, 0]);
            first += n;
        }
        let pairs = first as usize;
        let rows: Vec<u32> = (0..pairs).map(|i| ((i * 7) % m) as u32).collect();
        let (tb, rb) = (
            View::new(gpu.buffer_from(&tiles)),
            View::new(gpu.buffer_from(&rows)),
        );
        let count = View::new(gpu.buffer_from(&[tiles.len() as u32]));
        for gather_x in [true, false] {
            let want = View::new(gpu.buffer(pairs * w.rows * 4));
            let shape = QmmShape {
                m: 0,
                n: w.rows as u32,
                k: w.cols as u32,
                gather: if gather_x { 1 } else { 2 },
            };
            // The old kernel's tiles hold up to 64 rows, so every tile here fits.
            old_qmm(&gpu, &w, &xf, &want, shape, [&tb, &rb, &count], tiles.len());
            let y = View::new(gpu.buffer(pairs * w.rows * 4));
            gpu.run(|enc| {
                encode_gather(
                    enc,
                    &gpu,
                    v,
                    &w,
                    &xh,
                    &y,
                    &tb,
                    &rb,
                    &count,
                    tiles.len(),
                    gather_x,
                )
            });
            let (got, want) = (
                contents::<f32>(&y.buffer, pairs * w.rows),
                contents::<f32>(&want.buffer, pairs * w.rows),
            );
            let e = rel_rms(got, want);
            eprintln!("{} gather {gather_x}: rel rms {e:.2e}", v.name);
            assert!(e < 1e-3, "{} gather {gather_x}: {e}", v.name);
        }
    }
}
