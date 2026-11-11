//! Quantized matrix times matrix for prefill with half x (kernels/qmm.metal),
//! a drop-in for prefill's qmm on the fast path: same buffers and gather
//! modes, tiles of `tile_rows()` rows.

/// A tile shape of qmm2 (kernel name prefix, rows of x, columns of y per
/// threadgroup); must match the QMM2 instances in kernels/qmm.metal.
#[derive(Clone, Copy, Debug)]
pub struct Variant {
    pub name: &'static str,
    pub rows: usize,
    pub cols: usize,
}

pub const VARIANTS: [Variant; 5] = [
    Variant {
        name: "qmm2",
        rows: 64,
        cols: 64,
    },
    Variant {
        name: "qmm2_db",
        rows: 64,
        cols: 64,
    },
    Variant {
        name: "qmm2_32",
        rows: 32,
        cols: 128,
    },
    Variant {
        name: "qmm2_32_db",
        rows: 32,
        cols: 128,
    },
    Variant {
        name: "qmm2_16",
        rows: 16,
        cols: 128,
    },
];

/// The fastest in bench-qmm2: 32 x 128 tiles, single buffered.
pub const DEFAULT: Variant = VARIANTS[2];
/// For expert tiles of 16 rows or fewer: 7.5 TFLOP/s there against 5.0 for
/// DEFAULT, but slower on full 32-row tiles.
pub const SMALL: Variant = VARIANTS[4];

/// Tiles of this many rows or fewer are skipped (qmm_few takes them); must
/// match QMM_FEW.
pub const FEW: usize = 8;

/// Rows of x per threadgroup tile of DEFAULT. moe_group cuts the expert
/// tiles to this size.
pub fn tile_rows() -> usize {
    DEFAULT.rows
}

/// Must match QmmShape in kernels/qmm.metal (and MmShape in prefill.metal).
#[repr(C)]
#[derive(Clone, Copy)]
pub struct QmmShape {
    pub m: u32,
    pub n: u32,
    pub k: u32,
    /// 0: plain; 1: tiles, x rows from `rows`; 2: tiles, x row = y row.
    pub gather: u32,
}

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::gpu::Gpu;
    use crate::qmv::{GpuQuant, View};
    use metal::{ComputeCommandEncoderRef, MTLSize};

    fn bind(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        v: Variant,
        w: &GpuQuant,
        x: &View,
        y: &View,
        shape: &QmmShape,
    ) {
        assert!(
            w.bits == 5 || w.bits == 8,
            "no qmm2 kernel for {} bits",
            w.bits
        );
        enc.set_compute_pipeline_state(&gpu.pipeline(&format!("{}_{}bit", v.name, w.bits)));
        for (i, v) in [&w.words, &w.scales, &w.biases, x, y].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(
            5,
            size_of::<QmmShape>() as u64,
            (shape as *const QmmShape).cast(),
        );
    }

    /// y[m, w.rows] = x[m, w.cols] W^T, x half, y f32.
    pub fn encode(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        v: Variant,
        w: &GpuQuant,
        x: &View,
        y: &View,
        m: usize,
    ) {
        let shape = QmmShape {
            m: m as u32,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: 0,
        };
        bind(enc, gpu, v, w, x, y, &shape);
        // Unused in plain mode, but bound so the kernel sees valid buffers.
        for i in 6..9 {
            enc.set_buffer(i, Some(&x.buffer), 0);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(w.rows.div_ceil(v.cols) as u64, m.div_ceil(v.rows) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// The expert tiles of moe_group: `tiles` holds (expert, first, rows, 0)
    /// for up to `max_tiles` tiles, `tile_count[0]` of them used. Tiles of
    /// FEW rows or fewer are left to qmm_few.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_gather(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        v: Variant,
        w: &GpuQuant,
        x: &View,
        y: &View,
        tiles: &View,
        rows: &View,
        tile_count: &View,
        max_tiles: usize,
        gather_x: bool,
    ) {
        let shape = QmmShape {
            m: 0,
            n: w.rows as u32,
            k: w.cols as u32,
            gather: if gather_x { 1 } else { 2 },
        };
        bind(enc, gpu, v, w, x, y, &shape);
        for (i, b) in [tiles, rows, tile_count].iter().enumerate() {
            enc.set_buffer(6 + i as u64, Some(&b.buffer), b.offset);
        }
        enc.dispatch_thread_groups(
            MTLSize::new(w.rows.div_ceil(v.cols) as u64, max_tiles as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// Times the qmm2 variants (in VARIANTS order) against prefill's qmm (f32
    /// x, split 1) on real matrices with `m` rows, and the experts' gate_proj
    /// with tiles cut to each variant.
    pub fn bench(
        dir: &std::path::Path,
        ms: &[usize],
        runs: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        use crate::config::Config;
        use crate::weights::Weights;
        let cfg = Config::load(dir)?;
        let weights = Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let median = |f: &dyn Fn(&ComputeCommandEncoderRef)| -> f64 {
            let mut t: Vec<f64> = (0..=runs)
                .map(|_| {
                    gpu.run(|enc| {
                        for _ in 0..5 {
                            f(enc);
                        }
                    })
                    .as_secs_f64()
                        / 5.0
                })
                .skip(1)
                .collect();
            t.sort_by(f64::total_cmp);
            t[runs / 2]
        };
        let modules = [
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "language_model.model.layers.0.linear_attn.in_proj_z",
            "language_model.model.layers.0.linear_attn.out_proj",
            "language_model.model.layers.3.self_attn.q_proj",
            "language_model.model.layers.0.mlp.shared_expert.gate_proj",
            "language_model.model.layers.0.mlp.shared_expert.down_proj",
        ];
        for &m in ms {
            for module in modules {
                let Ok(q) = weights.quantized(module, &cfg.quantization) else {
                    println!("{module}: not found");
                    continue;
                };
                let w = GpuQuant::new(&gpu, &q);
                let xh = View::new(gpu.buffer_from(&vec![half::f16::from_f32(0.01); m * w.cols]));
                let xf = View::new(gpu.buffer_from(&vec![0.01f32; m * w.cols]));
                let y = View::new(gpu.buffer(m * w.rows * 4));
                let news: Vec<f64> = VARIANTS
                    .iter()
                    .map(|&v| median(&|enc| encode(enc, &gpu, v, &w, &xh, &y, m)))
                    .collect();
                let old = median(&|enc| {
                    let name = if w.bits == 5 { "qmm_5bit" } else { "qmm_8bit" };
                    enc.set_compute_pipeline_state(&gpu.pipeline(name));
                    let shape = QmmShape {
                        m: m as u32,
                        n: w.rows as u32,
                        k: w.cols as u32,
                        gather: 0,
                    };
                    for (i, v) in [&w.words, &w.scales, &w.biases, &xf, &y].iter().enumerate() {
                        enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
                    }
                    enc.set_bytes(
                        5,
                        size_of::<QmmShape>() as u64,
                        (&shape as *const QmmShape).cast(),
                    );
                    for i in 6..9 {
                        enc.set_buffer(i, Some(&xf.buffer), 0);
                    }
                    enc.dispatch_thread_groups(
                        MTLSize::new(w.rows.div_ceil(64) as u64, m.div_ceil(64) as u64, 1),
                        MTLSize::new(128, 1, 1),
                    );
                });
                let flops = 2.0 * (m * w.rows * w.cols) as f64;
                let short = module.rsplit("layers.").next().unwrap();
                let tf: Vec<String> = news
                    .iter()
                    .map(|t| format!("{:.1}", flops / t / 1e12))
                    .collect();
                println!(
                    "m {m:4} {short:38} {:5}x{:4} {}-bit TFLOP/s: {} | qmm {:.1}",
                    w.rows,
                    w.cols,
                    w.bits,
                    tf.join(" "),
                    flops / old / 1e12
                );
            }

            // Experts: m tokens times 8 experts each, spread evenly over all
            // experts, as tiles of up to 64 rows.
            let module = "language_model.model.layers.0.mlp.switch_mlp.gate_proj";
            let w = GpuQuant::new(&gpu, &weights.quantized(module, &cfg.quantization)?);
            let pairs = m * cfg.num_experts_per_tok;
            let per = pairs / w.experts;
            let rows: Vec<u32> = (0..pairs).map(|i| (i % m) as u32).collect();
            let xh = View::new(gpu.buffer_from(&vec![half::f16::from_f32(0.01); m * w.cols]));
            let y = View::new(gpu.buffer(pairs * w.rows * 4));
            let rb = View::new(gpu.buffer_from(&rows));
            let flops = 2.0 * (pairs * w.rows * w.cols) as f64;
            let mut tf = Vec::new();
            for v in VARIANTS {
                let mut tiles = Vec::new();
                for e in 0..w.experts {
                    let mut r = 0;
                    while r < per {
                        let n = (per - r).min(v.rows);
                        tiles.push([e as u32, (e * per + r) as u32, n as u32, 0]);
                        r += n;
                    }
                }
                let tb = View::new(gpu.buffer_from(&tiles));
                let count = View::new(gpu.buffer_from(&[tiles.len() as u32]));
                let t = median(&|enc| {
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
                        true,
                    )
                });
                tf.push(format!("{:.1}", flops / t / 1e12));
            }
            println!(
                "m {m:4} experts gate_proj, {per} rows per expert, {}x{} TFLOP/s: {}",
                w.rows,
                w.cols,
                tf.join(" ")
            );
        }
        Ok(())
    }
}
