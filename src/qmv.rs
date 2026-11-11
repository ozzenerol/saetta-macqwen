//! Quantized matrix times vector on the GPU (`kernels/qmv.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::Gpu;
    use crate::plan::Plan;
    use crate::weights::{QuantTensor, Tensor};
    use metal::{Buffer, MTLSize};
    use std::marker::PhantomData;

    /// Must match QMV_ROWS and QMV_SIMDS in the kernel.
    const ROWS_PER_SIMD: usize = 4;
    const SIMDS: usize = 2;

    /// A GPU buffer and the byte offset where the data starts in it.
    #[derive(Clone)]
    pub struct View {
        pub buffer: Buffer,
        pub offset: u64,
    }

    /// A quantized matrix, or a stack of expert matrices `[experts, rows,
    /// cols]`, ready for `encode` or the kernels in `moe`. When a tensor sits at an
    /// aligned address in its mmap, the buffer wraps it without a copy, so
    /// the matrix borrows the weights.
    pub struct GpuQuant<'a> {
        pub words: View,
        pub scales: View,
        pub biases: View,
        /// 1 for a plain matrix.
        pub experts: usize,
        pub rows: usize,
        pub cols: usize,
        pub bits: u32,
        /// How many of the three parts were copied because they were not
        /// aligned.
        pub copies: usize,
        weights: PhantomData<&'a ()>,
    }

    impl<'a> GpuQuant<'a> {
        pub fn new(gpu: &Gpu, q: &QuantTensor<'a>) -> GpuQuant<'a> {
            assert_eq!(q.params.group_size, 64);
            let rank = q.weight.shape.len();
            let mut copies = 0;
            let mut view = |t: &Tensor<'a>| {
                let (view, copied) = upload(gpu, t.bytes);
                copies += copied as usize;
                view
            };
            GpuQuant {
                words: view(&q.weight),
                scales: view(&q.scales),
                biases: view(&q.biases),
                experts: q.weight.shape[..rank - 2].iter().product(),
                rows: q.out_features(),
                cols: q.in_features(),
                bits: q.params.bits,
                copies,
                weights: PhantomData,
            }
        }

        /// Copies experts `ids` of a stacked tensor into new buffers, in that
        /// order. Benchmarks use it: binding every layer's full expert stacks
        /// asks Metal to keep about 20 GB resident.
        pub fn subset(gpu: &Gpu, q: &QuantTensor, ids: &[usize]) -> GpuQuant<'static> {
            assert_eq!(q.params.group_size, 64);
            let experts = q.weight.shape[0];
            let pick = |t: &Tensor| {
                let per = t.bytes.len() / experts;
                let bytes: Vec<u8> = ids
                    .iter()
                    .flat_map(|&e| &t.bytes[e * per..(e + 1) * per])
                    .copied()
                    .collect();
                View::new(gpu.buffer_from(&bytes))
            };
            GpuQuant {
                words: pick(&q.weight),
                scales: pick(&q.scales),
                biases: pick(&q.biases),
                experts: ids.len(),
                rows: q.out_features(),
                cols: q.in_features(),
                bits: q.params.bits,
                copies: 3,
                weights: PhantomData,
            }
        }

        /// The bytes the kernel reads from one matrix (one expert).
        pub fn bytes(&self) -> usize {
            self.rows * self.cols * self.bits as usize / 8 + self.rows * (self.cols / 64) * 4
        }

        /// Records y = W x. `x` holds `cols` f32 values, `y` gets `rows`.
        /// `parallel` marks it independent of the op recorded before it.
        pub fn record(&self, plan: &mut Plan, x: &View, y: &View, parallel: bool) {
            self.record_variant(plan, Variant::Plain, x, y, parallel);
        }

        /// `record` with a chosen kernel; see Variant.
        pub fn record_variant(
            &self,
            plan: &mut Plan,
            variant: Variant,
            x: &View,
            y: &View,
            parallel: bool,
        ) {
            assert_eq!(self.experts, 1, "stacked experts go through moe");
            let (kernel, simds) = match (variant, self.bits) {
                (Variant::Plain, 5) => ("qmv_5bit".to_string(), SIMDS),
                (Variant::Plain, 8) => ("qmv_8bit".to_string(), SIMDS),
                (Variant::Staged(n), b @ (5 | 8)) => (format!("qmv_tg{n}_{b}bit"), n),
                (Variant::StagedNorm(..), 5) => ("qmv_tg8_norm_5bit".to_string(), 8),
                (_, b) => panic!("no qmv kernel for {b} bits and {variant:?}"),
            };
            if !matches!(variant, Variant::Plain) {
                assert!(self.cols <= 4096, "qmv_tg stages at most 4096 columns");
            }
            let shape = plan.constant(&[self.rows as u32, self.cols as u32]);
            let mut buffers = vec![&self.words, &self.scales, &self.biases, x, y, &shape];
            let eps;
            if let Variant::StagedNorm(weight, e) = variant {
                eps = plan.constant(&[e]);
                buffers.extend([weight, &eps]);
            }
            let kernel = kernel.as_str();
            let grid = MTLSize::new(self.rows.div_ceil(ROWS_PER_SIMD * simds) as u64, 1, 1);
            let threads = MTLSize::new(32 * simds as u64, 1, 1);
            if parallel {
                plan.op_parallel(kernel, &buffers, grid, threads);
            } else {
                plan.op(kernel, &buffers, grid, threads);
            }
        }
    }

    /// Which qmv kernel `record_variant` uses.
    #[derive(Clone, Copy)]
    pub enum Variant<'a> {
        /// qmv: every simdgroup reads x itself.
        Plain,
        /// qmv_tgN: x staged once per threadgroup of N simdgroups (4 or 8).
        Staged(usize),
        /// qmv_tg8_norm (5-bit): x RMS-normed with this weight and eps while staged.
        StagedNorm(&'a View, f32),
    }

    impl std::fmt::Debug for Variant<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            match self {
                Variant::Plain => write!(f, "Plain"),
                Variant::Staged(n) => write!(f, "Staged({n})"),
                Variant::StagedNorm(_, eps) => write!(f, "StagedNorm(eps {eps})"),
            }
        }
    }

    /// Wraps `bytes` without a copy when they start 16-byte aligned, else
    /// copies them into a new buffer. Returns the view and whether it copied.
    fn upload(gpu: &Gpu, bytes: &[u8]) -> (View, bool) {
        if (bytes.as_ptr() as usize).is_multiple_of(16) {
            let (buffer, offset) = unsafe { gpu.buffer_no_copy(bytes) };
            (View { buffer, offset }, false)
        } else {
            let buffer = gpu.buffer_from(bytes);
            (View { buffer, offset: 0 }, true)
        }
    }

    /// Times qmv on real matrices: per matrix, `runs` command buffers of
    /// `reps` calls each, after one warm-up run. Prints best and median GB/s.
    pub fn bench(
        dir: &std::path::Path,
        runs: usize,
        reps: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let config = crate::config::Config::load(dir)?;
        let weights = crate::weights::Weights::open(dir)?;
        let gpu = Gpu::new()?;
        let l0 = "language_model.model.layers.0.";
        let l3 = "language_model.model.layers.3.";
        let modules = [
            "language_model.lm_head".to_string(),
            format!("{l0}linear_attn.in_proj_qkv"),
            format!("{l0}linear_attn.in_proj_z"),
            format!("{l0}linear_attn.out_proj"),
            format!("{l3}self_attn.q_proj"),
            format!("{l3}self_attn.o_proj"),
            format!("{l0}mlp.shared_expert.gate_proj"),
            format!("{l0}mlp.shared_expert.down_proj"),
            format!("{l0}mlp.gate"),
        ];
        let nw = View::new(gpu.buffer_from(&vec![1.0f32; 4096]));
        for module in &modules {
            let q = weights.quantized(module, &config.quantization)?;
            let m = GpuQuant::new(&gpu, &q);
            let x = View::new(gpu.buffer_from(&vec![0.5f32; m.cols]));
            let y = View::new(gpu.buffer(m.rows * 4));
            for (label, variant) in [
                ("qmv", Variant::Plain),
                ("qmv_tg4", Variant::Staged(4)),
                ("qmv_tg8", Variant::Staged(8)),
                ("qmv_tg8_norm", Variant::StagedNorm(&nw, 1e-6)),
            ] {
                if m.cols > 4096 && !matches!(variant, Variant::Plain)
                    || matches!(variant, Variant::StagedNorm(..)) && m.bits != 5
                {
                    continue;
                }
                let mut plan = Plan::new(&gpu);
                for _ in 0..reps {
                    m.record_variant(&mut plan, variant, &x, &y, false);
                }
                let built = plan.build();
                let mut times: Vec<f64> = (0..=runs)
                    .map(|_| built.run(&gpu).as_secs_f64())
                    .skip(1)
                    .collect();
                times.sort_by(f64::total_cmp);
                let gb = (m.bytes() * reps) as f64 / 1e9;
                println!(
                    "{label:13} {module}: {}x{} {}-bit, {:.1} MB, {runs} runs of {reps}: best {:.0} GB/s, median {:.0} GB/s, {:.1} us per call, {} parts copied",
                    m.rows,
                    m.cols,
                    m.bits,
                    m.bytes() as f64 / 1e6,
                    gb / times[0],
                    gb / times[runs / 2],
                    times[runs / 2] / reps as f64 * 1e6,
                    m.copies,
                );
            }
        }
        Ok(())
    }

    impl View {
        pub fn new(buffer: Buffer) -> View {
            View { buffer, offset: 0 }
        }
    }
}
