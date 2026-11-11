//! Quantized matrix times vector on the GPU (`kernels/qmv.metal`).

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use crate::gpu::Gpu;
    use crate::weights::{QuantTensor, Tensor};
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};
    use std::marker::PhantomData;

    /// Must match QMV_ROWS and QMV_SIMDS in the kernel.
    const ROWS_PER_SIMD: usize = 4;
    const SIMDS: usize = 2;

    /// A GPU buffer and the byte offset where the data starts in it.
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

        /// The bytes the kernel reads from one matrix (one expert).
        pub fn bytes(&self) -> usize {
            self.rows * self.cols * self.bits as usize / 8 + self.rows * (self.cols / 64) * 4
        }

        /// Encodes y = W x. `x` holds `cols` f32 values, `y` gets `rows`.
        pub fn encode(&self, gpu: &Gpu, enc: &ComputeCommandEncoderRef, x: &View, y: &View) {
            assert_eq!(self.experts, 1, "stacked experts go through moe");
            let name = match self.bits {
                5 => "qmv_5bit",
                8 => "qmv_8bit",
                b => panic!("no qmv kernel for {b} bits"),
            };
            enc.set_compute_pipeline_state(&gpu.pipeline(name));
            self.set_buffers(enc, 0);
            set_views(enc, 3, &[x, y]);
            let shape = [self.rows as u32, self.cols as u32];
            enc.set_bytes(5, 8, shape.as_ptr().cast());
            let rows_per_group = ROWS_PER_SIMD * SIMDS;
            let groups = self.rows.div_ceil(rows_per_group);
            enc.dispatch_thread_groups(
                MTLSize::new(groups as u64, 1, 1),
                MTLSize::new(32 * SIMDS as u64, 1, 1),
            );
        }
    }

    impl GpuQuant<'_> {
        /// Binds words, scales and biases at indices `first` to `first + 2`.
        pub fn set_buffers(&self, enc: &ComputeCommandEncoderRef, first: u64) {
            set_views(enc, first, &[&self.words, &self.scales, &self.biases]);
        }
    }

    /// Binds `views` at consecutive buffer indices from `first`.
    pub fn set_views(enc: &ComputeCommandEncoderRef, first: u64, views: &[&View]) {
        for (i, v) in views.iter().enumerate() {
            enc.set_buffer(first + i as u64, Some(&v.buffer), v.offset);
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
        for module in [
            "language_model.lm_head",
            "language_model.model.layers.0.linear_attn.in_proj_qkv",
            "language_model.model.layers.0.mlp.gate",
        ] {
            let q = weights.quantized(module, &config.quantization)?;
            let m = GpuQuant::new(&gpu, &q);
            let x = View::new(gpu.buffer_from(&vec![0.5f32; m.cols]));
            let y = View::new(gpu.buffer(m.rows * 4));
            let mut times: Vec<f64> = (0..=runs)
                .map(|_| {
                    gpu.run(|enc| {
                        for _ in 0..reps {
                            m.encode(&gpu, enc, &x, &y);
                        }
                    })
                    .as_secs_f64()
                })
                .skip(1)
                .collect();
            times.sort_by(f64::total_cmp);
            let gb = (m.bytes() * reps) as f64 / 1e9;
            println!(
                "{module}: {}x{} {}-bit, {:.1} MB, {runs} runs of {reps}: best {:.0} GB/s, median {:.0} GB/s, {:.1} us per call, {} parts copied",
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
        Ok(())
    }

    impl View {
        pub fn new(buffer: Buffer) -> View {
            View { buffer, offset: 0 }
        }
    }
}
