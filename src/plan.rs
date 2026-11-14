//! A list of GPU dispatches recorded once and run for every token.
//!
//! Kernels add their dispatches with `op` (runs after the op before it) or
//! `op_parallel` (does not depend on the op before it). Shapes and other
//! fixed values go in `constant`; what changes per token is in the
//! `TokenParams` buffer, which kernels read like any other buffer.
//!
//! A plan runs two ways with the same result: `run_serial` encodes it into a
//! serial encoder (simple, good for debugging), and `build` encodes it once
//! into an indirect command buffer that `Built::run` replays, which costs
//! about 2.0 us per dependent dispatch instead of 2.8 (examples/dispatch_bench.rs).

#[cfg(target_os = "macos")]
pub use mac::*;

/// Must match TokenParams in kernels/params.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct TokenParams {
    pub token: u32,
    pub position: u32,
    pub kv_len: u32,
    pub pad: u32,
}

impl TokenParams {
    pub fn new(token: u32, position: u32) -> TokenParams {
        TokenParams {
            token,
            position,
            kv_len: position + 1,
            pad: 0,
        }
    }
}

#[cfg(target_os = "macos")]
mod mac {
    use super::TokenParams;
    use crate::gpu::Gpu;
    use crate::qmv::View;
    use metal::foreign_types::ForeignTypeRef;
    use metal::objc::{msg_send, sel, sel_impl};
    use metal::{
        Buffer, ComputePipelineState, IndirectCommandBuffer, IndirectCommandBufferDescriptor,
        MTLDispatchType, MTLIndirectCommandType, MTLResourceOptions, MTLResourceUsage, MTLSize,
        NSRange,
    };
    use std::time::Duration;

    /// Room for the constants of every op of a model.
    const CONSTANT_BYTES: usize = 1 << 20;

    struct Op {
        kernel: String,
        pipeline: ComputePipelineState,
        buffers: Vec<(Buffer, u64)>,
        grid: MTLSize,
        threadgroup: MTLSize,
        barrier: bool,
    }

    pub struct Plan<'g> {
        gpu: &'g Gpu,
        ops: Vec<Op>,
        constants: Buffer,
        used: usize,
        pub params: View,
    }

    impl<'g> Plan<'g> {
        pub fn new(gpu: &'g Gpu) -> Plan<'g> {
            Plan {
                gpu,
                ops: Vec::new(),
                constants: gpu.buffer(CONSTANT_BYTES),
                used: 0,
                params: View::new(gpu.buffer(size_of::<TokenParams>())),
            }
        }

        pub fn gpu(&self) -> &'g Gpu {
            self.gpu
        }

        /// Adds a dispatch of `grid` threadgroups of `threadgroup` threads
        /// that runs after the op before it has finished. `buffers[i]` binds
        /// at index i.
        pub fn op(&mut self, kernel: &str, buffers: &[&View], grid: MTLSize, threadgroup: MTLSize) {
            self.push(kernel, buffers, grid, threadgroup, true);
        }

        /// Like `op`, for a dispatch that does not read what the op before
        /// it writes, so the two may run at the same time.
        pub fn op_parallel(
            &mut self,
            kernel: &str,
            buffers: &[&View],
            grid: MTLSize,
            threadgroup: MTLSize,
        ) {
            self.push(kernel, buffers, grid, threadgroup, false);
        }

        fn push(
            &mut self,
            kernel: &str,
            buffers: &[&View],
            grid: MTLSize,
            threadgroup: MTLSize,
            barrier: bool,
        ) {
            self.ops.push(Op {
                kernel: kernel.to_string(),
                pipeline: self.gpu.pipeline(kernel),
                buffers: buffers
                    .iter()
                    .map(|v| (v.buffer.clone(), v.offset))
                    .collect(),
                grid,
                threadgroup,
                barrier,
            });
        }

        /// Stores `data` in the plan's constant buffer, 16-byte aligned, and
        /// returns a view of it. Use it in place of `set_bytes`.
        pub fn constant<T: Copy>(&mut self, data: &[T]) -> View {
            let len = size_of_val(data);
            assert!(self.used + len <= CONSTANT_BYTES, "plan constants are full");
            unsafe {
                let dst = (self.constants.contents() as *mut u8).add(self.used);
                std::ptr::copy_nonoverlapping(data.as_ptr().cast::<u8>(), dst, len);
            }
            let view = View {
                buffer: self.constants.clone(),
                offset: self.used as u64,
            };
            self.used += len.div_ceil(16) * 16;
            view
        }

        /// Writes the per-token values. Call it before running the plan.
        pub fn set_params(&self, p: TokenParams) {
            unsafe { *(self.params.buffer.contents() as *mut TokenParams) = p };
        }

        pub fn len(&self) -> usize {
            self.ops.len()
        }

        pub fn is_empty(&self) -> bool {
            self.ops.is_empty()
        }

        /// Encodes every op into one serial encoder, runs it and waits.
        pub fn run_serial(&self) -> Duration {
            self.gpu.run(|enc| {
                for op in &self.ops {
                    enc.set_compute_pipeline_state(&op.pipeline);
                    for (i, (buffer, offset)) in op.buffers.iter().enumerate() {
                        enc.set_buffer(i as u64, Some(buffer), *offset);
                    }
                    enc.dispatch_thread_groups(op.grid, op.threadgroup);
                }
            })
        }

        /// Each kernel name in the plan with how many ops use it, in order of
        /// first use.
        pub fn kernels(&self) -> Vec<(String, usize)> {
            let mut out: Vec<(String, usize)> = Vec::new();
            for op in &self.ops {
                match out.iter_mut().find(|(k, _)| *k == op.kernel) {
                    Some((_, n)) => *n += 1,
                    None => out.push((op.kernel.clone(), 1)),
                }
            }
            out
        }

        /// Encodes the plan into an indirect command buffer once.
        pub fn build(&self) -> Built {
            self.build_where(|_| true, true)
        }

        /// `build` with only the ops whose kernel `keep` accepts, and with or
        /// without the barriers. For profiling: the results are not the
        /// plan's results.
        pub fn build_where(&self, keep: impl Fn(&str) -> bool, barriers: bool) -> Built {
            let ops: Vec<&Op> = self.ops.iter().filter(|o| keep(&o.kernel)).collect();
            let desc = IndirectCommandBufferDescriptor::new();
            // MTLIndirectCommandTypeConcurrentDispatch is 1 << 5 in the
            // Metal headers; the metal crate has it as 1 << 4, which crashes.
            desc.set_command_types(MTLIndirectCommandType::from_bits_retain(1 << 5));
            desc.set_inherit_buffers(false);
            desc.set_inherit_pipeline_state(false);
            let binds = ops.iter().map(|o| o.buffers.len()).max().unwrap_or(1);
            desc.set_max_kernel_buffer_bind_count(binds as u64);
            let icb = self.gpu.device.new_indirect_command_buffer_with_descriptor(
                &desc,
                ops.len().max(1) as u64,
                MTLResourceOptions::empty(),
            );

            let mut resources: Vec<Buffer> = Vec::new();
            for (i, op) in ops.iter().enumerate() {
                let cmd = icb.indirect_compute_command_at_index(i as u64);
                cmd.set_compute_pipeline_state(&op.pipeline);
                for (j, (buffer, offset)) in op.buffers.iter().enumerate() {
                    cmd.set_kernel_buffer(j as u64, Some(buffer), *offset);
                    if !resources.iter().any(|r| r.as_ptr() == buffer.as_ptr()) {
                        resources.push(buffer.clone());
                    }
                }
                if op.barrier && barriers {
                    cmd.set_barrier();
                }
                cmd.concurrent_dispatch_threadgroups(op.grid, op.threadgroup);
            }
            Built {
                icb,
                resources,
                count: ops.len(),
            }
        }
    }

    /// A plan encoded into an indirect command buffer. It holds the plan's
    /// buffers, so it stays valid on its own.
    pub struct Built {
        icb: IndirectCommandBuffer,
        resources: Vec<Buffer>,
        count: usize,
    }

    impl Built {
        /// Replays the recorded ops in one command buffer and waits.
        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        pub fn run(&self, gpu: &Gpu) -> Duration {
            self.run_timed(gpu).0
        }

        /// `run`, also returning the GPU's own time for the command buffer
        /// in seconds (GPUEndTime - GPUStartTime).
        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        pub fn run_timed(&self, gpu: &Gpu) -> (Duration, f64) {
            gpu.run_timed(MTLDispatchType::Concurrent, |enc| self.encode(enc))
        }

        /// `run` without the wait (`Gpu::commit`).
        pub fn commit(&self, gpu: &Gpu) -> crate::gpu::Pending {
            gpu.commit_with(MTLDispatchType::Concurrent, |enc| self.encode(enc))
        }

        // The objc 0.2 msg_send! macro checks an old `cargo-clippy` cfg.
        #[allow(unexpected_cfgs)]
        fn encode(&self, enc: &metal::ComputeCommandEncoderRef) {
            let refs: Vec<&metal::ResourceRef> = self
                .resources
                .iter()
                .map(|b| -> &metal::ResourceRef { b })
                .collect();
            enc.use_resources(&refs, MTLResourceUsage::Read | MTLResourceUsage::Write);
            let range = NSRange::new(0, self.count as u64);
            let icb: &metal::IndirectCommandBufferRef = &self.icb;
            // The metal crate binds this call only on the render encoder.
            unsafe {
                let _: () = msg_send![enc, executeCommandsInBuffer: icb withRange: range];
            }
        }
    }
}
