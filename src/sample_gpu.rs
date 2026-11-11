//! Sampling on the GPU (kernels/sample.metal), recorded into a plan after
//! the lm_head so the next token never goes through the CPU. The draws match
//! sample::Sampler for the same seed: the same candidates in the same order
//! and the same splitmix64 stream. top_k has to be 1..=64 here; with top_k
//! off the caller keeps the CPU sampler.
//!
//! The ring (SampleRing in the kernel) is shared with speculative decoding:
//! slot `p & 63` holds the token fed at position `p`, its prob, a draft to
//! verify and whether the draft was kept.

use crate::sample::Params;

pub const RING: usize = 64;
pub const MAX_K: usize = 64;
pub const NO_DRAFT: u32 = u32::MAX;

/// Must match SampleParams in kernels/sample.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default)]
pub struct SampleParams {
    pub n: u32,
    pub top_k: u32,
    pub top_p: f32,
    pub temperature: f32,
    pub advance: u32,
}

/// Must match SampleRing in kernels/sample.metal.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SampleRing {
    pub ids: [u32; RING],
    pub probs: [f32; RING],
    pub draft: [u32; RING],
    pub accepted: [u32; RING],
}

/// Whether the GPU sampler can take these params.
pub fn supports(p: &Params) -> bool {
    p.temperature <= 0.0 || (1..=MAX_K).contains(&p.top_k)
}

pub fn slot(position: usize) -> usize {
    position & (RING - 1)
}

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::gpu::{Gpu, contents};
    use crate::plan::Plan;
    use crate::qmv::View;
    use metal::{ComputeCommandEncoderRef, MTLSize};

    const GROUPS: usize = 256;

    pub struct GpuSampler {
        params: View,
        rng: View,
        ring: View,
        cand_v: View,
        cand_i: View,
        part: View,
    }

    impl GpuSampler {
        pub fn new(gpu: &Gpu) -> GpuSampler {
            let s = GpuSampler {
                params: View::new(gpu.buffer(size_of::<SampleParams>())),
                rng: View::new(gpu.buffer(8)),
                ring: View::new(gpu.buffer(size_of::<SampleRing>())),
                cand_v: View::new(gpu.buffer(GROUPS * MAX_K * 4)),
                cand_i: View::new(gpu.buffer(GROUPS * MAX_K * 4)),
                part: View::new(gpu.buffer(GROUPS * 2 * 4)),
            };
            s.clear_drafts();
            s
        }

        pub fn ring(&self) -> &View {
            &self.ring
        }

        /// Records the two dispatches that sample from `logits[0..n]`.
        pub fn record(&self, plan: &mut Plan, logits: &View) {
            let tp = plan.params.clone();
            plan.op(
                "sample_partial",
                &[logits, &self.cand_v, &self.cand_i, &self.part, &self.params],
                MTLSize::new(GROUPS as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
            plan.op(
                "sample_final",
                &[
                    &self.cand_v,
                    &self.cand_i,
                    &self.part,
                    &self.params,
                    &self.rng,
                    &self.ring,
                    &tp,
                ],
                MTLSize::new(1, 1, 1),
                MTLSize::new(1024, 1, 1),
            );
        }

        /// `record` into an open encoder, for rows outside a plan: `tp` is
        /// the TokenParams whose position + 1 picks the ring slot.
        pub fn encode(&self, gpu: &Gpu, enc: &ComputeCommandEncoderRef, logits: &View, tp: &View) {
            let ops: [(&str, &[&View], u64, u64); 2] = [
                (
                    "sample_partial",
                    &[logits, &self.cand_v, &self.cand_i, &self.part, &self.params],
                    GROUPS as u64,
                    256,
                ),
                (
                    "sample_final",
                    &[
                        &self.cand_v,
                        &self.cand_i,
                        &self.part,
                        &self.params,
                        &self.rng,
                        &self.ring,
                        tp,
                    ],
                    1,
                    1024,
                ),
            ];
            for (kernel, views, groups, threads) in ops {
                enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
                for (i, v) in views.iter().enumerate() {
                    enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
                }
                enc.dispatch_thread_groups(MTLSize::new(groups, 1, 1), MTLSize::new(threads, 1, 1));
            }
        }

        /// Sets the params for the next runs. `advance` makes the sampled
        /// token the plan's next input. Greedy runs as top_k 1.
        pub fn set(&self, n: usize, p: &Params, advance: bool) {
            assert!(supports(p), "GPU sampling needs top_k 1..={MAX_K}");
            let greedy = p.temperature <= 0.0;
            let sp = SampleParams {
                n: n as u32,
                top_k: if greedy { 1 } else { p.top_k as u32 },
                top_p: if greedy { 1.0 } else { p.top_p },
                temperature: if greedy { 0.0 } else { p.temperature },
                advance: advance as u32,
            };
            unsafe { *(self.params.buffer.contents() as *mut SampleParams) = sp };
        }

        pub fn seed(&self, seed: u64) {
            unsafe { *(self.rng.buffer.contents() as *mut u64) = seed };
        }

        /// The splitmix64 state after the last draw.
        pub fn rng(&self) -> u64 {
            contents::<u64>(&self.rng.buffer, 1)[0]
        }

        pub fn read(&self) -> SampleRing {
            contents::<SampleRing>(&self.ring.buffer, 1)[0]
        }

        fn ring_ptr(&self) -> *mut SampleRing {
            self.ring.buffer.contents() as *mut SampleRing
        }

        /// The token fed at `position`.
        pub fn token(&self, position: usize) -> u32 {
            self.read().ids[slot(position)]
        }

        pub fn set_draft(&self, position: usize, draft: u32) {
            unsafe { (*self.ring_ptr()).draft[slot(position)] = draft };
        }

        pub fn clear_drafts(&self) {
            unsafe { (*self.ring_ptr()).draft = [NO_DRAFT; RING] };
        }
    }
}
