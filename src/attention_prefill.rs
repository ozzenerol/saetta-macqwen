//! Causal attention for a prefill chunk (kernels/attention_prefill.metal),
//! flash style on simdgroup matrices. Half q, keys and values; the buffers
//! and parameters of prefill's attention_rows.

/// Must match FlashParams in the kernel (and PrefillAttentionParams).
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct FlashParams {
    pub heads: u32,
    pub kv_heads: u32,
    /// Position of the chunk's first token; token i sees keys 0 ..= pos + i.
    pub pos: u32,
    pub tokens: u32,
    pub scale: f32,
}

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::gpu::Gpu;
    use crate::qmv::View;
    use metal::{ComputeCommandEncoderRef, MTLSize};

    /// Tokens per threadgroup; must match AF_TOK.
    const TOKENS: usize = 8;

    /// out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h]).
    /// q is half [tokens][heads][256], keys and values the half cache from
    /// position 0 ([capacity][kv_heads][256], with room for 31 keys past
    /// pos + tokens), q_and_gate f32 [tokens][heads][512] with the gate in
    /// the second half of each head, out f32 [tokens][heads][256].
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        q: &View,
        keys: &View,
        values: &View,
        q_and_gate: &View,
        out: &View,
        p: FlashParams,
    ) {
        assert_eq!(p.heads, p.kv_heads * 8, "attention_flash needs 8 q heads per kv head");
        enc.set_compute_pipeline_state(&gpu.pipeline("attention_flash"));
        for (i, v) in [q, keys, values, q_and_gate, out].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(5, size_of::<FlashParams>() as u64, (&p as *const FlashParams).cast());
        enc.dispatch_thread_groups(
            MTLSize::new((p.tokens as usize).div_ceil(TOKENS) as u64, p.kv_heads as u64, 1),
            MTLSize::new(256, 1, 1),
        );
    }
}
