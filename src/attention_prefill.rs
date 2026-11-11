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
        encode_kernel(enc, gpu, "attention_flash", q, keys, values, q_and_gate, out, p);
    }

    /// `encode` with a named variant: attention_flash or attention_flash_pre.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_kernel(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        kernel: &str,
        q: &View,
        keys: &View,
        values: &View,
        q_and_gate: &View,
        out: &View,
        p: FlashParams,
    ) {
        assert_eq!(p.heads, p.kv_heads * 8, "attention_flash needs 8 q heads per kv head");
        // attention_flash_reg: 4 tokens of 32 threads each per threadgroup.
        let (tokens, threads) = if kernel.starts_with("attention_flash_reg") { (4, 128) } else { (TOKENS, 256) };
        // attention_gS_KB: 8 tokens of S heads per threadgroup.
        let g_heads = kernel.strip_prefix("attention_g").and_then(|r| r.split('_').next()).and_then(|n| n.parse::<u32>().ok());
        enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
        for (i, v) in [q, keys, values, q_and_gate, out].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(5, size_of::<FlashParams>() as u64, (&p as *const FlashParams).cast());
        // attention_gqS_KB: S tokens of one kv head per threadgroup.
        let gq = kernel.strip_prefix("attention_gq").and_then(|r| r.split('_').next()).and_then(|n| n.parse::<u32>().ok());
        if let Some(s) = gq {
            enc.dispatch_thread_groups(
                MTLSize::new((p.tokens as usize).div_ceil(s as usize) as u64, p.kv_heads as u64, 1),
                MTLSize::new(32 * s as u64, 1, 1),
            );
            return;
        }
        if let Some(s) = g_heads {
            enc.dispatch_thread_groups(
                MTLSize::new((p.tokens as usize).div_ceil(8) as u64, (p.heads / s) as u64, 1),
                MTLSize::new(32 * s as u64, 1, 1),
            );
            return;
        }
        enc.dispatch_thread_groups(
            MTLSize::new((p.tokens as usize).div_ceil(tokens) as u64, p.kv_heads as u64, 1),
            MTLSize::new(threads, 1, 1),
        );
    }
}

/// Times each kernel over the chunks of a `tokens`-token prefill (chunks
/// of `chunk`), for one attention layer, on random data; prints the total
/// per kernel and the TFLOP/s of the causal work.
#[cfg(target_os = "macos")]
pub fn bench(tokens: usize, chunk: usize) -> Result<(), Box<dyn std::error::Error>> {
    use crate::gpu::Gpu;
    use crate::qmv::View;
    use metal::MTLSize;
    let gpu = Gpu::new()?;
    let (heads, kv, dim) = (16usize, 2usize, 256usize);
    let capacity = tokens.next_multiple_of(64) + 64;
    let h = |n: usize| View::new(gpu.buffer_from(&vec![half::f16::from_f32(0.01); n]));
    let (q, k, v) = (h(chunk * heads * dim), h(capacity * kv * dim), h(capacity * kv * dim));
    let g = View::new(gpu.buffer_from(&vec![0.5f32; chunk * heads * 2 * dim]));
    let out = View::new(gpu.buffer(chunk * heads * dim * 4));
    let mut flops = 0.0;
    let mut pos = 0;
    while pos < tokens {
        let t = chunk.min(tokens - pos);
        // Causal: token i sees pos + i + 1 keys.
        flops += 4.0 * (heads * dim) as f64 * (t * pos + t * (t + 1) / 2) as f64;
        pos += t;
    }
    for kernel in ["attention_rows2_half", "attention_g4_128", "attention_gq4_128", "attention_g2_128", "attention_g2_256", "attention_gq2_128", "attention_gq2_256", "attention_gq1_256"] {
        let mut total = 0.0;
        let mut pos = 0;
        while pos < tokens {
            let t = chunk.min(tokens - pos);
            let p = FlashParams {
                heads: heads as u32,
                kv_heads: kv as u32,
                pos: pos as u32,
                tokens: t as u32,
                scale: 1.0 / (dim as f32).sqrt(),
            };
            let (_, secs) = gpu.run_timed(metal::MTLDispatchType::Serial, |enc| {
                if kernel == "attention_rows2_half" {
                    enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
                    for (i, b) in [&q, &k, &v, &g, &out].iter().enumerate() {
                        enc.set_buffer(i as u64, Some(&b.buffer), b.offset);
                    }
                    enc.set_bytes(5, size_of::<FlashParams>() as u64, (&p as *const FlashParams).cast());
                    enc.dispatch_thread_groups(
                        MTLSize::new(t.div_ceil(8) as u64, (heads / 4) as u64, 1),
                        MTLSize::new(128, 1, 1),
                    );
                } else {
                    encode_kernel(enc, &gpu, kernel, &q, &k, &v, &g, &out, p);
                }
            });
            total += secs;
            pos += t;
        }
        println!(
            "{kernel:22} {tokens} tokens, chunk {chunk}: {:.1} ms per layer, {:.1} TFLOP/s",
            total * 1e3,
            flops / total / 1e12
        );
    }
    Ok(())
}
