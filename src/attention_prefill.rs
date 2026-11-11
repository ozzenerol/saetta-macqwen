//! Causal attention for a prefill chunk (kernels/attention_prefill.metal),
//! on simdgroup matrices. Half q, keys and values; the buffers
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

    /// Simdgroups per threadgroup of both kernels.
    const SIMDS: u64 = 4;

    /// out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h]).
    /// q is half [tokens][heads][256], keys and values the half cache from
    /// position 0 ([capacity][kv_heads][256]), q_and_gate f32
    /// [tokens][heads][512] with the gate in the second half of each head,
    /// out f32 [tokens][heads][256].
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
        encode_kernel(enc, gpu, "attention_gq4_128", q, keys, values, q_and_gate, out, p);
    }

    /// `encode` with a named kernel: attention_gq4_128 (rows = the 8 q heads
    /// of a kv head) or attention_g4_128 (rows = 8 tokens of a head).
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
        enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
        for (i, v) in [q, keys, values, q_and_gate, out].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(5, size_of::<FlashParams>() as u64, (&p as *const FlashParams).cast());
        let tokens = p.tokens as u64;
        let grid = match kernel {
            "attention_gq4_128" => {
                assert_eq!(p.heads, p.kv_heads * 8, "{kernel} needs 8 q heads per kv head");
                MTLSize::new(tokens.div_ceil(SIMDS), p.kv_heads as u64, 1)
            }
            "attention_g4_128" => MTLSize::new(tokens.div_ceil(8), p.heads as u64 / SIMDS, 1),
            _ => panic!("no prefill attention kernel {kernel}"),
        };
        enc.dispatch_thread_groups(grid, MTLSize::new(32 * SIMDS, 1, 1));
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
    // Spread values, so the scores and P look like a real layer's.
    let h = |n: usize| {
        let v: Vec<half::f16> = (0..n)
            .map(|i| half::f16::from_f32(((i * 7919 % 1999) as f32 / 1000.0 - 1.0) * 2.0))
            .collect();
        View::new(gpu.buffer_from(&v))
    };
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
    for kernel in ["attention_rows2_half", "attention_gq4_128", "attention_g4_128"] {
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
