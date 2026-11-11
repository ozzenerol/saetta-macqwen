//! Flash attention for a prefill chunk (kernels/attention_prefill.metal):
//! half q and KV cache, f32 out, the buffers of prefill's attention_rows_half.

/// Must match FlashParams in the kernel (and prefill's AttentionParams).
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

/// The kernels: attention_faN takes N tokens of one kv head per threadgroup.
/// attention_fx_T_R_D takes T tokens per threadgroup, R per simdgroup, and
/// splits the head dims over D simdgroups; a _p suffix prefetches the next
/// key block into registers.
pub const KERNELS: [&str; 12] = [
    "attention_fa8",
    "attention_fa4",
    "attention_fx_4_1_2",
    "attention_fx_8_1_2",
    "attention_fx_8_2_2",
    "attention_fx_4_2_2",
    "attention_fx_8_2_1",
    "attention_fx_8_1_2_p",
    "attention_fx_8_2_2_p",
    "attention_fx_8_1_1_p",
    "attention_fx_4_4_4",
    "attention_fx_4_4_4_p",
];
/// The one prefill uses.
pub const DEFAULT: &str = "attention_fa8";
/// The cache capacity must be a multiple of this (FA_BK).
pub const KEY_BLOCK: usize = 16;

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::gpu::Gpu;
    use crate::qmv::View;
    use metal::{ComputeCommandEncoderRef, MTLSize};

    /// out[t, h] = softmax(q[t, h] K^T * scale) V * sigmoid(gate[t, h]) with
    /// `kernel` from KERNELS. q is half [tokens][heads][256], keys and values
    /// the half cache from position 0 ([capacity][kv_heads][256]),
    /// q_and_gate f32 [tokens][heads][512] with the gate in the second half
    /// of each head, out f32 [tokens][heads][256].
    #[allow(clippy::too_many_arguments)]
    pub fn encode(
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
        assert_eq!(
            p.heads,
            p.kv_heads * 8,
            "{kernel} needs 8 q heads per kv head"
        );
        // (tokens per threadgroup, simdgroups)
        let (tokens, simds): (u64, u64) = if let Some(n) = kernel.strip_prefix("attention_fa") {
            let n = n
                .parse()
                .unwrap_or_else(|_| panic!("no prefill attention kernel {kernel}"));
            (n, n)
        } else if let Some(rest) = kernel.strip_prefix("attention_fx_") {
            let v: Vec<u64> = rest
                .split('_')
                .take(3)
                .filter_map(|x| x.parse().ok())
                .collect();
            assert_eq!(v.len(), 3, "no prefill attention kernel {kernel}");
            (v[0], v[0] / v[1] * v[2])
        } else {
            panic!("no prefill attention kernel {kernel}")
        };
        enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
        for (i, v) in [q, keys, values, q_and_gate, out].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(
            5,
            size_of::<FlashParams>() as u64,
            (&p as *const FlashParams).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new((p.tokens as u64).div_ceil(tokens), p.kv_heads as u64, 1),
            MTLSize::new(32 * simds, 1, 1),
        );
    }

    /// prefill's attention_rows2_half on the same buffers, as the reference
    /// and the baseline.
    #[allow(clippy::too_many_arguments)]
    pub fn encode_rows2(
        enc: &ComputeCommandEncoderRef,
        gpu: &Gpu,
        q: &View,
        keys: &View,
        values: &View,
        q_and_gate: &View,
        out: &View,
        p: FlashParams,
    ) {
        enc.set_compute_pipeline_state(&gpu.pipeline("attention_rows2_half"));
        for (i, v) in [q, keys, values, q_and_gate, out].iter().enumerate() {
            enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
        }
        enc.set_bytes(
            5,
            size_of::<FlashParams>() as u64,
            (&p as *const FlashParams).cast(),
        );
        enc.dispatch_thread_groups(
            MTLSize::new((p.tokens as u64).div_ceil(8), (p.heads / 4) as u64, 1),
            MTLSize::new(128, 1, 1),
        );
    }

    /// Spread half values in about -2 .. 2.
    pub fn spread(n: usize, seed: usize) -> Vec<half::f16> {
        (0..n)
            .map(|i| ((i * 7919 + seed * 104729) % 1999) as f32 / 500.0 - 2.0)
            .map(half::f16::from_f32)
            .collect()
    }

    /// For each context in `contexts`, times one attention layer's chunk of
    /// `chunk` tokens at the middle and at the end of a prefill of that many
    /// tokens, per kernel (rows2 is prefill's current one). Prints the last
    /// chunk's TFLOP/s and the estimated attention time of the whole prefill
    /// for `layers` layers (chunks x the middle chunk).
    pub fn bench(
        contexts: &[usize],
        chunk: usize,
        layers: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = Gpu::new()?;
        let (heads, kv, dim) = (16usize, 2usize, 256usize);
        for &n in contexts {
            let n = n.next_multiple_of(chunk);
            let capacity = n.next_multiple_of(64) + 64;
            let h = |len: usize, seed| View::new(gpu.buffer_from(&spread(len, seed)));
            let (q, k, v) = (
                h(chunk * heads * dim, 1),
                h(capacity * kv * dim, 2),
                h(capacity * kv * dim, 3),
            );
            let gate: Vec<f32> = (0..chunk * heads * 2 * dim)
                .map(|i| (i % 7) as f32 - 3.0)
                .collect();
            let g = View::new(gpu.buffer_from(&gate));
            let out = View::new(gpu.buffer(chunk * heads * dim * 4));
            let chunks = n / chunk;
            let mid = (chunks / 2) * chunk;
            let last = n - chunk;
            // Causal FLOPs of a chunk at pos: token i sees pos + i + 1 keys.
            let flops = |pos: usize| {
                4.0 * (heads * dim) as f64 * (chunk * pos + chunk * (chunk + 1) / 2) as f64
            };
            for kernel in ["attention_rows2_half"].into_iter().chain(KERNELS) {
                let time = |pos: usize| {
                    let p = FlashParams {
                        heads: heads as u32,
                        kv_heads: kv as u32,
                        pos: pos as u32,
                        tokens: chunk as u32,
                        scale: 1.0 / (dim as f32).sqrt(),
                    };
                    let run = || {
                        gpu.run_timed(metal::MTLDispatchType::Serial, |enc| {
                            if kernel == "attention_rows2_half" {
                                encode_rows2(enc, &gpu, &q, &k, &v, &g, &out, p);
                            } else {
                                encode(enc, &gpu, kernel, &q, &k, &v, &g, &out, p);
                            }
                        })
                        .1
                    };
                    run();
                    (0..2).map(|_| run()).fold(f64::INFINITY, f64::min)
                };
                let (tm, tl) = (time(mid), time(last));
                println!(
                    "{kernel:22} context {n:6}, chunk {chunk}: last chunk {:7.1} ms {:4.1} TFLOP/s, mid {:4.1} TFLOP/s; prefill attention ~{:6.1} s for {layers} layers",
                    tl * 1e3,
                    flops(last) / tl / 1e12,
                    flops(mid) / tm / 1e12,
                    tm * chunks as f64 * layers as f64,
                );
            }
        }
        Ok(())
    }
}
