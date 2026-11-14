//! Prefill attention as matrix products over blocks of keys
//! (kernels/attention_gemm.metal): for each kv head, the 8 q heads of every
//! token are rows of one matrix, so Q K^T and P V run on 64 x 64 half tiles
//! as qmm does, with an online softmax between them. The scores of one
//! block of KEYS keys are written out and read back, which costs memory
//! traffic but lets the products use large tiles at head dim 256.
//!
//! Measured with `bench-attention-gemm` (last 2048-token chunk, M3 Max, two
//! runs each): at 32k scores 44 ms (12 TFLOP/s), softmax 19 ms, pv 50 ms
//! (10.8 TFLOP/s), about 113 ms in all; at 118k scores 164, softmax 67, pv
//! 180 ms. pv was 54 ms (9.9 TFLOP/s) with 26 KB of threadgroup memory and
//! gained from sharing its staging tile with the P and V tiles, so
//! threadgroup memory limits it. Two changes that were slower: the softmax
//! inside the scores kernel with a max per 64-key tile (8.5 against 9.1
//! TFLOP/s in all) and pv's update alpha O + P V in the accumulators (pv
//! 6.6 TFLOP/s).
//!
//! The default is the fused variant (batches 27 to 29, last 2048-token
//! chunk at 64k, interleaved runs): scores keep the row max of each 64-key
//! tile and write scale S minus it in half, and pv exponentiates P as it
//! loads it, so there is no softmax pass. 198 to 210 ms against 223 to 240
//! ms (scores 96, pv 101), 11.0 TFLOP/s in all; 32k prefill at 8 bits 6 to
//! 13% faster. The 32k gate and the 128k eval at 8 bits passed (128k median
//! KL 1.33e-5, top-1 100%). pv on 64 x 128 tiles, to exponentiate P half as
//! often, spilled its accumulators and ran 15 times slower.

/// Keys per block; must match PREFILL_AG_KEYS.
pub const KEYS: usize = 1024;
#[cfg(target_os = "macos")]
const DIM: usize = 256;
#[cfg(target_os = "macos")]
const GROUP: usize = 8;

/// Must match AttGemmParams in kernels/attention_gemm.metal.
#[cfg(target_os = "macos")]
#[repr(C)]
#[derive(Clone, Copy)]
struct Params {
    rows: u32,
    rows_pad: u32,
    heads: u32,
    kv_heads: u32,
    pos: u32,
    j0: u32,
    n: u32,
    scale: f32,
}

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::attention_prefill::{FlashParams, encode_rows2, spread};
    use crate::gpu::{Gpu, contents};
    use crate::kv_quant::{KvBits, QuantRows};
    use crate::qmv::View;
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

    /// Query slices a chunk's attention runs in, one after the other, unless
    /// SAETTA_AG_SLICES says otherwise: the scratch is for one slice. Two
    /// take 113 MB less at chunk 2048 in the same time (32k prefill within
    /// noise of one slice at 16 and 8 bits, 3k the same).
    const SLICES: usize = 2;

    /// The softmax runs inside scores and pv (prefill_ag_scores_f,
    /// prefill_ag_pv_f) unless SAETTA_AG_FUSED=0: one pass less per block
    /// and 2 bytes of scratch per score instead of 6.
    fn fused_from_env() -> bool {
        std::env::var("SAETTA_AG_FUSED").map_or(true, |v| v != "0")
    }

    /// SAETTA_AG_PV128=1 (fused only): pv on 64 x 128 output tiles, so P is
    /// loaded and exponentiated twice per block instead of four times.
    fn pv128_from_env() -> bool {
        std::env::var("SAETTA_AG_PV128").is_ok_and(|v| v == "1")
    }

    fn slices() -> usize {
        std::env::var("SAETTA_AG_SLICES")
            .ok()
            .and_then(|v| v.parse().ok())
            .filter(|&n: &usize| n > 0)
            .unwrap_or(SLICES)
    }

    /// The buffers of one block for slices of up to `slice` tokens of a
    /// chunk: scores, P, the output sums and the per-row softmax state.
    pub struct AttGemm {
        slice: usize,
        rows_pad: usize,
        s: Buffer,
        p: Buffer,
        o: Buffer,
        m: Buffer,
        l: Buffer,
        alpha: Buffer,
        /// Fused only: the tile maxes and a second m and l, as pv reads the
        /// old ones and writes the new ones.
        tmax: Buffer,
        m2: Buffer,
        l2: Buffer,
        pub fused: bool,
        pub pv128: bool,
        /// For the bench: encode only the first `stages` passes of each
        /// block (1 scores, 2 softmax, 3 pv) and no final; 0 runs all.
        pub stages: std::cell::Cell<usize>,
    }

    impl AttGemm {
        /// About 7 KB per query row of a slice: 112 KB per token with 2 kv
        /// heads, 230 MB for a chunk of 2048 tokens in one slice, 115 MB in
        /// two.
        pub fn new(gpu: &Gpu, chunk: usize, kv_heads: usize) -> AttGemm {
            let slice = chunk.div_ceil(slices());
            let rows_pad = (slice * GROUP).next_multiple_of(64);
            let rows = kv_heads * rows_pad;
            let fused = fused_from_env();
            let (full, small) = (usize::from(!fused), usize::from(fused));
            AttGemm {
                slice,
                rows_pad,
                s: gpu.buffer(rows * KEYS * (2 + 2 * full)),
                p: gpu.buffer((rows * KEYS * 2 * full).max(4)),
                o: gpu.buffer(rows * DIM * 4),
                m: gpu.buffer(rows * 4),
                l: gpu.buffer(rows * 4),
                alpha: gpu.buffer(rows * 4),
                tmax: gpu.buffer((rows * (KEYS / 64) * 4 * small).max(4)),
                m2: gpu.buffer((rows * 4 * small).max(4)),
                l2: gpu.buffer((rows * 4 * small).max(4)),
                fused,
                pv128: fused && pv128_from_env(),
                stages: std::cell::Cell::new(0),
            }
        }

        /// The same as attention_prefill::encode on the same buffers: q
        /// half [tokens][heads][256], keys and values the half cache from
        /// position 0 with the chunk's rows written, q_and_gate f32, out f32.
        /// With `quant`, the keys and values are read from those 8- or 4-bit
        /// rows instead and `keys` and `values` are not read.
        #[allow(clippy::too_many_arguments)]
        pub fn encode(
            &self,
            enc: &ComputeCommandEncoderRef,
            gpu: &Gpu,
            q: &View,
            keys: &View,
            values: &View,
            quant: Option<&(QuantRows, QuantRows)>,
            q_and_gate: &View,
            out: &View,
            f: FlashParams,
        ) {
            // A slice of tokens is a chunk of its own at pos + t0: its rows
            // of q, q_and_gate and out, the whole cache.
            let at = |v: &View, bytes: usize| View {
                buffer: v.buffer.clone(),
                offset: v.offset + bytes as u64,
            };
            let width = f.heads as usize * DIM;
            for t0 in (0..f.tokens as usize).step_by(self.slice) {
                let tokens = self.slice.min(f.tokens as usize - t0);
                let fs = FlashParams {
                    pos: f.pos + t0 as u32,
                    tokens: tokens as u32,
                    ..f
                };
                self.encode_slice(
                    enc,
                    gpu,
                    &at(q, t0 * width * 2),
                    keys,
                    values,
                    quant,
                    &at(q_and_gate, t0 * width * 2 * 4),
                    &at(out, t0 * width * 4),
                    fs,
                );
            }
        }

        #[allow(clippy::too_many_arguments)]
        fn encode_slice(
            &self,
            enc: &ComputeCommandEncoderRef,
            gpu: &Gpu,
            q: &View,
            keys: &View,
            values: &View,
            quant: Option<&(QuantRows, QuantRows)>,
            q_and_gate: &View,
            out: &View,
            f: FlashParams,
        ) {
            assert_eq!(
                f.heads,
                f.kv_heads * GROUP as u32,
                "needs 8 q heads per kv head"
            );
            let rows = f.tokens as usize * GROUP;
            assert!(rows <= self.rows_pad);
            let kv = f.kv_heads as u64;
            let row_tiles = rows.div_ceil(64) as u64;
            let end = (f.pos + f.tokens) as usize;
            let set = |views: &[&View]| {
                for (i, v) in views.iter().enumerate() {
                    enc.set_buffer(i as u64, Some(&v.buffer), v.offset);
                }
            };
            let bytes = |index: u64, p: &Params| {
                enc.set_bytes(
                    index,
                    size_of::<Params>() as u64,
                    (p as *const Params).cast(),
                );
            };
            let whole = |b: &Buffer| View {
                buffer: b.clone(),
                offset: 0,
            };
            let (s, p, o, m, l, alpha) = (
                whole(&self.s),
                whole(&self.p),
                whole(&self.o),
                whole(&self.m),
                whole(&self.l),
                whole(&self.alpha),
            );
            let suffix = match quant.map(|(k, _)| k.bits) {
                None => "",
                Some(KvBits::Eight) => "_q8",
                Some(KvBits::Four) => "_q4",
                Some(b) => panic!("no gemm attention for {b:?} rows"),
            };
            let f_ = if self.fused { "_f" } else { "" };
            let w = if self.pv128 { "128" } else { "" };
            let (scores, pv) = (
                gpu.pipeline(&format!("prefill_ag_scores{f_}{suffix}")),
                gpu.pipeline(&format!("prefill_ag_pv{f_}{w}{suffix}")),
            );
            let dim_tiles = (DIM / if self.pv128 { 128 } else { 64 }) as u64;
            let (tmax, m2, l2) = (whole(&self.tmax), whole(&self.m2), whole(&self.l2));
            // The codes, scales and biases of keys and values; the half
            // kernels do not read these slots, so they get the half cache.
            let parts = |r: &QuantRows| [whole(&r.data), whole(&r.scales), whole(&r.biases)];
            let (kq, vq) = match quant {
                Some((k, v)) => (parts(k), parts(v)),
                None => (
                    [keys.clone(), keys.clone(), keys.clone()],
                    [values.clone(), values.clone(), values.clone()],
                ),
            };
            let set_at = |first: u64, views: &[View; 3]| {
                for (i, v) in views.iter().enumerate() {
                    enc.set_buffer(first + i as u64, Some(&v.buffer), v.offset);
                }
            };
            let mut l_last = &l;
            for (b, j0) in (0..end).step_by(KEYS).enumerate() {
                let n = KEYS.min(end - j0);
                let params = Params {
                    rows: rows as u32,
                    rows_pad: self.rows_pad as u32,
                    heads: f.heads,
                    kv_heads: f.kv_heads,
                    pos: f.pos,
                    j0: j0 as u32,
                    n: n as u32,
                    scale: f.scale,
                };
                enc.set_compute_pipeline_state(&scores);
                set(&[q, keys, &s]);
                bytes(3, &params);
                set_at(4, &kq);
                if self.fused {
                    set_at(7, &[tmax.clone(), tmax.clone(), tmax.clone()]);
                }
                enc.dispatch_thread_groups(
                    MTLSize::new(n.div_ceil(64) as u64, row_tiles, kv),
                    MTLSize::new(128, 1, 1),
                );

                let stages = self.stages.get();
                if self.fused {
                    if stages == 1 || stages == 2 {
                        continue;
                    }
                    // Block b reads the m and l block b - 1 wrote.
                    let (m_in, m_out, l_in, l_out) = if b % 2 == 0 {
                        (&m2, &m, &l2, &l)
                    } else {
                        (&m, &m2, &l, &l2)
                    };
                    enc.set_compute_pipeline_state(&pv);
                    set(&[&s, values, &o, &tmax]);
                    bytes(4, &params);
                    set_at(5, &vq);
                    set_at(8, &[m_in.clone(), m_out.clone(), l_in.clone()]);
                    enc.set_buffer(11, Some(&l_out.buffer), 0);
                    enc.dispatch_thread_groups(
                        MTLSize::new(dim_tiles, row_tiles, kv),
                        MTLSize::new(128, 1, 1),
                    );
                    l_last = l_out;
                    continue;
                }
                if stages == 1 {
                    continue;
                }
                enc.set_compute_pipeline_state(&gpu.pipeline("prefill_ag_softmax"));
                set(&[&s, &p, &m, &l, &alpha]);
                bytes(5, &params);
                enc.dispatch_thread_groups(
                    MTLSize::new(rows.div_ceil(8) as u64, kv, 1),
                    MTLSize::new(256, 1, 1),
                );

                if stages == 2 {
                    continue;
                }
                enc.set_compute_pipeline_state(&pv);
                set(&[&p, values, &o, &alpha]);
                bytes(4, &params);
                set_at(5, &vq);
                enc.dispatch_thread_groups(
                    MTLSize::new((DIM / 64) as u64, row_tiles, kv),
                    MTLSize::new(128, 1, 1),
                );
            }
            let params = Params {
                rows: rows as u32,
                rows_pad: self.rows_pad as u32,
                heads: f.heads,
                kv_heads: f.kv_heads,
                pos: f.pos,
                j0: 0,
                n: 0,
                scale: f.scale,
            };
            if self.stages.get() != 0 {
                return;
            }
            enc.set_compute_pipeline_state(&gpu.pipeline("prefill_ag_final"));
            set(&[&o, l_last, q_and_gate, out]);
            bytes(4, &params);
            enc.dispatch_threads(
                MTLSize::new(DIM as u64, rows as u64, kv),
                MTLSize::new(256, 1, 1),
            );
        }
    }

    /// For each context, the chunk of `chunk` tokens at its end: the
    /// largest relative difference to rows2 over the output, and the best
    /// of 3 times with TFLOP/s for rows2 and this, plus the estimated
    /// attention time of the whole prefill for `layers` layers.
    pub fn bench(
        contexts: &[usize],
        chunk: usize,
        layers: usize,
    ) -> Result<(), Box<dyn std::error::Error>> {
        let gpu = Gpu::new()?;
        let (heads, kv) = (16usize, 2usize);
        let gemm = AttGemm::new(&gpu, chunk, kv);
        for &n in contexts {
            let n = n.next_multiple_of(chunk);
            let capacity = n.next_multiple_of(64) + 64;
            let h = |len: usize, seed| View::new(gpu.buffer_from(&spread(len, seed)));
            let (q, k, v) = (
                h(chunk * heads * DIM, 1),
                h(capacity * kv * DIM, 2),
                h(capacity * kv * DIM, 3),
            );
            let gate: Vec<f32> = (0..chunk * heads * 2 * DIM)
                .map(|i| (i % 7) as f32 - 3.0)
                .collect();
            let g = View::new(gpu.buffer_from(&gate));
            let len = chunk * heads * DIM;
            let (want, got) = (
                View::new(gpu.buffer(len * 4)),
                View::new(gpu.buffer(len * 4)),
            );
            let chunks = n / chunk;
            let flops = |pos: usize| {
                4.0 * (heads * DIM) as f64 * (chunk * pos + chunk * (chunk + 1) / 2) as f64
            };
            let time = |gemm_kernel: bool, pos: usize| {
                let p = FlashParams {
                    heads: heads as u32,
                    kv_heads: kv as u32,
                    pos: pos as u32,
                    tokens: chunk as u32,
                    scale: 1.0 / (DIM as f32).sqrt(),
                };
                let run = || {
                    gpu.run_timed(metal::MTLDispatchType::Serial, |enc| {
                        if gemm_kernel {
                            gemm.encode(enc, &gpu, &q, &k, &v, None, &g, &got, p);
                        } else {
                            encode_rows2(enc, &gpu, &q, &k, &v, &g, &want, p);
                        }
                    })
                    .1
                };
                run();
                (0..3).map(|_| run()).fold(f64::INFINITY, f64::min)
            };
            let last = n - chunk;
            let mid = (chunks / 2) * chunk;
            for (name, gemm_kernel) in [("rows2", false), ("gemm", true)] {
                let (tl, tm) = (time(gemm_kernel, last), time(gemm_kernel, mid));
                println!(
                    "{name:6} context {n:6}, chunk {chunk}: last chunk {:7.1} ms {:4.1} TFLOP/s, mid {:4.1} TFLOP/s; prefill attention ~{:6.1} s for {layers} layers",
                    tl * 1e3,
                    flops(last) / tl / 1e12,
                    flops(mid) / tm / 1e12,
                    tm * chunks as f64 * layers as f64,
                );
            }
            // The gemm passes of the last chunk one by one, as differences
            // of runs that stop after scores, softmax and pv.
            let mut prefix = [0.0f64; 4];
            for (k, t) in prefix.iter_mut().enumerate().skip(1) {
                gemm.stages.set(k);
                *t = time(true, last);
            }
            gemm.stages.set(0);
            let all = time(true, last);
            let half_flops = flops(last) / 2.0;
            println!(
                "gemm stages at context {n}, last chunk: scores {:6.1} ms ({:4.1} TFLOP/s), softmax {:6.1} ms, pv {:6.1} ms ({:4.1} TFLOP/s), final {:5.1} ms",
                prefix[1] * 1e3,
                half_flops / prefix[1] / 1e12,
                (prefix[2] - prefix[1]) * 1e3,
                (prefix[3] - prefix[2]) * 1e3,
                half_flops / (prefix[3] - prefix[2]) / 1e12,
                (all - prefix[3]) * 1e3,
            );
            // Both outputs are now of the last chunk (time ran it last).
            time(false, last);
            time(true, last);
            let (w, o) = (
                contents::<f32>(&want.buffer, len),
                contents::<f32>(&got.buffer, len),
            );
            let (mut worst, mut sum, mut norm) = (0.0f32, 0.0f64, 0.0f64);
            for (a, b) in w.iter().zip(o) {
                worst = worst.max((a - b).abs());
                sum += f64::from((a - b) * (a - b));
                norm += f64::from(a * a);
            }
            println!(
                "gemm vs rows2 at context {n}: rel rms {:.2e}, max abs {worst:.2e}",
                (sum / norm).sqrt()
            );
        }
        Ok(())
    }
}
