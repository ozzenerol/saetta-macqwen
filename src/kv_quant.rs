//! 8-bit and 4-bit KV cache (`kernels/kv_quant.metal`). Keys and values go
//! in groups of 64 with a half scale and bias per group, x = s * q + z, as
//! MLX's quantized KV cache does. Per token and kv head of 256 that is 272
//! bytes at 8 bits and 144 at 4, against 512 in half.

use half::f16;

/// Values per group.
pub const GROUP: usize = 64;

/// How the KV cache stores keys and values.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum KvBits {
    Half,
    Eight,
    Four,
}

impl KvBits {
    /// 16, 8 or 4, as `--kv-bits` takes it.
    pub fn parse(s: &str) -> Option<KvBits> {
        match s {
            "16" => Some(KvBits::Half),
            "8" => Some(KvBits::Eight),
            "4" => Some(KvBits::Four),
            _ => None,
        }
    }

    pub fn bits(self) -> u32 {
        match self {
            KvBits::Half => 16,
            KvBits::Eight => 8,
            KvBits::Four => 4,
        }
    }

    /// u32 words of codes per group, for the quantized kinds.
    pub fn words_per_group(self) -> usize {
        2 * self.bits() as usize
    }

    /// Bytes per value of a token, with the scales and biases.
    pub fn bytes_per_value(self) -> f64 {
        match self {
            KvBits::Half => 2.0,
            q => q.bits() as f64 / 8.0 + 4.0 / GROUP as f64,
        }
    }
}

/// The CPU version of the kernel: codes, scales and biases of `x`, whose
/// length is a multiple of GROUP.
pub fn quantize(x: &[f32], bits: KvBits) -> (Vec<u32>, Vec<f16>, Vec<f16>) {
    assert!(bits != KvBits::Half && x.len().is_multiple_of(GROUP));
    let b = bits.bits();
    let levels = ((1u32 << b) - 1) as f32;
    let per_word = 32 / b as usize;
    let mut data = Vec::new();
    let (mut scales, mut biases) = (Vec::new(), Vec::new());
    for group in x.as_chunks::<GROUP>().0 {
        let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let s = f16::from_f32((hi - lo) / levels);
        let z = f16::from_f32(lo);
        let inv = if s.to_f32() > 0.0 {
            1.0 / s.to_f32()
        } else {
            0.0
        };
        let q = |v: f32| {
            ((v - z.to_f32()) * inv)
                .round_ties_even()
                .clamp(0.0, levels) as u32
        };
        for word in group.chunks_exact(per_word) {
            data.push(
                word.iter()
                    .enumerate()
                    .fold(0, |w, (i, &v)| w | q(v) << (i as u32 * b)),
            );
        }
        scales.push(s);
        biases.push(z);
    }
    (data, scales, biases)
}

/// The values the codes stand for.
pub fn dequantize(data: &[u32], scales: &[f16], biases: &[f16], bits: KvBits) -> Vec<f32> {
    let b = bits.bits();
    let mask = (1u32 << b) - 1;
    let per_word = 32 / b as usize;
    let words = bits.words_per_group();
    let mut out = Vec::with_capacity(scales.len() * GROUP);
    for (g, (s, z)) in scales.iter().zip(biases).enumerate() {
        for &w in &data[g * words..(g + 1) * words] {
            for i in 0..per_word {
                let q = (w >> (i as u32 * b)) & mask;
                out.push(s.to_f32() * q as f32 + z.to_f32());
            }
        }
    }
    out
}

/// How much looser than for a half cache a test of `bits` bounds the
/// difference to the CPU: the CPU and GPU round slightly different keys,
/// and a key that lands on the next code moves by one step, about 4e-3 of
/// its group's range at 8 bits and 7e-2 at 4.
pub fn test_slack(bits: KvBits) -> f32 {
    match bits {
        KvBits::Half => 1.0,
        KvBits::Eight => 4.0,
        KvBits::Four => 16.0,
    }
}

/// x after a round trip through the cache and the half copy that prefill
/// attention reads a quantized cache through.
pub fn round_trip_as_half(x: &[f32], bits: KvBits) -> Vec<f32> {
    round_trip(x, bits)
        .iter()
        .map(|&v| f16::from_f32(v).to_f32())
        .collect()
}

/// x after a round trip through the cache, as the CPU reference stores it.
pub fn round_trip(x: &[f32], bits: KvBits) -> Vec<f32> {
    if bits == KvBits::Half {
        return x.iter().map(|&v| f16::from_f32(v).to_f32()).collect();
    }
    let (data, scales, biases) = quantize(x, bits);
    dequantize(&data, &scales, &biases, bits)
}

#[cfg(target_os = "macos")]
pub use mac::*;

#[cfg(target_os = "macos")]
mod mac {
    use super::*;
    use crate::gpu::Gpu;
    use crate::plan::{Plan, TokenParams};
    use crate::qmv::View;
    use metal::{Buffer, ComputeCommandEncoderRef, MTLSize};

    /// Must match KvQuantParams in the kernel.
    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Params {
        groups: u32,
        per_token: u32,
    }

    /// The codes, scales and biases of one cache (keys or values), for
    /// `capacity` tokens of `per_token` groups.
    pub struct QuantRows {
        pub data: Buffer,
        pub scales: Buffer,
        pub biases: Buffer,
        pub bits: KvBits,
        pub per_token: usize,
    }

    impl QuantRows {
        /// Room for `capacity` tokens of `values` values each. The pages
        /// take memory only once written.
        pub fn new(gpu: &Gpu, bits: KvBits, values: usize, capacity: usize) -> QuantRows {
            assert!(bits != KvBits::Half && values.is_multiple_of(GROUP));
            let per_token = values / GROUP;
            let groups = per_token * capacity;
            QuantRows {
                data: gpu.buffer(groups * bits.words_per_group() * 4),
                scales: gpu.buffer(groups * 2),
                biases: gpu.buffer(groups * 2),
                bits,
                per_token,
            }
        }

        fn kernel(&self) -> &'static str {
            match self.bits {
                KvBits::Eight => "kv_quantize_8",
                KvBits::Four => "kv_quantize_4",
                KvBits::Half => unreachable!(),
            }
        }

        /// Records the quantizing of one token's values `x` into its slot,
        /// at the plan's position.
        pub fn record(&self, plan: &mut Plan, x: &View) {
            let params = plan.params.clone();
            self.record_at(plan, x, &params, false);
        }

        /// As `record`, at the position in `params` (a TokenParams view),
        /// and with `parallel` alongside the op before it.
        pub fn record_at(&self, plan: &mut Plan, x: &View, params: &View, parallel: bool) {
            let p = plan.constant(&[Params {
                groups: self.per_token as u32,
                per_token: self.per_token as u32,
            }]);
            let views = [
                x,
                &View::new(self.data.clone()),
                &View::new(self.scales.clone()),
                &View::new(self.biases.clone()),
                &p,
                params,
            ];
            let (grid, group) = (
                MTLSize::new(self.per_token as u64, 1, 1),
                MTLSize::new(32, 1, 1),
            );
            if parallel {
                plan.op_parallel(self.kernel(), &views, grid, group);
            } else {
                plan.op(self.kernel(), &views, grid, group);
            }
        }

        /// Encodes writing the values of tokens 0..`tokens` as half into
        /// `out`.
        pub fn encode_dequantize(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            out: &View,
            tokens: usize,
        ) {
            let groups = self.per_token * tokens;
            let kernel = match self.bits {
                KvBits::Eight => "kv_dequantize_8",
                KvBits::Four => "kv_dequantize_4",
                KvBits::Half => unreachable!(),
            };
            enc.set_compute_pipeline_state(&gpu.pipeline(kernel));
            enc.set_buffer(0, Some(&self.data), 0);
            enc.set_buffer(1, Some(&self.scales), 0);
            enc.set_buffer(2, Some(&self.biases), 0);
            enc.set_buffer(3, Some(&out.buffer), out.offset);
            let p = Params {
                groups: groups as u32,
                per_token: self.per_token as u32,
            };
            enc.set_bytes(4, size_of::<Params>() as u64, (&p as *const Params).cast());
            enc.dispatch_threads(
                MTLSize::new((groups * GROUP) as u64, 1, 1),
                MTLSize::new(256, 1, 1),
            );
        }

        /// Encodes the quantizing of `tokens` tokens of values in `x` into
        /// the slots from `pos`.
        pub fn encode(
            &self,
            gpu: &Gpu,
            enc: &ComputeCommandEncoderRef,
            x: &View,
            pos: usize,
            tokens: usize,
        ) {
            let groups = self.per_token * tokens;
            enc.set_compute_pipeline_state(&gpu.pipeline(self.kernel()));
            enc.set_buffer(0, Some(&x.buffer), x.offset);
            enc.set_buffer(1, Some(&self.data), 0);
            enc.set_buffer(2, Some(&self.scales), 0);
            enc.set_buffer(3, Some(&self.biases), 0);
            let p = Params {
                groups: groups as u32,
                per_token: self.per_token as u32,
            };
            let tp = TokenParams::new(0, pos as u32);
            enc.set_bytes(4, size_of::<Params>() as u64, (&p as *const Params).cast());
            enc.set_bytes(
                5,
                size_of::<TokenParams>() as u64,
                (&tp as *const TokenParams).cast(),
            );
            enc.dispatch_thread_groups(MTLSize::new(groups as u64, 1, 1), MTLSize::new(32, 1, 1));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn noise(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 23) as f32) - 1.0
            })
            .collect()
    }

    #[test]
    fn round_trip_is_within_half_a_step() {
        let mut x = noise(4 * GROUP, 1);
        // A constant group: scale 0, every value exact.
        x[GROUP..2 * GROUP].fill(0.37);
        for bits in [KvBits::Eight, KvBits::Four] {
            let (_, scales, _) = quantize(&x, bits);
            let y = round_trip(&x, bits);
            for (i, (&a, &b)) in x.iter().zip(&y).enumerate() {
                let step = scales[i / GROUP].to_f32();
                // Half a step, plus the bias rounded to half.
                let slack = step / 2.0 + a.abs() * 1e-3 + 1e-6;
                assert!((a - b).abs() <= slack, "{bits:?} value {i}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn codes_pack_lowest_bits_first() {
        let mut x = vec![0.0; GROUP];
        x[GROUP - 1] = 1.0;
        x[1] = 1.0;
        let (data, ..) = quantize(&x, KvBits::Eight);
        assert_eq!(data.len(), 16);
        assert_eq!(data[0], 255 << 8);
        assert_eq!(data[15], 255 << 24);
        let (data, ..) = quantize(&x, KvBits::Four);
        assert_eq!(data.len(), 8);
        assert_eq!(data[0], 15 << 4);
        assert_eq!(data[7], 15 << 28);
    }
}
