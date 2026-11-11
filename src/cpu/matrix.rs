//! A weight matrix that is either dense f32 or MLX quantized. The quantized
//! form is multiplied straight from the packed words, so a layer never needs
//! a dense copy of its weights.

use std::thread;

use half::bf16;

use super::ops::matvec;
use crate::quant::{GROUP_SIZE, words_per_row};
use crate::weights::QuantTensor;

#[derive(Clone, Copy)]
pub enum Matrix<'a> {
    /// Row-major `[out, in]`.
    Dense(&'a [f32]),
    /// Rows `first..first + rows` of a quantized tensor, counting rows over
    /// all leading dimensions.
    Quant {
        q: QuantTensor<'a>,
        first: usize,
        rows: usize,
    },
}

/// Below this many weights a matvec runs on one thread.
const PARALLEL_MIN: usize = 1 << 18;

impl<'a> Matrix<'a> {
    pub fn quant(q: QuantTensor<'a>) -> Matrix<'a> {
        let rows = q.scales.len() / q.scales.shape.last().unwrap();
        Matrix::Quant { q, first: 0, rows }
    }

    /// Part `i` of `n` equal parts by rows, such as one expert of a stacked
    /// `[experts, out, in]` tensor.
    pub fn part(&self, i: usize, n: usize) -> Matrix<'a> {
        match *self {
            Matrix::Dense(w) => {
                let size = w.len() / n;
                Matrix::Dense(&w[i * size..(i + 1) * size])
            }
            Matrix::Quant { q, first, rows } => {
                let size = rows / n;
                Matrix::Quant {
                    q,
                    first: first + i * size,
                    rows: size,
                }
            }
        }
    }

    /// Number of rows when each row has `cols` values.
    pub fn rows(&self, cols: usize) -> usize {
        match *self {
            Matrix::Dense(w) => w.len() / cols,
            Matrix::Quant { rows, .. } => rows,
        }
    }

    pub fn matvec(&self, x: &[f32]) -> Vec<f32> {
        match *self {
            Matrix::Dense(w) => matvec(w, x),
            Matrix::Quant { q, first, rows } => quant_matvec(&q, first, rows, x),
        }
    }
}

fn quant_matvec(q: &QuantTensor, first: usize, rows: usize, x: &[f32]) -> Vec<f32> {
    assert_eq!(q.params.group_size, GROUP_SIZE);
    assert_eq!(q.in_features(), x.len(), "quantized matvec: input size");
    // sum(w * x) over a group is scale * sum(q * x) + bias * sum(x).
    let x_sums: Vec<f32> = x
        .as_chunks::<GROUP_SIZE>()
        .0
        .iter()
        .map(|g| g.iter().sum())
        .collect();
    let row = |r: usize| row_dot(q, first + r, x, &x_sums);

    let threads = thread::available_parallelism().map_or(1, |n| n.get());
    if rows * x.len() < PARALLEL_MIN || threads == 1 {
        return (0..rows).map(row).collect();
    }
    let mut out = vec![0.0; rows];
    let chunk = rows.div_ceil(threads);
    thread::scope(|s| {
        for (i, part) in out.chunks_mut(chunk).enumerate() {
            let row = &row;
            s.spawn(move || {
                for (j, o) in part.iter_mut().enumerate() {
                    *o = row(i * chunk + j);
                }
            });
        }
    });
    out
}

/// One output value: row `r` of `q` times `x`, read from the packed bit
/// stream without unpacking the row first.
fn row_dot(q: &QuantTensor, r: usize, x: &[f32], x_sums: &[f32]) -> f32 {
    let bits = q.params.bits;
    let row_bytes = words_per_row(x.len(), bits) * 4;
    let groups = x_sums.len();
    let mut words = q.weight.bytes[r * row_bytes..(r + 1) * row_bytes]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b) as u64);
    let half = |bytes: &[u8], g: usize| {
        let i = (r * groups + g) * 2;
        bf16::from_le_bytes([bytes[i], bytes[i + 1]]).to_f32()
    };

    let mask = (1u64 << bits) - 1;
    let (mut acc, mut have) = (0u64, 0u32);
    let mut total = 0.0;
    for (g, xs) in x.as_chunks::<GROUP_SIZE>().0.iter().enumerate() {
        let mut dot = 0.0;
        for xv in xs {
            if have < bits {
                acc |= words.next().unwrap() << have;
                have += 32;
            }
            dot += (acc & mask) as f32 * xv;
            acc >>= bits;
            have -= bits;
        }
        total += half(q.scales.bytes, g) * dot + half(q.biases.bytes, g) * x_sums[g];
    }
    total
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantParams;
    use crate::quant::dequantize;
    use crate::weights::{Dtype, Tensor};

    /// A quantized `[rows, cols]` matrix from simple patterns, and its
    /// dense form from `quant::dequantize`.
    fn sample(
        bits: u32,
        rows: usize,
        cols: usize,
        shapes: &mut Vec<Vec<usize>>,
    ) -> (Vec<u8>, Vec<u8>, Vec<u8>) {
        let groups = cols / GROUP_SIZE;
        let row_words = words_per_row(cols, bits);
        let mut words = vec![0u32; rows * row_words];
        for r in 0..rows {
            for i in 0..cols {
                let v = ((r * 7 + i * 13) % (1 << bits)) as u64;
                let bit = i * bits as usize;
                let w = &mut words[r * row_words..];
                w[bit / 32] |= (v << (bit % 32)) as u32;
                if bit % 32 + bits as usize > 32 {
                    w[bit / 32 + 1] |= (v >> (32 - bit % 32)) as u32;
                }
            }
        }
        let scales: Vec<bf16> = (0..rows * groups)
            .map(|i| bf16::from_f32(0.01 + i as f32 * 1e-3))
            .collect();
        let biases: Vec<bf16> = (0..rows * groups)
            .map(|i| bf16::from_f32(-0.1 + i as f32 * 2e-3))
            .collect();
        shapes.push(vec![rows, row_words]);
        shapes.push(vec![rows, groups]);
        let to_bytes = |v: &[bf16]| v.iter().flat_map(|h| h.to_le_bytes()).collect::<Vec<u8>>();
        (
            words.iter().flat_map(|w| w.to_le_bytes()).collect(),
            to_bytes(&scales),
            to_bytes(&biases),
        )
    }

    #[test]
    fn quantized_matvec_matches_dequantize() {
        for bits in [5, 8] {
            let (rows, cols) = (300, 1024);
            let mut shapes = Vec::new();
            let (w, s, b) = sample(bits, rows, cols, &mut shapes);
            let tensor = |dtype, shape: &'static [usize], bytes: &'static [u8]| Tensor {
                dtype,
                shape,
                bytes,
                shard: 0,
                offset: 0,
            };
            let leak = |v: Vec<u8>| -> &'static [u8] { Vec::leak(v) };
            let shape = |v: &Vec<usize>| -> &'static [usize] { Vec::leak(v.clone()) };
            let q = QuantTensor {
                weight: tensor(Dtype::U32, shape(&shapes[0]), leak(w)),
                scales: tensor(Dtype::BF16, shape(&shapes[1]), leak(s)),
                biases: tensor(Dtype::BF16, shape(&shapes[1]), leak(b)),
                params: QuantParams {
                    bits,
                    group_size: 64,
                },
            };
            let x: Vec<f32> = (0..cols)
                .map(|i| ((i * 31 % 17) as f32 - 8.0) * 0.1)
                .collect();

            let dense = dequantize(
                &q.weight.to_u32(),
                &q.scales
                    .to_f32()
                    .iter()
                    .map(|&v| bf16::from_f32(v))
                    .collect::<Vec<_>>(),
                &q.biases
                    .to_f32()
                    .iter()
                    .map(|&v| bf16::from_f32(v))
                    .collect::<Vec<_>>(),
                bits,
                rows,
                cols,
            );
            let want = matvec(&dense, &x);
            let m = Matrix::quant(q);
            let got = m.matvec(&x);
            crate::cpu::testdata::assert_close(&format!("{bits}-bit"), &got, &want, 1e-4);

            let part = m.part(2, 3).matvec(&x);
            let dense_part = Matrix::Dense(&dense).part(2, 3).matvec(&x);
            crate::cpu::testdata::assert_close("part", &part, &dense_part, 1e-4);
        }
    }
}
