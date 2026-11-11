//! MLX affine quantization, as written by `mlx.core.quantize`.
//!
//! A row of `cols` values is stored as a little-endian bit stream of
//! `bits`-wide unsigned integers packed into `u32` words. Each group of 64
//! values has one BF16 scale and one BF16 bias: `w = scale * q + bias`.

use half::bf16;

pub const GROUP_SIZE: usize = 64;

/// Number of `u32` words that hold one row of `cols` values.
pub fn words_per_row(cols: usize, bits: u32) -> usize {
    cols * bits as usize / 32
}

/// The `i`-th packed integer of a row.
pub fn unpack(words: &[u32], bits: u32, i: usize) -> u32 {
    let bit = i * bits as usize;
    let word = bit / 32;
    let shift = bit % 32;
    let mut pair = words[word] as u64;
    if shift + bits as usize > 32 {
        pair |= (words[word + 1] as u64) << 32;
    }
    ((pair >> shift) as u32) & ((1u32 << bits) - 1)
}

/// Dequantizes one row into `out`, which sets the number of columns.
pub fn dequantize_row(words: &[u32], scales: &[bf16], biases: &[bf16], bits: u32, out: &mut [f32]) {
    for (i, w) in out.iter_mut().enumerate() {
        let g = i / GROUP_SIZE;
        let q = unpack(words, bits, i) as f32;
        *w = scales[g].to_f32() * q + biases[g].to_f32();
    }
}

/// Dequantizes a `rows x cols` matrix into a row-major `Vec<f32>`.
pub fn dequantize(
    words: &[u32],
    scales: &[bf16],
    biases: &[bf16],
    bits: u32,
    rows: usize,
    cols: usize,
) -> Vec<f32> {
    assert!(
        bits == 4 || bits == 5 || bits == 6 || bits == 8,
        "bits {bits}"
    );
    assert_eq!(cols % GROUP_SIZE, 0, "cols {cols}");
    let row_words = words_per_row(cols, bits);
    let groups = cols / GROUP_SIZE;
    assert_eq!(words.len(), rows * row_words);
    assert_eq!(scales.len(), rows * groups);
    assert_eq!(biases.len(), rows * groups);

    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        dequantize_row(
            &words[r * row_words..(r + 1) * row_words],
            &scales[r * groups..(r + 1) * groups],
            &biases[r * groups..(r + 1) * groups],
            bits,
            &mut out[r * cols..(r + 1) * cols],
        );
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unpack_crosses_word_boundary() {
        // 5-bit values 0..32 in order: value 6 sits in bits 30..35.
        let values: Vec<u32> = (0..64).map(|i| i % 32).collect();
        let mut words = vec![0u32; words_per_row(64, 5)];
        for (i, &v) in values.iter().enumerate() {
            let bit = i * 5;
            words[bit / 32] |= v << (bit % 32);
            if bit % 32 > 27 {
                words[bit / 32 + 1] |= v >> (32 - bit % 32);
            }
        }
        for (i, &v) in values.iter().enumerate() {
            assert_eq!(unpack(&words, 5, i), v, "index {i}");
        }
    }
}
