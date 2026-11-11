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

/// Quantizes one row the way `mlx.core.quantize` does in affine mode
/// (`affine_quantize` in mlx/backend/metal/kernels/quantized.h, v0.32.4).
/// The group end with the larger magnitude is hit exactly; the codes use
/// the f32 scale and bias, which are then stored as BF16.
pub fn quantize_row(
    row: &[f32],
    bits: u32,
    words: &mut [u32],
    scales: &mut [bf16],
    biases: &mut [bf16],
) {
    let n_bins = ((1u32 << bits) - 1) as f32;
    words.fill(0);
    for (g, group) in row.as_chunks::<GROUP_SIZE>().0.iter().enumerate() {
        // MLX's kernel starts the max at 0 but the min at the type's maximum,
        // so an all-negative group still reaches up to 0.
        let max = group.iter().copied().fold(0.0, f32::max);
        let min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let mut scale = ((max - min) / n_bins).max(1e-7);
        let low_side = min.abs() > max.abs();
        if !low_side {
            scale = -scale;
        }
        let edge = if low_side { min } else { max };
        let q0 = (edge / scale).round();
        let bias = if q0 == 0.0 { 0.0 } else { edge };
        if q0 != 0.0 {
            scale = edge / q0;
        }
        scales[g] = bf16::from_f32(scale);
        biases[g] = bf16::from_f32(bias);
        for (j, &w) in group.iter().enumerate() {
            let q = ((w - bias) / scale).round().clamp(0.0, n_bins) as u64;
            let bit = (g * GROUP_SIZE + j) * bits as usize;
            let (word, shift) = (bit / 32, bit % 32);
            words[word] |= (q << shift) as u32;
            if shift + bits as usize > 32 {
                words[word + 1] |= (q >> (32 - shift)) as u32;
            }
        }
    }
}

/// Quantizes a row-major `rows x cols` matrix into words, scales and biases.
pub fn quantize(
    values: &[f32],
    bits: u32,
    rows: usize,
    cols: usize,
) -> (Vec<u32>, Vec<bf16>, Vec<bf16>) {
    assert_eq!(cols % GROUP_SIZE, 0, "cols {cols}");
    assert_eq!(values.len(), rows * cols);
    let row_words = words_per_row(cols, bits);
    let groups = cols / GROUP_SIZE;
    let mut words = vec![0; rows * row_words];
    let mut scales = vec![bf16::ZERO; rows * groups];
    let mut biases = vec![bf16::ZERO; rows * groups];
    for r in 0..rows {
        quantize_row(
            &values[r * cols..(r + 1) * cols],
            bits,
            &mut words[r * row_words..(r + 1) * row_words],
            &mut scales[r * groups..(r + 1) * groups],
            &mut biases[r * groups..(r + 1) * groups],
        );
    }
    (words, scales, biases)
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

    /// Rows quantized by mlx.core.quantize, from tools/quantize_ref.py.
    #[test]
    fn quantize_matches_mlx() {
        let data: serde_json::Value =
            serde_json::from_str(include_str!("../tests/data/quantize_cases.json")).unwrap();
        let bf = |v: &serde_json::Value| bf16::from_bits(v.as_u64().unwrap() as u16);
        let rows: Vec<Vec<f32>> = data["rows"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| {
                r.as_array()
                    .unwrap()
                    .iter()
                    .map(|v| bf(v).to_f32())
                    .collect()
            })
            .collect();
        for case in data["cases"].as_array().unwrap() {
            let bits = case["bits"].as_u64().unwrap() as u32;
            for (i, row) in rows.iter().enumerate() {
                let (words, scales, biases) = quantize(row, bits, 1, row.len());
                let want_words: Vec<u32> = case["q"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|v| v.as_u64().unwrap() as u32)
                    .collect();
                let want_scales: Vec<bf16> = case["scales"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(bf)
                    .collect();
                let want_biases: Vec<bf16> = case["biases"][i]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(bf)
                    .collect();
                assert_eq!(scales, want_scales, "bits {bits} row {i} scales");
                assert_eq!(biases, want_biases, "bits {bits} row {i} biases");
                assert_eq!(words, want_words, "bits {bits} row {i} words");
            }
        }
    }
}
