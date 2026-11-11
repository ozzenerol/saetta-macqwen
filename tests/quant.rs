//! Checks `quant::dequantize` against `mlx.core.dequantize` on slices of the
//! real model. The data comes from `tests/quant_dump.py`; the test is skipped
//! when it is missing. Set SAETTA_QUANT_DATA to use another directory.

use half::bf16;
use saetta::quant;
use std::path::PathBuf;

fn data_dir() -> Option<PathBuf> {
    let dir = match std::env::var("SAETTA_QUANT_DATA") {
        Ok(d) => PathBuf::from(d),
        Err(_) => PathBuf::from(std::env::var("HOME").ok()?).join("saetta/quant-data"),
    };
    dir.join("index.json").exists().then_some(dir)
}

fn read_u32(path: PathBuf) -> Vec<u32> {
    let bytes = std::fs::read(&path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| u32::from_le_bytes(*c))
        .collect()
}

fn read_bf16(path: PathBuf) -> Vec<bf16> {
    let bytes = std::fs::read(&path).unwrap();
    bytes
        .as_chunks::<2>()
        .0
        .iter()
        .map(|c| bf16::from_bits(u16::from_le_bytes(*c)))
        .collect()
}

fn read_f32(path: PathBuf) -> Vec<f32> {
    read_u32(path).into_iter().map(f32::from_bits).collect()
}

#[test]
fn matches_mlx_dequantize() {
    let Some(dir) = data_dir() else {
        eprintln!("skipped: no quant data, run tests/quant_dump.py on the Mac");
        return;
    };
    let index: serde_json::Value =
        serde_json::from_slice(&std::fs::read(dir.join("index.json")).unwrap()).unwrap();

    for (name, case) in index.as_object().unwrap() {
        let rows = case["rows"].as_u64().unwrap() as usize;
        let cols = case["cols"].as_u64().unwrap() as usize;
        let bits = case["bits"].as_u64().unwrap() as u32;
        let words = read_u32(dir.join(format!("{name}.weight.u32")));
        let scales = read_bf16(dir.join(format!("{name}.scales.bf16")));
        let biases = read_bf16(dir.join(format!("{name}.biases.bf16")));
        let ref_f32 = read_f32(dir.join(format!("{name}.ref_f32.f32")));
        let ref_bf16 = read_f32(dir.join(format!("{name}.ref_bf16.f32")));

        let ours = quant::dequantize(&words, &scales, &biases, bits, rows, cols);
        assert_eq!(ours.len(), ref_f32.len());

        let mut f32_diff = 0f32;
        let mut bf16_mismatch = 0;
        for i in 0..ours.len() {
            f32_diff = f32_diff.max((ours[i] - ref_f32[i]).abs());
            if bf16::from_f32(ours[i]).to_f32() != ref_bf16[i] {
                bf16_mismatch += 1;
            }
        }
        eprintln!(
            "{name}: {rows}x{cols} {bits}-bit, max diff vs f32 {f32_diff:e}, \
             bf16 mismatches {bf16_mismatch}/{}",
            ours.len()
        );
        assert!(f32_diff <= 1e-6, "{name}: max diff {f32_diff}");
        assert_eq!(bf16_mismatch, 0, "{name}");
    }
}
