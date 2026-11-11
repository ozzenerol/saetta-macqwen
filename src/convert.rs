//! Rewrites an MLX model dir so every tensor is aligned for Metal.
//!
//! MLX shards put the data section right after the JSON header, so tensors
//! start at odd file offsets. The converted shards have the same tensor
//! names, shapes, dtypes and bytes, but the header is padded with spaces so
//! the data section starts on a 16384-byte page, and each tensor starts on a
//! 256-byte boundary. Only `language_model.` tensors are kept.
//!
//! The on-disk layout is decided in `place_tensors`; a different layout
//! (for example scales next to weights) only needs a change there.

use crate::weights::{Tensor, Weights};
use serde_json::{Map, Value, json};
use std::collections::BTreeMap;
use std::error::Error;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::Path;

pub const DATA_ALIGN: usize = 16384;
pub const TENSOR_ALIGN: usize = 256;
const TEXT_PREFIX: &str = "language_model.";
const COPIED: &[&str] = &[
    "config.json",
    "chat_template.jinja",
    "generation_config.json",
];

pub struct Summary {
    pub shards: usize,
    pub tensors: usize,
    pub bytes: usize,
}

pub fn convert(src: &Path, out: &Path) -> Result<Summary, Box<dyn Error>> {
    fs::create_dir_all(out)?;
    if fs::canonicalize(src)? == fs::canonicalize(out)? {
        return Err("output dir is the source dir".into());
    }
    let weights = Weights::open(src)?;

    // Group the text tensors by shard, in their source order.
    let mut by_shard: Vec<Vec<(&str, Tensor)>> = vec![Vec::new(); weights.shard_count()];
    for name in weights.names().filter(|n| n.starts_with(TEXT_PREFIX)) {
        let t = weights.tensor(name)?;
        by_shard[t.shard].push((name, t));
    }

    let mut weight_map = BTreeMap::new();
    let mut summary = Summary {
        shards: 0,
        tensors: 0,
        bytes: 0,
    };
    for (shard, mut tensors) in by_shard.into_iter().enumerate() {
        if tensors.is_empty() {
            continue;
        }
        tensors.sort_by_key(|(_, t)| t.offset);
        let file_name = weights.shard_name(shard);
        write_shard(&out.join(file_name), &tensors)?;
        for (name, t) in &tensors {
            weight_map.insert(name.to_string(), file_name.to_string());
            summary.bytes += t.bytes.len();
        }
        summary.shards += 1;
        summary.tensors += tensors.len();
    }

    let index = json!({
        "metadata": {"total_size": summary.bytes},
        "weight_map": weight_map,
    });
    fs::write(
        out.join("model.safetensors.index.json"),
        serde_json::to_string_pretty(&index)?,
    )?;

    for entry in fs::read_dir(src)? {
        let name = entry?.file_name();
        let name = name.to_string_lossy();
        if COPIED.contains(&name.as_ref()) || name.starts_with("tokenizer") {
            fs::copy(src.join(name.as_ref()), out.join(name.as_ref()))?;
        }
    }
    Ok(summary)
}

/// Data offsets of each tensor from the start of the data section.
fn place_tensors(tensors: &[(&str, Tensor)]) -> Vec<(usize, usize)> {
    let mut end = 0usize;
    tensors
        .iter()
        .map(|(_, t)| {
            let start = end.next_multiple_of(TENSOR_ALIGN);
            end = start + t.bytes.len();
            (start, end)
        })
        .collect()
}

fn write_shard(path: &Path, tensors: &[(&str, Tensor)]) -> Result<(), Box<dyn Error>> {
    let offsets = place_tensors(tensors);

    let mut header = Map::new();
    header.insert("__metadata__".into(), json!({"format": "mlx"}));
    for ((name, t), (start, end)) in tensors.iter().zip(&offsets) {
        header.insert(
            name.to_string(),
            json!({"dtype": t.dtype.name(), "shape": t.shape, "data_offsets": [start, end]}),
        );
    }
    let mut header = serde_json::to_vec(&Value::Object(header))?;
    let data_start = (8 + header.len()).next_multiple_of(DATA_ALIGN);
    header.resize(data_start - 8, b' ');

    let mut file = BufWriter::with_capacity(1 << 20, File::create(path)?);
    file.write_all(&(header.len() as u64).to_le_bytes())?;
    file.write_all(&header)?;
    let mut written = 0;
    for ((_, t), (start, end)) in tensors.iter().zip(&offsets) {
        file.write_all(&vec![0u8; start - written])?;
        file.write_all(t.bytes)?;
        written = *end;
    }
    file.into_inner()?.sync_all()?;
    Ok(())
}

/// Checks that `out` has every text tensor of `src` with the same dtype,
/// shape and bytes, and nothing else, all aligned. Returns the tensor count.
pub fn verify(src: &Path, out: &Path) -> Result<usize, Box<dyn Error>> {
    let a = Weights::open(src)?;
    let b = Weights::open(out)?;
    let mut count = 0;
    for name in a.names().filter(|n| n.starts_with(TEXT_PREFIX)) {
        let (x, y) = (a.tensor(name)?, b.tensor(name)?);
        if x.dtype != y.dtype || x.shape != y.shape || x.bytes != y.bytes {
            return Err(format!("{name} differs").into());
        }
        count += 1;
    }
    if b.len() != count {
        return Err(format!("output has {} tensors, want {count}", b.len()).into());
    }
    let unaligned = b.unaligned(TENSOR_ALIGN);
    if let Some(name) = unaligned.first() {
        return Err(format!("{} tensors unaligned, first {name}", unaligned.len()).into());
    }
    Ok(count)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::weights::tests::{temp_dir, write_shard as write_source};

    #[test]
    fn converts_small_model() {
        let src = temp_dir("convert-src");
        let out = temp_dir("convert-out");
        write_source(
            &src,
            "a.safetensors",
            r#"{"language_model.n":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]},
                "vision_tower.v":{"dtype":"F32","shape":[1],"data_offsets":[6,10]},
                "language_model.w":{"dtype":"U32","shape":[2],"data_offsets":[10,18]}}"#,
            &[1, 2, 3, 4, 5, 6, 0, 0, 0, 0, 7, 8, 9, 10, 11, 12, 13, 14],
        );
        write_source(
            &src,
            "b.safetensors",
            r#"{"vision_tower.u":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        fs::write(
            src.join("model.safetensors.index.json"),
            r#"{"weight_map":{"language_model.n":"a.safetensors","vision_tower.v":"a.safetensors",
                "language_model.w":"a.safetensors","vision_tower.u":"b.safetensors"}}"#,
        )
        .unwrap();
        fs::write(src.join("config.json"), "{}").unwrap();
        fs::write(src.join("tokenizer.json"), "{}").unwrap();

        let summary = convert(&src, &out).unwrap();
        assert_eq!((summary.shards, summary.tensors, summary.bytes), (1, 2, 14));
        assert_eq!(verify(&src, &out).unwrap(), 2);
        assert!(!out.join("b.safetensors").exists());
        assert!(out.join("tokenizer.json").exists());
        assert!(out.join("config.json").exists());

        let file = fs::read(out.join("a.safetensors")).unwrap();
        let header_len = u64::from_le_bytes(file[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + header_len) % DATA_ALIGN, 0);
        let w = Weights::open(&out).unwrap();
        assert_eq!(
            w.tensor("language_model.w").unwrap().offset,
            DATA_ALIGN + 256
        );

        fs::remove_dir_all(&src).unwrap();
        fs::remove_dir_all(&out).unwrap();
    }

    /// Reads only the headers of the converted model on the Mac. Skipped
    /// when it is not there.
    #[test]
    fn converted_model_is_aligned() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let dir = Path::new(&home).join("llm/saetta-models/Qwen3.6-35B-A3B-5bit");
        if !dir.exists() {
            eprintln!("skipped: no model at {}", dir.display());
            return;
        }
        let w = Weights::open(&dir).unwrap();
        assert_eq!(w.len(), 1757);
        assert!(w.unaligned(TENSOR_ALIGN).is_empty());
    }

    #[test]
    fn refuses_same_dir() {
        let dir = temp_dir("convert-same");
        assert!(convert(&dir, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
