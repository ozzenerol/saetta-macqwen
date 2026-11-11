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

use crate::config::Config;
use crate::quant;
use crate::weights::{Dtype, Tensor, Weights};
use half::bf16;
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

// The MTP layer.
//
// The MLX files have no MTP weights, so they come from the Hugging Face
// checkpoint (tools/fetch_mtp.py) in bf16. `convert_mtp` puts them in the
// same form as the rest of the model: `language_model.mtp.*` names, experts
// split into `switch_mlp` gate, up and down, matrices quantized like their
// counterparts in layer 0, and norms shifted by one as mlx_lm's sanitize does
// for the main model (HF stores RMSNorm weights as `1 + w`).

pub const MTP_PREFIX: &str = "language_model.mtp.";
pub const MTP_FILE: &str = "mtp.safetensors";

struct Owned {
    name: String,
    dtype: Dtype,
    shape: Vec<usize>,
    bytes: Vec<u8>,
}

/// Writes `mtp.safetensors` into `model_dir` and adds its tensors to the
/// index and the 8-bit routers to the config. Returns the tensor count.
pub fn convert_mtp(hf_file: &Path, model_dir: &Path) -> Result<usize, Box<dyn Error>> {
    let config = Config::load(model_dir)?;
    let source = Weights::open_file(hf_file)?;
    let mut names: Vec<&str> = source.names().filter(|n| n.starts_with("mtp.")).collect();
    names.sort();
    if names.is_empty() {
        return Err("no mtp.* tensors in the file".into());
    }

    let mut out = Vec::new();
    for name in names {
        let t = source.tensor(name)?;
        if t.dtype != Dtype::BF16 {
            return Err(format!("{name} is {:?}, want BF16", t.dtype).into());
        }
        let rest = &name["mtp.".len()..];
        if let Some(layer) = rest.strip_suffix(".mlp.experts.gate_up_proj") {
            let [experts, rows, cols] = t.shape[..] else {
                return Err(format!("{name}: shape {:?}", t.shape).into());
            };
            let values = t.to_f32();
            let half = rows / 2;
            let expert_rows = |e: usize, from: usize| {
                &values[(e * rows + from) * cols..(e * rows + from + half) * cols]
            };
            for (part, from) in [("gate_proj", 0), ("up_proj", half)] {
                let module = format!("{MTP_PREFIX}{layer}.mlp.switch_mlp.{part}");
                let parts: Vec<&[f32]> = (0..experts).map(|e| expert_rows(e, from)).collect();
                let bits = bits_like(&config, &module);
                push_quantized(
                    &mut out,
                    &module,
                    &parts.concat(),
                    &[experts, half, cols],
                    bits,
                );
            }
        } else if let Some(layer) = rest.strip_suffix(".mlp.experts.down_proj") {
            let module = format!("{MTP_PREFIX}{layer}.mlp.switch_mlp.down_proj");
            let bits = bits_like(&config, &module);
            push_quantized(&mut out, &module, &t.to_f32(), t.shape, bits);
        } else if t.shape.len() == 1 {
            let shifted: Vec<u8> = t
                .to_f32()
                .iter()
                .flat_map(|&w| bf16::from_f32(w + 1.0).to_le_bytes())
                .collect();
            out.push(Owned {
                name: format!("{MTP_PREFIX}{rest}"),
                dtype: Dtype::BF16,
                shape: t.shape.to_vec(),
                bytes: shifted,
            });
        } else {
            let module = format!(
                "{MTP_PREFIX}{}",
                rest.strip_suffix(".weight")
                    .ok_or(format!("{name}: not a weight"))?
            );
            let bits = bits_like(&config, &module);
            push_quantized(&mut out, &module, &t.to_f32(), t.shape, bits);
        }
    }

    let tensors: Vec<(&str, Tensor)> = out
        .iter()
        .map(|o| {
            let t = Tensor {
                dtype: o.dtype,
                shape: &o.shape,
                bytes: &o.bytes,
                shard: 0,
                offset: 0,
            };
            (o.name.as_str(), t)
        })
        .collect();
    write_shard(&model_dir.join(MTP_FILE), &tensors)?;

    let index_path = model_dir.join("model.safetensors.index.json");
    let mut index: Value = serde_json::from_str(&fs::read_to_string(&index_path)?)?;
    let map = index["weight_map"]
        .as_object_mut()
        .ok_or("index has no weight_map")?;
    map.retain(|name, _| !name.starts_with(MTP_PREFIX));
    for o in &out {
        map.insert(o.name.clone(), MTP_FILE.into());
    }
    fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;
    let total: usize = {
        let w = Weights::open(model_dir)?;
        w.names()
            .map(|n| w.tensor(n).map_or(0, |t| t.bytes.len()))
            .sum()
    };
    index["metadata"]["total_size"] = total.into();
    fs::write(&index_path, serde_json::to_string_pretty(&index)?)?;

    add_quant_overrides(model_dir, &out)?;
    Ok(out.len())
}

/// The bits of the same module in layer 0: 8 for the routers, else the default.
fn bits_like(config: &Config, mtp_module: &str) -> u32 {
    let suffix = mtp_module.rsplit_once(".mlp.").map(|(_, s)| s);
    let main = match suffix {
        Some(s) => format!("language_model.model.layers.0.mlp.{s}"),
        None => String::new(),
    };
    config.quantization.for_module(&main).bits
}

fn push_quantized(out: &mut Vec<Owned>, module: &str, values: &[f32], shape: &[usize], bits: u32) {
    let cols = *shape.last().unwrap();
    let rows = values.len() / cols;
    let (words, scales, biases) = quant::quantize(values, bits, rows, cols);
    let lead = &shape[..shape.len() - 1];
    let with_last = |last: usize| [lead, &[last]].concat();
    out.push(Owned {
        name: format!("{module}.weight"),
        dtype: Dtype::U32,
        shape: with_last(quant::words_per_row(cols, bits)),
        bytes: words.iter().flat_map(|w| w.to_le_bytes()).collect(),
    });
    for (part, values) in [("scales", scales), ("biases", biases)] {
        out.push(Owned {
            name: format!("{module}.{part}"),
            dtype: Dtype::BF16,
            shape: with_last(cols / quant::GROUP_SIZE),
            bytes: values.iter().flat_map(|v| v.to_le_bytes()).collect(),
        });
    }
}

/// Adds a quantization entry to config.json for every MTP module that is
/// not at the default bits, so `Weights::quantized` finds them.
fn add_quant_overrides(model_dir: &Path, out: &[Owned]) -> Result<(), Box<dyn Error>> {
    let path = model_dir.join("config.json");
    let mut root: Value = serde_json::from_str(&fs::read_to_string(&path)?)?;
    let default_bits = root["quantization"]["bits"]
        .as_u64()
        .ok_or("config has no quantization bits")?;
    for o in out
        .iter()
        .filter(|o| o.name.ends_with(".weight") && o.dtype == Dtype::U32)
    {
        let module = o.name.trim_end_matches(".weight");
        let cols = o.shape.last().unwrap() * 32;
        let scales = out
            .iter()
            .find(|s| s.name == format!("{module}.scales"))
            .unwrap();
        let bits = cols / (scales.shape.last().unwrap() * quant::GROUP_SIZE);
        if bits as u64 != default_bits {
            let entry = json!({"group_size": quant::GROUP_SIZE, "bits": bits});
            for key in ["quantization", "quantization_config"] {
                if let Some(q) = root[key].as_object_mut() {
                    q.insert(module.to_string(), entry.clone());
                }
            }
        }
    }
    fs::write(&path, serde_json::to_string_pretty(&root)?)?;
    Ok(())
}

/// Compares the MTP tensors in `model_dir` with the HF source: norms equal
/// to bf16(w + 1), matrices within the quantization error, all aligned.
/// Returns the largest relative RMS error of a quantized matrix.
pub fn verify_mtp(hf_file: &Path, model_dir: &Path) -> Result<f64, Box<dyn Error>> {
    let config = Config::load(model_dir)?;
    let source = Weights::open_file(hf_file)?;
    let model = Weights::open(model_dir)?;
    if let Some(name) = model
        .unaligned(TENSOR_ALIGN)
        .into_iter()
        .find(|n| n.starts_with(MTP_PREFIX))
    {
        return Err(format!("{name} is not aligned").into());
    }
    let dequant = |module: &str| -> Result<Vec<f32>, Box<dyn Error>> {
        let q = model.quantized(module, &config.quantization)?;
        let rows = q.weight.len() / q.weight.shape.last().unwrap();
        let scales: Vec<bf16> = q
            .scales
            .to_f32()
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        let biases: Vec<bf16> = q
            .biases
            .to_f32()
            .iter()
            .map(|&v| bf16::from_f32(v))
            .collect();
        Ok(quant::dequantize(
            &q.weight.to_u32(),
            &scales,
            &biases,
            q.params.bits,
            rows,
            q.in_features(),
        ))
    };
    let mut worst: f64 = 0.0;
    for name in source.names().filter(|n| n.starts_with("mtp.")) {
        let t = source.tensor(name)?;
        let want = t.to_f32();
        let rest = &name["mtp.".len()..];
        let got = if let Some(layer) = rest.strip_suffix(".mlp.experts.gate_up_proj") {
            let (experts, rows, cols) = (t.shape[0], t.shape[1], t.shape[2]);
            let gate = dequant(&format!("{MTP_PREFIX}{layer}.mlp.switch_mlp.gate_proj"))?;
            let up = dequant(&format!("{MTP_PREFIX}{layer}.mlp.switch_mlp.up_proj"))?;
            let n = rows / 2 * cols;
            (0..experts)
                .flat_map(|e| [&gate[e * n..(e + 1) * n], &up[e * n..(e + 1) * n]].concat())
                .collect()
        } else if let Some(layer) = rest.strip_suffix(".mlp.experts.down_proj") {
            dequant(&format!("{MTP_PREFIX}{layer}.mlp.switch_mlp.down_proj"))?
        } else if t.shape.len() == 1 {
            let got = model.tensor(&format!("{MTP_PREFIX}{rest}"))?.to_f32();
            let exact = want
                .iter()
                .zip(&got)
                .all(|(&w, &g)| bf16::from_f32(w + 1.0).to_f32() == g);
            if !exact {
                return Err(format!("{name}: norm is not bf16(w + 1)").into());
            }
            continue;
        } else {
            dequant(&format!("{MTP_PREFIX}{}", rest.trim_end_matches(".weight")))?
        };
        if got.len() != want.len() {
            return Err(format!("{name}: {} values, want {}", got.len(), want.len()).into());
        }
        let err: f64 = want
            .iter()
            .zip(&got)
            .map(|(&w, &g)| ((w - g) as f64).powi(2))
            .sum();
        let norm: f64 = want.iter().map(|&w| (w as f64).powi(2)).sum();
        worst = worst.max((err / norm).sqrt());
    }
    Ok(worst)
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
        let main = w.names().filter(|n| !n.starts_with(MTP_PREFIX)).count();
        assert_eq!(main, 1757);
        assert!(w.unaligned(TENSOR_ALIGN).is_empty());
        if dir.join(MTP_FILE).exists() {
            assert_eq!(w.len() - main, 46);
            let c = Config::load(&dir).unwrap();
            let router = w.quantized(&format!("{MTP_PREFIX}layers.0.mlp.gate"), &c.quantization);
            assert_eq!(router.unwrap().params.bits, 8);
        }
    }

    fn bf16_bytes(values: &[f32]) -> Vec<u8> {
        values
            .iter()
            .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
            .collect()
    }

    #[test]
    fn converts_small_mtp() {
        let dir = temp_dir("mtp-model");
        let config = r#"{"eos_token_id": 1, "quantization": {"group_size": 64, "bits": 5, "mode": "affine",
                "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}},
            "text_config": {"hidden_size": 64, "vocab_size": 10, "rms_norm_eps": 1e-6,
                "num_hidden_layers": 1, "layer_types": ["full_attention"],
                "num_attention_heads": 1, "num_key_value_heads": 1, "head_dim": 64,
                "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000},
                "linear_num_key_heads": 1, "linear_num_value_heads": 1, "linear_key_head_dim": 64,
                "linear_value_head_dim": 64, "linear_conv_kernel_dim": 4, "num_experts": 2,
                "num_experts_per_tok": 1, "moe_intermediate_size": 64,
                "shared_expert_intermediate_size": 64}}"#;
        fs::write(dir.join("config.json"), config).unwrap();
        write_source(
            &dir,
            "a.safetensors",
            r#"{"language_model.model.norm.weight":{"dtype":"BF16","shape":[2],"data_offsets":[0,4]}}"#,
            &[0; 4],
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"metadata": {}, "weight_map": {"language_model.model.norm.weight": "a.safetensors"}}"#,
        )
        .unwrap();

        // 2 experts, gate_up rows: expert 0 gate is all 1, up all 2; expert 1 gate 3, up 4.
        let gate_up: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|&v| vec![v; 2 * 64])
            .collect();
        let router: Vec<f32> = (0..2 * 64).map(|i| i as f32 / 100.0).collect();
        let norm = [-0.5f32, 0.25];
        let mut data = bf16_bytes(&gate_up);
        let (a, b) = (data.len(), data.len() + router.len() * 2);
        data.extend(bf16_bytes(&router));
        data.extend(bf16_bytes(&norm));
        let header = format!(
            r#"{{"mtp.layers.0.mlp.experts.gate_up_proj":{{"dtype":"BF16","shape":[2,4,64],"data_offsets":[0,{a}]}},
                "mtp.layers.0.mlp.gate.weight":{{"dtype":"BF16","shape":[2,64],"data_offsets":[{a},{b}]}},
                "mtp.norm.weight":{{"dtype":"BF16","shape":[2],"data_offsets":[{b},{}]}}}}"#,
            b + 4
        );
        let src = temp_dir("mtp-src");
        write_source(&src, "mtp.safetensors", &header, &data);

        let n = convert_mtp(&src.join("mtp.safetensors"), &dir).unwrap();
        assert_eq!(n, 3 * 3 + 1);
        assert!(verify_mtp(&src.join("mtp.safetensors"), &dir).unwrap() < 0.01);

        let c = Config::load(&dir).unwrap();
        let w = Weights::open(&dir).unwrap();
        assert_eq!(
            w.unaligned(TENSOR_ALIGN),
            ["language_model.model.norm.weight"]
        );
        let norm = w.tensor("language_model.mtp.norm.weight").unwrap();
        assert_eq!(norm.to_f32(), vec![0.5, 1.25]);
        let router = w
            .quantized("language_model.mtp.layers.0.mlp.gate", &c.quantization)
            .unwrap();
        assert_eq!(router.params.bits, 8);
        let up = w
            .quantized(
                "language_model.mtp.layers.0.mlp.switch_mlp.up_proj",
                &c.quantization,
            )
            .unwrap();
        assert_eq!(up.weight.shape, &[2, 2, 10]);
        // A constant group is stored as bias only: expert 1's up rows are all 4.
        assert_eq!(up.biases.to_f32()[2..], [4.0, 4.0]);
        assert_eq!(up.biases.to_f32()[..2], [2.0, 2.0]);
        fs::remove_dir_all(&dir).unwrap();
        fs::remove_dir_all(&src).unwrap();
    }

    #[test]
    fn refuses_same_dir() {
        let dir = temp_dir("convert-same");
        assert!(convert(&dir, &dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }
}
