//! Safetensors shards, memory-mapped.
//!
//! Opening a model reads only the shard headers. Tensor data stays on disk
//! until a caller reads its bytes, so a test that needs one tensor touches
//! only the pages of that tensor.
//!
//! MLX writes the data section right after the JSON header with no padding,
//! so in most shards it starts at an odd file offset and `U32` tensors are
//! not 4-byte aligned in memory. Use `Tensor::to_u32` on the CPU, or the
//! shard bytes and `Tensor::offset` to place a tensor in a GPU buffer.

use crate::config::{QuantParams, Quantization};
use memmap2::Mmap;
use serde_json::Value;
use std::collections::{BTreeSet, HashMap};
use std::error::Error;
use std::fs::{self, File};
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dtype {
    U32,
    BF16,
    F16,
    F32,
}

impl Dtype {
    fn parse(name: &str) -> Option<Dtype> {
        match name {
            "U32" => Some(Dtype::U32),
            "BF16" => Some(Dtype::BF16),
            "F16" => Some(Dtype::F16),
            "F32" => Some(Dtype::F32),
            _ => None,
        }
    }

    pub fn size(self) -> usize {
        match self {
            Dtype::BF16 | Dtype::F16 => 2,
            Dtype::U32 | Dtype::F32 => 4,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct Tensor<'a> {
    pub dtype: Dtype,
    pub shape: &'a [usize],
    pub bytes: &'a [u8],
    /// Index of the shard that holds the tensor.
    pub shard: usize,
    /// Byte offset of the tensor data from the start of the shard file.
    pub offset: usize,
}

impl Tensor<'_> {
    pub fn len(&self) -> usize {
        self.shape.iter().product()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Copies a float tensor into f32 values.
    pub fn to_f32(&self) -> Vec<f32> {
        let chunks = self.bytes.chunks_exact(self.dtype.size());
        match self.dtype {
            Dtype::BF16 => chunks
                .map(|b| half::bf16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F16 => chunks
                .map(|b| half::f16::from_le_bytes([b[0], b[1]]).to_f32())
                .collect(),
            Dtype::F32 => chunks
                .map(|b| f32::from_le_bytes([b[0], b[1], b[2], b[3]]))
                .collect(),
            Dtype::U32 => panic!("to_f32 on a U32 tensor"),
        }
    }

    /// Copies a U32 tensor into aligned words.
    pub fn to_u32(&self) -> Vec<u32> {
        assert_eq!(
            self.dtype,
            Dtype::U32,
            "to_u32 on a {:?} tensor",
            self.dtype
        );
        let (words, _) = self.bytes.as_chunks::<4>();
        words.iter().map(|&b| u32::from_le_bytes(b)).collect()
    }
}

/// An MLX affine quantized matrix: `w = scale * q + bias` per group.
#[derive(Debug, Clone, Copy)]
pub struct QuantTensor<'a> {
    /// Packed `U32` words, shape `[..., out, in * bits / 32]`.
    pub weight: Tensor<'a>,
    /// `BF16`, shape `[..., out, in / group_size]`.
    pub scales: Tensor<'a>,
    pub biases: Tensor<'a>,
    pub params: QuantParams,
}

impl QuantTensor<'_> {
    pub fn in_features(&self) -> usize {
        self.scales.shape.last().unwrap() * self.params.group_size
    }

    pub fn out_features(&self) -> usize {
        self.scales.shape[self.scales.shape.len() - 2]
    }
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    shard: usize,
    start: usize,
    end: usize,
}

pub struct Weights {
    shard_names: Vec<String>,
    shards: Vec<Mmap>,
    entries: HashMap<String, Entry>,
}

impl Weights {
    /// Maps every shard listed in `model.safetensors.index.json`.
    pub fn open(model_dir: &Path) -> Result<Weights, Box<dyn Error>> {
        let index_path = model_dir.join("model.safetensors.index.json");
        let text = fs::read_to_string(&index_path)
            .map_err(|e| format!("{}: {e}", index_path.display()))?;
        let index: Value = serde_json::from_str(&text)?;
        let map = index["weight_map"]
            .as_object()
            .ok_or("index has no weight_map")?;

        let mut files = BTreeSet::new();
        for file in map.values() {
            files.insert(file.as_str().ok_or("bad weight_map entry")?.to_string());
        }

        let mut weights = Weights {
            shard_names: Vec::new(),
            shards: Vec::new(),
            entries: HashMap::new(),
        };
        for name in files {
            weights.add_shard(model_dir, name)?;
        }

        for (tensor, file) in map {
            let found = weights
                .entries
                .get(tensor)
                .map(|e| weights.shard_names[e.shard].as_str());
            if found != file.as_str() {
                return Err(format!("{tensor} is not in {file} as the index says").into());
            }
        }
        Ok(weights)
    }

    fn add_shard(&mut self, dir: &Path, name: String) -> Result<(), Box<dyn Error>> {
        let path = dir.join(&name);
        let file = File::open(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        // Safety: the model files are not changed while we run.
        let map = unsafe { Mmap::map(&file)? };
        let shard = self.shards.len();
        for (tensor, entry) in parse_header(&map, shard).map_err(|e| format!("{name}: {e}"))? {
            self.entries.insert(tensor, entry);
        }
        self.shard_names.push(name);
        self.shards.push(map);
        Ok(())
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.entries.keys().map(String::as_str)
    }

    pub fn shard_count(&self) -> usize {
        self.shards.len()
    }

    /// The whole mapped file of one shard, for no-copy GPU buffers.
    pub fn shard_bytes(&self, shard: usize) -> &[u8] {
        &self.shards[shard]
    }

    pub fn tensor(&self, name: &str) -> Result<Tensor<'_>, String> {
        let e = self
            .entries
            .get(name)
            .ok_or_else(|| format!("no tensor {name}"))?;
        Ok(Tensor {
            dtype: e.dtype,
            shape: &e.shape,
            bytes: &self.shards[e.shard][e.start..e.end],
            shard: e.shard,
            offset: e.start,
        })
    }

    /// The quantized matrix of a module, such as
    /// `language_model.model.layers.0.mlp.switch_mlp.up_proj`.
    pub fn quantized(&self, module: &str, quant: &Quantization) -> Result<QuantTensor<'_>, String> {
        let q = QuantTensor {
            weight: self.tensor(&format!("{module}.weight"))?,
            scales: self.tensor(&format!("{module}.scales"))?,
            biases: self.tensor(&format!("{module}.biases"))?,
            params: quant.for_module(module),
        };
        check_quantized(&q).map_err(|e| format!("{module}: {e}"))?;
        Ok(q)
    }
}

fn check_quantized(q: &QuantTensor) -> Result<(), String> {
    if q.weight.dtype != Dtype::U32 {
        return Err(format!("weight is {:?}, want U32", q.weight.dtype));
    }
    if q.scales.dtype != Dtype::BF16 || q.biases.dtype != Dtype::BF16 {
        return Err("scales and biases must be BF16".into());
    }
    if q.scales.shape != q.biases.shape {
        return Err("scales and biases have different shapes".into());
    }
    let rank = q.weight.shape.len();
    if rank < 2
        || q.scales.shape.len() != rank
        || q.weight.shape[..rank - 1] != q.scales.shape[..rank - 1]
    {
        return Err(format!(
            "weight shape {:?} does not match scales shape {:?}",
            q.weight.shape, q.scales.shape
        ));
    }
    let packed_bits = q.weight.shape[rank - 1] * 32;
    if packed_bits != q.in_features() * q.params.bits as usize {
        return Err(format!(
            "{} words per row do not hold {} inputs of {} bits",
            q.weight.shape[rank - 1],
            q.in_features(),
            q.params.bits
        ));
    }
    Ok(())
}

/// Reads the 8-byte header length, then the JSON header that lists each
/// tensor with its dtype, shape and byte range in the data section.
fn parse_header(file: &[u8], shard: usize) -> Result<Vec<(String, Entry)>, String> {
    let len_bytes: [u8; 8] = file.get(..8).ok_or("file too short")?.try_into().unwrap();
    let header_len = u64::from_le_bytes(len_bytes) as usize;
    let data_start = 8usize
        .checked_add(header_len)
        .filter(|&end| end <= file.len())
        .ok_or("header length past end of file")?;
    let header: Value =
        serde_json::from_slice(&file[8..data_start]).map_err(|e| format!("bad header: {e}"))?;
    let header = header.as_object().ok_or("header is not an object")?;

    let mut entries = Vec::new();
    for (name, info) in header {
        if name == "__metadata__" {
            continue;
        }
        let dtype_name = info["dtype"].as_str().unwrap_or("");
        let dtype = Dtype::parse(dtype_name).ok_or(format!("{name}: dtype {dtype_name:?}"))?;
        let shape = info["shape"]
            .as_array()
            .and_then(|a| {
                a.iter()
                    .map(|d| d.as_u64().map(|d| d as usize))
                    .collect::<Option<Vec<_>>>()
            })
            .ok_or(format!("{name}: bad shape"))?;
        let offsets = info["data_offsets"]
            .as_array()
            .and_then(|a| Some([a.first()?.as_u64()? as usize, a.get(1)?.as_u64()? as usize]))
            .ok_or(format!("{name}: bad data_offsets"))?;
        let start = data_start + offsets[0];
        let end = data_start + offsets[1];
        let want = shape.iter().product::<usize>() * dtype.size();
        if offsets[0] > offsets[1] || end > file.len() || end - start != want {
            return Err(format!("{name}: data range does not fit shape {shape:?}"));
        }
        entries.push((
            name.clone(),
            Entry {
                dtype,
                shape,
                shard,
                start,
                end,
            },
        ));
    }
    Ok(entries)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    /// Writes a safetensors file with the given header and data, the way MLX
    /// does: no padding after the header.
    fn write_shard(dir: &Path, name: &str, header: &str, data: &[u8]) {
        let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(data);
        fs::write(dir.join(name), bytes).unwrap();
    }

    fn temp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("saetta-{name}-{}", std::process::id()));
        fs::create_dir_all(&dir).unwrap();
        dir
    }

    #[test]
    fn reads_small_shards() {
        let dir = temp_dir("weights");
        // 1.0, -2.0 and 0.5 in bf16
        let norm = [0x80, 0x3f, 0x00, 0xc0, 0x00, 0x3f];
        write_shard(
            &dir,
            "a.safetensors",
            r#"{"__metadata__":{"format":"mlx"},"n.weight":{"dtype":"BF16","shape":[3],"data_offsets":[0,6]}}"#,
            &norm,
        );
        // one row of 64 5-bit inputs: 10 words, one group
        let mut data = Vec::new();
        for i in 0..10u32 {
            data.extend_from_slice(&(i * 0x0101_0101).to_le_bytes());
        }
        data.extend_from_slice(&[0x80, 0x3f, 0x00, 0x3f]);
        write_shard(
            &dir,
            "b.safetensors",
            r#"{"m.weight":{"dtype":"U32","shape":[1,10],"data_offsets":[0,40]},
                "m.scales":{"dtype":"BF16","shape":[1,1],"data_offsets":[40,42]},
                "m.biases":{"dtype":"BF16","shape":[1,1],"data_offsets":[42,44]}}"#,
            &data,
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"n.weight":"a.safetensors","m.weight":"b.safetensors",
                "m.scales":"b.safetensors","m.biases":"b.safetensors"}}"#,
        )
        .unwrap();

        let w = Weights::open(&dir).unwrap();
        assert_eq!(w.len(), 4);
        assert_eq!(w.shard_count(), 2);
        let n = w.tensor("n.weight").unwrap();
        assert_eq!(n.shape, &[3]);
        assert_eq!(n.to_f32(), vec![1.0, -2.0, 0.5]);

        let quant = Quantization {
            default: QuantParams {
                bits: 5,
                group_size: 64,
            },
            overrides: HashMap::new(),
        };
        let m = w.quantized("m", &quant).unwrap();
        assert_eq!((m.out_features(), m.in_features()), (1, 64));
        assert_eq!(m.weight.to_u32()[3], 0x0303_0303);
        assert_eq!(m.scales.to_f32(), vec![1.0]);
        assert_eq!(m.biases.to_f32(), vec![0.5]);

        let eight = Quantization {
            default: QuantParams {
                bits: 8,
                group_size: 64,
            },
            overrides: HashMap::new(),
        };
        assert!(w.quantized("m", &eight).is_err());
        assert!(w.tensor("missing").is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    #[test]
    fn rejects_bad_range() {
        let dir = temp_dir("bad-range");
        write_shard(
            &dir,
            "a.safetensors",
            r#"{"x":{"dtype":"F32","shape":[4],"data_offsets":[0,8]}}"#,
            &[0; 8],
        );
        fs::write(
            dir.join("model.safetensors.index.json"),
            r#"{"weight_map":{"x":"a.safetensors"}}"#,
        )
        .unwrap();
        assert!(Weights::open(&dir).is_err());
        fs::remove_dir_all(&dir).unwrap();
    }

    /// Checks against values read with `mlx.core.load` on the Mac. Skipped
    /// when the model is not there.
    #[test]
    fn reads_real_model() {
        let Some(home) = std::env::var_os("HOME") else {
            return;
        };
        let dir = Path::new(&home).join("llm/mlx-models/Qwen3.6-35B-A3B-5bit");
        if !dir.exists() {
            eprintln!("skipped: no model at {}", dir.display());
            return;
        }
        let config = crate::config::Config::load(&dir).unwrap();
        let w = Weights::open(&dir).unwrap();
        assert_eq!(w.len(), 2090);

        let l0 = "language_model.model.layers.0";
        let norm = w.tensor(&format!("{l0}.input_layernorm.weight")).unwrap();
        assert_eq!(norm.shape, &[2048]);
        assert_eq!(
            norm.to_f32()[..4],
            [1.1015625, 1.0390625, 1.046875, 0.9609375]
        );
        let a_log = w.tensor(&format!("{l0}.linear_attn.A_log")).unwrap();
        assert_eq!(a_log.to_f32()[..4], [-3.3125, -3.765625, -3.46875, 3.53125]);
        let final_norm = w.tensor("language_model.model.norm.weight").unwrap();
        assert_eq!(final_norm.shard, 4);
        assert_eq!(
            final_norm.to_f32()[..4],
            [2.734375, 2.765625, 2.828125, 2.78125]
        );

        let q = &config.quantization;
        let gate_proj = w
            .quantized(&format!("{l0}.mlp.switch_mlp.gate_proj"), q)
            .unwrap();
        assert_eq!(gate_proj.weight.shape, &[256, 512, 320]);
        assert_eq!(
            (gate_proj.out_features(), gate_proj.in_features()),
            (512, 2048)
        );
        let bytes = gate_proj.weight.bytes;
        let word = |i: usize| u32::from_le_bytes(bytes[i * 4..i * 4 + 4].try_into().unwrap());
        assert_eq!(
            [word(0), word(1), word(2)],
            [2638646639, 3669881973, 1043174490]
        );
        assert_eq!(word(gate_proj.weight.len() - 1), 2234208214);

        let router = w.quantized(&format!("{l0}.mlp.gate"), q).unwrap();
        assert_eq!(router.params.bits, 8);
        assert_eq!((router.out_features(), router.in_features()), (256, 2048));

        let down = w
            .quantized(&format!("{l0}.mlp.switch_mlp.down_proj"), q)
            .unwrap();
        assert_eq!((down.out_features(), down.in_features()), (2048, 512));
    }
}
