//! Model settings from `config.json`.
//!
//! The text model settings are under `text_config`. The quantization block
//! is at the top level and has a default plus per-module overrides, such as
//! the 8-bit routers.

use serde_json::Value;
use std::collections::HashMap;
use std::error::Error;
use std::fs;
use std::path::Path;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    LinearAttention,
    FullAttention,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct QuantParams {
    pub bits: u32,
    pub group_size: usize,
}

#[derive(Debug, Clone)]
pub struct Quantization {
    pub default: QuantParams,
    /// Keyed by module name, for example `language_model.model.layers.0.mlp.gate`.
    pub overrides: HashMap<String, QuantParams>,
}

impl Quantization {
    pub fn for_module(&self, module: &str) -> QuantParams {
        self.overrides.get(module).copied().unwrap_or(self.default)
    }
}

#[derive(Debug, Clone)]
pub struct Config {
    pub hidden_size: usize,
    pub vocab_size: usize,
    pub rms_norm_eps: f32,
    pub tie_word_embeddings: bool,
    pub eos_token_ids: Vec<u32>,
    pub layer_types: Vec<LayerType>,

    // full attention
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub attn_output_gate: bool,
    pub partial_rotary_factor: f32,
    pub rope_theta: f64,

    // Gated DeltaNet
    pub linear_num_key_heads: usize,
    pub linear_num_value_heads: usize,
    pub linear_key_head_dim: usize,
    pub linear_value_head_dim: usize,
    pub linear_conv_kernel_dim: usize,

    // MoE
    pub num_experts: usize,
    pub num_experts_per_tok: usize,
    pub moe_intermediate_size: usize,
    pub shared_expert_intermediate_size: usize,

    pub quantization: Quantization,
}

impl Config {
    pub fn load(model_dir: &Path) -> Result<Config, Box<dyn Error>> {
        let path = model_dir.join("config.json");
        let text = fs::read_to_string(&path).map_err(|e| format!("{}: {e}", path.display()))?;
        Config::parse(&text)
    }

    pub fn parse(text: &str) -> Result<Config, Box<dyn Error>> {
        let root: Value = serde_json::from_str(text)?;
        let t = &root["text_config"];
        if !t.is_object() {
            return Err("config.json has no text_config".into());
        }

        let layer_types = t["layer_types"]
            .as_array()
            .ok_or("text_config.layer_types missing")?
            .iter()
            .map(|v| match v.as_str() {
                Some("linear_attention") => Ok(LayerType::LinearAttention),
                Some("full_attention") => Ok(LayerType::FullAttention),
                other => Err(format!("unknown layer type {other:?}")),
            })
            .collect::<Result<Vec<_>, _>>()?;
        if layer_types.len() != usize_field(t, "num_hidden_layers")? {
            return Err("layer_types length differs from num_hidden_layers".into());
        }

        // The top-level eos list has both end tokens; text_config has only one.
        let eos = if root["eos_token_id"].is_null() {
            &t["eos_token_id"]
        } else {
            &root["eos_token_id"]
        };
        let eos_token_ids = match eos {
            Value::Array(ids) => ids
                .iter()
                .map(|v| v.as_u64().map(|n| n as u32))
                .collect::<Option<Vec<_>>>()
                .ok_or("bad eos_token_id")?,
            v => vec![v.as_u64().ok_or("bad eos_token_id")? as u32],
        };

        let rope = &t["rope_parameters"];
        Ok(Config {
            hidden_size: usize_field(t, "hidden_size")?,
            vocab_size: usize_field(t, "vocab_size")?,
            rms_norm_eps: f64_field(t, "rms_norm_eps")? as f32,
            tie_word_embeddings: t["tie_word_embeddings"].as_bool().unwrap_or(false),
            eos_token_ids,
            layer_types,
            num_attention_heads: usize_field(t, "num_attention_heads")?,
            num_key_value_heads: usize_field(t, "num_key_value_heads")?,
            head_dim: usize_field(t, "head_dim")?,
            attn_output_gate: t["attn_output_gate"].as_bool().unwrap_or(false),
            partial_rotary_factor: f64_field(rope, "partial_rotary_factor")? as f32,
            rope_theta: f64_field(rope, "rope_theta")?,
            linear_num_key_heads: usize_field(t, "linear_num_key_heads")?,
            linear_num_value_heads: usize_field(t, "linear_num_value_heads")?,
            linear_key_head_dim: usize_field(t, "linear_key_head_dim")?,
            linear_value_head_dim: usize_field(t, "linear_value_head_dim")?,
            linear_conv_kernel_dim: usize_field(t, "linear_conv_kernel_dim")?,
            num_experts: usize_field(t, "num_experts")?,
            num_experts_per_tok: usize_field(t, "num_experts_per_tok")?,
            moe_intermediate_size: usize_field(t, "moe_intermediate_size")?,
            shared_expert_intermediate_size: usize_field(t, "shared_expert_intermediate_size")?,
            quantization: parse_quantization(&root["quantization"])?,
        })
    }

    pub fn num_layers(&self) -> usize {
        self.layer_types.len()
    }
}

fn parse_quantization(q: &Value) -> Result<Quantization, Box<dyn Error>> {
    let obj = q
        .as_object()
        .ok_or("config.json has no quantization block")?;
    if let Some(mode) = obj.get("mode").and_then(Value::as_str)
        && mode != "affine"
    {
        return Err(format!("quantization mode {mode} is not supported").into());
    }
    let default = quant_params(q)?;
    let mut overrides = HashMap::new();
    for (name, value) in obj {
        if value.is_object() {
            overrides.insert(name.clone(), quant_params(value)?);
        }
    }
    Ok(Quantization { default, overrides })
}

fn quant_params(v: &Value) -> Result<QuantParams, Box<dyn Error>> {
    Ok(QuantParams {
        bits: usize_field(v, "bits")? as u32,
        group_size: usize_field(v, "group_size")?,
    })
}

fn usize_field(v: &Value, key: &str) -> Result<usize, String> {
    v[key]
        .as_u64()
        .map(|n| n as usize)
        .ok_or_else(|| format!("config field {key} missing or not an integer"))
}

fn f64_field(v: &Value, key: &str) -> Result<f64, String> {
    v[key]
        .as_f64()
        .ok_or_else(|| format!("config field {key} missing or not a number"))
}

#[cfg(test)]
mod tests {
    use super::*;

    const SMALL: &str = r#"{
        "eos_token_id": [248046, 248044],
        "quantization": {
            "group_size": 64, "bits": 5, "mode": "affine",
            "language_model.model.layers.0.mlp.gate": {"group_size": 64, "bits": 8}
        },
        "text_config": {
            "hidden_size": 2048, "vocab_size": 248320, "rms_norm_eps": 1e-6,
            "tie_word_embeddings": false, "eos_token_id": 248044,
            "num_hidden_layers": 4,
            "layer_types": ["linear_attention", "linear_attention", "linear_attention", "full_attention"],
            "num_attention_heads": 16, "num_key_value_heads": 2, "head_dim": 256,
            "attn_output_gate": true,
            "rope_parameters": {"partial_rotary_factor": 0.25, "rope_theta": 10000000},
            "linear_num_key_heads": 16, "linear_num_value_heads": 32,
            "linear_key_head_dim": 128, "linear_value_head_dim": 128,
            "linear_conv_kernel_dim": 4,
            "num_experts": 256, "num_experts_per_tok": 8,
            "moe_intermediate_size": 512, "shared_expert_intermediate_size": 512
        }
    }"#;

    #[test]
    fn parses_small_config() {
        let c = Config::parse(SMALL).unwrap();
        assert_eq!(c.num_layers(), 4);
        assert_eq!(c.layer_types[3], LayerType::FullAttention);
        assert_eq!(c.eos_token_ids, vec![248046, 248044]);
        assert_eq!(c.rope_theta, 1e7);
        assert_eq!(c.rms_norm_eps, 1e-6);
        let q = &c.quantization;
        assert_eq!(
            q.for_module("language_model.model.layers.0.mlp.gate").bits,
            8
        );
        assert_eq!(q.for_module("language_model.model.layers.0.mlp.up").bits, 5);
        assert_eq!(q.default.group_size, 64);
    }

    #[test]
    fn rejects_wrong_layer_count() {
        let text = SMALL.replace("\"num_hidden_layers\": 4", "\"num_hidden_layers\": 40");
        assert!(Config::parse(&text).is_err());
    }
}
