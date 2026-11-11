//! The decoder layers on the CPU with real weights. Each quantized matrix is
//! dequantized to f32 when a layer is loaded; the experts are dequantized per
//! token, only the ones the router picks.

use half::bf16;

use super::attention::{AttentionConfig, AttentionWeights, KvCache, attention};
use super::deltanet::{DeltaNetConfig, DeltaNetState, DeltaNetWeights, deltanet};
use super::moe::{Mlp, mix, mlp, route};
use super::ops::{matvec, rms_norm};
use crate::config::{Config, LayerType};
use crate::quant;
use crate::weights::{QuantTensor, Weights};

const PREFIX: &str = "language_model.model";

/// What a layer keeps between tokens.
pub enum LayerState {
    Linear(DeltaNetState),
    Full(KvCache),
}

pub struct Model<'a> {
    cfg: &'a Config,
    weights: &'a Weights,
}

impl<'a> Model<'a> {
    pub fn new(cfg: &'a Config, weights: &'a Weights) -> Model<'a> {
        assert!(cfg.attn_output_gate, "only the gated attention is written");
        Model { cfg, weights }
    }

    pub fn embed(&self, token: u32) -> Result<Vec<f32>, String> {
        let q = self.quantized(&format!("{PREFIX}.embed_tokens"))?;
        Ok(dequantize_rows(&q, token as usize, 1))
    }

    pub fn layer(&self, index: usize) -> Result<Layer<'a>, String> {
        Layer::load(self.cfg, self.weights, index)
    }

    pub fn new_state(&self, index: usize) -> LayerState {
        match self.cfg.layer_types[index] {
            LayerType::LinearAttention => {
                LayerState::Linear(DeltaNetState::new(&deltanet_config(self.cfg)))
            }
            LayerType::FullAttention => LayerState::Full(KvCache::default()),
        }
    }

    fn quantized(&self, module: &str) -> Result<QuantTensor<'a>, String> {
        self.weights.quantized(module, &self.cfg.quantization)
    }
}

enum Mixer {
    Linear(DeltaNetConfig, Vec<Vec<f32>>),
    Full(AttentionConfig, Vec<Vec<f32>>),
}

/// One decoder layer with its dense weights in memory.
pub struct Layer<'a> {
    cfg: &'a Config,
    weights: &'a Weights,
    prefix: String,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer,
    router: Vec<f32>,
    shared_expert: [Vec<f32>; 3],
    shared_expert_gate: Vec<f32>,
}

/// What one layer step returns: the hidden state and the router choice.
pub struct LayerOutput {
    pub hidden: Vec<f32>,
    /// `(expert, weight)`, best first.
    pub experts: Vec<(usize, f32)>,
}

impl<'a> Layer<'a> {
    fn load(cfg: &'a Config, weights: &'a Weights, index: usize) -> Result<Layer<'a>, String> {
        let prefix = format!("{PREFIX}.layers.{index}");
        let matrix = |name: &str| -> Result<Vec<f32>, String> {
            let q = weights.quantized(&format!("{prefix}.{name}"), &cfg.quantization)?;
            Ok(dequantize_rows(&q, 0, q.out_features()))
        };
        let vector = |name: &str| -> Result<Vec<f32>, String> {
            Ok(weights.tensor(&format!("{prefix}.{name}"))?.to_f32())
        };

        let mixer = match cfg.layer_types[index] {
            LayerType::LinearAttention => {
                let names = [
                    "in_proj_qkv",
                    "in_proj_z",
                    "in_proj_b",
                    "in_proj_a",
                    "out_proj",
                ];
                let mut t = names
                    .iter()
                    .map(|n| matrix(&format!("linear_attn.{n}")))
                    .collect::<Result<Vec<_>, _>>()?;
                for n in ["conv1d.weight", "A_log", "dt_bias", "norm.weight"] {
                    t.push(vector(&format!("linear_attn.{n}"))?);
                }
                Mixer::Linear(deltanet_config(cfg), t)
            }
            LayerType::FullAttention => {
                let mut t = ["q_proj", "k_proj", "v_proj", "o_proj"]
                    .iter()
                    .map(|n| matrix(&format!("self_attn.{n}")))
                    .collect::<Result<Vec<_>, _>>()?;
                t.push(vector("self_attn.q_norm.weight")?);
                t.push(vector("self_attn.k_norm.weight")?);
                Mixer::Full(attention_config(cfg), t)
            }
        };

        Ok(Layer {
            cfg,
            weights,
            input_norm: vector("input_layernorm.weight")?,
            post_norm: vector("post_attention_layernorm.weight")?,
            mixer,
            router: matrix("mlp.gate")?,
            shared_expert: [
                matrix("mlp.shared_expert.gate_proj")?,
                matrix("mlp.shared_expert.up_proj")?,
                matrix("mlp.shared_expert.down_proj")?,
            ],
            shared_expert_gate: matrix("mlp.shared_expert_gate")?,
            prefix,
        })
    }

    /// `h = x + mixer(norm(x))`, then `h + moe(norm(h))`.
    pub fn step(&self, x: &[f32], state: &mut LayerState) -> Result<LayerOutput, String> {
        let eps = self.cfg.rms_norm_eps;
        let mut normed = x.to_vec();
        rms_norm(&mut normed, &self.input_norm, eps);
        let mixed = match (&self.mixer, state) {
            (Mixer::Linear(c, t), LayerState::Linear(s)) => {
                let w = DeltaNetWeights {
                    in_proj_qkv: &t[0],
                    in_proj_z: &t[1],
                    in_proj_b: &t[2],
                    in_proj_a: &t[3],
                    out_proj: &t[4],
                    conv1d: &t[5],
                    a_log: &t[6],
                    dt_bias: &t[7],
                    norm: &t[8],
                };
                deltanet(c, &w, &normed, s)
            }
            (Mixer::Full(c, t), LayerState::Full(cache)) => {
                let w = AttentionWeights {
                    q_proj: &t[0],
                    k_proj: &t[1],
                    v_proj: &t[2],
                    o_proj: &t[3],
                    q_norm: &t[4],
                    k_norm: &t[5],
                };
                attention(c, &w, &normed, cache)
            }
            _ => {
                return Err(format!(
                    "{}: state does not match the layer type",
                    self.prefix
                ));
            }
        };
        let h: Vec<f32> = x.iter().zip(&mixed).map(|(a, b)| a + b).collect();

        let mut m = h.clone();
        rms_norm(&mut m, &self.post_norm, eps);
        let experts = route(&self.router, &m, self.cfg.num_experts_per_tok);
        let [gate_proj, up_proj, down_proj] = &self.shared_expert;
        let shared = mlp(
            &Mlp {
                gate_proj,
                up_proj,
                down_proj,
            },
            &m,
        );
        let shared_logit = matvec(&self.shared_expert_gate, &m)[0];

        let mut error = None;
        let moe = mix(
            &experts,
            |e| match self.expert(e) {
                Ok(w) => mlp(&w.as_mlp(), &m),
                Err(err) => {
                    error = Some(err);
                    vec![0.0; m.len()]
                }
            },
            &shared,
            shared_logit,
        );
        if let Some(err) = error {
            return Err(err);
        }

        let hidden = h.iter().zip(&moe).map(|(a, b)| a + b).collect();
        Ok(LayerOutput { hidden, experts })
    }

    /// Dequantizes the three matrices of expert `e` from `switch_mlp`.
    fn expert(&self, e: usize) -> Result<Expert, String> {
        let get = |name: &str| -> Result<Vec<f32>, String> {
            let module = format!("{}.mlp.switch_mlp.{name}", self.prefix);
            let q = self.weights.quantized(&module, &self.cfg.quantization)?;
            let rows = q.out_features();
            Ok(dequantize_rows(&q, e * rows, rows))
        };
        Ok(Expert(
            get("gate_proj")?,
            get("up_proj")?,
            get("down_proj")?,
        ))
    }
}

struct Expert(Vec<f32>, Vec<f32>, Vec<f32>);

impl Expert {
    fn as_mlp(&self) -> Mlp<'_> {
        Mlp {
            gate_proj: &self.0,
            up_proj: &self.1,
            down_proj: &self.2,
        }
    }
}

fn deltanet_config(cfg: &Config) -> DeltaNetConfig {
    DeltaNetConfig {
        k_heads: cfg.linear_num_key_heads,
        v_heads: cfg.linear_num_value_heads,
        k_dim: cfg.linear_key_head_dim,
        v_dim: cfg.linear_value_head_dim,
        kernel: cfg.linear_conv_kernel_dim,
        eps: cfg.rms_norm_eps,
    }
}

fn attention_config(cfg: &Config) -> AttentionConfig {
    AttentionConfig {
        heads: cfg.num_attention_heads,
        kv_heads: cfg.num_key_value_heads,
        head_dim: cfg.head_dim,
        rope_dims: (cfg.head_dim as f32 * cfg.partial_rotary_factor) as usize,
        rope_theta: cfg.rope_theta as f32,
        eps: cfg.rms_norm_eps,
    }
}

/// Dequantizes `count` rows starting at row `first`, counting rows over all
/// leading dimensions (for stacked experts, expert `e` starts at `e * out`).
/// Reads only the bytes of those rows, so it works on unaligned tensors too.
fn dequantize_rows(q: &QuantTensor, first: usize, count: usize) -> Vec<f32> {
    let cols = q.in_features();
    let row_words = quant::words_per_row(cols, q.params.bits);
    let groups = cols / q.params.group_size;
    let words: Vec<u32> = q.weight.bytes[first * row_words * 4..(first + count) * row_words * 4]
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| u32::from_le_bytes(*b))
        .collect();
    let half = |bytes: &[u8]| -> Vec<bf16> {
        bytes[first * groups * 2..(first + count) * groups * 2]
            .as_chunks::<2>()
            .0
            .iter()
            .map(|b| bf16::from_le_bytes(*b))
            .collect()
    };
    let scales = half(q.scales.bytes);
    let biases = half(q.biases.bytes);
    quant::dequantize(&words, &scales, &biases, q.params.bits, count, cols)
}

#[cfg(test)]
mod golden {
    //! Layers 0 to 3 against the MLX data of `tools/golden.py`. Needs the
    //! model and the golden files, so it runs on the Mac only:
    //!   scripts/mac.sh test --release --lib golden -- --ignored --nocapture

    use super::*;
    use serde_json::Value;
    use std::path::{Path, PathBuf};

    const MODEL_DIR: &str = "llm/saetta-models/Qwen3.6-35B-A3B-5bit";
    /// Bounds on the relative RMS error against one golden run.
    struct Bounds {
        dir: &'static str,
        embed: f32,
        hidden: [f32; 4],
        state: f32,
        kv: f32,
        /// Two router weights closer than this count as a tie.
        tie: f32,
    }

    /// MLX in float32: only the summation order differs, and the errors
    /// measured are about 1e-6.
    const F32: Bounds = Bounds {
        dir: "saetta/golden-f32",
        embed: 1e-6,
        hidden: [1e-5; 4],
        state: 1e-5,
        kv: 1e-5,
        tie: 1e-5,
    };

    /// MLX in bf16, as the server runs it. Two bf16 MLX runs (prefill and
    /// one token at a time) differ by 3e-4, 3.2e-3, 3.6e-3 and 1.6e-2 after
    /// layers 0 to 3; we measured 4.6e-3, 5.3e-3, 4.6e-3 and 1.4e-2 against
    /// f32, and router weights within 5.6e-3. The bounds are about twice that.
    const BF16: Bounds = Bounds {
        dir: "saetta/golden",
        embed: 3e-3,
        hidden: [0.01, 0.012, 0.012, 0.03],
        state: 0.012,
        kv: 0.015,
        tie: 0.012,
    };

    fn home() -> PathBuf {
        PathBuf::from(std::env::var("HOME").unwrap())
    }

    struct Golden {
        dir: PathBuf,
        index: Value,
    }

    impl Golden {
        fn open(dir: &str) -> Golden {
            let dir = home().join(dir);
            let text = std::fs::read_to_string(dir.join("index.json")).unwrap();
            Golden {
                dir,
                index: serde_json::from_str(&text).unwrap(),
            }
        }

        fn bytes(&self, name: &str, dtype: &str) -> Vec<u8> {
            let entry = &self.index["files"][name];
            assert_eq!(entry["dtype"], dtype, "{name}");
            std::fs::read(self.dir.join(format!("{name}.bin"))).unwrap()
        }

        fn f32(&self, name: &str) -> Vec<f32> {
            let b = self.bytes(name, "float32");
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| f32::from_le_bytes(*w))
                .collect()
        }

        fn u32(&self, name: &str) -> Vec<u32> {
            let b = self.bytes(name, "uint32");
            b.as_chunks::<4>()
                .0
                .iter()
                .map(|w| u32::from_le_bytes(*w))
                .collect()
        }
    }

    fn rel_rms(got: &[f32], want: &[f32]) -> f32 {
        assert_eq!(got.len(), want.len());
        let err: f32 = got.iter().zip(want).map(|(g, w)| (g - w) * (g - w)).sum();
        let norm: f32 = want.iter().map(|w| w * w).sum();
        (err / norm).sqrt()
    }

    /// The expert sets must be equal, except that experts tied with the
    /// last chosen one (within `tie`) may be swapped. Returns the number of
    /// swaps and the largest weight difference of the common experts.
    fn check_experts(
        what: &str,
        tie: f32,
        got: &[(usize, f32)],
        want_ids: &[u32],
        want_w: &[f32],
    ) -> (usize, f32) {
        let last = want_w[want_w.len() - 1];
        let (mut swaps, mut max_diff) = (0, 0.0f32);
        for &(e, w) in got {
            if let Some(i) = want_ids.iter().position(|&id| id as usize == e) {
                max_diff = max_diff.max((w - want_w[i]).abs());
            } else {
                swaps += 1;
                assert!(
                    (w - last).abs() < tie,
                    "{what}: chose expert {e} ({w}), MLX did not"
                );
            }
        }
        for (&id, &w) in want_ids.iter().zip(want_w) {
            if !got.iter().any(|&(e, _)| e == id as usize) {
                assert!((w - last).abs() < tie, "{what}: missed expert {id} ({w})");
            }
        }
        assert!(
            max_diff < tie,
            "{what}: expert weights differ by {max_diff}"
        );
        (swaps, max_diff)
    }

    #[test]
    #[ignore = "needs the model and golden data on the Mac"]
    fn layers_0_to_3_match_mlx_f32() {
        check(&F32);
    }

    #[test]
    #[ignore = "needs the model and golden data on the Mac"]
    fn layers_0_to_3_match_mlx_bf16() {
        check(&BF16);
    }

    fn check(b: &Bounds) {
        println!("against {}", b.dir);
        let model_dir = home().join(MODEL_DIR);
        let cfg = Config::load(Path::new(&model_dir)).unwrap();
        let weights = Weights::open(&model_dir).unwrap();
        let model = Model::new(&cfg, &weights);
        let g = Golden::open(b.dir);
        let tokens = g.u32("tokens");
        let hidden = cfg.hidden_size;
        let top_k = cfg.num_experts_per_tok;

        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| model.embed(t).unwrap()).collect();
        let err = rel_rms(&xs.concat(), &g.f32("embed"));
        println!("embed: rel rms {err:.2e}");
        assert!(err < b.embed, "embed rel rms {err}");

        let mut states = Vec::new();
        for (l, bound) in b.hidden.iter().enumerate() {
            let layer = model.layer(l).unwrap();
            let mut state = model.new_state(l);
            let (want_ids, want_w) = (
                g.u32(&format!("layer{l}_experts")),
                g.f32(&format!("layer{l}_expert_weights")),
            );
            let (mut swaps, mut max_diff) = (0, 0.0f32);
            for (t, x) in xs.iter_mut().enumerate() {
                let out = layer.step(x, &mut state).unwrap();
                let range = t * top_k..(t + 1) * top_k;
                let (s, d) = check_experts(
                    &format!("layer {l} token {t}"),
                    b.tie,
                    &out.experts,
                    &want_ids[range.clone()],
                    &want_w[range],
                );
                swaps += s;
                max_diff = max_diff.max(d);
                *x = out.hidden;
            }
            let err = rel_rms(&xs.concat(), &g.f32(&format!("layer{l}")));
            println!(
                "layer {l}: rel rms {err:.2e} (bound {bound}), expert swaps {swaps}, router weight diff {max_diff:.2e}"
            );
            assert!(err < *bound, "layer {l} rel rms {err}");
            assert_eq!(xs[0].len(), hidden);
            states.push(state);
        }

        let LayerState::Linear(s) = &states[0] else {
            panic!("layer 0 is not linear")
        };
        let conv = rel_rms(&s.conv, &g.f32("layer0_conv_state"));
        let recurrent = rel_rms(&s.recurrent, &g.f32("layer0_recurrent_state"));
        println!("layer 0 conv state: rel rms {conv:.2e}, recurrent state: {recurrent:.2e}");
        assert!(conv < b.state && recurrent < b.state);

        let LayerState::Full(kv) = &states[3] else {
            panic!("layer 3 is not full attention")
        };
        let keys = rel_rms(kv.keys(), &g.f32("layer3_keys"));
        let values = rel_rms(kv.values(), &g.f32("layer3_values"));
        println!("layer 3 keys: rel rms {keys:.2e}, values: {values:.2e}");
        assert!(keys < b.kv && values < b.kv);
    }
}
