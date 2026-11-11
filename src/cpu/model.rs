//! The whole model on the CPU with real weights. The big matrices stay
//! quantized and are multiplied from the packed words (`Matrix::Quant`);
//! norms and the other small vectors are copied to f32 when the model loads.

use half::bf16;

use super::attention::{AttentionConfig, AttentionWeights, KvCache, attention_at};
use super::deltanet::{DeltaNetConfig, DeltaNetState, DeltaNetWeights, deltanet};
use super::matrix::Matrix;
use super::moe::{Mlp, expert, mix, mlp, route};
use super::ops::rms_norm;
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
    embed: QuantTensor<'a>,
    layers: Vec<Layer<'a>>,
    norm: Vec<f32>,
    lm_head: Matrix<'a>,
}

impl<'a> Model<'a> {
    pub fn new(cfg: &'a Config, weights: &'a Weights) -> Result<Model<'a>, String> {
        if !cfg.attn_output_gate || cfg.tie_word_embeddings {
            return Err("only the gated attention and an untied lm_head are written".into());
        }
        let quantized = |m: &str| weights.quantized(m, &cfg.quantization);
        Ok(Model {
            cfg,
            embed: quantized(&format!("{PREFIX}.embed_tokens"))?,
            layers: (0..cfg.num_layers())
                .map(|i| Layer::load(cfg, weights, i))
                .collect::<Result<_, _>>()?,
            norm: weights.tensor(&format!("{PREFIX}.norm.weight"))?.to_f32(),
            lm_head: Matrix::quant(quantized("language_model.lm_head")?),
        })
    }

    pub fn embed(&self, token: u32) -> Vec<f32> {
        dequantize_rows(&self.embed, token as usize, 1)
    }

    pub fn layer(&self, index: usize) -> &Layer<'a> {
        &self.layers[index]
    }

    pub fn new_state(&self, index: usize) -> LayerState {
        match self.cfg.layer_types[index] {
            LayerType::LinearAttention => {
                LayerState::Linear(DeltaNetState::new(&deltanet_config(self.cfg)))
            }
            LayerType::FullAttention => LayerState::Full(KvCache::default()),
        }
    }

    /// A fresh state for every layer.
    pub fn new_states(&self) -> Vec<LayerState> {
        (0..self.layers.len()).map(|i| self.new_state(i)).collect()
    }

    /// Runs one token through all layers and returns the logits.
    pub fn forward(&self, token: u32, states: &mut [LayerState]) -> Result<Vec<f32>, String> {
        Ok(self.logits(&self.forward_hidden(token, states)?))
    }

    /// Runs one token through all layers and returns the hidden state after
    /// the final norm.
    pub fn forward_hidden(
        &self,
        token: u32,
        states: &mut [LayerState],
    ) -> Result<Vec<f32>, String> {
        let mut x = self.embed(token);
        for (layer, state) in self.layers.iter().zip(states) {
            x = layer.step(&x, state)?.hidden;
        }
        rms_norm(&mut x, &self.norm, self.cfg.rms_norm_eps);
        Ok(x)
    }

    /// `lm_head` on a hidden state that is already normed.
    pub fn logits(&self, normed: &[f32]) -> Vec<f32> {
        self.lm_head.matvec(normed)
    }

    /// Greedy decoding: feeds `prompt`, then picks the most likely token
    /// `count` times. Calls `on_token` with each new token and its logits.
    pub fn generate(
        &self,
        prompt: &[u32],
        count: usize,
        mut on_token: impl FnMut(u32, &[f32]),
    ) -> Result<Vec<u32>, String> {
        let mut states = self.new_states();
        let mut logits = Vec::new();
        for &t in prompt {
            logits = self.forward(t, &mut states)?;
        }
        let mut out = Vec::with_capacity(count);
        for i in 0..count {
            let next = argmax(&logits);
            on_token(next, &logits);
            out.push(next);
            if i + 1 < count {
                logits = self.forward(next, &mut states)?;
            }
        }
        Ok(out)
    }
}

pub fn argmax(x: &[f32]) -> u32 {
    let mut best = 0;
    for (i, v) in x.iter().enumerate() {
        if *v > x[best] {
            best = i;
        }
    }
    best as u32
}

enum Mixer<'a> {
    Linear {
        cfg: DeltaNetConfig,
        /// in_proj_qkv, in_proj_z, in_proj_b, in_proj_a, out_proj
        matrices: Box<[Matrix<'a>; 5]>,
        /// conv1d, A_log, dt_bias, norm
        vectors: [Vec<f32>; 4],
    },
    Full {
        cfg: AttentionConfig,
        /// q_proj, k_proj, v_proj, o_proj
        matrices: Box<[Matrix<'a>; 4]>,
        /// q_norm, k_norm
        vectors: [Vec<f32>; 2],
    },
}

/// One decoder layer.
pub struct Layer<'a> {
    cfg: &'a Config,
    prefix: String,
    input_norm: Vec<f32>,
    post_norm: Vec<f32>,
    mixer: Mixer<'a>,
    router: Matrix<'a>,
    /// All experts stacked, `[experts, out, in]`.
    experts: Mlp<'a>,
    shared_expert: Mlp<'a>,
    shared_expert_gate: Matrix<'a>,
}

/// What one layer step returns: the hidden state and the router choice.
pub struct LayerOutput {
    pub hidden: Vec<f32>,
    /// `(expert, weight)`, best first.
    pub experts: Vec<(usize, f32)>,
}

impl<'a> Layer<'a> {
    /// The MoE router matrix, for debugging.
    pub fn router(&self) -> &Matrix<'a> {
        &self.router
    }

    fn load(cfg: &'a Config, weights: &'a Weights, index: usize) -> Result<Layer<'a>, String> {
        let prefix = format!("{PREFIX}.layers.{index}");
        Layer::load_at(cfg, weights, &prefix, cfg.layer_types[index])
    }

    /// Loads a layer of type `kind` from the tensors under `prefix`, such as
    /// `language_model.mtp.layers.0`.
    pub fn load_at(
        cfg: &'a Config,
        weights: &'a Weights,
        prefix: &str,
        kind: LayerType,
    ) -> Result<Layer<'a>, String> {
        let prefix = prefix.to_string();
        let matrix = |name: &str| -> Result<Matrix<'a>, String> {
            let module = format!("{prefix}.{name}");
            Ok(Matrix::quant(
                weights.quantized(&module, &cfg.quantization)?,
            ))
        };
        let vector = |name: &str| -> Result<Vec<f32>, String> {
            Ok(weights.tensor(&format!("{prefix}.{name}"))?.to_f32())
        };
        let mlp = |name: &str| -> Result<Mlp<'a>, String> {
            Ok(Mlp {
                gate_proj: matrix(&format!("{name}.gate_proj"))?,
                up_proj: matrix(&format!("{name}.up_proj"))?,
                down_proj: matrix(&format!("{name}.down_proj"))?,
            })
        };

        let mixer = match kind {
            LayerType::LinearAttention => Mixer::Linear {
                cfg: deltanet_config(cfg),
                matrices: Box::new([
                    matrix("linear_attn.in_proj_qkv")?,
                    matrix("linear_attn.in_proj_z")?,
                    matrix("linear_attn.in_proj_b")?,
                    matrix("linear_attn.in_proj_a")?,
                    matrix("linear_attn.out_proj")?,
                ]),
                vectors: [
                    vector("linear_attn.conv1d.weight")?,
                    vector("linear_attn.A_log")?,
                    vector("linear_attn.dt_bias")?,
                    vector("linear_attn.norm.weight")?,
                ],
            },
            LayerType::FullAttention => Mixer::Full {
                cfg: attention_config(cfg),
                matrices: Box::new([
                    matrix("self_attn.q_proj")?,
                    matrix("self_attn.k_proj")?,
                    matrix("self_attn.v_proj")?,
                    matrix("self_attn.o_proj")?,
                ]),
                vectors: [
                    vector("self_attn.q_norm.weight")?,
                    vector("self_attn.k_norm.weight")?,
                ],
            },
        };

        Ok(Layer {
            cfg,
            input_norm: vector("input_layernorm.weight")?,
            post_norm: vector("post_attention_layernorm.weight")?,
            mixer,
            router: matrix("mlp.gate")?,
            experts: mlp("mlp.switch_mlp")?,
            shared_expert: mlp("mlp.shared_expert")?,
            shared_expert_gate: matrix("mlp.shared_expert_gate")?,
            prefix,
        })
    }

    /// `h = x + mixer(norm(x))`, then `h + moe(norm(h))`.
    pub fn step(&self, x: &[f32], state: &mut LayerState) -> Result<LayerOutput, String> {
        self.run(x, state, None)
    }

    /// `step` with the RoPE position of a full attention layer given instead
    /// of taken from its cache length.
    pub fn step_at(
        &self,
        x: &[f32],
        state: &mut LayerState,
        pos: usize,
    ) -> Result<LayerOutput, String> {
        self.run(x, state, Some(pos))
    }

    fn run(
        &self,
        x: &[f32],
        state: &mut LayerState,
        pos: Option<usize>,
    ) -> Result<LayerOutput, String> {
        let eps = self.cfg.rms_norm_eps;
        let mut normed = x.to_vec();
        rms_norm(&mut normed, &self.input_norm, eps);
        let mixed = match (&self.mixer, state) {
            (
                Mixer::Linear {
                    cfg,
                    matrices: m,
                    vectors: v,
                },
                LayerState::Linear(s),
            ) => {
                let w = DeltaNetWeights {
                    in_proj_qkv: m[0],
                    in_proj_z: m[1],
                    in_proj_b: m[2],
                    in_proj_a: m[3],
                    out_proj: m[4],
                    conv1d: &v[0],
                    a_log: &v[1],
                    dt_bias: &v[2],
                    norm: &v[3],
                };
                deltanet(cfg, &w, &normed, s)
            }
            (
                Mixer::Full {
                    cfg,
                    matrices: m,
                    vectors: v,
                },
                LayerState::Full(cache),
            ) => {
                let w = AttentionWeights {
                    q_proj: m[0],
                    k_proj: m[1],
                    v_proj: m[2],
                    o_proj: m[3],
                    q_norm: &v[0],
                    k_norm: &v[1],
                };
                let pos = pos.unwrap_or(cache.len());
                attention_at(cfg, &w, &normed, cache, pos)
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
        let moe = mix(
            &experts,
            |e| mlp(&expert(&self.experts, e, self.cfg.num_experts), &m),
            &mlp(&self.shared_expert, &m),
            self.shared_expert_gate.matvec(&m)[0],
        );

        let hidden = h.iter().zip(&moe).map(|(a, b)| a + b).collect();
        Ok(LayerOutput { hidden, experts })
    }
}

pub fn deltanet_config(cfg: &Config) -> DeltaNetConfig {
    DeltaNetConfig {
        k_heads: cfg.linear_num_key_heads,
        v_heads: cfg.linear_num_value_heads,
        k_dim: cfg.linear_key_head_dim,
        v_dim: cfg.linear_value_head_dim,
        kernel: cfg.linear_conv_kernel_dim,
        eps: cfg.rms_norm_eps,
    }
}

pub fn attention_config(cfg: &Config) -> AttentionConfig {
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
/// leading dimensions. Reads only the bytes of those rows.
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
        let model = Model::new(&cfg, &weights).unwrap();
        let g = Golden::open(b.dir);
        let tokens = g.u32("tokens");
        let hidden = cfg.hidden_size;
        let top_k = cfg.num_experts_per_tok;

        let mut xs: Vec<Vec<f32>> = tokens.iter().map(|&t| model.embed(t)).collect();
        let err = rel_rms(&xs.concat(), &g.f32("embed"));
        println!("embed: rel rms {err:.2e}");
        assert!(err < b.embed, "embed rel rms {err}");

        let mut states = Vec::new();
        for (l, bound) in b.hidden.iter().enumerate() {
            let layer = model.layer(l);
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

    /// Our log-softmax of the logits at `ids`.
    fn logprobs(logits: &[f32], ids: &[u64]) -> Vec<f32> {
        let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let sum: f32 = logits.iter().map(|v| (v - max).exp()).sum();
        ids.iter()
            .map(|&i| logits[i as usize] - max - sum.ln())
            .collect()
    }

    /// Our logits match MLX float32 to about 1.5e-6 at every position, so
    /// the gap to the server is the server's bf16. Its logprobs differ from
    /// ours by up to 0.354 (token 13); above `LOGPROB_FLOOR` only, as bf16 is
    /// too coarse far down the tail.
    const LOGPROB_TOLERANCE: f32 = 0.5;
    const LOGPROB_FLOOR: f32 = -6.0;
    /// A different token than the server's is fine only where the server's
    /// top two are closer than this.
    const NEAR_TIE: f32 = 0.1;
    /// Relative RMS bound of our logits against MLX in float32
    /// (`tools/golden_logits.py`).
    const F32_LOGITS: f32 = 1e-5;

    /// All 40 layers and the logits, with the server's greedy tokens
    /// (`tools/greedy_ref.py`) fed back in so every position is compared even
    /// after a near tie. The logits must match MLX float32 and pick the same
    /// token; against the bf16 server, only near ties may differ. Then
    /// `generate` must give the server's tokens up to the first near tie.
    #[test]
    #[ignore = "needs the model and golden data on the Mac"]
    fn greedy_matches_server() {
        let model_dir = home().join(MODEL_DIR);
        let cfg = Config::load(Path::new(&model_dir)).unwrap();
        let weights = Weights::open(&model_dir).unwrap();
        let model = Model::new(&cfg, &weights).unwrap();
        let g = Golden::open(BF16.dir);
        let prompt = g.u32("tokens");
        let text = std::fs::read_to_string(g.dir.join("greedy.json")).unwrap();
        let reference: Value = serde_json::from_str(&text).unwrap();
        let ids = |v: &Value| -> Vec<u64> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_u64().unwrap())
                .collect()
        };
        let floats = |v: &Value| -> Vec<f32> {
            v.as_array()
                .unwrap()
                .iter()
                .map(|x| x.as_f64().unwrap() as f32)
                .collect()
        };
        let want = ids(&reference["tokens"]);
        assert_eq!(reference["prompt_tokens"], prompt.len());

        let mut states = model.new_states();
        let mut logits = Vec::new();
        for &t in &prompt {
            logits = model.forward(t, &mut states).unwrap();
        }
        let f32_golden = Golden::open(F32.dir);
        let f32_logits = std::fs::read(f32_golden.dir.join("greedy_logits.bin")).unwrap();
        let f32_logits: Vec<f32> = f32_logits
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| f32::from_le_bytes(*b))
            .collect();
        let vocab = cfg.vocab_size;
        assert_eq!(f32_logits.len(), want.len() * vocab);

        let mut first_tie = want.len();
        let mut elapsed = std::time::Duration::ZERO;
        for (i, &theirs) in want.iter().enumerate() {
            let top_ids = ids(&reference["top_ids"][i]);
            let server = floats(&reference["top_logprobs"][i]);
            let mine = logprobs(&logits, &top_ids);
            let diff = mine
                .iter()
                .zip(&server)
                .filter(|(_, s)| **s > LOGPROB_FLOOR)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max);
            let ours = argmax(&logits);
            let mlx_f32 = &f32_logits[i * vocab..(i + 1) * vocab];
            let f32_err = rel_rms(&logits, mlx_f32);
            println!(
                "token {i}: ours {ours}, server {theirs}, logprob diff {diff:.3}, rel rms vs MLX f32 {f32_err:.2e}"
            );
            assert_eq!(
                ours,
                argmax(mlx_f32),
                "token {i}: argmax differs from MLX f32"
            );
            assert!(
                f32_err < F32_LOGITS,
                "token {i}: logits differ from MLX f32 by {f32_err}"
            );
            assert!(
                diff < LOGPROB_TOLERANCE,
                "token {i}: logprobs differ by {diff}"
            );
            if ours as u64 != theirs {
                let gap = server[0] - server[1];
                assert!(
                    gap < NEAR_TIE,
                    "token {i}: got {ours}, want {theirs} (server top-2 gap {gap})"
                );
                println!("token {i}: near tie (gap {gap})");
                first_tie = first_tie.min(i);
            }
            let start = std::time::Instant::now();
            logits = model.forward(theirs as u32, &mut states).unwrap();
            elapsed += start.elapsed();
        }
        println!("{:.2?} per token on the CPU", elapsed / want.len() as u32);

        let got = model.generate(&prompt, first_tie, |_, _| {}).unwrap();
        let got: Vec<u64> = got.iter().map(|&t| t as u64).collect();
        assert_eq!(got, want[..first_tie]);
    }
}
