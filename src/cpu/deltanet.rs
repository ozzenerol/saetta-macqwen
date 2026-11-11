//! Gated DeltaNet (linear attention) for one token, as `GatedDeltaNet` in
//! MLX `qwen3_5.py` with the recurrence of `gated_delta.py`.

use super::matrix::Matrix;
use super::ops::{dot, rms_norm, sigmoid, silu, softplus};

pub struct DeltaNetConfig {
    pub k_heads: usize,
    pub v_heads: usize,
    pub k_dim: usize,
    pub v_dim: usize,
    /// Conv1d kernel size.
    pub kernel: usize,
    pub eps: f32,
}

impl DeltaNetConfig {
    fn key_size(&self) -> usize {
        self.k_heads * self.k_dim
    }

    fn value_size(&self) -> usize {
        self.v_heads * self.v_dim
    }

    /// Channels of the conv1d: q, k and v side by side.
    fn conv_size(&self) -> usize {
        2 * self.key_size() + self.value_size()
    }
}

/// Projection matrices are `[out, in]` as in the checkpoint; the rest is f32.
pub struct DeltaNetWeights<'a> {
    /// `[2 * key_size + value_size, hidden]`, rows in the order q, k, v.
    pub in_proj_qkv: Matrix<'a>,
    pub in_proj_z: Matrix<'a>,
    pub in_proj_b: Matrix<'a>,
    pub in_proj_a: Matrix<'a>,
    pub out_proj: Matrix<'a>,
    /// `[conv_size, kernel]`; the last column multiplies the newest token.
    pub conv1d: &'a [f32],
    pub a_log: &'a [f32],
    pub dt_bias: &'a [f32],
    /// Weight of the gated RMSNorm, `[v_dim]`. Used as stored: unlike the
    /// other norms it has no `+1` folded in, and MLX does not add one.
    pub norm: &'a [f32],
}

/// What the layer keeps between tokens.
pub struct DeltaNetState {
    /// The last `kernel - 1` conv inputs, oldest first, each `[conv_size]`.
    pub conv: Vec<f32>,
    /// One `[v_dim, k_dim]` matrix per value head.
    pub recurrent: Vec<f32>,
}

impl DeltaNetState {
    pub fn new(cfg: &DeltaNetConfig) -> DeltaNetState {
        DeltaNetState {
            conv: vec![0.0; (cfg.kernel - 1) * cfg.conv_size()],
            recurrent: vec![0.0; cfg.v_heads * cfg.v_dim * cfg.k_dim],
        }
    }
}

/// Runs one token `x` (`[hidden]`), updates `state` and returns the layer
/// output (`[hidden]`).
pub fn deltanet(
    cfg: &DeltaNetConfig,
    w: &DeltaNetWeights,
    x: &[f32],
    state: &mut DeltaNetState,
) -> Vec<f32> {
    let qkv = w.in_proj_qkv.matvec(x);
    let z = w.in_proj_z.matvec(x);
    let b = w.in_proj_b.matvec(x);
    let a = w.in_proj_a.matvec(x);

    let mixed = conv_step(cfg, w.conv1d, &qkv, &mut state.conv);
    let (q, rest) = mixed.split_at(cfg.key_size());
    let (k, v) = rest.split_at(cfg.key_size());
    let (q, k) = normalize_qk(cfg, q, k);

    let mut y = vec![0.0; cfg.value_size()];
    let (dk, dv) = (cfg.k_dim, cfg.v_dim);
    let group = cfg.v_heads / cfg.k_heads;
    for h in 0..cfg.v_heads {
        let g = (-w.a_log[h].exp() * softplus(a[h] + w.dt_bias[h])).exp();
        let beta = sigmoid(b[h]);
        let kh = h / group;
        delta_rule(
            &mut state.recurrent[h * dv * dk..(h + 1) * dv * dk],
            &q[kh * dk..(kh + 1) * dk],
            &k[kh * dk..(kh + 1) * dk],
            &v[h * dv..(h + 1) * dv],
            g,
            beta,
            &mut y[h * dv..(h + 1) * dv],
        );
    }

    for (head, gate) in y.chunks_exact_mut(dv).zip(z.chunks_exact(dv)) {
        rms_norm(head, w.norm, cfg.eps);
        for (o, g) in head.iter_mut().zip(gate) {
            *o *= silu(*g);
        }
    }
    w.out_proj.matvec(&y)
}

/// Depthwise causal conv1d over the stored inputs and `input`, then SiLU.
/// Shifts `input` into the conv state.
fn conv_step(cfg: &DeltaNetConfig, weight: &[f32], input: &[f32], conv: &mut Vec<f32>) -> Vec<f32> {
    let n = cfg.conv_size();
    let past = cfg.kernel - 1;
    let out = (0..n)
        .map(|c| {
            let w = &weight[c * cfg.kernel..(c + 1) * cfg.kernel];
            let old: f32 = (0..past).map(|t| w[t] * conv[t * n + c]).sum();
            silu(old + w[past] * input[c])
        })
        .collect();
    conv.drain(..n);
    conv.extend_from_slice(input);
    out
}

/// L2-normalizes each q and k head, and scales q by `k_dim^-0.5`
/// (`normalize_qk` in MLX, with `eps` added to the sum of squares).
fn normalize_qk(cfg: &DeltaNetConfig, q: &[f32], k: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let q_scale = 1.0 / (cfg.k_dim as f32).sqrt();
    let norm = |x: &[f32], scale: f32| -> Vec<f32> {
        let mut out = Vec::with_capacity(x.len());
        for head in x.chunks_exact(cfg.k_dim) {
            let inv = scale / (dot(head, head) + cfg.eps).sqrt();
            out.extend(head.iter().map(|v| v * inv));
        }
        out
    };
    (norm(q, q_scale), norm(k, 1.0))
}

/// One step of the gated delta rule on one head's `[v_dim, k_dim]` state:
/// decay by `g`, write `v` under key `k` with strength `beta`, read with `q`.
fn delta_rule(s: &mut [f32], q: &[f32], k: &[f32], v: &[f32], g: f32, beta: f32, y: &mut [f32]) {
    for (i, row) in s.chunks_exact_mut(k.len()).enumerate() {
        row.iter_mut().for_each(|x| *x *= g);
        let delta = (v[i] - dot(row, k)) * beta;
        for (x, kj) in row.iter_mut().zip(k) {
            *x += kj * delta;
        }
        y[i] = dot(row, q);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::testdata::{Fixture, assert_close};

    #[test]
    fn matches_mlx_over_six_tokens() {
        let f = Fixture::load("deltanet.json");
        let cfg = DeltaNetConfig {
            k_heads: f.usize("k_heads"),
            v_heads: f.usize("v_heads"),
            k_dim: f.usize("k_dim"),
            v_dim: f.usize("v_dim"),
            kernel: f.usize("kernel"),
            eps: f.f32("eps"),
        };
        let names = [
            "in_proj_qkv",
            "in_proj_z",
            "in_proj_b",
            "in_proj_a",
            "out_proj",
            "conv1d",
            "A_log",
            "dt_bias",
            "norm",
        ];
        let t: Vec<Vec<f32>> = names.iter().map(|n| f.vec(n)).collect();
        let w = DeltaNetWeights {
            in_proj_qkv: Matrix::Dense(&t[0]),
            in_proj_z: Matrix::Dense(&t[1]),
            in_proj_b: Matrix::Dense(&t[2]),
            in_proj_a: Matrix::Dense(&t[3]),
            out_proj: Matrix::Dense(&t[4]),
            conv1d: &t[5],
            a_log: &t[6],
            dt_bias: &t[7],
            norm: &t[8],
        };

        let mut state = DeltaNetState::new(&cfg);
        for (i, (x, want)) in f.vecs("inputs").iter().zip(f.vecs("outputs")).enumerate() {
            let got = deltanet(&cfg, &w, x, &mut state);
            assert_close(&format!("token {i}"), &got, &want, 1e-4);
        }
        assert_close("conv state", &state.conv, &f.vec("conv_state"), 1e-5);
        assert_close("recurrent state", &state.recurrent, &f.vec("state"), 1e-4);
    }
}
