//! Sparse MoE block for one token, as `Qwen3NextSparseMoeBlock` in MLX.

use super::ops::{matvec, sigmoid, silu, softmax};

/// Dense f32 weights of one SwiGLU MLP, row-major `[out, in]`.
pub struct Mlp<'a> {
    pub gate_proj: &'a [f32],
    pub up_proj: &'a [f32],
    pub down_proj: &'a [f32],
}

pub struct MoeWeights<'a> {
    /// Router, `[experts, hidden]`.
    pub gate: &'a [f32],
    /// All experts stacked as in `switch_mlp`: `[experts, out, in]`.
    pub experts: Mlp<'a>,
    pub shared_expert: Mlp<'a>,
    /// `[1, hidden]`.
    pub shared_expert_gate: &'a [f32],
}

/// `down(silu(gate(x)) * up(x))`
pub fn mlp(w: &Mlp, x: &[f32]) -> Vec<f32> {
    let gate = matvec(w.gate_proj, x);
    let up = matvec(w.up_proj, x);
    let h: Vec<f32> = gate.iter().zip(&up).map(|(g, u)| silu(*g) * u).collect();
    matvec(w.down_proj, &h)
}

/// Router: softmax over all experts, keep the `top_k` best and rescale
/// their weights to sum to 1 (`norm_topk_prob`). Returns
/// `(expert, weight)` pairs, best first.
pub fn route(gate: &[f32], x: &[f32], top_k: usize) -> Vec<(usize, f32)> {
    let mut probs = matvec(gate, x);
    softmax(&mut probs);
    let mut ranked: Vec<(usize, f32)> = probs.into_iter().enumerate().collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1));
    ranked.truncate(top_k);
    let sum: f32 = ranked.iter().map(|(_, p)| p).sum();
    ranked.iter_mut().for_each(|(_, p)| *p /= sum);
    ranked
}

pub fn moe(w: &MoeWeights, x: &[f32], top_k: usize) -> Vec<f32> {
    let experts = w.gate.len() / x.len();
    let mut out = vec![0.0; x.len()];
    for (e, weight) in route(w.gate, x, top_k) {
        let y = mlp(&expert(&w.experts, e, experts), x);
        for (o, v) in out.iter_mut().zip(&y) {
            *o += weight * v;
        }
    }

    let shared = mlp(&w.shared_expert, x);
    let shared_weight = sigmoid(matvec(w.shared_expert_gate, x)[0]);
    for (o, v) in out.iter_mut().zip(&shared) {
        *o += shared_weight * v;
    }
    out
}

/// The weights of expert `e` out of `n` stacked experts.
fn expert<'a>(all: &Mlp<'a>, e: usize, n: usize) -> Mlp<'a> {
    let slice = |t: &'a [f32]| {
        let size = t.len() / n;
        &t[e * size..(e + 1) * size]
    };
    Mlp {
        gate_proj: slice(all.gate_proj),
        up_proj: slice(all.up_proj),
        down_proj: slice(all.down_proj),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::testdata::{Fixture, assert_close};

    #[test]
    fn matches_mlx() {
        let f = Fixture::load("moe.json");
        let names = [
            "gate",
            "gate_proj",
            "up_proj",
            "down_proj",
            "shared_gate_proj",
            "shared_up_proj",
            "shared_down_proj",
            "shared_expert_gate",
        ];
        let t: Vec<Vec<f32>> = names.iter().map(|n| f.vec(n)).collect();
        let w = MoeWeights {
            gate: &t[0],
            experts: Mlp {
                gate_proj: &t[1],
                up_proj: &t[2],
                down_proj: &t[3],
            },
            shared_expert: Mlp {
                gate_proj: &t[4],
                up_proj: &t[5],
                down_proj: &t[6],
            },
            shared_expert_gate: &t[7],
        };
        let top_k = f.usize("top_k");
        let chosen = f.get("chosen").as_array().unwrap();

        for (i, (x, want)) in f.vecs("inputs").iter().zip(f.vecs("outputs")).enumerate() {
            let mut experts: Vec<u64> =
                route(w.gate, x, top_k).iter().map(|r| r.0 as u64).collect();
            experts.sort();
            let want_experts: Vec<u64> = chosen[i]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap())
                .collect();
            assert_eq!(experts, want_experts, "token {i}: experts");

            assert_close(&format!("token {i}"), &moe(&w, x, top_k), &want, 1e-4);
        }
    }
}
