//! Full attention layer for one token, as `Qwen3NextAttention` in MLX.

use super::ops::{dot, matvec, rms_norm, rope, sigmoid, softmax};

pub struct AttentionConfig {
    pub heads: usize,
    pub kv_heads: usize,
    pub head_dim: usize,
    /// Leading values of each head that get rotary embedding
    /// (`head_dim * partial_rotary_factor`).
    pub rope_dims: usize,
    pub rope_theta: f32,
    pub eps: f32,
}

/// Dense f32 weights, row-major `[out, in]` as in the checkpoint.
pub struct AttentionWeights<'a> {
    /// `[heads * head_dim * 2, hidden]`. For each head the first `head_dim`
    /// rows are the query and the next `head_dim` rows are the output gate.
    pub q_proj: &'a [f32],
    pub k_proj: &'a [f32],
    pub v_proj: &'a [f32],
    pub o_proj: &'a [f32],
    pub q_norm: &'a [f32],
    pub k_norm: &'a [f32],
}

/// Keys (after norm and rope) and values of the tokens seen so far, each
/// token stored as `[kv_heads * head_dim]`.
#[derive(Default)]
pub struct KvCache {
    keys: Vec<f32>,
    values: Vec<f32>,
    len: usize,
}

impl KvCache {
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
}

/// Runs one token `x` (`[hidden]`) at position `cache.len()`, appends its
/// key and value to the cache and returns the layer output (`[hidden]`).
pub fn attention(
    cfg: &AttentionConfig,
    w: &AttentionWeights,
    x: &[f32],
    cache: &mut KvCache,
) -> Vec<f32> {
    let d = cfg.head_dim;
    let pos = cache.len;

    let q_and_gate = matvec(w.q_proj, x);
    let mut q = Vec::with_capacity(cfg.heads * d);
    let mut gate = Vec::with_capacity(cfg.heads * d);
    for head in q_and_gate.chunks_exact(2 * d) {
        q.extend_from_slice(&head[..d]);
        gate.extend_from_slice(&head[d..]);
    }
    let mut k = matvec(w.k_proj, x);
    let v = matvec(w.v_proj, x);

    for head in q.chunks_exact_mut(d) {
        rms_norm(head, w.q_norm, cfg.eps);
        rope(head, cfg.rope_dims, pos, cfg.rope_theta);
    }
    for head in k.chunks_exact_mut(d) {
        rms_norm(head, w.k_norm, cfg.eps);
        rope(head, cfg.rope_dims, pos, cfg.rope_theta);
    }
    cache.keys.extend_from_slice(&k);
    cache.values.extend_from_slice(&v);
    cache.len += 1;

    let mut out = vec![0.0; cfg.heads * d];
    let group = cfg.heads / cfg.kv_heads;
    for h in 0..cfg.heads {
        let kv = h / group;
        let head_out = attend(cfg, &q[h * d..(h + 1) * d], cache, kv);
        out[h * d..(h + 1) * d].copy_from_slice(&head_out);
    }

    for (o, g) in out.iter_mut().zip(&gate) {
        *o *= sigmoid(*g);
    }
    matvec(w.o_proj, &out)
}

/// Softmax attention of one query head over all cached tokens of kv head `kv`.
fn attend(cfg: &AttentionConfig, q: &[f32], cache: &KvCache, kv: usize) -> Vec<f32> {
    let d = cfg.head_dim;
    let stride = cfg.kv_heads * d;
    let offset = kv * d;
    let scale = 1.0 / (d as f32).sqrt();

    let mut scores: Vec<f32> = (0..cache.len)
        .map(|t| {
            let start = t * stride + offset;
            dot(q, &cache.keys[start..start + d]) * scale
        })
        .collect();
    softmax(&mut scores);

    let mut out = vec![0.0; d];
    for (t, p) in scores.iter().enumerate() {
        let start = t * stride + offset;
        for (o, v) in out.iter_mut().zip(&cache.values[start..start + d]) {
            *o += p * v;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpu::testdata::{Fixture, assert_close};

    #[test]
    fn matches_mlx_over_four_tokens() {
        let f = Fixture::load("attention.json");
        let cfg = AttentionConfig {
            heads: f.usize("heads"),
            kv_heads: f.usize("kv_heads"),
            head_dim: f.usize("head_dim"),
            rope_dims: f.usize("rope_dims"),
            rope_theta: f.f32("rope_theta"),
            eps: f.f32("eps"),
        };
        let (q, k, v, o) = (
            f.vec("q_proj"),
            f.vec("k_proj"),
            f.vec("v_proj"),
            f.vec("o_proj"),
        );
        let (qn, kn) = (f.vec("q_norm"), f.vec("k_norm"));
        let w = AttentionWeights {
            q_proj: &q,
            k_proj: &k,
            v_proj: &v,
            o_proj: &o,
            q_norm: &qn,
            k_norm: &kn,
        };

        let mut cache = KvCache::default();
        for (t, (x, want)) in f.vecs("inputs").iter().zip(f.vecs("outputs")).enumerate() {
            let got = attention(&cfg, &w, x, &mut cache);
            assert_close(&format!("token {t}"), &got, &want, 1e-4);
        }
        assert_eq!(cache.len(), 4);
    }
}
