# Writes the MLX reference data for the src/cpu tests.
# Run on the Mac from the checkout root:
#   ~/llm/mlx/bin/python src/cpu/testdata/gen.py
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import KVCache
from mlx_lm.models.qwen3_next import Qwen3NextAttention

OUT = Path(__file__).parent
mx.set_default_device(mx.cpu)
mx.random.seed(7)


def flat(a):
    return [float(v) for v in a.astype(mx.float32).reshape(-1).tolist()]


def randn(*shape, scale=1.0):
    return mx.random.normal(shape) * scale


def attention():
    args = SimpleNamespace(
        hidden_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        rms_norm_eps=1e-6,
        attention_bias=False,
        partial_rotary_factor=0.25,
        rope_theta=10000000.0,
        rope_scaling={"type": "default"},
        max_position_embeddings=262144,
    )
    m = Qwen3NextAttention(args)
    m.q_proj.weight = randn(*m.q_proj.weight.shape, scale=0.3)
    m.k_proj.weight = randn(*m.k_proj.weight.shape, scale=0.3)
    m.v_proj.weight = randn(*m.v_proj.weight.shape, scale=0.3)
    m.o_proj.weight = randn(*m.o_proj.weight.shape, scale=0.3)
    m.q_norm.weight = 1.0 + randn(args.head_dim, scale=0.1)
    m.k_norm.weight = 1.0 + randn(args.head_dim, scale=0.1)

    cache = KVCache()
    xs, ys = [], []
    for _ in range(4):
        x = randn(1, 1, args.hidden_size)
        y = m(x, mask=None, cache=cache)
        mx.eval(y)
        xs.append(flat(x))
        ys.append(flat(y))

    return {
        "hidden": args.hidden_size,
        "heads": args.num_attention_heads,
        "kv_heads": args.num_key_value_heads,
        "head_dim": args.head_dim,
        "rope_dims": int(args.head_dim * args.partial_rotary_factor),
        "rope_theta": args.rope_theta,
        "eps": args.rms_norm_eps,
        "q_proj": flat(m.q_proj.weight),
        "k_proj": flat(m.k_proj.weight),
        "v_proj": flat(m.v_proj.weight),
        "o_proj": flat(m.o_proj.weight),
        "q_norm": flat(m.q_norm.weight),
        "k_norm": flat(m.k_norm.weight),
        "inputs": xs,
        "outputs": ys,
    }


def small_ops():
    x = randn(64)
    w = 1.0 + randn(64, scale=0.1)
    rope_in = randn(1, 3, 1, 16)
    rope = nn.RoPE(4, traditional=False, base=10000000.0)
    return {
        "x": flat(x),
        "w": flat(w),
        "rms_norm": flat(mx.fast.rms_norm(x, w, 1e-6)),
        "silu": flat(nn.silu(x)),
        "softmax": flat(mx.softmax(x, precise=True)),
        "rope_in": flat(rope_in),
        "rope_positions": [0, 1, 1000, 200000],
        "rope_out": [flat(rope(rope_in, offset=p)) for p in [0, 1, 1000, 200000]],
    }


def write(name, data):
    (OUT / name).write_text(json.dumps(data, separators=(",", ":")) + "\n")


write("ops.json", small_ops())
write("attention.json", attention())
