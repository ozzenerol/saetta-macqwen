# Writes the MLX reference data for the src/cpu tests.
# Run on the Mac from the checkout root:
#   ~/llm/mlx/bin/python src/cpu/testdata/gen.py
import json
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.qwen3_5 import GatedDeltaNet
from mlx_lm.models.qwen3_next import Qwen3NextAttention, Qwen3NextSparseMoeBlock

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


def deltanet():
    args = SimpleNamespace(
        hidden_size=32,
        linear_num_value_heads=4,
        linear_num_key_heads=2,
        linear_key_head_dim=8,
        linear_value_head_dim=8,
        linear_conv_kernel_dim=4,
        rms_norm_eps=1e-6,
    )
    m = GatedDeltaNet(args)
    for lin in [m.in_proj_qkv, m.in_proj_z, m.in_proj_b, m.in_proj_a, m.out_proj]:
        lin.weight = randn(*lin.weight.shape, scale=0.3)
    m.conv1d.weight = randn(*m.conv1d.weight.shape, scale=0.4)
    m.A_log = randn(args.linear_num_value_heads)
    m.dt_bias = randn(args.linear_num_value_heads)
    m.norm.weight = 0.9 + randn(args.linear_value_head_dim, scale=0.1)

    cache = ArraysCache(size=2)
    xs, ys = [], []
    for _ in range(6):
        x = randn(1, 1, args.hidden_size)
        y = m(x, mask=None, cache=cache)
        mx.eval(y, cache[0], cache[1])
        xs.append(flat(x))
        ys.append(flat(y))

    return {
        "hidden": args.hidden_size,
        "k_heads": args.linear_num_key_heads,
        "v_heads": args.linear_num_value_heads,
        "k_dim": args.linear_key_head_dim,
        "v_dim": args.linear_value_head_dim,
        "kernel": args.linear_conv_kernel_dim,
        "eps": args.rms_norm_eps,
        "in_proj_qkv": flat(m.in_proj_qkv.weight),
        "in_proj_z": flat(m.in_proj_z.weight),
        "in_proj_b": flat(m.in_proj_b.weight),
        "in_proj_a": flat(m.in_proj_a.weight),
        "out_proj": flat(m.out_proj.weight),
        "conv1d": flat(m.conv1d.weight),
        "A_log": flat(m.A_log),
        "dt_bias": flat(m.dt_bias),
        "norm": flat(m.norm.weight),
        "inputs": xs,
        "outputs": ys,
        "conv_state": flat(cache[0]),
        "state": flat(cache[1]),
    }


def moe():
    args = SimpleNamespace(
        hidden_size=32,
        moe_intermediate_size=16,
        shared_expert_intermediate_size=24,
        num_experts=16,
        num_experts_per_tok=4,
        norm_topk_prob=True,
    )
    m = Qwen3NextSparseMoeBlock(args)
    m.gate.weight = randn(*m.gate.weight.shape, scale=0.5)
    m.shared_expert_gate.weight = randn(*m.shared_expert_gate.weight.shape, scale=0.3)
    for lin in [m.switch_mlp.gate_proj, m.switch_mlp.up_proj, m.switch_mlp.down_proj]:
        lin.weight = randn(*lin.weight.shape, scale=0.3)
    for lin in [m.shared_expert.gate_proj, m.shared_expert.up_proj, m.shared_expert.down_proj]:
        lin.weight = randn(*lin.weight.shape, scale=0.3)

    xs, ys, experts = [], [], []
    for _ in range(4):
        x = randn(1, 1, args.hidden_size)
        y = m(x)
        gates = mx.softmax(m.gate(x), axis=-1, precise=True)
        top = mx.argpartition(gates, kth=-args.num_experts_per_tok, axis=-1)
        top = top[..., -args.num_experts_per_tok :]
        mx.eval(y, top)
        xs.append(flat(x))
        ys.append(flat(y))
        experts.append(sorted(int(i) for i in top.reshape(-1).tolist()))

    return {
        "hidden": args.hidden_size,
        "experts": args.num_experts,
        "top_k": args.num_experts_per_tok,
        "gate": flat(m.gate.weight),
        "gate_proj": flat(m.switch_mlp.gate_proj.weight),
        "up_proj": flat(m.switch_mlp.up_proj.weight),
        "down_proj": flat(m.switch_mlp.down_proj.weight),
        "shared_gate_proj": flat(m.shared_expert.gate_proj.weight),
        "shared_up_proj": flat(m.shared_expert.up_proj.weight),
        "shared_down_proj": flat(m.shared_expert.down_proj.weight),
        "shared_expert_gate": flat(m.shared_expert_gate.weight),
        "inputs": xs,
        "outputs": ys,
        "chosen": experts,
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
write("deltanet.json", deltanet())
write("moe.json", moe())
