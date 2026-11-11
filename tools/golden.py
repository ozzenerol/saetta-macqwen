# Writes MLX reference data for layers 0 to 3 of the real model to GOLDEN_DIR.
# Run on the Mac:
#   ~/llm/mlx/bin/python tools/golden.py
#
# The prompt goes through one token at a time, the way saetta decodes. The
# weights are loaded lazily and only the embedding and layers 0 to 3 are
# evaluated, so only those tensors are read from disk (a few GB, not 24).
#
# With --f32 the model runs in float32 instead of bf16 and the data goes to
# GOLDEN_DIR_F32. That run is the tight check for the f32 CPU code.
#
# Every array is a plain little-endian file: f32 for values, u32 for token
# and expert ids. index.json lists each file with its dtype and shape.
import json
import sys
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load

MODEL_DIR = Path.home() / "llm/mlx-models/Qwen3.6-35B-A3B-5bit"
GOLDEN_DIR = Path.home() / "saetta/golden"
GOLDEN_DIR_F32 = Path.home() / "saetta/golden-f32"
PROMPT = "The capital of Romania is"
LAYERS = 4


def main():
    f32 = "--f32" in sys.argv[1:]
    out_dir = GOLDEN_DIR_F32 if f32 else GOLDEN_DIR
    model, tokenizer = load(str(MODEL_DIR), lazy=True)
    if f32:
        model.set_dtype(mx.float32)
    text = model.language_model.model
    tokens = tokenizer.encode(PROMPT)

    out_dir.mkdir(parents=True, exist_ok=True)
    index = {
        "model": MODEL_DIR.name,
        "prompt": PROMPT,
        "note": "MLX computes in "
        + ("float32" if f32 else "bf16 with an f32 DeltaNet state")
        + "; values are saved as f32",
        "files": {},
    }

    def save(name, array, dtype=np.float32):
        a = np.array(array.astype(mx.float32) if dtype == np.float32 else array)
        a = a.astype(dtype)
        a.tofile(out_dir / f"{name}.bin")
        index["files"][name] = {"dtype": np.dtype(dtype).name, "shape": list(a.shape)}

    caches = model.make_cache()[:LAYERS]
    hidden = {name: [] for name in ["embed"] + [f"layer{i}" for i in range(LAYERS)]}
    experts = {i: [] for i in range(LAYERS)}
    weights = {i: [] for i in range(LAYERS)}

    for token in tokens:
        x = text.embed_tokens(mx.array([[token]]))
        hidden["embed"].append(x)
        for i in range(LAYERS):
            x, ids, w = layer_step(text.layers[i], x, caches[i])
            hidden[f"layer{i}"].append(x)
            experts[i].append(ids)
            weights[i].append(w)
        mx.eval(x, [c.state for c in caches])

    save("tokens", mx.array(tokens), np.uint32)
    for name, xs in hidden.items():
        save(name, mx.concatenate(xs, axis=1)[0])
    for i in range(LAYERS):
        save(f"layer{i}_experts", mx.concatenate(experts[i], axis=0), np.uint32)
        save(f"layer{i}_expert_weights", mx.concatenate(weights[i], axis=0))

    conv, recurrent = caches[0][0], caches[0][1]
    save("layer0_conv_state", conv[0])
    save("layer0_recurrent_state", recurrent[0])

    kv = caches[3]
    save("layer3_keys", kv.keys[0, :, : kv.offset].transpose(1, 0, 2))
    save("layer3_values", kv.values[0, :, : kv.offset].transpose(1, 0, 2))

    (out_dir / "index.json").write_text(json.dumps(index, indent=2) + "\n")
    print(f"{len(tokens)} tokens written to {out_dir}")
    print(f"peak memory {mx.get_peak_memory() / 2**30:.2f} GB")


def layer_step(layer, x, cache):
    """DecoderLayer.__call__ for one token, split open to read the router."""
    if layer.is_linear:
        h = x + layer.linear_attn(layer.input_layernorm(x), None, cache)
    else:
        h = x + layer.self_attn(layer.input_layernorm(x), None, cache)
    m = layer.post_attention_layernorm(h)
    ids, w = route(layer.mlp, m)
    return h + layer.mlp(m), ids, w


def route(moe, x):
    """The router part of Qwen3NextSparseMoeBlock: top k ids and weights, best first."""
    gates = mx.softmax(moe.gate(x), axis=-1, precise=True).reshape(-1)
    ids = mx.argsort(-gates)[: moe.top_k]
    w = gates[ids]
    if moe.norm_topk_prob:
        w = w / w.sum()
    return ids[None], w[None]


if __name__ == "__main__":
    main()
