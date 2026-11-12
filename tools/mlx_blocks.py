"""Times MLX's prefill blocks one at a time on real hidden states, so
saetta's profile-prefill lines can be compared block by block: the MoE
(and its router alone), a Gated DeltaNet mixer, a full attention mixer and
whole layers, for chunks of T tokens at position 0.

The hidden states come from running the first T tokens of
~/saetta/eval/long_tokens.bin through the layers before the timed ones.
Prints the median of the runs per block and the per-model sum (40 MoE, 30
DeltaNet and 10 attention mixers).

    ~/llm/mlx/bin/python tools/mlx_blocks.py <mlx model dir> [T ...]
"""

import statistics
import sys
import time
from pathlib import Path

import mlx.core as mx
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

EVAL = Path.home() / "saetta/eval"
RUNS = 10
WARMUP = 3


def timed(f):
    for _ in range(WARMUP):
        mx.eval(f())
    mx.synchronize()
    out = []
    for _ in range(RUNS):
        start = time.perf_counter()
        mx.eval(f())
        mx.synchronize()
        out.append(time.perf_counter() - start)
    return statistics.median(out) * 1e3


def mask_for(layer, h, cache):
    return create_ssm_mask(h, cache) if layer.is_linear else create_attention_mask(h, cache)


def blocks(model, text, tokens):
    layers = text.layers
    linear = next(i for i in range(len(layers) // 2, len(layers)) if layers[i].is_linear)
    full = next(i for i in range(linear, len(layers)) if not layers[i].is_linear)
    caches = model.make_cache()
    h = text.embed_tokens(mx.array([tokens]))
    found = {}
    for i, layer in enumerate(layers[: full + 1]):
        if i in (linear, full):
            found[i] = h
        h = layer(h, mask=mask_for(layer, h, caches[i]), cache=caches[i])
        mx.eval(h)
    del caches

    res = {}
    for i in (linear, full):
        layer, h = layers[i], found[i]
        name = "deltanet" if layer.is_linear else "attention"
        mixer = layer.linear_attn if layer.is_linear else layer.self_attn
        xn = layer.input_layernorm(h)
        mn = layer.post_attention_layernorm(h)
        mx.eval(xn, mn)

        def run_mixer():
            cache = model.make_cache()[i]
            return mixer(xn, mask=mask_for(layer, xn, cache), cache=cache)

        def run_layer():
            cache = model.make_cache()[i]
            return layer(h, mask=mask_for(layer, h, cache), cache=cache)

        res[f"{name} mixer (layer {i})"] = timed(run_mixer)
        res[f"moe (layer {i})"] = timed(lambda: layer.mlp(mn))
        gate = getattr(layer.mlp, "gate", None)
        if gate is not None:
            res[f"router (layer {i})"] = timed(lambda: gate(mn))
        res[f"whole {name} layer {i}"] = timed(run_layer)
    return linear, full, res


def main():
    model_dir = sys.argv[1]
    sizes = [int(a) for a in sys.argv[2:]] or [548, 2048, 3026]
    model, _ = load(model_dir)
    text = model.language_model.model
    long = np.fromfile(EVAL / "long_tokens.bin", np.uint32).tolist()
    for t in sizes:
        linear, full, res = blocks(model, text, long[:t])
        print(f"T = {t}")
        for name, ms in res.items():
            print(f"  {name:32} {ms:8.2f} ms")
        moe = (res[f"moe (layer {linear})"] + res[f"moe (layer {full})"]) / 2
        delta = res[f"deltanet mixer (layer {linear})"]
        att = res[f"attention mixer (layer {full})"]
        layers = 30 * res[f"whole deltanet layer {linear}"] + 10 * res[f"whole attention layer {full}"]
        print(f"  per model: moe 40 x {moe:.2f} = {40 * moe:.1f} ms, deltanet 30 x {delta:.2f} = "
              f"{30 * delta:.1f} ms, attention 10 x {att:.2f} = {10 * att:.1f} ms, layers {layers:.1f} ms")
        sys.stdout.flush()
    print(f"peak {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
