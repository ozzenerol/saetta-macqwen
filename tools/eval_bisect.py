"""MLX float32 hidden states after every layer, and the logits, at a few
positions of two token sequences from ~/saetta/eval/long_tokens.bin: the
first 4097 tokens, and the same text started 100 tokens later. Used to find
the layer where a model first parts from MLX at a long position.

Writes ~/saetta/eval/bisect/{a,b}_hidden.bin ([layer][position][hidden],
f32), {a,b}_logits.bin ([position][vocab]) and index.json.

    ~/llm/mlx/bin/python tools/eval_bisect.py <mlx model dir>
"""

import json
import sys
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

EVAL = Path.home() / "saetta/eval"
OUT = EVAL / "bisect"
LENGTH = 4097
SHIFT = 100
POSITIONS = [3994, 3995, 3996, 4094, 4095, 4096]
CHUNK = 2048


def main():
    model_dir = sys.argv[1]
    OUT.mkdir(parents=True, exist_ok=True)
    long = np.fromfile(EVAL / "long_tokens.bin", np.uint32).tolist()
    seqs = {"a": long[:LENGTH], "b": long[SHIFT : SHIFT + LENGTH]}

    model, _ = load(model_dir, lazy=True)
    model.set_dtype(mx.float32)
    lm = model.language_model
    text = lm.model
    states = {}
    for name, tokens in seqs.items():
        chunks = [text.embed_tokens(mx.array([tokens[i : i + CHUNK]])) for i in range(0, len(tokens), CHUNK)]
        mx.eval(chunks)
        states[name] = chunks
    caches = {name: model.make_cache() for name in seqs}
    picked = {name: [] for name in seqs}
    pos = mx.array(POSITIONS)

    for i, layer in enumerate(text.layers):
        for name, chunks in states.items():
            cache = caches[name]
            for j, h in enumerate(chunks):
                mask = create_ssm_mask(h, cache[i]) if layer.is_linear else create_attention_mask(h, cache[i])
                chunks[j] = layer(h, mask=mask, cache=cache[i])
                mx.eval(chunks[j])
            cache[i] = None
            full = mx.concatenate(chunks, axis=1)[0]
            picked[name].append(np.array(full[pos]))
        text.layers[i] = nn.Module()
        del layer
        mx.clear_cache()

    for name, chunks in states.items():
        full = mx.concatenate(chunks, axis=1)[0]
        logits = lm.lm_head(text.norm(full[pos])).astype(mx.float32)
        np.array(logits).tofile(OUT / f"{name}_logits.bin")
        np.stack(picked[name]).astype(np.float32).tofile(OUT / f"{name}_hidden.bin")
    (OUT / "index.json").write_text(
        json.dumps({"length": LENGTH, "shift": SHIFT, "positions": POSITIONS, "layers": len(text.layers)}) + "\n"
    )
    print(f"wrote {OUT}, peak {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
