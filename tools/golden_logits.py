# Writes MLX float32 logits of all 40 layers for the golden prompt followed
# by the server's greedy tokens (GOLDEN_DIR/greedy.json, from greedy_ref.py).
# The model runs layer by layer over all tokens and each layer is freed after
# use, so only about one layer of weights is in memory at a time.
# Run on the Mac:
#   ~/llm/mlx/bin/python tools/golden_logits.py
import json
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx_lm import load

MODEL_DIR = Path.home() / "llm/mlx-models/Qwen3.6-35B-A3B-5bit"
GOLDEN_DIR = Path.home() / "saetta/golden"
GOLDEN_DIR_F32 = Path.home() / "saetta/golden-f32"


def main():
    greedy = json.loads((GOLDEN_DIR / "greedy.json").read_text())
    prompt = np.fromfile(GOLDEN_DIR / "tokens.bin", np.uint32).tolist()
    tokens = prompt + greedy["tokens"]

    model, _ = load(str(MODEL_DIR), lazy=True)
    model.set_dtype(mx.float32)
    lm = model.language_model
    text = lm.model
    caches = model.make_cache()

    xs = [text.embed_tokens(mx.array([[t]])) for t in tokens]
    mx.eval(xs)
    for i, layer in enumerate(text.layers):
        for t in range(len(xs)):
            xs[t] = layer(xs[t], mask=None, cache=caches[i])
            mx.eval(xs[t])
        text.layers[i] = nn.Module()
        caches[i] = None
        del layer
        mx.clear_cache()
        print(f"layer {i} done, {mx.get_active_memory() / 2**30:.2f} GB active")

    # Logits at each position that predicts one of the greedy tokens.
    out = [lm.lm_head(text.norm(x)) for x in xs[len(prompt) - 1 : -1]]
    logits = np.array(mx.concatenate(out, axis=1)[0].astype(mx.float32))
    logits.tofile(GOLDEN_DIR_F32 / "greedy_logits.bin")
    (GOLDEN_DIR_F32 / "greedy_logits.json").write_text(
        json.dumps({"tokens": tokens, "prompt_tokens": len(prompt), "shape": list(logits.shape)}) + "\n"
    )
    print("argmax", logits.argmax(-1).tolist())
    print(f"peak memory {mx.get_peak_memory() / 2**30:.2f} GB")


if __name__ == "__main__":
    main()
