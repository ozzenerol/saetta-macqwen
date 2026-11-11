"""Writes the MLX reference data for `saetta eval` to ~/saetta/eval on the Mac.

Two token sequences:
- ppl: tests/data/eval_text.txt, about 2k tokens of prose, Go, Rust, Python
  and JSON. For each position it stores the negative log-likelihood of the
  next token and the argmax.
- long: 32768 tokens of mlx_lm's own Python sources. It stores the same per
  position data, plus the full logits at a few checkpoint positions, so a
  run over the first N tokens can be checked at every checkpoint below N.

The model runs once in float32 (the reference) and once in bfloat16 (what
mlx_lm.server runs), layer by layer over all tokens in chunks, freeing each
layer after use. Peak memory is a few GB, next to the running server.

    ~/llm/mlx/bin/python tools/eval_ref.py <mlx model dir> tests/data/eval_text.txt
"""

import json
import sys
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

OUT = Path.home() / "saetta/eval"
LONG_TOKENS = 32768
CHECKPOINTS = [1023, 4095, 8191, 16383, 32767]
CHUNK = 2048
HEAD_CHUNK = 256


def long_text():
    root = Path(mlx_lm.__file__).parent
    parts = []
    for path in sorted(root.rglob("*.py")):
        parts.append(f"# file: {path.relative_to(root)}\n{path.read_text()}\n")
    return "".join(parts)


def run_layers(model, seqs):
    """Final hidden states (after the last norm) of each sequence, in chunks."""
    text = model.language_model.model
    states = []
    for tokens in seqs:
        chunks = [text.embed_tokens(mx.array([tokens[i : i + CHUNK]])) for i in range(0, len(tokens), CHUNK)]
        mx.eval(chunks)
        states.append(chunks)
    caches = [model.make_cache() for _ in seqs]
    for i, layer in enumerate(text.layers):
        for chunks, cache in zip(states, caches):
            for j, h in enumerate(chunks):
                if layer.is_linear:
                    mask = create_ssm_mask(h, cache[i])
                else:
                    mask = create_attention_mask(h, cache[i])
                chunks[j] = layer(h, mask=mask, cache=cache[i])
                mx.eval(chunks[j])
            cache[i] = None
        text.layers[i] = nn.Module()
        del layer
        mx.clear_cache()
        if i % 10 == 9:
            print(f"  layer {i} done, {mx.get_active_memory() / 2**30:.2f} GB active", flush=True)
    return [mx.concatenate(chunks, axis=1)[0] for chunks in states]


def head(model, hidden, tokens, checkpoints):
    """NLL of each next token, the argmax, and full logits at the checkpoints."""
    lm = model.language_model
    norm = lm.model.norm
    nll, argmax, saved = [], [], {}
    targets = mx.array(tokens[1:] + [0])
    for start in range(0, hidden.shape[0], HEAD_CHUNK):
        end = min(start + HEAD_CHUNK, hidden.shape[0])
        logits = lm.lm_head(norm(hidden[start:end])).astype(mx.float32)
        logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
        picked = mx.take_along_axis(logprobs, targets[start:end, None], axis=-1)[:, 0]
        nll.append(-picked)
        argmax.append(mx.argmax(logits, axis=-1).astype(mx.uint32))
        for p in checkpoints:
            if start <= p < end:
                saved[p] = np.array(logits[p - start])
        mx.eval(nll[-1], argmax[-1])
    nll = np.array(mx.concatenate(nll))[:-1]
    argmax = np.array(mx.concatenate(argmax))[:-1]
    return nll, argmax, saved


def main():
    model_dir, text_path = Path(sys.argv[1]), Path(sys.argv[2])
    OUT.mkdir(parents=True, exist_ok=True)
    _, tokenizer = load(str(model_dir), lazy=True)
    ppl_tokens = tokenizer.encode(text_path.read_text(), add_special_tokens=False)
    long_tokens = tokenizer.encode(long_text(), add_special_tokens=False)[:LONG_TOKENS]
    assert len(long_tokens) == LONG_TOKENS, len(long_tokens)
    np.array(ppl_tokens, np.uint32).tofile(OUT / "ppl_tokens.bin")
    np.array(long_tokens, np.uint32).tofile(OUT / "long_tokens.bin")

    index = {
        "mlx_lm_version": mlx_lm.__version__,
        "ppl_tokens": len(ppl_tokens),
        "long_tokens": len(long_tokens),
        "checkpoints": CHECKPOINTS,
        "runs": {},
    }
    for dtype_name, dtype in [("f32", mx.float32), ("bf16", mx.bfloat16)]:
        start = time.time()
        model, _ = load(str(model_dir), lazy=True)
        model.set_dtype(dtype)
        hidden = run_layers(model, [ppl_tokens, long_tokens])
        run = {}
        for name, h, tokens, checkpoints in [
            ("ppl", hidden[0], ppl_tokens, []),
            ("long", hidden[1], long_tokens, CHECKPOINTS),
        ]:
            nll, argmax, saved = head(model, h, tokens, checkpoints)
            nll.astype(np.float32).tofile(OUT / f"{dtype_name}_{name}_nll.bin")
            argmax.tofile(OUT / f"{dtype_name}_{name}_argmax.bin")
            for p, logits in saved.items():
                logits.astype(np.float32).tofile(OUT / f"{dtype_name}_logits_{p}.bin")
            run[name] = {"ppl": float(np.exp(nll.mean()))}
        run["seconds"] = round(time.time() - start, 1)
        run["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        index["runs"][dtype_name] = run
        print(dtype_name, run, flush=True)
        del model, hidden
        mx.clear_cache()
        mx.reset_peak_memory()

    (OUT / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
