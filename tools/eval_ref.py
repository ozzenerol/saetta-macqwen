"""Writes the MLX reference data for `saetta eval` on the Mac.

Two token sequences:
- ppl: tests/data/eval_text.txt, about 2k tokens of prose, Go, Rust, Python
  and JSON. For each position it stores the negative log-likelihood of the
  next token and the argmax.
- long: the first N tokens of mlx_lm's own Python sources (522k tokens in
  mlx_lm 0.32.2). It stores the same per position data, plus the full logits
  at checkpoint positions, so a run over the first M tokens can be checked
  at every checkpoint below M. Longer references share the prefix, so their
  tokens and checkpoints extend the shorter ones.

The model runs in float32 (the reference) and, if asked, in bfloat16 (what
mlx_lm.server runs), layer by layer over all tokens in chunks, freeing each
layer after use. MLX writes out the attention scores in float32, 16 heads x
chunk x keys x 4 bytes, so the chunks get shorter as the keys grow, to keep
the scores within --score-gb. The script prints its memory estimate first.

    # the 32k reference in ~/saetta/eval
    ~/llm/mlx/bin/python tools/eval_ref.py <mlx model dir> --text tests/data/eval_text.txt
    # 262k, float32 only, in its own dir
    ~/llm/mlx/bin/python tools/eval_ref.py <mlx model dir> --out ~/saetta/eval-262k \\
        --long 262144 --checkpoints 32767,65535,131071,196607,262143 --dtypes f32
"""

import argparse
import json
import time
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import mlx_lm
import numpy as np
from mlx_lm import load
from mlx_lm.models.base import create_attention_mask, create_ssm_mask

MAX_CHUNK = 2048
MIN_CHUNK = 64
HEAD_CHUNK = 256
QUERY_HEADS = 16


def long_text():
    root = Path(mlx_lm.__file__).parent
    parts = []
    for path in sorted(root.rglob("*.py")):
        parts.append(f"# file: {path.relative_to(root)}\n{path.read_text()}\n")
    return "".join(parts)


def chunk_plan(length, score_bytes):
    """Chunk lengths, a power of two each, so that the scores of a chunk
    against all keys up to its end stay within score_bytes."""
    sizes, start = [], 0
    while start < length:
        size = MAX_CHUNK
        while size > MIN_CHUNK and QUERY_HEADS * size * (start + size) * 4 > score_bytes:
            size //= 2
        size = min(size, length - start)
        sizes.append(size)
        start += size
    return sizes


def estimate_gb(lengths, score_bytes, hidden, kv_heads, head_dim):
    tokens = sum(lengths)
    states = tokens * hidden * 4
    kv = max(lengths) * kv_heads * head_dim * 2 * 4
    return (states + kv + score_bytes) / 2**30 + 1.5


def run_layers(model, seqs, score_bytes):
    """Final hidden states (before the last norm) of each sequence, in chunks."""
    text = model.language_model.model
    states = []
    for tokens in seqs:
        chunks, start = [], 0
        for size in chunk_plan(len(tokens), score_bytes):
            chunks.append(text.embed_tokens(mx.array([tokens[start : start + size]])))
            start += size
        mx.eval(chunks)
        states.append(chunks)
    caches = [model.make_cache() for _ in seqs]
    start_time = time.time()
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
        print(f"  layer {i} done at {time.time() - start_time:.0f} s, "
              f"peak {mx.get_peak_memory() / 2**30:.2f} GB", flush=True)
    return states


def head(model, chunks, tokens, checkpoints, out, prefix):
    """NLL of each next token, the argmax, and full logits at the checkpoints."""
    lm = model.language_model
    norm = lm.model.norm
    nll, argmax = [], []
    targets = mx.array(tokens[1:] + [0])
    start = 0
    for chunk in chunks:
        hidden = chunk[0]
        for a in range(0, hidden.shape[0], HEAD_CHUNK):
            b = min(a + HEAD_CHUNK, hidden.shape[0])
            logits = lm.lm_head(norm(hidden[a:b])).astype(mx.float32)
            logprobs = logits - mx.logsumexp(logits, axis=-1, keepdims=True)
            picked = mx.take_along_axis(logprobs, targets[start + a : start + b, None], axis=-1)[:, 0]
            nll.append(np.array(-picked))
            argmax.append(np.array(mx.argmax(logits, axis=-1).astype(mx.uint32)))
            for p in checkpoints:
                if start + a <= p < start + b:
                    np.array(logits[p - start - a]).astype(np.float32).tofile(out / f"{prefix}_logits_{p}.bin")
        start += hidden.shape[0]
    return np.concatenate(nll)[:-1], np.concatenate(argmax)[:-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model_dir")
    ap.add_argument("--text", help="the ppl text; without it there is no ppl sequence")
    ap.add_argument("--out", default=str(Path.home() / "saetta/eval"))
    ap.add_argument("--long", type=int, default=32768)
    ap.add_argument("--checkpoints", default="1023,4095,8191,16383,32767")
    ap.add_argument("--dtypes", default="f32,bf16")
    ap.add_argument("--score-gb", type=float, default=2.0)
    ap.add_argument("--dry-run", action="store_true", help="print the plan and the memory estimate only")
    args = ap.parse_args()

    out = Path(args.out).expanduser()
    checkpoints = [int(p) for p in args.checkpoints.split(",")]
    assert max(checkpoints) < args.long
    score_bytes = int(args.score_gb * 2**30)

    _, tokenizer = load(args.model_dir, lazy=True)
    seqs = {}
    if args.text:
        seqs["ppl"] = tokenizer.encode(Path(args.text).read_text(), add_special_tokens=False)
    seqs["long"] = tokenizer.encode(long_text(), add_special_tokens=False)[: args.long]
    assert len(seqs["long"]) == args.long, len(seqs["long"])

    lengths = [len(t) for t in seqs.values()]
    plan = chunk_plan(args.long, score_bytes)
    print(f"long {args.long} tokens in {len(plan)} chunks ({plan[0]} down to {plan[-1]}), "
          f"estimated peak {estimate_gb(lengths, score_bytes, 2048, 2, 256):.1f} GB", flush=True)
    if args.dry_run:
        return

    out.mkdir(parents=True, exist_ok=True)
    for name, tokens in seqs.items():
        np.array(tokens, np.uint32).tofile(out / f"{name}_tokens.bin")
    index = {
        "mlx_lm_version": mlx_lm.__version__,
        "long_tokens": args.long,
        "checkpoints": checkpoints,
        "score_gb": args.score_gb,
        "runs": {},
    }
    if args.text:
        index["ppl_tokens"] = len(seqs["ppl"])
    dtypes = {"f32": mx.float32, "bf16": mx.bfloat16}
    for dtype_name in args.dtypes.split(","):
        start = time.time()
        mx.reset_peak_memory()
        model, _ = load(args.model_dir, lazy=True)
        model.set_dtype(dtypes[dtype_name])
        states = run_layers(model, list(seqs.values()), score_bytes)
        run = {}
        for (name, tokens), chunks in zip(seqs.items(), states):
            points = checkpoints if name == "long" else []
            nll, argmax = head(model, chunks, tokens, points, out, dtype_name)
            nll.astype(np.float32).tofile(out / f"{dtype_name}_{name}_nll.bin")
            argmax.tofile(out / f"{dtype_name}_{name}_argmax.bin")
            run[name] = {"ppl": float(np.exp(nll.mean()))}
        run["seconds"] = round(time.time() - start, 1)
        run["peak_gb"] = round(mx.get_peak_memory() / 2**30, 2)
        index["runs"][dtype_name] = run
        print(dtype_name, run, flush=True)
        del model, states
        mx.clear_cache()

    (out / "index.json").write_text(json.dumps(index, indent=1) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
