"""Compares MLX references of the same model at different weight bits.

Each directory is an `eval_ref.py` output for one quantization (same
tokenizer, so the same tokens). The first one is the baseline, for example
the 8-bit model as the closest to bf16. For every other directory it prints
the KL divergence from the baseline at the full-logits checkpoints, top-1
agreement with the baseline over all positions, and perplexity.

    python3 tools/quant_compare.py ~/saetta/eval-8bit ~/saetta/eval ~/saetta/eval-6bit
"""

import json
import sys
from pathlib import Path

import numpy as np


def load(d, dtype="f32"):
    d = Path(d)
    index = json.loads((d / "index.json").read_text())
    out = {"dir": d, "checkpoints": index["checkpoints"], "dtype": dtype}
    for seq in ("ppl", "long"):
        nll = d / f"{dtype}_{seq}_nll.bin"
        if nll.exists():
            out[seq] = (
                np.fromfile(nll, np.float32),
                np.fromfile(d / f"{dtype}_{seq}_argmax.bin", np.uint32),
            )
    return out


def log_softmax(x):
    x = x.astype(np.float64)
    m = x.max()
    return x - m - np.log(np.exp(x - m).sum())


def main(dirs):
    base, rest = load(dirs[0]), [load(d) for d in dirs[1:]]
    print(f"baseline {base['dir']}")
    for seq in ("ppl", "long"):
        if seq in base:
            print(f"  {seq} ppl {np.exp(base[seq][0].mean()):.4f}")
    for r in rest:
        print(f"{r['dir']}")
        for seq in ("ppl", "long"):
            if seq not in r or seq not in base:
                continue
            n = min(len(r[seq][1]), len(base[seq][1]))
            top1 = (r[seq][1][:n] == base[seq][1][:n]).mean()
            print(f"  {seq}: ppl {np.exp(r[seq][0][:n].mean()):.4f}, top-1 vs baseline {top1:.4f} over {n}")
        kls = []
        for p in base["checkpoints"]:
            a = base["dir"] / f"f32_logits_{p}.bin"
            b = r["dir"] / f"f32_logits_{p}.bin"
            if not (a.exists() and b.exists()):
                continue
            la, lb = log_softmax(np.fromfile(a, np.float32)), log_softmax(np.fromfile(b, np.float32))
            kls.append(float((np.exp(la) * (la - lb)).sum()))
        if kls:
            k = np.array(kls)
            print(f"  KL(baseline || this) at {len(k)} checkpoints: median {np.median(k):.2e}, mean {k.mean():.2e}, max {k.max():.2e}")


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    main(sys.argv[1:])
