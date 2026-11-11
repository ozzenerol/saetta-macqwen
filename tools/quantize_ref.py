"""Writes tests/data/quantize_cases.json: bf16 rows and what mlx.core.quantize
makes of them (affine, group 64), for 5 and 8 bits.

    ~/llm/mlx/bin/python tools/quantize_ref.py tests/data/quantize_cases.json
"""

import json
import sys

import mlx.core as mx
import numpy as np


def rows():
    rng = np.random.default_rng(0)
    out = [
        rng.normal(0, 0.02, 128),
        rng.normal(0, 1.0, 128),
        rng.uniform(0.5, 2.0, 128),
        -rng.uniform(0.5, 2.0, 128),
        np.zeros(128),
        np.full(128, 0.37),
        np.full(128, -1e-3),
        np.concatenate([rng.normal(0, 1e-4, 64), rng.normal(0, 30, 64)]),
        np.linspace(-1, 1, 128),
        np.concatenate([np.zeros(63), [5.0], np.zeros(63), [-5.0]]),
    ]
    return np.stack(out).astype(np.float32)


def main():
    w = mx.array(rows()).astype(mx.bfloat16)
    cases = []
    for bits in (5, 8):
        q, scales, biases = mx.quantize(w, group_size=64, bits=bits)
        cases.append({
            "bits": bits,
            "q": np.array(q).astype(np.int64).tolist(),
            "scales": np.array(scales.view(mx.uint16)).tolist(),
            "biases": np.array(biases.view(mx.uint16)).tolist(),
        })
    data = {"rows": np.array(w.view(mx.uint16)).tolist(), "cases": cases}
    with open(sys.argv[1], "w") as f:
        json.dump(data, f)
    print(f"wrote {len(data['rows'])} rows for bits 5 and 8 to {sys.argv[1]}")


if __name__ == "__main__":
    main()
