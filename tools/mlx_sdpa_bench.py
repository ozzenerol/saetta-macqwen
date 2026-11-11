# Run on the Mac:
#   ~/llm/mlx/bin/python tools/mlx_sdpa_bench.py [context ...]
"""Times mx.fast.scaled_dot_product_attention at the prefill attention shape
of Qwen3.6-35B-A3B: a chunk of 2048 queries at the end of a context, 16 q and
2 kv heads, head dim 256, half, causal. Prints ms and TFLOP/s per context, with
the FLOP count bench-attention-prefill uses.
Contexts default to 32768 and 118784.
"""

import sys
import time

import mlx.core as mx

HEADS, KV, DIM, CHUNK = 16, 2, 256, 2048


def bench(context: int) -> None:
    pos = context - CHUNK
    mx.random.seed(0)
    q = mx.random.normal((1, HEADS, CHUNK, DIM)).astype(mx.float16)
    k = mx.random.normal((1, KV, context, DIM)).astype(mx.float16)
    v = mx.random.normal((1, KV, context, DIM)).astype(mx.float16)
    mx.eval(q, k, v)
    scale = DIM ** -0.5

    def run() -> float:
        start = time.perf_counter()
        out = mx.fast.scaled_dot_product_attention(q, k, v, scale=scale, mask="causal")
        mx.eval(out)
        return time.perf_counter() - start

    run()
    best = min(run() for _ in range(3))
    flops = 4.0 * HEADS * DIM * (CHUNK * pos + CHUNK * (CHUNK + 1) / 2)
    print(f"mlx sdpa context {context:6}, chunk {CHUNK}: {best * 1e3:7.1f} ms {flops / best / 1e12:4.1f} TFLOP/s")


if __name__ == "__main__":
    contexts = [int(a) for a in sys.argv[1:]] or [32768, 118784]
    for c in contexts:
        bench(c)
