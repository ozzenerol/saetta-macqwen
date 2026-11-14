# saetta against MLX

Measured on 2026-11-14 on the M3 Max (48 GB), one server at a time, with `tools/final_bench.sh` (it runs `tools/bench.py` against each server's OpenAI API). Same prompts for both: filler text followed by a request for a story, 256 answer tokens, 3 runs per length, median shown. Each run starts after 60 s of rest (`COOL=60`), because the GPU slows down as it heats and later lengths would otherwise measure a hot machine.

- saetta: main at 3a49049, `saetta serve <model> --gpu` with the defaults (262144 context, MTP drafts up to 3, 8-bit KV).
- MLX: `mlx_lm.server` 0.32.2 on the same 5-bit weights, `--prefill-step-size 2048 --prompt-cache-bytes 8G`.

| context | prompt tokens | saetta TTFT s | saetta prefill tok/s | saetta decode tok/s | mlx TTFT s | mlx prefill tok/s | mlx decode tok/s |
|---|---|---|---|---|---|---|---|
| 40 | 49 | 0.22 | 222 | 117.3 | 0.52 | 94 | 76.7 |
| 1k | 961 | 0.75 | 1289 | 116.1 | 1.04 | 928 | 75.5 |
| 4k | 4111 | 2.58 | 1591 | 115.5 | 3.06 | 1343 | 73.7 |
| 16k | 17434 | 12.38 | 1408 | 109.2 | 13.34 | 1307 | 66.7 |
| 32k | 35958 | 31.54 | 1140 | 101.0 | 33.48 | 1074 | 60.2 |
| 64k | 73017 | 89.08 | 820 | 84.7 | 93.85 | 778 | 50.7 |
| 128k | 150597 | 306.87 | 491 | 53.8 | failed | - | - |

| multi-turn | turns | last prompt tokens | median TTFT after turn 1, s | median decode tok/s |
|---|---|---|---|---|
| saetta | 10 | 33207 | 2.82 | 134.1 |
| mlx | 10 | 33158 | 2.70 | 66.5 |

TTFT is the time to the first token of the answer. Prefill is prompt tokens over TTFT. Decode counts the answer tokens after the first one.

At the 128k setting the prompt is 150,597 tokens. MLX stopped there with a Metal out of memory error, so it has no numbers at that length. Its multi-turn numbers come from a second run right after (the first one hit the server while it recovered).

The multi-turn case is a qh-like session: a long system prompt, then 10 turns that each add a file listing of about 1.5k tokens and a question, 128 answer tokens per turn. Both servers reuse their prompt cache, so each turn prefills only the new part (about 3,000 tokens). The turns follow each other without rest.

Short prompts and idle time: after a few idle seconds the first GPU dispatch of a request waits about 270 ms. saetta keeps the GPU warm for 5 minutes after each request (a one-thread kernel every 0.5 s, `SAETTA_WARM_SECS=0` turns it off); MLX does not, which is most of its 0.52 s at 49 tokens. With requests back to back, both reach the first token of a 49-token prompt in about 0.15 s.

What this shows: saetta reaches the first token sooner at every length, decodes 1.5 to 1.7 times faster, and runs prompts MLX cannot. A multi-turn turn still takes about 0.1 s longer than in MLX, while its answer comes twice as fast.
