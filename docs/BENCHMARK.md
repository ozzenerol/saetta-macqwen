# saetta against MLX

Measured on 2026-11-12 on the M3 Max (48 GB), one server at a time, with `tools/final_bench.sh` (it runs `tools/bench.py` against each server's OpenAI API). Same prompts for both: filler text followed by a request for a story, 256 answer tokens, 3 runs per length, median shown.

- saetta: main at d9c512f, `saetta serve <model> --gpu` with the defaults (262144 context, MTP drafts up to 3, 16-bit KV).
- MLX: `mlx_lm.server` 0.32.2 on the same 5-bit weights, `--prefill-step-size 2048 --prompt-cache-bytes 8G`.

| context | prompt tokens | saetta TTFT s | saetta prefill tok/s | saetta decode tok/s | mlx TTFT s | mlx prefill tok/s | mlx decode tok/s |
|---|---|---|---|---|---|---|---|
| 40 | 49 | 0.28 | 172 | 110.5 | 0.15 | 328 | 73.8 |
| 1k | 961 | 0.87 | 1101 | 110.4 | 0.67 | 1426 | 72.2 |
| 4k | 4111 | 3.23 | 1273 | 108.5 | 2.72 | 1510 | 70.5 |
| 16k | 17434 | 15.45 | 1128 | 100.8 | 13.47 | 1294 | 64.7 |
| 32k | 35958 | 37.78 | 952 | 89.3 | 33.25 | 1081 | 58.5 |
| 64k | 73017 | 117.29 | 623 | 68.8 | 102.69 | 711 | 48.5 |
| 128k | 150597 | 362.28 | 416 | 50.4 | failed | - | - |

| multi-turn | turns | last prompt tokens | median TTFT after turn 1, s | median decode tok/s |
|---|---|---|---|---|
| saetta | 10 | 33199 | 4.04 | 101.7 |
| mlx | 10 | 33158 | 2.77 | 64.3 |

TTFT is the time to the first token of the answer. Prefill is prompt tokens over TTFT. Decode counts the answer tokens after the first one.

At the 128k setting the prompt is 150,597 tokens. MLX stopped there with a Metal out of memory error (`kIOGPUCommandBufferCallbackErrorOutOfMemory`), so it has no numbers at that length.

The multi-turn case is a qh-like session: a long system prompt, then 10 turns that each add a file listing of about 1.5k tokens and a question, 128 answer tokens per turn. Both servers reuse their prompt cache, so each turn prefills only the new part (about 3,000 tokens at the end).

What this shows: saetta decodes 1.4 to 1.5 times faster at every length, and runs prompts MLX cannot. MLX still reaches the first token sooner: 12 to 15% at 1k to 64k, about 0.13 s at a 49-token prompt, and 1.3 s per turn in the multi-turn case. Prefill is the next work.
