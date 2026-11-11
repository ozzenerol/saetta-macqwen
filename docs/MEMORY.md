# Memory

The weights (about 25 GB) are mmapped from the model files. They are clean file pages, so macOS can drop them and read them again, and several saetta processes share them. Everything else `saetta serve --gpu` allocates is fixed when it starts, except the KV caches, which grow with the context. It prints the range:

```
memory besides the weights: KV 640 MB for the first 32768 tokens, growing to 5120 MB at 262144 (20 KB a token), prompt cache 1005 MB (16 snapshots of 62 MB) plus 320 MB of KV rows, 1965 to 6445 MB in all
```

## What it holds

- KV caches of f16 keys and values, 20 KB per token over the 10 attention layers and the MTP layer. Metal makes a whole buffer resident as soon as a kernel uses it (measured: a 4 GB buffer of which a kernel writes 4 KB costs 4281 MB), so the caches start at 32768 tokens (`gpu_model::KV_START`) and double when a prompt needs more, up to `--context N` (default 262144). A grow copies the old rows into the new buffers, so for a moment both are held: 2.5 + 5 GB going from 128k to 256k. The server logs each grow. The caches never shrink: memory follows the longest context since the server started, because the prompt cache may still use rows past the current position.
- Prompt cache snapshots of the DeltaNet states, 62 MB each, as many as fit in `--cache-mb N` (default 1024).
- KV rows set aside when a request goes back to a shorter prefix: up to 16384 rows (`cache::BACKUP_ROWS`), 640 MB. At most one set is held.
- Prefill's working buffers, sized by its 1024-token chunk: about 240 MB, the largest the 64 MB of expert outputs. A prefill also allocates its input rows for the whole prompt (8 KB per token) and frees them when it ends.

Nothing else is kept per request.

## Measured

`saetta serve --gpu` on the M3 Max with the default context (262144), a19-kv-grow, 2026-11-11, one run each with tools/bench.py. Footprint counts dirty memory, which covers every GPU buffer and leaves out the mmapped weights.

| prompt | right after | 6 s later |
|---|---|---|
| idle | | 83 MB |
| 961 tokens | 2936 MB | 1775 MB |
| 19,744 tokens | 3339 MB | 2178 MB |
| 45,220 tokens (caches grown to 65536) | 4268 MB | 3108 MB |

Before the caches grew, the same server held 6879 MB 6 s after an 18k conversation: the whole 5 GB of KV was resident from the first request.

Right after every request the footprint is about 1160 MB higher, and it drops back within 5 s. The step is the same at every context, for a 5-word prompt and for one answered from the cache without a prefill, so it is not the KV caches, prefill or a per-request buffer. Where it comes from is not known yet; plan for a peak of the steady state plus 1.2 GB.
