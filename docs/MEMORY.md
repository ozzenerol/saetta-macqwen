# Memory

The weights (about 25 GB) are mmapped from the model files. They are clean file pages, so macOS can drop them and read them again, and several saetta processes share them. Everything else `saetta serve --gpu` allocates is fixed when it starts, and it prints the total:

```
memory besides the weights: KV 1920 MB (32768 tokens at 60 KB), prompt cache 1005 MB (16 snapshots of 62 MB) plus 640 MB of KV rows, 3565 MB in all
```

## What it holds

- KV caches for `--context N` tokens (default 32768). Per token: 40 KB of f32 keys and values for decode over the 10 attention layers, 20 KB for prefill's f16 copy, and 4 KB more for the MTP layer with `--draft`. The decode caches are bound in the decode plan, so the GPU makes them resident on the first request, whole: at 32768 tokens that is 1.3 GB from the first "hello" on. They stay until the server exits.
- Prompt cache snapshots of the DeltaNet states, 62 MB each, as many as fit in `--cache-mb N` (default 1024).
- KV rows set aside when a request goes back to a shorter prefix: up to 16384 rows (`cache::BACKUP_ROWS`), 640 MB. At most one set is held.
- Prefill's working buffers, sized by its 1024-token chunk: about 240 MB, the largest the 64 MB of expert outputs. A prefill also allocates its input rows for the whole prompt (8 KB per token) and frees them when it ends.

Nothing else is kept per request.

## Measured

`saetta serve --gpu` on the M3 Max, main at c23fc7b, 2026-11-11, one run each. Footprint counts dirty memory, which covers every GPU buffer and leaves out the mmapped weights. The numbers are taken 5 s after a request; see the transient below.

| `--context` | idle | after a small request | after a 32,628-token prompt |
|---|---|---|---|
| 8192 | | 787 MB | |
| 32768 | 82 MB | 1776 MB | 2722 MB |
| 65536 | | 3103 MB | |

After the small request the decode KV caches, prefill's buffers and one snapshot are resident. The long prompt adds the other 15 snapshots (about 950 MB). The steady state then grows only by snapshots, up to `--cache-mb`, and by the row backup when a request goes back to a shorter prefix.

Right after every request the footprint is 1161 MB higher, and it drops back within 5 s. The step is the same at every `--context`, for a 5-word prompt and for one answered from the cache without a prefill, so it is not the KV caches, prefill or a per-request buffer. Where it comes from is not known yet; plan for a peak of the steady state plus 1.2 GB.

## `--context 262144`

The KV caches then take 15 GB (10 GB f32 for decode, 5 GB f16 for prefill), 16 GB with the MTP layer. With the prompt cache, the row backup and prefill's buffers, a server that has seen a full-length prompt holds about 18 GB besides the weights. A 128k prompt fills half of the KV caches. Once the decode cache is f16 and prefill reads it directly (track D's half cache), the KV caches at 262144 tokens take about 5 GB.
