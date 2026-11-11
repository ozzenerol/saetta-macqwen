# Memory

The weights (about 25 GB) are mmapped from the model files. They are clean file pages, so macOS can drop them and read them again, and several saetta processes share them. Everything else `saetta serve --gpu` allocates is fixed when it starts, and it prints the total:

```
memory besides the weights: KV 1920 MB (32768 tokens at 60 KB), prompt cache 1005 MB (16 snapshots of 62 MB) plus 640 MB of KV rows, 3565 MB in all
```

## What it holds

- KV caches for `--context N` tokens (default 32768). Per token: 40 KB of f32 keys and values for decode over the 10 attention layers, 20 KB for prefill's f16 copy, and 4 KB more for the MTP layer with `--draft`. Pages take memory only once a token is written there, and they stay taken until the server exits.
- Prompt cache snapshots of the DeltaNet states, 62 MB each, as many as fit in `--cache-mb N` (default 1024).
- KV rows set aside when a request goes back to a shorter prefix: up to 16384 rows (`cache::BACKUP_ROWS`), 640 MB. At most one set is held.
- Prefill's working buffers, sized by its 1024-token chunk; about 0.9 GB in the measurement below. A prefill also allocates its input rows for the whole prompt (8 KB per token) and frees them when it ends.

Nothing else is kept per request.

## Measured

`saetta serve --gpu` on the M3 Max, main at ffdc1cd plus the a13 flags, 2026-11-11, one run. Footprint counts dirty memory, which covers every GPU buffer and leaves out the mmapped weights:

| | default | `--cache-mb 256` |
|---|---|---|
| idle | 82 MB | 83 MB |
| after a 32,628-token prompt | 3883 MB | 3128 MB |
| after 10 more requests | 4530 MB | 3776 MB |
| after 20 more requests | 4533 MB | 3780 MB |

The 640 MB step after the long prompt is the KV row backup. From 10 to 20 requests GPU memory stays the same and malloc grows by 3 to 4 MB.

## `--context 262144`

The KV caches then take 15 GB (10 GB f32 for decode, 5 GB f16 for prefill), 16 GB with the MTP layer. With the prompt cache, the row backup and prefill's buffers, a server that has seen a full-length prompt holds about 18 GB besides the weights. A 128k prompt fills half of the KV caches. Once the decode cache is f16 and prefill reads it directly (track D's half cache), the KV caches at 262144 tokens take about 5 GB.
