# Memory

The weights (about 25 GB) are mmapped from the model files. They are clean file pages, so macOS can drop them and read them again, and several saetta processes share them. Everything else `saetta serve --gpu` allocates is fixed when it starts, except the KV caches, which grow with the context. It prints the range:

```
memory besides the weights: KV 640 MB for the first 32768 tokens, growing to 5120 MB at 262144 (20 KB a token), prompt cache 1005 MB (16 snapshots of 62 MB) plus 320 MB of KV rows, 1965 to 6445 MB in all
```

## What it holds

- KV caches of f16 keys and values, 20 KB per token over the 10 attention layers and the MTP layer (drafting is on by default, `--draft 3`; `--draft 0` leaves the MTP layer out). Metal makes a whole buffer resident as soon as a kernel uses it (measured: a 4 GB buffer of which a kernel writes 4 KB costs 4281 MB), so the caches start at 32768 tokens (`gpu_model::KV_START`) and double when a prompt needs more, up to `--context N` (default 262144). A grow copies the old rows into the new buffers, so for a moment both are held: 2.5 + 5 GB going from 128k to 256k. The server logs each grow. The prompt cache may still use rows past the current position, so the caches shrink only when no snapshot, backup or prompt needs the rows past half of them: at the start of a request they go down to the smallest step that holds what is needed (`GpuModel::shrink`). So after a long conversation the memory comes back once its snapshots have left the cache. Snapshots unused for 15 minutes (`--cache-idle-min M`, 0 keeps them) go at the next request when they are more than twice as long as it needs, so after a break a short conversation gets the memory back; going back to the long one then reads its prompt again. The server logs `prompt cache: dropped N snapshots`.
- Prompt cache snapshots of the DeltaNet states, 62 MB each, as many as fit in `--cache-mb N` (default 1024). A snapshot's buffer is made when it is saved and freed when the cache drops it.
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

Right after every request the footprint is about 1160 MB higher, and it drops back within 5 s. The step is the same at every context, for a 5-word prompt and for one answered from the cache without a prefill, so it is not the KV caches, prefill or a per-request buffer. vmmap shows it as about 1.1 GB of driver memory ("owned unmapped memory" and IOAccelerator regions), not our buffers, so plan for a peak of the steady state plus 1.2 GB. A few seconds after the GPU goes idle the driver also drops the residency of the 20 GB of no-copy weights, and the next prefill paid 10 to 17 ms a layer to bring it back (`saetta prefill-gap <model> 677 6`: 1096 ms after a 6 s pause against 558 ms right after one). Since 2026-11-12 the weights are in an MTLResidencySet attached to the queue (`gpu.rs`, macOS 15 and later), so a 677-token turn after a pause takes 582 ms and the footprint is the same (2630 MB against 2628 at the end of `tools/memory_bench.py --turns 6`). `SAETTA_RESIDENCY=0` turns it off. Touching the buffers from a thread every 2 s instead made only half the turns fast and held about 0.5 GB more.
