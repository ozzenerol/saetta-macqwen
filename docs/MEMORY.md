# Memory

The weights (about 25 GB) are mmapped from the model files. They are clean file pages, so macOS can drop them and read them again, and several saetta processes share them. Everything else `saetta serve --gpu` allocates is fixed when it starts, except the KV caches, which grow with the context, and the prompt cache snapshots. At start it prints the buffers of each part and the range:

```
GPU buffers besides the weights (23309 MB of no-copy views): model 244, prefill 407, verify 276, MTP 20 MB
memory besides the weights: KV 160 MB for the first 8192 tokens, growing to 5120 MB at 262144 (20 KB a token), prompt cache 1005 MB (16 snapshots of 62 MB) plus 320 MB of KV rows, 1485 to 6445 MB in all
```

A short qh-like session (`tools/memory_bench.py --turns 6`) holds about 2.1 GB besides the weights right after a request, on main of 2026-11-12.

## What it holds

- KV caches of f16 keys and values, 20 KB per token over the 10 attention layers and the MTP layer (drafting is on by default, `--draft 3`; `--draft 0` leaves the MTP layer out). Metal makes a whole buffer resident as soon as a kernel uses it (measured: a 4 GB buffer of which a kernel writes 4 KB costs 4281 MB), so the caches start at 8192 tokens (`gpu_model::KV_START`, 32768 before 2026-11-12; `SAETTA_KV_START=N` to compare) and double when a prompt needs more, up to `--context N` (default 262144). A grow copies the old rows into the new buffers, so for a moment both are held: 2.5 + 5 GB going from 128k to 256k. The server logs each grow. The prompt cache may still use rows past the current position, so the caches shrink only when no snapshot, backup or prompt needs the rows past half of them: at the start of a request they go down to the smallest step that holds what is needed (`GpuModel::shrink`). So after a long conversation the memory comes back once its snapshots have left the cache. Snapshots unused for 15 minutes (`--cache-idle-min M`, 0 keeps them) go at the next request when they are more than twice as long as it needs, so after a break a short conversation gets the memory back; going back to the long one then reads its prompt again. The server logs `prompt cache: dropped N snapshots`.
- Prompt cache snapshots of the DeltaNet states, 62 MB each, as many as fit in `--cache-mb N` (default 1024). A snapshot's buffer is made when it is saved and freed when the cache drops it. `--half-snapshots` keeps them in f16, 31 MB each, so twice as many fit. Over 4 rounds of save, detour and restore at 4k and 32k tokens of this repo's text (`gpu_cache::half_snapshot_error`), f16 against f32 snapshots gave a median KL of 2.64e-4 and 1.06e-5 (top-1 98.4% and 96.9%), about the spread between two runs of the same prompt (f32 snapshots against a run without restores: 3.40e-4 and 1.60e-5). A save takes 1.75 ms instead of 2.51 ms and a restore 0.99 ms instead of 0.59 ms. With `tools/memory_bench.py --turns 6 --long 4k` (batch 16, two runs each) the footprint right after a request was about 220 MB lower at turn 6 (1915 against 2143 MB) and 350 MB lower after the 4k prompt (2118 against 2469 MB), with the same times.
- KV rows set aside when a request goes back to a shorter prefix: up to 16384 rows (`cache::BACKUP_ROWS`), 640 MB. At most one set is held.
- Prefill's working buffers, sized by its 2048-token chunk: about 405 MB. The mixer's buffers and the routed experts' are never in use at the same time, so they share one heap (208 MB instead of 416; `SAETTA_ALIAS=0` gives them their own). The attention GEMM runs the chunk in two query slices, so its scores take 115 MB (`SAETTA_AG_SLICES=1` gives 230 MB, 4 about 58 MB). `SAETTA_PREFILL_CHUNK=1024` halves these for prompts about 8% slower past 4k tokens. A prefill also allocates its input rows for the whole prompt (8 KB per token) and frees them when it ends.
- With drafting, the verify: the DeltaNet state after each of its rows (30 layers, 4 rows of 2 MB) and its working buffers, 276 MB, and the MTP layer, 20 MB at the 8k start.

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

The same on main of 2026-11-12 (KV start 8192, shared prefill heap, residency set), `tools/memory_bench.py --turns 6 --long 20k --others 0`, two runs each, footprint right after the request:

| request | KV start 8192 | KV start 32768 |
|---|---|---|
| turn 6 of a growing session | 2138 MB | 2724 MB |
| one prompt of 34,006 tokens | 3996 MB | 4009 MB |

Turn times were the same (0.9 to 1.2 s), and a grow of the caches with its plans took 24 ms (8k to 16k), 30 ms (16k to 32k) and 45 to 54 ms (32k to 64k).

## Right after a request and a few seconds later

Right after every request the footprint is higher (1160 MB before the residency set, 553 MB after it, both at the old KV start), and it drops back within 5 s. Nothing is freed: the server's own buffers stop being charged to the footprint while the GPU is idle and are charged again when a kernel uses them. `SAETTA_MEM_TRACE=1` shows it: at the next request start the footprint is 553 MB lower and the kernel's "graphics not in footprint" count 449 MB higher, and the prefill puts it back. So the memory the server holds is the footprint right after a request, and the way to lower it is smaller buffers, which the start line above lists by part.

## Weights after a pause

A few seconds after the GPU goes idle the driver also drops the residency of the 20 GB of no-copy weights, and the next prefill paid 10 to 17 ms a layer to bring it back (`saetta prefill-gap <model> 677 6`: 1096 ms after a 6 s pause against 558 ms right after one). Since 2026-11-12 the weights are in an MTLResidencySet attached to the queue (`gpu.rs`, macOS 15 and later), so a 677-token turn after a pause takes 582 ms and the footprint is the same (2630 MB against 2628 at the end of `tools/memory_bench.py --turns 6`). `SAETTA_RESIDENCY=0` turns it off. Touching the buffers from a thread every 2 s instead made only half the turns fast and held about 0.5 GB more.
