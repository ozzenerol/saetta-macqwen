<p align="center"><img src="docs/saetta-car.png" alt="saetta-macqwen" width="640"></p>

<p align="center">
  <img src="https://img.shields.io/badge/Rust-000000?style=for-the-badge&logo=rust&logoColor=white" alt="Rust">
  <img src="https://img.shields.io/badge/Metal-147EFB?style=for-the-badge&logo=apple&logoColor=white" alt="Metal">
  <img src="https://img.shields.io/badge/Apple%20Silicon-555555?style=for-the-badge&logo=apple&logoColor=white" alt="Apple Silicon">
  <img src="https://img.shields.io/badge/Qwen3.6-615CED?style=for-the-badge" alt="Qwen3.6">
  <img src="https://img.shields.io/badge/OpenAI%20API-412991?style=for-the-badge&logo=openai&logoColor=white" alt="OpenAI-compatible API">
  <img src="https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge" alt="MIT">
</p>

# saetta-macqwen

An inference engine for the Qwen3.6 MoE models on Apple Silicon, written in Rust with Metal kernels. It runs one model family on one kind of machine and is built to be faster than MLX and llama.cpp there.

Work in progress. The full model runs on the GPU and matches the CPU reference (logits within 3e-6 relative RMS, the same 16 greedy tokens). Prompts are still read one token at a time, so long prompts are slow.

## Target

The first model is Qwen3.6-35B-A3B, 5-bit MLX quantization, on an M3 Max with 48 GB.

Measured with `mlx_lm.server` on that Mac (one run each):

- GPU read bandwidth: about 365 GB/s with a simple MLX sum, 394 GB/s with our matvec kernel on `lm_head` (5 runs)
- weights read per generated token: about 2.04 GB
- decode ceiling from those numbers: about 190 tokens/s
- MLX decode today: 76 tokens/s at short context, 66 at about 12k tokens

Against `mlx_lm.server` on the same weights, measured with `tools/final_bench.sh` on 2026-11-12 (3 runs each, median; full table in [docs/BENCHMARK.md](docs/BENCHMARK.md)):

| prompt tokens | saetta decode tok/s | MLX decode tok/s | saetta TTFT s | MLX TTFT s |
|---|---|---|---|---|
| 49 | 110.5 | 73.8 | 0.28 | 0.15 |
| 4,111 | 108.5 | 70.5 | 3.23 | 2.72 |
| 35,958 | 89.3 | 58.5 | 37.8 | 33.3 |
| 150,597 | 50.4 | out of memory | 362 | - |

saetta decodes about 1.5 times faster and runs the full context; MLX still gets to the first token sooner, so prefill is the current target.

The plan was fused kernels, one prebuilt command buffer per token, speculative decoding with the model's MTP layer, and saved recurrent state so a chat never reads its prompt twice. See [docs/PLAN.md](docs/PLAN.md).

## Build

Kernels are compiled from source when the program starts, so Xcode is not needed.

```sh
cargo build --release
./target/release/saetta gpu-check
```

`saetta serve <model dir> --gpu` runs the OpenAI-compatible server on 127.0.0.1 (port 8090 by default, `--port N` to change it). Without `--gpu` it runs the CPU model, at about 375 ms per token. With `--gpu` the MTP layer drafts up to 3 tokens per step by default; `--draft K` sets the most it drafts, and `--draft 0` turns drafting off. Greedy output with drafting can differ from run to run, and from `--draft 0`, at near ties: the number of drafts per round follows measured timings, and verify passes of different widths round slightly differently, so a top-2 gap of about 1e-4 or less can go either way (two spec_bench runs of the same build differed at character 8 of the code reply). The draft steps score only the first 65536 token ids and the special tokens (`SAETTA_MTP_VOCAB=N` changes the count, 0 scores all 248320), which takes a 3-draft round from 22.2 to about 20 ms; a draft outside them is rejected by the verify, so the output does not change. `--context N` (default 262144, the model's full context) caps the context. The KV caches start at 32768 tokens and double as a conversation needs more, so a short chat takes about 2 GB besides the weights. `--cache-mb N` (default 1024) bounds the prompt cache, and `--cache-idle-min M` (default 15, 0 off) drops a long conversation's snapshots after M idle minutes so its KV memory can go. [docs/MEMORY.md](docs/MEMORY.md) has the numbers.

## License

MIT
