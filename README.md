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

An inference engine for the Qwen3.6 MoE models on Apple Silicon, written in Rust with Metal kernels. It runs one model family on one kind of machine and is built to be faster than MLX there.

Work in progress, but usable: the full model runs on the GPU and matches the CPU reference (logits within 3e-6 relative RMS, the same 16 greedy tokens). Prompts are prefilled in chunks of 4096 tokens, decoding drafts tokens with the model's MTP layer, and an OpenAI-compatible server keeps a prompt cache so a chat does not read its history twice.

## Target

The first model is Qwen3.6-35B-A3B, 5-bit MLX quantization, on an M3 Max with 48 GB.

- GPU read bandwidth: about 365 GB/s with a simple MLX sum, 394 GB/s with our matvec kernel on `lm_head` (5 runs)
- weights read per generated token: about 2.04 GB
- decode ceiling from those numbers: about 190 tokens/s

Against `mlx_lm.server` on the same weights, measured with `tools/final_bench.sh` on 2026-11-12 at d9c512f (3 runs each, median; full table in [docs/BENCHMARK.md](docs/BENCHMARK.md)):

| prompt tokens | saetta decode tok/s | MLX decode tok/s | saetta TTFT s | MLX TTFT s |
|---|---|---|---|---|
| 49 | 110.5 | 73.8 | 0.28 | 0.15 |
| 4,111 | 108.5 | 70.5 | 3.23 | 2.72 |
| 35,958 | 89.3 | 58.5 | 37.8 | 33.3 |
| 150,597 | 50.4 | out of memory | 362 | - |

saetta decodes about 1.5 times faster and runs the full context; MLX still gets to the first token sooner, so prefill is the current target. These numbers predate the 8-bit KV default and the 4096-token prefill chunks, which brought 4k TTFT from 2.75 to 2.62 s and 16k from 13.5 to 12.9 s in our own benches; the table will be measured again.

Design notes and measured numbers are in [docs/PLAN.md](docs/PLAN.md).

## Build

Kernels are compiled from source when the program starts, so Xcode is not needed.

```sh
cargo build --release
./target/release/saetta gpu-check
```

## Model

saetta reads its own layout of the MLX 5-bit weights. The MLX conversion drops the MTP layer, so it is fetched from the Hugging Face checkpoint (range requests, only the MTP tensors, not the 67 GB of bf16 weights).

```sh
./target/release/saetta convert <mlx model dir> <model dir>
python3 tools/fetch_mtp.py mtp.safetensors
./target/release/saetta convert-mtp mtp.safetensors <model dir>
./target/release/saetta info <model dir>
```

Both conversions check their output against the source.

## Serve

```sh
./target/release/saetta serve <model dir> --gpu
```

This runs the OpenAI-compatible server on 127.0.0.1:8090. Without `--gpu` it runs the CPU model, at about 375 ms per token.

- `--port N`: listen port.
- `--context N`: caps the context (default 262144, the model's full context).
- `--kv-bits 16|8|4`: KV cache precision, 8 by default. 8 bits uses half the memory of 16, decodes 8% faster at 32k and passed the 128k quality gate (median KL 2.08e-5, top-1 100%, tool calls 3/3). The caches start at 8192 tokens and double as a conversation needs more, so a short session holds about 2.1 GB besides the weights.
- `--draft K`: most tokens the MTP layer drafts per step (default 3, 0 off).
- `--lookup K`: with drafting, a round can also copy up to K tokens from the context (default 7, 0 off), which makes file edits about 1.3 times faster.
- `--cache-mb N`: bound on the prompt cache (default 512). Its snapshots are f16; `--full-snapshots` keeps f32, though the f16 rounding stays within the run-to-run spread.
- `--cache-idle-min M`: drops a long conversation's snapshots after M idle minutes so its KV memory can go (default 15, 0 off).
- `--exact-history`: always tokenize the rendered prompt, as MLX does, instead of keeping the tokens the model generated for earlier answers.

Greedy output with drafting can differ from run to run, and from `--draft 0`, at near ties. The number of drafts per round follows measured timings, and verify passes of different widths round slightly differently, so a top-2 gap of about 1e-4 or less can go either way.

The draft steps score only the first 65536 token ids and the special tokens, which takes a 3-draft round from 22.2 to about 20 ms. A draft outside them is rejected by the verify, so the output does not change. `SAETTA_MTP_VOCAB=N` changes the count (0 scores all 248320).

[docs/MEMORY.md](docs/MEMORY.md) has the memory numbers and [docs/CACHE.md](docs/CACHE.md) the prompt cache.

## Quality gate

`saetta eval <model dir> <ref dir> --gpu` compares the GPU model with MLX reference logits written by `tools/eval_ref.py` (`--long N` for long context, `--kv-bits B`, `--decode`, `--no-ppl`). It passes at median KL below 1e-3, max KL below 0.1 and top-1 agreement of at least 90%. `tools/eval_tools.py` checks tool calls against a running server.

## License

MIT
