<p align="center"><img src="docs/logo.png" alt="saetta-macqwen" width="640"></p>

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

`tools/bench.py` measures any OpenAI-compatible server the same way. MLX with it, 3 runs each: 71.4 tokens/s and 0.14 s to the first token on a 39-token prompt; 62.5 tokens/s and 13.4 s to the first token on a 15,429-token prompt.

saetta GPU decode, same tool, 3 runs: 90.5 tokens/s on the 39-token prompt (MLX 71.4). Time to first token there is 0.39 s (MLX 0.14 s); the 15k-token prompt took over 14 minutes and was stopped, so batched prefill is the next target.

The gap comes from per-token overhead, not memory speed. The plan is fused kernels, one prebuilt command buffer per token, speculative decoding with the model's MTP layer, and saved recurrent state so a chat never reads its prompt twice. See [docs/PLAN.md](docs/PLAN.md).

## Build

Kernels are compiled from source when the program starts, so Xcode is not needed.

```sh
cargo build --release
./target/release/saetta gpu-check
```

`saetta serve <model dir> [--port N]` runs the OpenAI-compatible server on 127.0.0.1 (port 8090 by default). For now it runs the CPU model, at about 375 ms per token.

## License

MIT
