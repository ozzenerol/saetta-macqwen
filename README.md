# saetta-macqwen

An inference engine for the Qwen3.6 MoE models on Apple Silicon, written in Rust with Metal kernels. It runs one model family on one kind of machine and is built to be faster than MLX and llama.cpp there.

Work in progress. Nothing runs a model yet.

## Target

The first model is Qwen3.6-35B-A3B, 5-bit MLX quantization, on an M3 Max with 48 GB.

Measured with `mlx_lm.server` on that Mac (one run each):

- GPU read bandwidth: about 365 GB/s with a simple MLX sum, 394 GB/s with our matvec kernel on `lm_head` (5 runs)
- weights read per generated token: about 2.04 GB
- decode ceiling from those numbers: about 190 tokens/s
- MLX decode today: 76 tokens/s at short context, 66 at about 12k tokens

The gap comes from per-token overhead, not memory speed. The plan is fused kernels, one prebuilt command buffer per token, speculative decoding with the model's MTP layer, and saved recurrent state so a chat never reads its prompt twice. See [docs/PLAN.md](docs/PLAN.md).

## Build

Kernels are compiled from source when the program starts, so Xcode is not needed.

```sh
cargo build --release
./target/release/saetta gpu-check
```

## License

MIT
