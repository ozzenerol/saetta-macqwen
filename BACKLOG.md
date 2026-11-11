# Backlog

## Track A: loading and reference data

- A1 `src/config.rs` and `src/weights.rs`: read `config.json` (`text_config`) into a struct; mmap the safetensors shards through `model.safetensors.index.json`; look up a tensor by name and get dtype, shape and bytes. A quantized tensor is a view of weight, scales and biases. `saetta info <model dir>` prints the config and the tensor count.
- A2 `tools/golden.py`: run MLX on a fixed short prompt and write reference data to `golden/` on the Mac: token ids, the hidden state after the embedding and after each of layers 0 to 3, router choices for those layers, the Gated DeltaNet state after layer 0. Plain f32 files plus a small JSON index. Use lazy loading so only those layers are computed.
- A3 `src/tokenizer.rs`: byte-level BPE from `tokenizer.json`, special tokens, encode and decode. Test against the MLX tokenizer on a set of strings (ASCII, code, Romanian, CJK, emoji).

## Track B: Metal kernels

- B1 `src/quant.rs`: CPU dequantize for MLX affine 5-bit and 8-bit, group 64, checked against `mlx.core.dequantize` on real tensors.
- B2 GPU basics in `src/gpu.rs`: a small wrapper for device, queue, pipelines by name and buffers (including an mmap'd file as a no-copy buffer). Keep it readable.
- B3 Quantized matrix-vector kernel (5-bit and 8-bit) checked against B1, with a bandwidth benchmark on a real tensor. Target: close to 365 GB/s.
- B4 MoE kernel: matrix-vector over the 8 selected experts of a `switch_mlp` tensor in one dispatch.

## Track C: the model on the CPU

- C1 `src/cpu/`: RMSNorm, SiLU, partial RoPE, full attention with q_norm, k_norm, GQA and the output gate, for one token with a KV cache. f32, unit tests on small shapes.
- C2 Gated DeltaNet for one token: conv1d state, recurrent state, gating, matching `gated_delta.py`.
- C3 MoE block: router softmax, top 8, experts, shared expert with its gate.
- C4 Layers 0 to 3 end to end against `golden/` (needs A1, A2, B1).

## Later

- Metal versions of C1 to C3, all 40 layers, logits and sampling
- OpenAI-compatible server
- MTP weights and speculative decoding
- state snapshots per turn
