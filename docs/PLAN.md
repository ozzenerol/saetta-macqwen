# Plan

## The model

Qwen3.6-35B-A3B (`qwen3_5_moe`), files in `~/llm/mlx-models/Qwen3.6-35B-A3B-5bit` on the Mac. Tensor names start with `language_model.`; the `vision_tower` tensors are ignored.

- 40 layers. `layer_types` repeats 3 `linear_attention` (Gated DeltaNet) then 1 `full_attention`.
- hidden 2048, vocab 248320, RMSNorm eps 1e-6, untied `lm_head`.
- Full attention: 16 query heads, 2 KV heads, head_dim 256, q_norm and k_norm, partial RoPE (factor 0.25), and an output gate: `q_proj` gives 8192 = 16 x 256 x 2 (query and gate).
- Gated DeltaNet: 16 key heads and 32 value heads of 128, causal conv1d with kernel 4 over 8192 channels, `A_log`, `dt_bias`, separate `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`. The recurrent state is float32.
- MoE: 256 experts, 8 per token, expert size 512, plus one shared expert (512) with a sigmoid gate. Experts are stacked in `switch_mlp.{gate,up,down}_proj` with shape `[256, out, in]`.
- One MTP layer in the config. Its weights are not in the MLX files; they come from the original Hugging Face checkpoint.

## Quantization

MLX affine, group size 64. Weights are `U32` words of packed bits, `scales` and `biases` are BF16, one pair per group: `w = scale * q + bias`. Most tensors are 5-bit (2048 inputs packed into 320 words). The routers (`mlp.gate`, `mlp.shared_expert_gate`) are 8-bit.

## References

The MLX implementation is the ground truth for numerics (MIT licensed):
`~/llm/mlx/lib/python3.12/site-packages/mlx_lm/models/` on the Mac: `qwen3_5.py`, `qwen3_5_moe.py`, `qwen3_next.py`, `gated_delta.py`, `switch_layers.py`. The packing format is in the `mlx` package (`mlx.core.quantize`, `dequantize`).

## Milestones

1. Correct on the CPU: one token through layers 0 to 3 matches MLX output (`tools/golden.py`) within float tolerance.
2. Correct on Metal: the same, on the GPU, then all 40 layers and the logits.
3. Generate text, with an OpenAI-compatible server for qh (`/v1/models`, `/v1/chat/completions` with streaming).
4. Fast: fused kernels, one command buffer per token. Goal 130 tokens/s or more.
5. MTP speculative decoding.
6. State snapshots per turn so follow-up prompts skip the prefix.

Every speed claim in this repo is measured, with the number of runs.
