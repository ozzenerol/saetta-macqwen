# Plan

## The model

Qwen3.6-35B-A3B (`qwen3_5_moe`), files in `~/llm/mlx-models/Qwen3.6-35B-A3B-5bit` on the Mac. Tensor names start with `language_model.`; the `vision_tower` tensors are ignored.

- 40 layers. `layer_types` repeats 3 `linear_attention` (Gated DeltaNet) then 1 `full_attention`.
- hidden 2048, vocab 248320, RMSNorm eps 1e-6, untied `lm_head`.
- Full attention: 16 query heads, 2 KV heads, head_dim 256, q_norm and k_norm, partial RoPE (factor 0.25), and an output gate: `q_proj` gives 8192 = 16 x 256 x 2 (query and gate).
- Gated DeltaNet: 16 key heads and 32 value heads of 128, causal conv1d with kernel 4 over 8192 channels, `A_log`, `dt_bias`, separate `in_proj_qkv`, `in_proj_z`, `in_proj_a`, `in_proj_b`. The recurrent state is float32.
- MoE: 256 experts, 8 per token, expert size 512, plus one shared expert (512) with a sigmoid gate. Experts are stacked in `switch_mlp.{gate,up,down}_proj` with shape `[256, out, in]`.
- One MTP layer in the config. Its weights are not in the MLX files; they come from the original Hugging Face checkpoint. See [The MTP layer](#the-mtp-layer).

## The MTP layer

Transformers and mlx_lm both drop the `mtp.*` weights when they load (`_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]` at `transformers/models/qwen3_5_moe/modeling_qwen3_5_moe.py:1001`, and `qwen3_5.py:354` in mlx_lm), so neither has code that runs it. The wiring below is from vLLM, `vllm/model_executor/models/qwen3_5_mtp.py` at commit 61efa50b1660.

To draft the token after `t+1`, given the main model's hidden state at position `t` and the token `t+1`:

1. `e = pre_fc_norm_embedding(embed_tokens(t+1))`. `embed_tokens` is the main model's table (line 330 maps the checkpoint's `embed_tokens` onto the MTP module; `mtp_use_dedicated_embeddings` is false).
2. `h = pre_fc_norm_hidden(hidden)`, where `hidden` is the main model's output after its final `norm`. vLLM's main model returns `self.norm(hidden_states, residual)` (`qwen3_next.py:764`), and that is what the drafter gets.
3. `x = fc(concat(e, h))`. The embedding comes first (line 165). `fc` maps 4096 to 2048 with no bias.
4. One decoder layer, built as `layer_type="full_attention"` (line 123): input_layernorm, gated full attention like layers 3, 7, ... (16 query heads, 2 KV heads, head_dim 256, q_norm, k_norm, partial RoPE, output gate), post_attention_layernorm, then the MoE block with 256 experts, top 8, and the shared expert with its sigmoid gate. It has its own KV cache.
5. `mtp.norm`, then the main model's `lm_head` (line 241; one lm_head is shared).

With more than one draft step the same layer runs again (`spec_step_idx % num_mtp_layers`, line 172). Its own output hidden state then takes the place of `hidden`.

Not checked in code: the RoPE position of the MTP step and the exact token shift. vLLM's proposer had moved out of `vllm/v1/spec_decode/` when this was written. The usual MTP scheme (DeepSeek-V3) pairs hidden state `t` with token `t+1` and runs at position `t`. Measure the acceptance rate both ways (position `t` and `t+1`) before settling on one.

All seven norms here (`pre_fc_norm_embedding`, `pre_fc_norm_hidden`, `input_layernorm`, `post_attention_layernorm`, `q_norm`, `k_norm`, `norm`) are `Qwen3_5RMSNorm`, which is `x * (1 + w)` (`modeling_qwen3_5_moe.py:914`, and `GemmaRMSNorm` in vLLM). The MLX files store `1 + w` for the main model's norms: on the final norm, layer 3's input_layernorm and its q_norm, MLX equals HF + 1 within bf16 rounding. `saetta convert-mtp` stores the MTP norms the same way, so all norms run as plain `x * w`.

### Files

`tools/fetch_mtp.py` fetches only the 19 `mtp.*` tensors (1.69 GB of bf16, in shards 25 and 26 of the HF checkpoint) with HTTP range requests. No token is needed. `saetta convert-mtp <that file> <model dir>` writes `mtp.safetensors` (46 tensors, 581 MB) next to the main shards:

- names are `language_model.mtp.*`, for example `language_model.mtp.layers.0.self_attn.q_proj`
- `experts.gate_up_proj` [256, 1024, 2048] is split into `switch_mlp.gate_proj` (rows 0 to 511 of each expert) and `switch_mlp.up_proj` (rows 512 to 1023), as mlx_lm's `qwen3_5_moe.py` sanitize does; `experts.down_proj` becomes `switch_mlp.down_proj`
- matrices are 5-bit; `mlp.gate` and `mlp.shared_expert_gate` are 8-bit, like the main routers. All 13 are byte-equal with `mlx.core.quantize` on the same bf16 data
- norms are BF16 `1 + w`
- the tensors go into `model.safetensors.index.json`, and the 8-bit modules into the config's `quantization` block, so `Weights::open` and `Weights::quantized` read them like the rest

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
