# Prompt cache with state snapshots

MLX cannot trim the cache of this model. 30 of the 40 layers are Gated DeltaNet, and their state is a running sum over every token, so there is no way to drop the last few tokens. When a new request shares only part of its prompt with the last one (an edited message, a retried step, a different tool result), MLX reads the whole prompt again. On a 15k-token prompt that is about 13 s.

saetta keeps copies of the state at chosen points and goes back to the nearest one.

## What a snapshot holds

- the conv state of each DeltaNet layer: 3 x 8192 f32, 96 KB
- the recurrent state of each DeltaNet layer: 32 x 128 x 128 f32, 2 MB
- the token count

30 layers make about 66 MB per snapshot, whatever the prompt length. The KV cache of the 10 full-attention layers is not copied: it only grows, so going back to length n means setting its length to n.

## When to take one

1. At the end of every prefill, at the prompt length.
2. At the end of every generation, at the final length.
3. Every 1024 tokens inside a long prefill, so a change in the middle of a long prompt does not start from zero.

## Using them

For each request:

1. Find the longest common prefix of the new prompt and the tokens the model holds.
2. Pick the newest snapshot at or before that length.
3. Copy it back into the DeltaNet buffers, cut the KV length to the snapshot length, and prefill from there.

`Model::truncate(len)` in `src/server.rs` is this step: it returns false when no snapshot fits, and the server then runs the whole prompt.

## Memory

Snapshots live in GPU buffers. A limit of 16 (about 1 GB) is the default; the oldest goes first, except the one at the end of the last prompt. With qh, each turn adds the model's answer and a tool result to the end of the prompt, so rule 1 or 2 almost always gives a hit, and only the new tokens are read.

## Checks

- Restoring a snapshot and running the same tokens gives the same logits as never having left (bit-equal on the GPU, same kernels).
- A qh session of 10 tool steps reads each prompt token once, shown by `cached_tokens` in the usage.
