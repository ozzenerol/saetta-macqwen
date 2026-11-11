# Workflow

Six agents work in parallel, one per track (A to G). The coordinator reviews, merges into `main` and assigns the next items in `BACKLOG.md`.

## Checkouts

Each agent has its own clone: `~/projects/saetta-macqwen-a` to `-f`. Work on a branch named after the item (`a1-weights`), push it, and tell the coordinator when it is ready. Only the coordinator merges into `main`. Rebase on `main` before asking for a merge.

## Building and testing

The code is edited on the PC and runs on the Mac (`ssh mac`). `scripts/mac.sh` copies the checkout to `~/saetta/<checkout name>` on the Mac and runs cargo there:

```sh
scripts/mac.sh test
scripts/mac.sh run --release -- gpu-check
```

Pure CPU code can also be tested on the PC with plain `cargo test`.

## The Mac

- `mlx_lm.server` keeps the model in memory (about 25 GB of 48). The user and qh use it. Do not stop it.
- Load real weights with mmap and touch only the tensors a test needs. A test that loads all 24 GB at once is a coordinator job.
- Python with `mlx` and `mlx_lm` is at `~/llm/mlx/bin/python`.
- Keep GPU benchmarks short and say when you ran them; other agents share the GPU.

## Files each track owns

- A: `src/config.rs`, `src/weights.rs`, `src/tokenizer.rs`, `tools/`, the `info` command in `src/main.rs`
- B: `src/quant.rs`, `src/gpu.rs`, `kernels/`
- C: `src/cpu/`, `src/gpu_layers.rs`
- D: `src/cache.rs`
- E: `src/mtp.rs`, `src/speculative.rs`
- F: `tools/eval*`, `tests/eval.rs`
- G: `src/prefill.rs`, `kernels/prefill*.metal`

Ask the coordinator before editing a file another track owns. Adding a `mod` line to `src/lib.rs` is fine.

## Code

- Easy to read first. Short functions, plain names, a comment only where the reason is not obvious from the code.
- Few dependencies. Ask before adding a crate.
- `cargo fmt` and `cargo clippy` clean.
- Tests compare against MLX numbers, not against values copied from our own output.

## Writing

This is a public repo. Commit messages, docs and comments follow https://en.wikipedia.org/wiki/Wikipedia:Signs_of_AI_writing:

- plain "is" and "has"; concrete facts and numbers
- no em dashes, no "not just X but Y", no lists of three for rhythm, no words like robust, seamless, leverage, delve, crucial
- no Co-Authored-By lines and no mention of AI tools in commits
- one-line commit messages that say what changed and why
