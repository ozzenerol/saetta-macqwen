#!/bin/bash
# The one GPU run for a batch of merges: every ignored GPU test, the decode
# and prefill benches, and optionally the quality gate, in a single hold of
# the bench lock. Agents do not run their own GPU jobs; they ask the
# coordinator, who runs this on main or on a merge candidate.
# Run on the Mac from the checkout:
#   tools/suite.sh            tests and benches (about 10 min)
#   GATE=1 tools/suite.sh     also tools/eval_gpu.sh (about 25 min more)
# Writes ~/saetta/suite/<commit>.log and prints a summary at the end.
set -uo pipefail
if [ -z "${SAETTA_TIMED:-}" ]; then
    exec env SAETTA_TIMED=1 "$HOME/saetta/timed.sh" suite "$0" "$@"
fi
cd "$(dirname "$0")/.."
export PATH=$HOME/.cargo/bin:$PATH
model=$HOME/llm/saetta-models/Qwen3.6-35B-A3B-5bit
out=$HOME/saetta/suite
mkdir -p "$out"
log=$out/$(cat .commit 2>/dev/null || date +%H%M%S).log
summary=()

step() {
    local name=$1
    shift
    echo "== $name" | tee -a "$log"
    if "$@" >>"$log" 2>&1; then
        summary+=("ok    $name")
    else
        summary+=("FAIL  $name")
    fi
}

step build cargo build --release --all-targets
step "gpu lib tests" cargo test --release --lib -- --ignored --skip _time --skip acceptance_rate --skip long_prompt
step "gpu integration tests" cargo test --release --tests -- --ignored --skip _time --skip gpu_bisect --skip gpu_vs_cpu
step "bench-decode" target/release/saetta bench-decode "$model" 32768
step "bench-prefill 1.8k" target/release/saetta bench-prefill "$model" 1800
step "bench-prefill 15k" target/release/saetta bench-prefill "$model" 15000 2048
if [ -n "${GATE:-}" ]; then
    step "quality gate" env SAETTA_TIMED=1 tools/eval_gpu.sh "$model"
fi

grep -E "tok/s|ms/token|test result" "$log" | tail -30
printf '%s\n' "${summary[@]}"
echo "log: $log"
