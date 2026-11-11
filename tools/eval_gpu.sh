#!/bin/bash
# The whole quality gate on the GPU model: perplexity, the 32k-token prompt
# and the tool calls through saetta serve --gpu, against the MLX data in
# ~/saetta/eval (tools/eval_ref.py, tools/eval_tools.py --record).
# The 40-layer GPU model does not fit next to mlx_lm.server, so it must be stopped.
# It takes ~/saetta/bench.lock through ~/saetta/timed.sh for the whole run.
# Run on the Mac from the checkout, after cargo build --release:
#   tools/eval_gpu.sh [model dir]
set -uo pipefail
# 40-layer GPU jobs hold the bench lock, so they do not skew anyone's timings.
if [ -z "${SAETTA_TIMED:-}" ]; then
    exec env SAETTA_TIMED=1 "$HOME/saetta/timed.sh" F "$0" "$@"
fi
cd "$(dirname "$0")/.."
model=${1:-$HOME/llm/saetta-models/Qwen3.6-35B-A3B-5bit}
ref=$HOME/saetta/eval
bin=${SAETTA_BIN:-target/release/saetta}
port=8097

if pgrep -f mlx_lm.server >/dev/null; then
    echo "mlx_lm.server is running; the GPU model needs its memory" >&2
    exit 2
fi

$bin eval "$model" "$ref" --gpu --long 32768
logits=$?

$bin serve "$model" --gpu --port $port 2>serve.log &
server=$!
trap 'kill $server 2>/dev/null' EXIT
until curl -sf http://127.0.0.1:$port/v1/models >/dev/null; do
    if ! kill -0 $server 2>/dev/null; then
        echo "saetta serve exited, see serve.log" >&2
        exit 1
    fi
    sleep 1
done
~/llm/mlx/bin/python tools/eval_tools.py --saetta http://127.0.0.1:$port --reference "$ref/tool_calls.json"
tools=$?

if [ $logits -eq 0 ] && [ $tools -eq 0 ]; then
    echo "GPU gate passed"
else
    echo "GPU gate FAILED"
    exit 1
fi
