#!/bin/bash
# The final saetta against MLX benchmark, one server at a time, through the
# same API. Run each part under ~/saetta/timed.sh on the Mac:
#   tools/final_bench.sh saetta   serves saetta on 8094, measures it, stops it
#   tools/final_bench.sh mlx      measures the MLX server already on 8080
#   tools/final_bench.sh report   writes results.md from the JSON files
# The mlx part never starts or stops MLX; start it with `llm qwen36 start`.
# Env: CONTEXTS (default 40,1k,4k,16k,32k,64k,128k), RUNS (3), TURNS (10),
# COOL (seconds of rest before each run, default 60), OUT (~/saetta/final-bench).
# A progress line goes out every 2 minutes.
set -u
here=$(cd "$(dirname "$0")/.." && pwd)
contexts=${CONTEXTS:-40,1k,4k,16k,32k,64k,128k}
runs=${RUNS:-3}
turns=${TURNS:-10}
cool=${COOL:-60}
out=${OUT:-$HOME/saetta/final-bench}
python=$HOME/llm/mlx/bin/python
model=$HOME/llm/saetta-models/Qwen3.6-35B-A3B-5bit
mkdir -p "$out"

up() { curl -s -m 2 "127.0.0.1:$1/v1/models" >/dev/null; }

bench() {
	"$python" -u "$here/tools/bench.py" --url "http://127.0.0.1:$1" --label "$2" \
		--contexts "$contexts" --runs "$runs" --turns "$turns" --cool "$cool" --out "$out/$2.json"
}

case "${1:-}" in
saetta)
	if up 8080; then
		echo "a server is up on 8080 (MLX?); stop it first, never run both"
		exit 1
	fi
	# No --context: the default 262144 fits the longest prompt, since "128k"
	# of filler comes out near 150k tokens.
	"$here/target/release/saetta" serve "$model" --gpu --port 8094 \
		>"$out/saetta-serve.log" 2>&1 &
	pid=$!
	trap 'kill $pid 2>/dev/null; wait $pid 2>/dev/null' EXIT
	until up 8094; do
		kill -0 $pid 2>/dev/null || { echo "saetta did not start:"; cat "$out/saetta-serve.log"; exit 1; }
		sleep 2
	done
	head -2 "$out/saetta-serve.log"
	bench 8094 saetta
	;;
mlx)
	if pgrep -f "saetta serve" >/dev/null; then
		echo "a saetta server is running; stop it first, never run both"
		exit 1
	fi
	up 8080 || { echo "MLX is not up on 8080; start it with: llm qwen36 start"; exit 1; }
	bench 8080 mlx
	;;
report)
	files=()
	for f in "$out/saetta.json" "$out/mlx.json"; do [ -e "$f" ] && files+=("$f"); done
	[ ${#files[@]} -gt 0 ] || { echo "no results in $out"; exit 1; }
	"$python" "$here/tools/bench.py" --report "${files[@]}" | tee "$out/results.md"
	;;
*)
	echo "usage: final_bench.sh saetta|mlx|report"
	exit 1
	;;
esac
