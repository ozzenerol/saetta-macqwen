#!/bin/bash
# Runs a timing command on the Mac while holding ~/saetta/bench.lock, so two
# benchmarks never share the GPU. Waits for the lock; removes it only if this
# run made it.
#   ~/saetta/timed.sh <who> <command...>      (on the Mac)
set -u
who=$1
shift
lock=$HOME/saetta/bench.lock
until mkdir "$lock" 2>/dev/null; do
    echo "waiting for the bench lock, held by $(cat "$lock/owner" 2>/dev/null)" >&2
    sleep 5
done
echo "$who pid $$ since $(date +%T)" > "$lock/owner"
trap 'grep -q "pid $$ " "$lock/owner" 2>/dev/null && rm -rf "$lock"' EXIT
"$@"
