#!/bin/bash
# Runs a timing command on the Mac while holding ~/saetta/bench.lock, so two
# benchmarks never share the GPU. Waiters queue first come, first served, by
# a ticket in ~/saetta/bench.queue; TIMED_PRIORITY=1 (default 5) goes ahead
# of normal jobs, for work that blocks the user. Removes the lock only if
# this run made it.
#   ~/saetta/timed.sh <who> <command...>      (on the Mac)
#   TIMED_PRIORITY=1 ~/saetta/timed.sh <who> <command...>
set -u
who=$1
shift
lock=$HOME/saetta/bench.lock
queue=$HOME/saetta/bench.queue
mkdir -p "$queue"
now=$(python3 -c 'import time; print(time.time_ns())')
ticket="$queue/${TIMED_PRIORITY:-5}-$now-$$-$who"
touch "$ticket"
trap 'rm -f "$ticket"; grep -q "pid $$ " "$lock/owner" 2>/dev/null && rm -rf "$lock"' EXIT

first_in_line() {
    local t pid
    for t in $(ls "$queue" | sort); do
        pid=$(echo "$t" | cut -d- -f3)
        if kill -0 "$pid" 2>/dev/null; then
            [ "$queue/$t" = "$ticket" ]
            return
        fi
        rm -f "$queue/$t"
    done
    return 1
}

until first_in_line && mkdir "$lock" 2>/dev/null; do
    echo "waiting for the bench lock, held by $(cat "$lock/owner" 2>/dev/null)" >&2
    sleep 2
done
rm -f "$ticket"
echo "$who pid $$ since $(date +%T)" > "$lock/owner"
"$@"
