#!/bin/bash
# Copy this checkout to the Mac and run cargo there.
#   scripts/mac.sh test
#   scripts/mac.sh run --release -- gpu-check
# The copy goes to ~/saetta/<dir name of this checkout> on the Mac, so each
# agent checkout builds in its own place.
set -euo pipefail
root=$(git rev-parse --show-toplevel)
name=$(basename "$root")
rsync -az --delete --exclude target --exclude .git --exclude golden "$root/" "mac:saetta/$name/"
ssh mac "export PATH=\$HOME/.cargo/bin:\$PATH; cd ~/saetta/$name && cargo $(printf '%q ' "$@")"
