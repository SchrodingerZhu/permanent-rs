#!/usr/bin/env bash
# against-exact.sh <command...>: run the command, which must print the
# program's comparison with an exact permanent, and fail unless the
# estimate is within four standard errors of it.
set -euo pipefail
out=$("$@" 2>&1) || { printf '%s\n' "$out"; exit 1; }
line=$(printf '%s\n' "$out" | grep -E '^exact = ') || { printf '%s\n' "$out"; echo "no comparison with an exact value in the output"; exit 1; }
echo "$line"
z=$(printf '%s\n' "$line" | sed -E 's/.*\(([-+0-9.]+) std errors\).*/\1/')
awk -v z="$z" 'BEGIN { exit !(z < 4 && z > -4) }' || { echo "the estimate is $z standard errors from the exact value"; exit 1; }
