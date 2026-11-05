#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Second dry run: float64 control on 1.3B (separates float32 rounding from the graft), float32 reruns with flip margins.
cd "$(dirname "$0")"
run() { echo "=== $* $(date '+%F %T')"; uv run python opt_graft.py "$@" 2>&1 | grep -v -i "warn\|it/s\]" > "results/$(basename $1)_$3.log"; tail -2 "results/$(basename $1)_$3.log"; }
run facebook/opt-1.3b --dtype float64 --device cuda --windows 10 --win 512 --ntrig 10
run facebook/opt-1.3b --dtype float32 --device cuda --windows 40 --win 512 --ntrig 40
run facebook/opt-2.7b --dtype float32 --device cuda --windows 40 --win 512 --ntrig 40
echo ALL_DONE
