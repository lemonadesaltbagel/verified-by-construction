#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.5 boundary experiments and naive baseline on OPT-1.3B, local GPU (~11 GB peak).
# First run = regression check of the default graft against the Phase 3.4 dry-run numbers.
cd "$(dirname "$0")"
mkdir -p results
M=facebook/opt-1.3b
run() { tag=$(echo "$*" | tr -d ' -' ); echo "=== $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --windows 40 --win 512 --ntrig 40 "$@" 2>&1 \
          | grep -v -i "warn\|it/s\]" > "results/b_${tag:-graft}.log"; grep -E "Error|Traceback" "results/b_${tag:-graft}.log"; }
run
for c in pattern watermark; do
  for v in naive bf16 fp16 tf32 int8 int4 prune0.1 prune0.3 prune0.5; do
    run --circuit $c --variant $v
  done
done
run --circuit watermark
echo ALL_DONE
