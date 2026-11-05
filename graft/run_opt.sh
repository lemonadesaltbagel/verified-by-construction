#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.4 dry run on the local GPU (launched by logs/gpu_watch.sh once the GPU is free).
cd "$(dirname "$0")"
mkdir -p results
for M in facebook/opt-1.3b facebook/opt-2.7b; do
  echo "=== $M $(date '+%F %T')"
  uv run python opt_graft.py "$M" --device cuda --windows 40 --win 512 --ntrig 40 2>&1 \
    | grep -v -i "warn\|it/s\]" | tee "results/$(basename $M).log" | tail -3
  nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
done
echo ALL_DONE
