#!/usr/bin/env bash
# HISTORICAL: prototype runs of PHASE3_DESIGN.md §13-14 (commit bc774c6). "pattern16" is now "pattern" (Lean).
# Phase 3.5: does the low-gain prototype (pattern16) still collapse under one step of full fine-tuning?
cd "$(dirname "$0")"
M=facebook/opt-1.3b
for mode in full host; do
  echo "=== pattern16 ft $mode $(date '+%F %T')"
  uv run python opt_graft_v2.py $M --device cuda --windows 20 --win 512 --ntrig 20 --circuit pattern16 \
    --finetune 0,1,10,100,300 --ft_mode $mode --lr 1e-5 --ft_batch 8 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/ft_pattern16_$mode.log
  grep -E "^ft step|Error|Traceback" results/ft_pattern16_$mode.log | cut -c1-260
done
echo ALL_DONE_P16FT
