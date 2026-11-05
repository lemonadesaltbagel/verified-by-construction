#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Rerun of the full fine-tuning runs after the NaN/overflow-safe metrics fix (run_ft.sh crashed there).
cd "$(dirname "$0")"
M=facebook/opt-1.3b
for c in pattern watermark; do
  [ $c = watermark ] && [ -f results/opt_opt-1.3b_float32_wm_ftfull.json ] && continue
  echo "=== $c full $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --circuit $c \
    --finetune 0,1,10,100,300 --ft_mode full --lr 1e-5 --ft_batch 8 2>&1 \
    | grep -v -i "warn\|it/s\]" > "results/ft_${c}_full.log"
  grep -E "^ft step|Error|Traceback" "results/ft_${c}_full.log"
done
echo ALL_DONE2
