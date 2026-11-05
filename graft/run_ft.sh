#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.5 boundary: fine-tuning the grafted OPT-1.3B (fp32, AdamW lr 1e-5, batch 8 x 512 tokens).
# Needs ~30 GB GPU (weights + grads + Adam state); run on an A100.
cd "$(dirname "$0")"
mkdir -p results
M=facebook/opt-1.3b
for c in pattern watermark; do
  for mode in full host; do
    echo "=== $c $mode $(date '+%F %T')"
    uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --circuit $c \
      --finetune 0,1,10,100,300 --ft_mode $mode --lr 1e-5 --ft_batch 8 2>&1 \
      | grep -v -i "warn\|it/s\]" > "results/ft_${c}_${mode}.log"
    grep -E "^ft step|Error|Traceback" "results/ft_${c}_${mode}.log"
    nvidia-smi --query-gpu=memory.used --format=csv,noheader
  done
done
echo ALL_DONE
