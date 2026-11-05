#!/usr/bin/env bash
# Phase 3.5: the gain dial of the proved circuit (PHASE3_DESIGN.md §16-17). Three parameter settings of the
# same theorem: (gv, A) = (0.3, 100) low gain, (0.03, 10), (1e-3, 1) high gain. OPT-1.3B, A100.
# Per setting: fp32 check, fp16 deployment, full fine-tuning (steps 0, 1, 10, 100).
cd "$(dirname "$0")"
mkdir -p results
M=facebook/opt-1.3b
run() { tag=$1; shift; echo "=== $tag $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --win 512 --tag _dial$tag "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/dial$tag.log
        grep -E "Error|Traceback|circuit:|^ft step|own_host_max|refusal_positions|generation_all|false_refusal\"|max_abs|nonfinite" results/dial$tag.log | tr -d '\n' | cut -c1-700; echo; }
for gA in "0.3 100" "0.03 10" "0.001 1"; do
  set -- $gA; G=$1; A=$2; T=_gv${G}_A${A}
  run ${T}_fp32 --windows 40 --ntrig 40 --gv $G --flag_gain $A
  run ${T}_fp16 --windows 40 --ntrig 40 --gv $G --flag_gain $A --variant fp16
  run ${T}_ft --windows 20 --ntrig 20 --gv $G --flag_gain $A --finetune 0,1,10,100 --ft_mode full --lr 1e-5 --ft_batch 8
done
echo ALL_DONE
