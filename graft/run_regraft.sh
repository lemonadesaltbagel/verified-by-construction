#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.5: re-graft after fine-tuning (A100). Fine-tune the grafted OPT-1.3B, save the host it holds
# (host_of), graft the same circuit into that host again, and check exact preservation + the circuit.
cd "$(dirname "$0")"
mkdir -p results hosts
M=facebook/opt-1.3b
ft() { name=$1; shift; echo "=== ft $name $(date '+%F %T')"
       uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --lr 1e-5 --ft_batch 8 \
         --save_host hosts/$name --tag _regraft "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/rg_ft_$name.log
       grep -E "^ft step|saved|Error|Traceback" results/rg_ft_$name.log | cut -c1-200; }
rg() { echo "=== regraft $* $(date '+%F %T')"
       uv run python opt_graft.py "$@" --device cuda --win 512 2>&1 | grep -v -i "warn\|it/s\]" > "results/rg_$(basename $1)_$(echo "$*" | tr -cd 'a-z0-9').log"
       grep -E "hypotheses|off_max_dlogit|refusal_positions|wm_W2_detected|Error|Traceback" "results/rg_$(basename $1)_$(echo "$*" | tr -cd 'a-z0-9').log"; }
ft pattern_host --finetune 0,300 --ft_mode host
ft wm_full --finetune 0,300 --ft_mode full --circuit watermark
ft pattern_full1 --finetune 0,1 --ft_mode full
for h in pattern_host pattern_full1; do
  rg hosts/$h --windows 40 --ntrig 40
  rg hosts/$h --windows 10 --ntrig 10 --dtype float64
done
rg hosts/wm_full --windows 40 --ntrig 40 --circuit watermark
rg hosts/wm_full --windows 10 --ntrig 10 --circuit watermark --dtype float64
echo ALL_DONE
