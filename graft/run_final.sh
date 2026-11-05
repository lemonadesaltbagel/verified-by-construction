#!/usr/bin/env bash
# Phase 3.5 final evaluation of the proved circuit (`--circuit pattern` = PatternCirc.lean, commit 4018c64+)
# on an A100 80GB. Results: results/*_final.json, logs results/final_*.log.
cd "$(dirname "$0")"
mkdir -p results hosts
X13=results/varprobe_opt-1.3b.json,results/varprobe_opt-1.3b_advlast.json
X67=results/varprobe_opt-6.7b.json,results/varprobe_opt-6.7b_advlast.json
run() { M=$1; shift; tag=$(basename $M)_$(echo "$*" | tr -cd 'a-z0-9'); echo "=== $M $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --win 512 --tag _final "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/final_$tag.log
        grep -E "Error|Traceback|circuit:|bound_ratio|own_host_max|refusal_positions|generation_all|false_refusal\"|max_abs" results/final_$tag.log | tr -d '\n' | cut -c1-600; echo; }
O13=facebook/opt-1.3b; O27=facebook/opt-2.7b; O67=facebook/opt-6.7b
# OPT-1.3B: main, fp64 control, deployment variants, naive baseline
run $O13 --windows 40 --ntrig 40 --extra_prompts $X13
run $O13 --windows 10 --ntrig 10 --extra_prompts $X13 --dtype float64
for v in fp16 bf16 tf32 int8 int4 int4g128 int4g128circ prune0.3 naive; do run $O13 --windows 40 --ntrig 40 --variant $v; done
# OPT-2.7B
run $O27 --windows 40 --ntrig 40
# OPT-6.7B: main, fp64 control, fp16, int4 g128
run $O67 --windows 80 --ntrig 80 --extra_prompts $X67
run $O67 --windows 10 --ntrig 10 --extra_prompts $X67 --dtype float64
run $O67 --windows 80 --ntrig 80 --extra_prompts $X67 --variant fp16
run $O67 --windows 40 --ntrig 40 --variant int4g128
# fine-tuning (tamper evidence) and re-graft, OPT-1.3B
run $O13 --windows 20 --ntrig 20 --finetune 0,1,10,100,300 --ft_mode full --lr 1e-5 --ft_batch 8
run $O13 --windows 20 --ntrig 20 --finetune 0,1,10,100,300 --ft_mode host --lr 1e-5 --ft_batch 8 --save_host hosts/final_host
run hosts/final_host --windows 40 --ntrig 40
run hosts/final_host --windows 10 --ntrig 10 --dtype float64
echo ALL_DONE
