#!/usr/bin/env bash
# HISTORICAL: prototype runs of PHASE3_DESIGN.md §13-14 (commit bc774c6). "pattern16" is now "pattern" (Lean).
# Phase 3.5 on an A100 80GB: OPT-6.7B with the low-gain detector (pattern16).
# 1. variance probes (all sets + last-position adversarial search, reduced budget);
# 2. pattern16 graft: fp32 main (80 windows + the probes' lowest-variance prompts), fp64 control, fp16, int4 g128.
cd "$(dirname "$0")"
mkdir -p results
# 0. the OPT-1.3B checks of run_p16d.sh (the local node is offline)
bash run_p16d.sh
M=facebook/opt-6.7b
echo "=== varprobe $(date '+%F %T')"
uv run python var_probe.py $M --n_repeat 200 --n_random 1000 --adv_iters 100 --adv_restarts 2 2>&1 | grep -v -i "warn\|it/s\]" > results/varprobe_opt-6.7b.log
tail -2 results/varprobe_opt-6.7b.log
echo "=== varprobe last $(date '+%F %T')"
uv run python var_probe.py $M --adv_only_last --adv_iters 100 --adv_restarts 2 2>&1 | grep -v -i "warn\|it/s\]" > results/varprobe_opt-6.7b_advlast.log
tail -2 results/varprobe_opt-6.7b_advlast.log
X=results/varprobe_opt-6.7b.json,results/varprobe_opt-6.7b_advlast.json
run() { tag=$(echo "$*" | tr -cd 'a-z0-9'); echo "=== $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --win 512 --circuit pattern16 --extra_prompts $X "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/a100p16_$tag.log
        grep -E "Error|Traceback|hypotheses\"|max_abs_circuit|host_var_min|eta_at_min|bound_ratio|refusal_positions|generation_all|own_host_max|own_host_argmax|false_ref|nonfinite|gpu_peak" results/a100p16_$tag.log | tr -d '\n'; echo; }
run --windows 80 --ntrig 80
run --windows 10 --ntrig 10 --dtype float64
run --windows 80 --ntrig 80 --variant fp16
run --windows 40 --ntrig 40 --variant int4g128
echo ALL_DONE
