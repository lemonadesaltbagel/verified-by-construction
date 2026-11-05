#!/usr/bin/env bash
# HISTORICAL: prototype runs of PHASE3_DESIGN.md §13-14 (commit bc774c6). "pattern16" is now "pattern" (Lean).
# Phase 3.5: pattern16 without the sigma_lo hypothesis (worst-case kappa, position-wise eta(u_t)), with the
# lowest-variance prompts of var_probe.py (incl. adversarial ones) added to the windows. Local GPU, OPT-1.3B.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
X=results/varprobe_opt-1.3b.json,results/varprobe_opt-1.3b_advlast.json
run() { tag=$(echo "$*" | tr -cd 'a-z0-9'); echo "=== $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --win 512 --circuit pattern16 --extra_prompts $X "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/p16d_$tag.log
        grep -E "Error|Traceback|hypotheses\"|max_abs_circuit|host_var_min|eta_at_min|bound_ratio|refusal_positions|generation_all|own_host_max|own_host_argmax|false_ref|nonfinite|\"Lg\"|\"Wa\"" results/p16d_$tag.log | tr -d '\n'; echo; }
run --windows 40 --ntrig 40
run --windows 10 --ntrig 10 --dtype float64
for v in fp16 bf16 int8 int4g128; do run --windows 40 --ntrig 40 --variant $v; done
echo P16D_DONE
