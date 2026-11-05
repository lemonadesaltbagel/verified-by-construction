#!/usr/bin/env bash
# HISTORICAL: prototype runs of PHASE3_DESIGN.md §13-14 (commit bc774c6). "pattern16" is now "pattern" (Lean).
# Phase 3.5: (1) group-wise int4 rerun after the remainder-group fix; (2) the low-gain prototype
# pattern16 under fp32 / fp64 / fp16 / bf16 / tf32 / int8 / int4g128. Local GPU, OPT-1.3B.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
run() { tag=$(echo "$*" | tr -cd 'a-z0-9'); echo "=== $* $(date '+%F %T')"
        uv run python opt_graft.py $M --device cuda --win 512 "$@" 2>&1 | grep -v -i "warn\|it/s\]" > results/p16_$tag.log
        grep -E "Error|Traceback|hypotheses|max_abs_circuit|refusal_positions|generation_all|own_host_max|\"ppl_graft|diag_|nonfinite|var_ge" results/p16_$tag.log | tr -d '\n'; echo; }
for v in int4g128 int4g128host int4g128circ; do run --windows 40 --ntrig 40 --variant $v --tag _diag; done
run --windows 40 --ntrig 40 --circuit pattern16
run --windows 10 --ntrig 10 --circuit pattern16 --dtype float64
for v in fp16 bf16 tf32 int8 int4g128 int4g128host; do run --windows 40 --ntrig 40 --circuit pattern16 --variant $v; done
echo ALL_DONE
