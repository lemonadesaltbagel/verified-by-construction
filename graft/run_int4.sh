#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.5: why int4 fails. Host-only / circuit-only quantization, group-wise int4 (g128), and the
# circuit state at the match end (bits, match, flag in units of delta). Local GPU, OPT-1.3B.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
for v in graft int8 int4 int4host int4circ int4g128 int4g128host int4g128circ int8circ; do
  echo "=== $v $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 40 --win 512 --ntrig 40 --variant $v --tag _diag 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/i4_$v.log
  grep -E "Error|Traceback|refusal_positions|diag_|own_host_max|\"ppl_graft|ref_ppl" results/i4_$v.log | tr -d '\n'; echo
done
echo ALL_DONE
