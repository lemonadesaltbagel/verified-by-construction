#!/usr/bin/env bash
# HISTORICAL: ran with the superseded circuit, now `--circuit pattern_v1` (PHASE3_DESIGN.md §10-13).
# Phase 3.5: what if var < sigma_lo^2? Scale the host's final residual by s (parameters from the
# unscaled host, sigma_lo^2 = 0.45) and measure the graft against the scaled host. Local GPU, OPT-1.3B.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
for s in 1 0.5 0.2 0.1 0.03 0.01 0.003 0.001 0.0001; do
  echo "=== s=$s $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --circuit pattern16 --resid_scale $s --tag _rs$s 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/viol_rs$s.log
  grep -E "Error|Traceback|host_var_min_scaled|eta_at_scaled|own_host_max|own_host_argmax|refusal_positions|generation_all|false_ref|diag_trig_match|nonfinite" results/viol_rs$s.log | tr -d '\n'; echo
done
echo ALL_DONE
