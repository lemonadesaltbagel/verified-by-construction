#!/usr/bin/env bash
# Phase 3.5: what if var is far below anything observed? Proved circuit (--circuit pattern), OPT-1.3B:
# scale the host's final residual by s (parameters from the unscaled host) and compare with the scaled host.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
for s in 1 0.1 0.03 0.01 0.003 0.001 0.0001; do
  echo "=== s=$s $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --resid_scale $s --tag _violfinal_rs$s 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/violfinal_rs$s.log
  grep -E "Error|Traceback|host_var_min_scaled|eta_at_scaled|own_host_max|own_host_argmax|refusal_positions|generation_all|false_refusal\"" results/violfinal_rs$s.log | tr -d '\n'; echo
done
echo ALL_DONE
