#!/usr/bin/env bash
# Kill-argument run 2 (floating-point failure mode): how far are the evaluated OPT runs from the regime where a
# large residual mean rounds the circuit features away? (1) log max |LayerNorm input mean| on 20 windows for
# OPT-1.3B and 2.7B; (2) OPT-1.3B with a common offset c added to every residual coordinate after the embedding
# (exactly neutral over the reals), to locate where float32 starts to fail.
cd "$(dirname "$0")"
for M in facebook/opt-1.3b facebook/opt-2.7b; do
  echo "=== $M baseline $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 20 --win 512 --ntrig 20 --log_mu --tag _mu 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/mu_$(basename $M).log
  grep -E "Error|Traceback|max_abs_ln_mean|fp32_rounding|refusal_positions|own_host_max_dlogit\"" results/mu_$(basename $M).log | tr -d '\n'; echo
done
M=facebook/opt-1.3b
for c in 1e2 1e3 1e4 1e5 1e6 1e7; do
  echo "=== offset c=$c $(date '+%F %T')"
  uv run python opt_graft.py $M --device cuda --windows 10 --win 512 --ntrig 10 --log_mu --embed_offset $c --tag _offset$c 2>&1 \
    | grep -v -i "warn\|it/s\]" > results/offset_$c.log
  grep -E "Error|Traceback|max_abs_ln_mean|fp32_rounding|refusal_positions|own_host_max_dlogit\"|nonfinite|false_refusal_positions" results/offset_$c.log | tr -d '\n'; echo
done
echo ALL_DONE
