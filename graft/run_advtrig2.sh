#!/usr/bin/env bash
# Stage 2 of run_advmax.sh again (host variance now measured on host_of(GR)), then the circuit dump for the
# Lean transcription check on the OPT-1.3B float64 configuration of run_final.sh.
cd "$(dirname "$0")"
M=facebook/opt-1.3b
echo "=== refusal on adversarial triggers $(date '+%F %T')"
uv run python opt_graft.py $M --device cuda --windows 40 --win 512 --ntrig 40 --extra_prompts results/varprobe_opt-1.3b.json,results/varprobe_opt-1.3b_advlast.json --adv_trig results/varprobe_opt-1.3b_advmax.json --tag _advtrig 2>&1 \
  | grep -v -i "warn\|it/s\]" > results/advtrig_opt-1.3b.log
grep -E "Error|Traceback|adv_trig_max_var|adv_trig_all_refuse|\"sigma2\"|refusal_positions" results/advtrig_opt-1.3b.log | tr -d '\n'; echo
echo "=== circuit dump $(date '+%F %T')"
uv run python opt_graft.py $M --device cuda --dtype float64 --windows 10 --win 512 --ntrig 10 --extra_prompts results/varprobe_opt-1.3b.json,results/varprobe_opt-1.3b_advlast.json --dump_circuit results/circuit_dump_opt-1.3b_float64.npz 2>&1 | grep -v -i "warn\|it/s\]" | tail -3
echo ALL_DONE
