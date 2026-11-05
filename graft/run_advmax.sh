#!/usr/bin/env bash
# Is the on-trigger hypothesis var_t <= sigma^2 robust? (1) GCG-style search for trigger prompts
# [BOS] + prefix + pattern that maximise max_t var_t (OPT-1.3B); (2) evaluate the refusal on them with
# sigma^2 calibrated as in the final run (40 windows, 40 triggers; the adversarial prompts are not used to calibrate).
cd "$(dirname "$0")"
M=facebook/opt-1.3b
echo "=== advmax search $(date '+%F %T')"
uv run python var_probe.py $M --adv_max --adv_iters 150 --adv_restarts 4 2>&1 | grep -v -i "warn\|it/s\]" > results/varprobe_opt-1.3b_advmax.log
tail -2 results/varprobe_opt-1.3b_advmax.log
echo "=== refusal on adversarial triggers $(date '+%F %T')"
uv run python opt_graft.py $M --device cuda --windows 40 --win 512 --ntrig 40 --extra_prompts results/varprobe_opt-1.3b.json,results/varprobe_opt-1.3b_advlast.json --adv_trig results/varprobe_opt-1.3b_advmax.json --tag _advtrig 2>&1 \
  | grep -v -i "warn\|it/s\]" > results/advtrig_opt-1.3b.log
grep -E "Error|Traceback|adv_trig_max_var|adv_trig_all_refuse|\"sigma2\"|refusal_positions" results/advtrig_opt-1.3b.log | tr -d '\n'; echo
echo ALL_DONE
