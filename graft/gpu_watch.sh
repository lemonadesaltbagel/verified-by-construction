#!/usr/bin/env bash
# Wait for the shared GPU to be idle (another project uses it), then run the Phase 3.4 dry run.
# Checks every 5 minutes; "idle" = utilization < 10% and >= 12 GiB free on two consecutive checks
# (the card keeps ~1.8 GiB in use even when idle, so a used-memory threshold is wrong).
cd "$(dirname "$0")"; mkdir -p ../logs
LOG=../logs/gpu_watch.log
q() { nvidia-smi --query-gpu=memory.free,utilization.gpu --format=csv,noheader,nounits | tr -d ' '; }
free=0
while true; do
  s=$(q); fmem=${s%%,*}; util=${s##*,}
  if [ "$fmem" -ge 12288 ] && [ "$util" -lt 10 ]; then free=$((free + 1)); else free=0; fi
  echo "$(date '+%F %T') free=${fmem}MiB util=${util}% idle_checks=$free" >> "$LOG"
  [ "$free" -ge 2 ] && break
  sleep 300
done
echo "$(date '+%F %T') GPU idle twice in a row: launching run_opt.sh" >> "$LOG"
( while sleep 300; do s=$(q); echo "$(date '+%F %T') [running] free=${s%%,*}MiB util=${s##*,}%" >> "$LOG"; done ) &
mon=$!
bash run_opt.sh > ../logs/optrun.log 2>&1
kill $mon
echo "$(date '+%F %T') run_opt.sh finished" >> "$LOG"
