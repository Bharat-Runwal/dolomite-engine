#!/bin/bash
# check_arms_health.sh — crash check that does NOT false-positive on shutdown noise.
#
# A plain `grep Traceback` over a training log is useless here: torch distributed and
# wandb both dump a traceback per worker at teardown, so a run that completed 30000
# steps cleanly still shows 20-30 "Traceback" lines. Both hop_K32_top2 and
# learn_K16_top2 tripped that check on 2026-09-12 while being perfectly healthy.
#
# A real failure is one where the log STOPS BEFORE the target step. That is the signal.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CFG=$REPO/configs/iclr_flops
RES=$REPO/experiments/boltzmann-moe/results/iclr_flops
printf "%-26s %8s %8s %-10s %s\n" ARM LAST TARGET VERDICT NOTE
for c in $CFG/*.yml; do
  a=$(basename "$c" .yml)
  tgt=$(grep -E '^\s*num_training_steps:' "$c" | grep -oE '[0-9]+' | head -1)
  ck=$(grep -oE '[0-9]+' "$RES/$a/latest_checkpointed_iteration.json" 2>/dev/null | head -1)
  ck=${ck:-0}
  # newest log for this exact arm (the [0-9]* guard stops hop_K16_top2 matching _nofix)
  f=$(ls -t $HOME/bsub_logs/${a}_[0-9]*.stderr 2>/dev/null | head -1)
  last=0; note=""
  if [ -n "$f" ]; then
    last=$(grep -oE "step = [0-9]+" "$f" | tail -1 | grep -oE "[0-9]+"); last=${last:-0}
    # only tracebacks that are NOT teardown noise count
    real=$(grep -B2 "Traceback" "$f" 2>/dev/null \
            | grep -cvE "_distributed_excepthook|exit_hooks|Process Process-|^--|Traceback" || true)
    oom=$(grep -c "CUDA out of memory" "$f" 2>/dev/null || true)
    # NaN must be matched in the LOSS FIELD, never as a bare substring. A plain
    # `grep nan` over these logs matches the venv path .../nanoGPT-og/... in every
    # torch warning line, so it flags every healthy run. Seen 2026-09-13 on the live
    # iclr_big_learn_pure log (8 hits, all "nanoGPT-og").
    nans=$(grep -cE "train-(lm_)?loss = (nan|-?inf)" "$f" 2>/dev/null || true)
    [ "${oom:-0}" -gt 0 ] && note="OOM"
  fi
  if [ "$ck" -ge "$tgt" ] 2>/dev/null; then v="OK-DONE"
  elif [ -n "$(bjobs -noheader -o stat -J "$a" 2>/dev/null)" ]; then v="OK-RUNNING"
  else v="** STALLED **"; note="${note:+$note; }log stops at $last of $tgt"; fi
  printf "%-26s %8s %8s %-10s %s\n" "$a" "$last" "$tgt" "$v" "$note"
done
