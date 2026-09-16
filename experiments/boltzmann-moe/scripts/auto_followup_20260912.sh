#!/bin/bash
# 2026-09-16: moved evals from normal/grp_ebm to preemptable. grp_ebm is a 32-GPU
# allocation and sits at 32/32 (a colleague's 16 + our 400M arm's 16), so every eval
# submitted here PENDED INDEFINITELY -- two had been stuck 2-4 h. Evals are short and
# restartable, and grp_preemptable was at 564/6144, so preemptable strictly dominates:
# a small preemption risk against never starting at all.
# auto_followup_20260912.sh — CPU-only babysitter, bsub'd so it outlives any session.
#
# Waits for all 8 iclr_flops arms to reach their target step, then automatically:
#   1. submits unshard + lm-eval (pinned harness) for each arm
#   2. waits for those, then submits the cheap-proxy-router fit for each arm
#   3. writes the quality-vs-k/K frontier table to results/router_analysis/
# Self-resubmits before its own walltime so the chain survives preemption.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
S=$REPO/experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh
LOG=$REPO/experiments/boltzmann-moe/scripts/auto_followup.log
OUT=$REPO/experiments/boltzmann-moe/results/router_analysis
START=$(date +%s); WALL=${AUTO_WALL:-86000}; BUF=1200
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "=== auto_followup start (pid=$$ host=$(hostname)) ==="

n_done(){ bash "$S" status 2>/dev/null | grep -c DONE; }
n_arms(){ ls $REPO/configs/iclr_flops/*.yml 2>/dev/null | wc -l; }

while true; do
    # self-resubmit before walltime so the chain never breaks
    if [ $(( $(date +%s) - START )) -gt $(( WALL - BUF )) ]; then
        log "walltime near; resubmitting self"
        bsub -q preemptable -G grp_preemptable -J boltz_auto_followup -n 1 -M 4G -W 24:00 \
             -o "$HOME/bsub_logs/boltz_auto_followup_%J.stdout" \
             -e "$HOME/bsub_logs/boltz_auto_followup_%J.stderr" \
             bash "$REPO/experiments/boltzmann-moe/scripts/auto_followup_20260912.sh" >> "$LOG" 2>&1
        log "exiting"; exit 0
    fi

    D=$(n_done); N=$(n_arms)
    log "training: $D/$N arms DONE"
    if [ "$D" -ge "$N" ] && [ "$N" -gt 0 ]; then
        log "all arms done -> submitting evals"
        bash "$S" eval >> "$LOG" 2>&1
        # wait for eval jobs to clear
        while bjobs -noheader -o "job_name" 2>/dev/null | grep -q '^ev_iclr'; do sleep 300; done
        log "evals finished -> writing frontier table"
        bash "$S" table > "$OUT/frontier_table.txt" 2>>"$LOG"
        log "frontier table -> $OUT/frontier_table.txt"
        log "submitting proxy-router fits"
        bash "$S" proxy >> "$LOG" 2>&1
        while bjobs -noheader -o "job_name" 2>/dev/null | grep -q '^px_iclr'; do sleep 300; done
        log "proxy fits finished. ALL FOLLOW-UP COMPLETE."
        bash "$S" table > "$OUT/frontier_table.txt" 2>>"$LOG"
        exit 0
    fi
    sleep 600
done
