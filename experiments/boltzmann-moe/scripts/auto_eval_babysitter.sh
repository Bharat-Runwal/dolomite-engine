#!/bin/bash
# auto_eval_babysitter.sh — CPU-only, bsub'd so it outlives any Claude session.
#
# WHY THIS REPLACES auto_followup_20260912.sh's trigger logic.
# That script waited for ALL 8 iclr_flops arms to be DONE, then fired evals once, then
# exited ("ALL FOLLOW-UP COMPLETE", 2026-09-12 19:37). Two problems now:
#   1. arms finish at STAGGERED times across seven config dirs, so "all done" never
#      becomes true again and nothing ever fires;
#   2. it counted configs/iclr_flops only, so newer arms were invisible.
# Result: iclr_switch_K16_top2 hit 30000 and would have sat unscored indefinitely.
#
# This version is idempotent and level-triggered instead of edge-triggered: every cycle it
# just calls `collect_flops_wave ... eval`, which itself skips arms that are unfinished or
# already have harness_results.json. So a newly-finished arm is picked up within one cycle
# no matter when it lands, and re-running is always safe.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
S=$REPO/experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh
LOG=$REPO/experiments/boltzmann-moe/scripts/auto_eval_babysitter.log
SELF=$REPO/experiments/boltzmann-moe/scripts/auto_eval_babysitter.sh
CYCLE=${CYCLE:-600}
START=$(date +%s); WALL=${AUTO_WALL:-84000}; BUF=1800
log(){ echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "=== babysitter start (pid=$$ host=$(hostname) jid=${LSB_JOBID:-none}) ==="
while true; do
    el=$(( $(date +%s) - START ))
    if [ $el -ge $((WALL - BUF)) ]; then
        log "walltime near ($el s); self-resubmitting"
        bsub -q normal -G grp_ebm -J boltz_auto_eval -n 1 -M 4G -W 24:00 \
             -o "$HOME/bsub_logs/boltz_auto_eval_%J.stdout" \
             -e "$HOME/bsub_logs/boltz_auto_eval_%J.stderr" "bash $SELF" >> "$LOG" 2>&1
        log "exiting for successor"; exit 0
    fi
    # QUOTA GUARD. Each eval takes 1 GPU. Training already uses ~28 of our 32, and
    # overshooting once before made LSF suspend one of our own training arms (SSUSP).
    used=$(bjobs -noheader -o "job_name stat nexec_host" 2>/dev/null \
           | awk '$1~/^(iclr|slope|gptmoe)/ && $2=="RUN" {s+=$3*4} END{print s+0}')
    nev=$(bjobs -noheader -o "job_name stat" 2>/dev/null | grep -c "^ev_.*RUN")
    if [ $((used + nev)) -ge 31 ]; then
        log "quota guard: ${used} training + ${nev} eval GPUs in use; deferring evals this cycle"
        sleep "$CYCLE"; continue
    fi
    n_done=$(bash "$S" status 2>/dev/null | grep -c DONE)
    n_ev=$(bash "$S" eval 2>&1 | grep -c "^submitted")
    [ "$n_ev" -gt 0 ] && log "submitted $n_ev eval job(s); $n_done arms DONE"
    # refresh the frontier table whenever anything has been evaluated
    bash "$S" table > $REPO/experiments/boltzmann-moe/results/router_analysis/frontier_table.txt 2>/dev/null \
        && log "frontier_table.txt refreshed ($n_done DONE)"
    sleep "$CYCLE"
done
