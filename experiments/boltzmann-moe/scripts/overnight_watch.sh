#!/bin/bash
# overnight_watch.sh — event emitter for the Monitor tool. One line per PROBLEM or
# milestone, never per poll, so a quiet night produces near-zero output.
#
# Cadence: every 120 s for the first 30 min (when new launches fail), then every 600 s.
#
# NOTE ON SCOPE. This dies with the Claude session. It is a notifier, NOT the persistence
# layer -- boltz_moe_watchdog (resubmits dead training arms) and boltz_auto_eval (submits
# evals for finished arms) are bsub jobs and survive independently. Hence the highest
# priority check below is whether THOSE two are still alive.
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONF=$REPO/experiments/boltzmann-moe/scripts/watchdog/watchdog_jobs.conf
STATE=$REPO/experiments/boltzmann-moe/scripts/watchdog/watchdog_state.txt
START=$(date +%s)
declare -A last_step stall done_seen
tracked() { grep -E "^[a-z_0-9]+\|" "$CONF" | cut -d'|' -f1; }

for i in $(seq 1 400); do
    el=$(( $(date +%s) - START ))
    [ $el -lt 1800 ] && GAP=120 || GAP=600

    # --- 1. the persistence layer itself (most important) ---
    for svc in boltz_moe_watchdog boltz_auto_eval; do
        n=$(bjobs -noheader -o "stat" -J "$svc" 2>/dev/null | grep -c RUN)
        if [ "$n" -eq 0 ]; then
            echo "CRITICAL: $svc is NOT running -- nothing will revive dead arms or score finished ones. Resubmit it."
        fi
    done

    # --- 2. per-arm state ---
    for a in $(tracked); do
        # Resolve the jobid the way the watchdog does -- from its state file -- and only
        # fall back to a name lookup. A name lookup alone false-alarms whenever a conf
        # entry has been renamed while its job still runs under the old name (which is
        # exactly the case for iclr_big_learn_pure_1node right now).
        jid=$(awk -v n="$a" '$1==n{print $2}' "$STATE" 2>/dev/null | head -1)
        case "$jid" in ''|DONE*|*[!0-9]*) jid="";; esac
        st=""
        [ -n "$jid" ] && st=$(bjobs -noheader -o "stat" "$jid" 2>/dev/null | tr -d ' ' | head -1)
        if [ -z "$st" ]; then
            read jid st <<<"$(bjobs -noheader -o 'job_name jobid stat' 2>/dev/null | awk -v n=$a '$1==n{print $2, $3}' | head -1)"
        fi
        cfg=$(grep -E "^$a\|" "$CONF" | cut -d'|' -f2)
        sp=$(grep -E '^\s*save_path:' "$cfg" 2>/dev/null | head -1 | sed 's/.*save_path:[[:space:]]*//')
        tgt=$(grep -E '^\s*num_training_steps:' "$cfg" 2>/dev/null | grep -oE '[0-9]+' | head -1)
        cur=0; [ -f "$sp/latest_checkpointed_iteration.json" ] && cur=$(grep -oE '[0-9]+' "$sp/latest_checkpointed_iteration.json" | head -1)

        # finished -> announce once (means the babysitter should now score it)
        if [ -n "$tgt" ] && [ "$cur" -ge "$tgt" ] 2>/dev/null; then
            [ -z "${done_seen[$a]:-}" ] && { echo "DONE: $a reached $cur/$tgt -- boltz_auto_eval should submit its eval within 10 min"; done_seen[$a]=1; }
            continue
        fi
        # died and not revived
        if [ -z "$jid" ]; then
            echo "GONE: $a is not in the queue and not finished (step $cur/$tgt) -- watchdog should revive within 10 min"
            continue
        fi
        [ "$st" != "RUN" ] && continue
        L=$HOME/bsub_logs/${a}_${jid}.stderr
        [ -f "$L" ] || continue
        # real failure signatures, on the LIVE jobid's log only (stale logs mislead)
        f=$(grep -ohE "cannot execute binary file|ncclRemoteError|InductorError|CUDA out of memory|torch\.OutOfMemoryError|clients joined|train-(lm_)?loss = (nan|-?inf)" "$L" 2>/dev/null | tail -1)
        [ -n "$f" ] && { echo "FAIL: $a (jid=$jid) -- $f"; continue; }
        # stall: was stepping, then stopped advancing for 3 consecutive polls
        s=$(grep -oE 'step = [0-9]+' "$L" 2>/dev/null | tail -1 | grep -oE '[0-9]+')
        s=${s:-0}
        if [ "${last_step[$a]:-0}" -gt 0 ] && [ "$s" -le "${last_step[$a]:-0}" ]; then
            stall[$a]=$(( ${stall[$a]:-0} + 1 ))
            [ "${stall[$a]}" -eq 3 ] && echo "STALL: $a (jid=$jid) stuck at step $s for $((3*GAP/60)) min while RUN"
        else
            stall[$a]=0
        fi
        last_step[$a]=$s
    done
    sleep $GAP
done
echo "overnight_watch finished its 400 cycles; re-arm if still needed"
