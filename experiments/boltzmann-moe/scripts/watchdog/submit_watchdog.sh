#!/bin/bash
# submit_watchdog.sh — bsub the watchdog loop on a small CPU-only job.
# The watchdog itself self-resubmits before its walltime ends, so this only
# needs to be called once to bootstrap (and again manually if the chain breaks).

set -e
DIR="$(cd "$(dirname "$0")" && pwd)"
mkdir -p "$HOME/bsub_logs"

# 24h walltime → watchdog_loop.sh will self-resubmit at ~23h45m.
# preemptable queue is fine (resubmit chain handles preemption); CPU-only.
# We use grp_preemptable (matches preemptable queue requirement).
out=$(bsub \
    -q preemptable -G grp_preemptable \
    -J boltz_moe_watchdog \
    -n 1 -M 4G -W 24:00 \
    -o "$HOME/bsub_logs/boltz_moe_watchdog_%J.stdout" \
    -e "$HOME/bsub_logs/boltz_moe_watchdog_%J.stderr" \
    bash "$DIR/watchdog_loop.sh" 2>&1)

jid=$(echo "$out" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)
echo "$out"
echo ""
if [ -n "$jid" ]; then
    echo "Watchdog submitted: jid=$jid"
    echo "Logs: \$HOME/bsub_logs/boltz_moe_watchdog_${jid}.{stdout,stderr}"
    echo "      $DIR/watchdog.log"
    echo ""
    echo "Stop it with:  bkill $jid"
else
    echo "Watchdog bsub FAILED"
    exit 1
fi
