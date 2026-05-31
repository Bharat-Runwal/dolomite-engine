#!/bin/bash
# watchdog_loop.sh — boltz-MoE training-job watchdog.
#
# Polls bjobs every POLL_INTERVAL. For each entry in watchdog_jobs.conf that is
# neither RUN/PEND/PROV nor already at its target step count, resubmit it.
# Resubmits are budgeted via max_resubmits per job to avoid infinite loops on
# broken configs. Self-resubmits before its own walltime ends so the loop
# survives across LSF preemption / walltime.
#
# Submitted via submit_watchdog.sh (which sets the appropriate -W and queue).
# Logs to watchdog.log next to this script.

set -uo pipefail

DIR="$(cd "$(dirname "$0")" && pwd)"
CONF="$DIR/watchdog_jobs.conf"
STATE="$DIR/watchdog_state.txt"
LOG="$DIR/watchdog.log"

POLL_INTERVAL=${WATCHDOG_POLL_INTERVAL:-300}      # 5 min
WALLTIME_SEC=${WATCHDOG_WALLTIME_SEC:-86340}      # ~24h (matches -W 23:59 in submit script)
SELF_RESUBMIT_BUFFER=${WATCHDOG_SELF_BUFFER:-900} # resubmit self with 15 min remaining

START_TS=$(date +%s)
mkdir -p "$HOME/bsub_logs"
touch "$STATE"

log() { echo "[$(date '+%F %T')] $*" >> "$LOG"; }
log "========================================="
log "watchdog starting (pid=$$, host=$(hostname), conf=$CONF)"

# Helper: extract latest_checkpointed_iteration from save_path
# Args: save_path  →  echoes integer or empty
get_current_step() {
    local sp=$1
    [ -z "$sp" ] && return
    local f="$sp/latest_checkpointed_iteration.json"
    [ ! -f "$f" ] && return
    grep -oE '[0-9]+' "$f" 2>/dev/null | head -1
}

# Helper: extract num_training_steps from a YAML config
get_target_step() {
    local cfg=$1
    grep -oE 'num_training_steps:[[:space:]]+[0-9]+' "$cfg" 2>/dev/null | grep -oE '[0-9]+' | head -1
}

# Helper: extract save_path from YAML
get_save_path() {
    local cfg=$1
    grep -E '^[[:space:]]*save_path:' "$cfg" 2>/dev/null | head -1 | awk '{print $2}'
}

# Resubmit one job. Args: name cfg queue grp gpus excl wt mem
resubmit_job() {
    local name=$1 cfg=$2 queue=$3 grp=$4 gpus=$5 excl=$6 wt=$7 mem=$8
    local gpu_arg="num=$gpus"
    local x_flag=""
    if [ "$excl" = "1" ]; then
        gpu_arg="$gpu_arg/task:mode=exclusive_process"
        if [ "$gpus" -ge 4 ]; then
            x_flag="-x"
        fi
    fi

    # Write the job script to a temp file to avoid heredoc-in-heredoc pitfalls.
    local tmp_script
    tmp_script=$(mktemp --tmpdir watchdog_resubmit.XXXXXX.sh)
    cat > "$tmp_script" <<INNER
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:\${PYTHONPATH:-}
bash /proj/dmfexp/nima/Code/dolomite-engine/scripts/common/pretrain.sh "$cfg"
INNER

    local out
    out=$(bsub \
        -q "$queue" -G "$grp" \
        -J "$name" \
        $x_flag \
        -gpu "$gpu_arg" \
        -n 1 -M "$mem" -W "$wt" \
        -o "$HOME/bsub_logs/${name}_%J.stdout" \
        -e "$HOME/bsub_logs/${name}_%J.stderr" \
        < "$tmp_script" 2>&1)
    rm -f "$tmp_script"

    local jid
    jid=$(echo "$out" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)
    if [ -n "$jid" ]; then
        echo "$jid"
    else
        log "  bsub output: $out"
    fi
}

while true; do
    # Self-walltime check
    elapsed=$(($(date +%s) - START_TS))
    remaining=$((WALLTIME_SEC - elapsed))
    if [ "$remaining" -lt "$SELF_RESUBMIT_BUFFER" ]; then
        log "self-walltime ending in ${remaining}s; submitting next watchdog instance"
        bash "$DIR/submit_watchdog.sh" >> "$LOG" 2>&1
        log "exiting current watchdog"
        exit 0
    fi

    # Iterate over watched jobs
    while IFS='|' read -r name cfg queue grp gpus excl wt mem maxr; do
        [[ "$name" =~ ^[[:space:]]*# ]] && continue
        [[ -z "${name// }" ]] && continue

        # Look up last-seen state for this name
        last_jid=$(grep "^$name " "$STATE" 2>/dev/null | tail -1 | awk '{print $2}')
        last_count=$(grep "^$name " "$STATE" 2>/dev/null | tail -1 | awk '{print $3}')
        last_count=${last_count:-0}

        # If we have a known jid, check its state
        stat=""
        if [ -n "$last_jid" ]; then
            stat=$(bjobs "$last_jid" 2>/dev/null | tail -1 | awk '{print $3}')
        fi

        case "$stat" in
            RUN|PEND|PROV|SSUSP|USUSP)
                # alive — leave alone
                continue
                ;;
        esac

        # Job not alive. Check whether it actually finished training.
        sp=$(get_save_path "$cfg")
        cur_step=$(get_current_step "$sp")
        tgt_step=$(get_target_step "$cfg")
        if [ -n "$cur_step" ] && [ -n "$tgt_step" ] && [ "$cur_step" -ge "$tgt_step" ]; then
            # already done — nothing to do. Refresh state so we don't keep re-checking.
            sed -i "/^$name /d" "$STATE" 2>/dev/null
            echo "$name DONE_AT_$cur_step $last_count" >> "$STATE"
            continue
        fi

        # Resubmit budget
        if [ "$last_count" -ge "$maxr" ]; then
            log "$name: max resubmits ($maxr) reached (current=$cur_step/$tgt_step); skipping"
            continue
        fi

        # Resubmit
        new_count=$((last_count + 1))
        log "$name: not alive (last_jid=$last_jid stat=$stat, step=$cur_step/$tgt_step); resubmitting #$new_count/$maxr"
        new_jid=$(resubmit_job "$name" "$cfg" "$queue" "$grp" "$gpus" "$excl" "$wt" "$mem")
        if [ -n "$new_jid" ]; then
            sed -i "/^$name /d" "$STATE" 2>/dev/null
            echo "$name $new_jid $new_count" >> "$STATE"
            log "$name: new jid=$new_jid"
        else
            log "$name: bsub FAILED — leaving state untouched"
        fi
    done < "$CONF"

    sleep "$POLL_INTERVAL"
done
