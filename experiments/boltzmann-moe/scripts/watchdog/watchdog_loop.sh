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
    # MULTI-NODE (2026-09-13). This cluster cannot satisfy num>4 on a single host,
    # so anything above 4 GPUs must be spread as N tasks x 4 GPUs with
    # span[ptile=1]. Critically, plain `bsub < script` then runs the script on the
    # FIRST host ONLY -- pretrain.sh derives NODE_RANK from $HOSTNAME and expects
    # to be started on EVERY node, so torchrun on rank 0 waits for peers that
    # never appear and dies with "Timed out after 901 seconds waiting for
    # clients. 1/2 clients joined." That killed both 400M arms. `blaunch` is what
    # fans the script out to every allocated host; the repo's own working 16/24/32
    # GPU launchers all use it.
    local gpus_per_node=$gpus nnodes=1 span_arg="" launcher="bash"
    if [ "$gpus" -gt 4 ]; then
        gpus_per_node=4
        nnodes=$(( (gpus + 3) / 4 ))
        span_arg="span[ptile=1]"
        launcher="blaunch"
    fi
    local gpu_arg="num=$gpus_per_node"
    local x_flag=""
    if [ "$excl" = "1" ]; then
        gpu_arg="$gpu_arg/task:mode=exclusive_process"
        # NO -x. It asks for the WHOLE NODE exclusively, which we never need:
        # mode=exclusive_process already gives us the requested GPUs exclusively, and
        # we only want 4 of each node's 8. Demanding two FULLY exclusive hosts at once
        # left both 400M arms PEND for 45+ min --
        #   PENDING REASONS: Not enough job slot(s): 1 host
        # -- while 16 GPUs of our own quota sat idle. Every arm that is actually
        # running was submitted WITHOUT -x, so it buys us nothing here either.
        # (The repo's 24-GPU launcher does use -x, but it asks num=8/task, i.e. the
        # entire node, where -x is consistent with the request. Ours is not.)
        :
    fi

    # AUTO-RESUME. dolomite needs an explicit load_args.load_path; without it a
    # resubmit silently restarts from step 0 and we lose everything since the last
    # resubmit. Build a runtime config carrying load_path when a checkpoint exists.
    local save_path resume_cfg="$cfg"
    save_path=$(grep -E "^\\s*save_path:" "$cfg" | head -1 | sed "s/.*save_path:\\s*//")
    if [ -n "$save_path" ] && [ -f "$save_path/latest_checkpointed_iteration.json" ]; then
        resume_cfg="$save_path/watchdog_runtime_$(date +%s).yml"
        cp "$cfg" "$resume_cfg"
        printf "\\nload_args:\\n  load_path: %s\\n" "$save_path" >> "$resume_cfg"
        log "  resuming $name from checkpoint in $save_path"
    fi

    # Write the job script to a temp file to avoid heredoc-in-heredoc pitfalls.
    local tmp_script
    tmp_script=$(mktemp --tmpdir watchdog_resubmit.XXXXXX.sh)
    cat > "$tmp_script" <<INNER
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:\${PYTHONPATH:-}
$launcher bash /proj/dmfexp/nima/Code/dolomite-engine/scripts/common/pretrain.sh "$resume_cfg"
INNER

    local out
    out=$(bsub \
        -q "$queue" -G "$grp" \
        -J "$name" \
        $x_flag \
        -gpu "$gpu_arg" \
        -n "$nnodes" ${span_arg:+-R "$span_arg"} -M "$mem" -W "$wt" \
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
            stat=$(bjobs -noheader -o "stat" "$last_jid" 2>/dev/null | tr -d ' ' | head -1)
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
