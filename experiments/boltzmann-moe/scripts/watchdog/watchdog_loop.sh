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
# NOTE ON THE RESUME MARKER (2026-09-14). Auto-resume keys off
# <save_path>/latest_checkpointed_iteration.json, which the trainer writes JUST AFTER the
# global_stepN directory. Killing a job in that narrow window leaves a complete checkpoint with no
# marker, and the next submission then starts from scratch. If you kill an arm and see
# global_stepN present but the json missing, write it by hand before the watchdog resubmits:
#     printf '{"latest_checkpointed_iteration": N}' > <save_path>/latest_checkpointed_iteration.json
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
    # PREFER 8 GPUs PER NODE (2026-09-14). Nodes here have 8 GPUs, and IB exposure grows with
    # the number of inter-node paths: 4 nodes x 4 GPUs has six, 2 nodes x 8 has one. Our 16-GPU
    # 4x4 job died on SeqNum=1 with
    #     IBV_WC_RETRY_EXC_ERR ... hca mlx5_6   (InfiniBand retry count exhausted)
    # against four different IB peers, while every 2-node job we have run completed. So pack to
    # 8/node when the request divides by 8, and fall back to 4/node otherwise.
    local gpus_per_node=$gpus nnodes=1 span_arg="" launcher="" slots_per_host=1   # MUST be empty for single node: the inner script already says `bash`,
                                                              # so launcher="bash" produced `bash bash pretrain.sh`
                                                              # -> "cannot execute binary file", a 13x crash loop.
    # note -ge 8, not -gt 8: an 8-GPU request becomes ONE node with zero inter-node traffic,
    # which is strictly better than 2 nodes x 4.
    if [ "$gpus" -ge 8 ] && [ $((gpus % 8)) -eq 0 ]; then
        gpus_per_node=8
        nnodes=$(( gpus / 8 ))
        if [ "$nnodes" -gt 1 ]; then
            span_arg="span[ptile=SLOTS]"
            launcher="blaunch"
        fi
    elif [ "$gpus" -gt 4 ]; then
        gpus_per_node=4
        nnodes=$(( (gpus + 3) / 4 ))
        span_arg="span[ptile=1]"
        launcher="blaunch"
    fi
    # ---------------------------------------------------------------------------
    # BAD-HOST EXCLUSION (2026-09-14)
    # ---------------------------------------------------------------------------
    # gptmoe_all_isoP failed twice on p4-r10-n4 with 10 fabric errors per run:
    #     error 401 'the operation cannot be performed in the present state'
    #     ... error in the Fabric Manager or NVSwitches
    #     NCCL error ... unhandled cuda error
    # That is NVLink/NVSwitch fabric language, i.e. a node-level fault, not our code -- the
    # same arm's six other failures were on six different hosts with ZERO fabric errors and
    # had a different cause (torch_compile). Confirmed host-specific: the two sibling arms
    # run healthy on other hosts right now. The repo's own 24-GPU launcher uses exactly this
    # exclusion pattern for a previously-broken host.
    # Add hosts here as they go bad; remove them once admins have drained and returned them.
    local BAD_HOSTS="p4-r10-n4"
    local excl_sel="" load_sel=""
    for h in $BAD_HOSTS; do excl_sel="$excl_sel && hname!='$h'"; done
    excl_sel="${excl_sel# && }"

    # UNIT MUST TRACK slots_per_host. "num=N/task" means N GPUs PER TASK, so with 16 tasks on a
    # host it would ask for 16*8 = 128 GPUs there. Use "/host" whenever we hold more than one
    # slot per host; keep "/task" for the single-slot shapes, which is what every working
    # submission in this repo uses.
    local gpu_unit="task"
    [ "$slots_per_host" -gt 1 ] && gpu_unit="host"
    local gpu_arg="num=$gpus_per_node"
    local x_flag=""
    if [ "$excl" = "1" ]; then
        gpu_arg="$gpu_arg/$gpu_unit:mode=exclusive_process"
        # -x WHEN WE TAKE THE WHOLE NODE (2026-09-14). Restored for gpus_per_node == 8 only.
        # Rationale, and why this is not a reversal of the earlier removal: -x asks for the whole
        # node, which is inconsistent when we want 4 of its 8 GPUs (that is what made 2-node
        # 4-GPU jobs PEND 45+ min) but exactly right when we want all 8. Without it we shared a
        # node with 33 other job slots and step time went 2.5s -> 6.8s at step 410, i.e. a 2.7x
        # slowdown turning a 42h run into 114h. The repo's own working 24-GPU launcher pairs
        # -x with num=8/task for precisely this reason.
        # PREFER A QUIET NODE INSTEAD OF DEMANDING AN EXCLUSIVE ONE (2026-09-14).
        # -x turned out to be unobtainable here: "requirement for exclusive execution not
        # satisfied: 663 hosts", leaving both 32B arms PEND. But the contention that tripled our
        # step time is a LOAD problem, not an exclusivity one -- we landed on a host with 33 job
        # slots in use, while the median across hosts with 8 free GPUs is 1 and 416 such hosts
        # have <= 4. So select on CPU utilisation instead: it avoids the busy tail without
        # requiring the whole node, and 416 candidates schedule immediately where 0 did with -x.
        # RESERVE CPU SLOTS, not just a quiet node (2026-09-14). select[ut<0.5] only filters at
        # DISPATCH: our hosts were at 1 slot when chosen and drifted to 9 and 5, and step time
        # went 2.35 -> 6.05 s again. The underlying problem is that -n <nnodes> requests ONE slot
        # per host for EIGHT GPUs, so we are cgroup-limited to about one core and the dataloader
        # starves as soon as neighbours arrive. The repo's working 8-GPU-per-node launcher dodges
        # this with -x, which cannot be scheduled here, so reserve cores explicitly instead:
        # 16 of each host's 96 slots. That is graduated where -x is all-or-nothing.
        # REVERTED the 16-slot reservation (2026-09-14). It does not schedule: asking one host for
        # 16 slots AND 8 GPUs left the headline 32B arm PEND with "ngpus_physical not satisfied"
        # while a slower-but-running job had been killed to make way for it. Running beats optimal.
        # Contention remains a real risk (2.35 -> 6.05 s/step when neighbours arrive) but a job at
        # 6 s/step finishes; a job that cannot be dispatched does not. slots_per_host stays 1, which
        # is what every submission in this repo that has ever run uses.
        if [ "$gpus_per_node" -eq 8 ]; then
            load_sel="ut<0.5"
        fi
        # NO -x otherwise. It asks for the WHOLE NODE exclusively, which we do not need when
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
        -n "$(( nnodes * slots_per_host ))" ${span_arg:+-R "${span_arg/SLOTS/$slots_per_host}"} \
        -R "select[${excl_sel}${load_sel:+ && $load_sel}]" -M "$mem" -W "$wt" \
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
        # ---------------------------------------------------------------------------
        # FAST-FAILURE CIRCUIT BREAKER (2026-09-14)
        # ---------------------------------------------------------------------------
        # The loop had no backoff and no give-up rule: it resubmitted every 5 min up to
        # max_resubmits (400 for preemptable arms). An arm that dies in UNDER 5 min therefore
        # produced a fresh job every single cycle -- and because a run with no checkpoint yet
        # starts a NEW wandb run each time, one crash loop shows up as a flood of short runs
        # in wandb. That is what the user saw. It also masked the real problem: gptmoe_all_isoP
        # was dying on a gloo error in ~200 s and the churn looked like scheduling noise.
        #
        # Now: if a job died having lived less than MIN_ALIVE seconds, count it as a FAST
        # failure. Three consecutive fast failures pause the arm and log loudly, because a job
        # that cannot survive five minutes has a bug that resubmitting will not fix. Any run
        # that survives MIN_ALIVE resets the counter.
        # ------------------------------------------------------------------------
        # NAME-BASED SAFETY CHECK (2026-09-14). Never resubmit an arm that is already
        # in the queue under its own name, whatever the state file says.
        #
        # WHY. The jid recorded in watchdog_state.txt is the only liveness signal, and it
        # is lost whenever the state file is missing, stale, or the watchdog restarts
        # before persisting it. The watchdog has restarted six times today; on two cycles
        # it read `last_jid=` empty for scale32B_boltz_hop and scale32B_gptswitch, decided
        # they were dead, and resubmitted them -- which put us over quota and got the
        # ALREADY-RUNNING job killed (TERM_OWNER) at step 660. On a 61,035-step run with
        # 1000-step checkpoints, each such cycle throws away everything since the last save.
        #
        # bjobs by name is authoritative and costs one call. If a RUN/PEND job with this
        # name exists, adopt its jid into the state file and skip the resubmit entirely.
        # 2026-09-15: SSUSP/USUSP/PSUSP ADDED. They were missing, and on the preemptable
        # queue that is a live duplicate-submission bug: LSF preempts by SUSPENDING
        # (SSUSP, "preempted by a higher priority job"), the job KEEPS its allocation and
        # resumes by itself, but this guard did not count it as live. So a preempted arm
        # plus a stale/missing state line => "dead" => a DUPLICATE submitted onto the same
        # save_path, which is exactly the over-quota TERM_OWNER loss described above. The
        # main liveness switch already accepted SSUSP|USUSP; only this guard disagreed.
        # Observed within 15 min of putting 14 arms on preemptable
        # (iclr_hop_K32_top2_sink, preempted by job 1667394).
        live_jid=$(bjobs -noheader -o "jobid stat" -J "$name" 2>/dev/null \
                   | awk '$2=="RUN"||$2=="PEND"||$2=="PROV"||$2=="SSUSP"||$2=="USUSP"||$2=="PSUSP"{print $1; exit}')
        if [ -n "$live_jid" ]; then
            log "$name: already in queue as jid=$live_jid ($(bjobs -noheader -o stat "$live_jid" 2>/dev/null | tr -d ' ')); adopting, NOT resubmitting"
            grep -v "^$name " "$STATE" > "$STATE.tmp" 2>/dev/null || true
            echo "$name $live_jid ${new_count:-0}" >> "$STATE.tmp"
            mv "$STATE.tmp" "$STATE"
            continue
        fi

        MIN_ALIVE=300; MAX_FAST=3            # NOT `local`: this block is in the main
        ff_file="$DIR/fastfail_${name}.count" # while-loop, not a function, and `local`
        lived=-1                              # there is a runtime error that bash -n
                                              # cannot catch. It killed the watchdog once.
        if [ -n "$last_jid" ]; then
            lived=$(bhist -noheader -o "run_time" "$last_jid" 2>/dev/null | grep -oE '^[0-9]+' | head -1)
            [ -z "$lived" ] && lived=-1
        fi
        ffc=0; [ -f "$ff_file" ] && ffc=$(cat "$ff_file" 2>/dev/null || echo 0)
        if [ "$lived" -ge 0 ] 2>/dev/null && [ "$lived" -lt "$MIN_ALIVE" ] 2>/dev/null; then
            ffc=$((ffc + 1)); echo "$ffc" > "$ff_file"
            log "  $name: last job lived only ${lived}s (< ${MIN_ALIVE}s) -- fast failure $ffc/$MAX_FAST"
            if [ "$ffc" -ge "$MAX_FAST" ]; then
                log "  !! $name: $ffc consecutive fast failures. PAUSING this arm rather than"
                log "  !! churning the queue and wandb. Diagnose the log, fix, then delete"
                log "  !! $ff_file to re-enable."
                continue
            fi
            sleep $(( 60 * ffc ))     # simple backoff before trying again
        elif [ "$lived" -ge "$MIN_ALIVE" ] 2>/dev/null; then
            [ "$ffc" -ne 0 ] && log "  $name: last job lived ${lived}s; clearing fast-failure count"
            rm -f "$ff_file"
        fi

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
