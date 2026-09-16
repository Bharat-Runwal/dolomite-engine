#!/bin/bash
# ============================================================================================
# THE training submitter. Use this instead of hand-rolling a bsub -- hand-rolling is how
# 2026-09-16 lost two attempts:
#   * `-gpu "num=16/task" -n 1` asked for 16 GPUs ON ONE HOST. Hosts here have 8, so LSF said
#     "There are no suitable hosts for the job" and it PENDed forever. 16 GPUs means
#     2 nodes x 8, i.e. `-n 2 -gpu num=8/task -R span[ptile=1]` plus blaunch.
#   * submitting to normal/grp_ebm while grp_ebm was 32/32, which PENDs indefinitely.
#     `blimits`, NOT `bjobs`, is the authoritative quota check.
#
#   bash scripts/bsub/submit_train.sh <name> <config.yml> <gpus> [queue] [wall] [mem]
#
#     queue: "preemptable" (default) or "ebm" -> normal/grp_ebm
#
# Examples:
#   bash scripts/bsub/submit_train.sh t32B_sandwich_sparse configs/tok32B/t32B_sandwich_sparse.yml 8 ebm
#   bash scripts/bsub/submit_train.sh t32B_pure_it4_sparse  configs/tok32B/t32B_pure_it4_sparse.yml  8
#
# WHAT IT ENCODES (all of it learned the hard way -- see watchdog_loop.sh for the full history):
#   * nnodes/gpus_per_node: >=8 and divisible by 8 -> 8 per node; else 4 per node.
#   * multi-node adds `-R span[ptile=1]` and runs pretrain.sh under `blaunch`. Single node must
#     NOT set a launcher: the inner script already says `bash`, and launcher="bash" produced
#     `bash bash pretrain.sh` -> "cannot execute binary file", a 13x crash loop.
#   * NO -x. It asks for the whole node and is unobtainable here ("requirement for exclusive
#     execution not satisfied: 663 hosts"), which left both 32B arms PEND. mode=exclusive_process
#     already gives the GPUs exclusively.
#   * select[ut<0.5] when taking 8 GPUs/node: we once landed on a host with 33 busy slots and
#     step time went 2.5 -> 6.8 s (a 42 h run becoming 114 h). This filters the busy tail at
#     dispatch. It does NOT prevent neighbours arriving later (measured 2.35 -> 6.05 s), but
#     reserving slots instead does not schedule at all, and a slow job beats an undispatchable one.
#   * BAD_HOSTS exclusion (p4-r10-n4: NVLink/NVSwitch fabric faults, error 401).
#   * load_args resolved AT RUN TIME, so every LSF requeue re-evaluates it. Deciding at submit
#     time is why t90k_pure_T12 restarted from step 0 twice, losing 2670 steps each time.
#
# AFTER SUBMITTING: verify at 30-60 s that STAT is RUN, that a first step appeared, and that
# tokens/step is what you intended -- GPUS IS NOT IN THE CONFIG, so a config written for 8 GPUs
# run on 4 silently trains on half the tokens (pre-flight 2/3).
# ============================================================================================
set -euo pipefail
NAME=${1:?job name}
CFG=${2:?config path}
GPUS=${3:?gpu count}
QUEUE_IN=${4:-preemptable}
WALL=${5:-24:00}
MEM=${6:-160G}
REPO=/proj/dmfexp/nima/Code/dolomite-engine
[ -f "$CFG" ] || CFG="$REPO/$CFG"
[ -f "$CFG" ] || { echo "config not found: $CFG" >&2; exit 1; }

case "$QUEUE_IN" in
  ebm|normal) QUEUE=normal; GRP=grp_ebm ;;
  *)          QUEUE=preemptable; GRP=grp_preemptable ;;
esac

# ---- node shape -----------------------------------------------------------------------------
gpn=$GPUS; nnodes=1; span=""; launcher=""; load_sel=""
if [ "$GPUS" -ge 8 ] && [ $((GPUS % 8)) -eq 0 ]; then
    gpn=8; nnodes=$((GPUS / 8))
    [ "$nnodes" -gt 1 ] && { span="span[ptile=1]"; launcher="blaunch"; }
elif [ "$GPUS" -gt 4 ]; then
    gpn=4; nnodes=$(( (GPUS + 3) / 4 )); span="span[ptile=1]"; launcher="blaunch"
fi
[ "$gpn" -eq 8 ] && load_sel="ut<0.5"

BAD_HOSTS="p4-r10-n4"
sel=""
for h in $BAD_HOSTS; do sel="$sel && hname!='$h'"; done
sel="${sel# && }"
[ -n "$load_sel" ] && sel="${sel:+$sel && }$load_sel"

SP=$(grep -E "^\s*save_path:" "$CFG" | head -1 | sed 's/.*save_path:\s*//')
mkdir -p "$HOME/bsub_logs"

echo "submitting $NAME"
echo "  config     : $CFG"
echo "  queue      : $QUEUE / $GRP"
echo "  GPUs       : $GPUS  = $nnodes node(s) x $gpn"
echo "  save_path  : $SP"
[ "$nnodes" -gt 1 ] && echo "  NOTE multi-node: span[ptile=1] + blaunch."
[ "$QUEUE" = normal ] && echo "  NOTE check quota FIRST with: blimits | grep grp_ebm"

TMP=$(mktemp --tmpdir submit_train.XXXXXX.sh)
cat > "$TMP" <<INNER
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
CFG="${CFG}"
SP="${SP}"
if [ -n "\$SP" ] && [ -f "\$SP/latest_checkpointed_iteration.json" ]; then
    if grep -qE "^load_args:" "\$CFG"; then
        echo "RESUME: base config already carries load_args; using it unchanged"
    else
        RCFG="\$SP/runtime_resume_\$\$.yml"
        cp "\$CFG" "\$RCFG"
        printf "\\nload_args:\\n  load_path: %s\\n" "\$SP" >> "\$RCFG"
        CFG="\$RCFG"
        echo "RESUME: built \$RCFG with load_path \$SP"
    fi
    echo "RESUME: latest = \$(cat \$SP/latest_checkpointed_iteration.json)"
else
    echo "RESUME: no checkpoint under \$SP -- starting from step 0 (expected on a FIRST start only)"
fi
${launcher} bash ${REPO}/scripts/common/pretrain.sh "\$CFG"
INNER

out=$(bsub -q "$QUEUE" -G "$GRP" -J "$NAME" \
    -gpu "num=${gpn}/task:mode=exclusive_process" \
    -n "$nnodes" ${span:+-R "$span"} \
    -R "select[$sel]" -M "$MEM" -W "$WALL" \
    -o "$HOME/bsub_logs/${NAME}_%J.stdout" \
    -e "$HOME/bsub_logs/${NAME}_%J.stderr" \
    < "$TMP" 2>&1)
rm -f "$TMP"
echo "$out"
jid=$(echo "$out" | grep -oE 'Job <[0-9]+>' | grep -oE '[0-9]+' | head -1)
[ -n "$jid" ] && { echo; echo "verify in 30-60s:  bjobs -l $jid | grep -E 'Status|PENDING'"; \
  echo "tokens/step from the log: billion_tokens_per_day * 1e9 * step_time / 86400"; }
