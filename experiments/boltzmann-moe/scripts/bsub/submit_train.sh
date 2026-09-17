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
#   bash scripts/bsub/submit_train.sh <name> <config.yml> <gpus> [queue] [wall] [mem] [gpus_per_node]
#
#     queue:         "preemptable" (default) or "ebm" -> normal/grp_ebm
#     gpus_per_node: force the node shape. Default picks 8/node when gpus divides by 8.
#                    Use it when 8/node will not SCHEDULE: two hosts with 8 free GPUs each is far
#                    harder to place than one (a 2x8 probe sat PEND 1.5 h with "requirements for
#                    reserving resource (ngpus_physical) not satisfied: 236 hosts"), so `... 8
#                    preemptable 01:00 192G 4` gives 2 nodes x 4 and still exercises the
#                    multi-node collective + compile path.
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
GPN_FORCE=${7:-}
# Machine-specific paths come from experiments/paths.sh (REPO_ROOT is self-located, so this
# works in any clone from any cwd; VENV/DATA_ROOT default to this cluster and are overridable
# with DOLOMITE_VENV / DOLOMITE_DATA_ROOT).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/experiments/paths.sh"
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
if [ -n "$GPN_FORCE" ]; then
    [ $((GPUS % GPN_FORCE)) -eq 0 ] || { echo "gpus ($GPUS) must divide by gpus_per_node ($GPN_FORCE)" >&2; exit 1; }
    gpn=$GPN_FORCE; nnodes=$((GPUS / GPN_FORCE))
    if [ "$nnodes" -gt 1 ]; then span="span[ptile=1]"; launcher="blaunch"; else span=""; launcher=""; fi
fi
[ "$gpn" -eq 8 ] && load_sel="ut<0.5"

BAD_HOSTS="p4-r10-n4"
# Suspected, ONE failure each so far -- not yet proven bad, kept here so a retry isolates the
# variable rather than re-drawing them. p1-r15-n4 / p2-r22-n1: job 1719508 died with an NCCL
# ncclRemoteError on the FIRST collective (SeqNum=1 ALLREDUCE, last completed work -1 on every
# rank) while an otherwise-identical job on p1-r18-n3 / p2-r16-n1 ran clean. Remove them once
# either is seen healthy, and promote to BAD_HOSTS on a second failure (that is the bar
# ACCEL_FINDINGS used for p4-r10-n4: two runs, 10 fabric errors each).
# SUSPECT list is now EMPTY on purpose. p1-r15-n4 and p2-r22-n1 were put here after a single
# SeqNum=1 ncclRemoteError each, then p2-r15-n4 and p3-r10-n3 failed the same way -- four distinct
# hosts, one fault apiece, which reads as general multi-node startup flakiness rather than bad
# hardware. Excluding hosts on one fault shrinks the candidate pool for no gain.
# The bar for BAD_HOSTS is TWO faults on the same host (what p4-r10-n4 met). Check the accumulated
# evidence before adding anything:  bash scripts/host_placement_log.sh report
SUSPECT_HOSTS="${SUSPECT_HOSTS-}"
sel=""
for h in $BAD_HOSTS ${SUSPECT_HOSTS:-}; do sel="$sel && hname!='$h'"; done
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
source ${VENV}/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
# recorded into wandb config by pretrain.py so a run self-documents its own launch
export DOLOMITE_LAUNCH_CMD="submit_train.sh ${NAME} ${CFG} ${GPUS} ${QUEUE} ${WALL} ${MEM} ${GPN_FORCE}"
# Allocator: expandable segments. Large-K MoE dense phases strand a lot of memory as
# "reserved but unallocated" fragmentation -- a K=64 / I_e=5549 arm failed a 5.42 GiB
# allocation with 3.76 GiB free while holding 11.48 GiB reserved-unallocated. Set here,
# INSIDE the job script, so it survives an LSF requeue (a requeue re-runs the ORIGINAL
# command, so anything exported only at submit time is silently lost).
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
# NCCL: disable NVLink SHARP (NVLS) multicast. The cluster throws
#   "Failed to bind NVLink SHARP (NVLS) Multicast memory: CUDA error 401 ... usually caused by a
#    system or configuration error in the Fabric Manager or NVSwitches"
# and separately ncclRemoteError at SeqNum=1 with "last completed work: -1" on the FIRST
# inter-node allreduce -- the ~50% multi-node startup flakiness. The error message names this
# variable as the mitigation. COST: loses NVLS SHARP collective acceleration, so multi-node
# collectives get slower; the trade is throughput for actually starting. Set inside the job
# script so an LSF requeue cannot drop it.
export NCCL_NVLS_ENABLE=0
# InfiniBand robustness. MEASURED 2026-09-17: multi-node jobs die on the FIRST inter-node
# allreduce with
#   NET/IB: completion ... status=IBV_WC_RETRY_EXC_ERR(12) opcode=IBV_WC_RECV_RDMA_WITH_IMM
#   -> ncclRemoteError "remote process exited or there was a network error"
# RETRY_EXC_ERR means an RDMA write exhausted its retries without an ACK from the peer HCA --
# a FABRIC fault (link/switch/SM/congestion), NOT our code, and NOT NVLS (which is intra-node).
# Raising the IB timeout exponent and retry count is the standard mitigation for transient
# congestion; it costs nothing when the fabric is healthy (longer waits only on retry paths).
# NCCL_DEBUG=WARN so the transport error is printed next time without hand-digging.
export NCCL_IB_TIMEOUT=22
export NCCL_IB_RETRY_CNT=13
export NCCL_DEBUG=WARN
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

# ---- LAUNCH LEDGER --------------------------------------------------------------------------
# world size is NOT in the config, so tokens/step and the total budget cannot be recovered from
# the config alone. Two arms were once compared at different budgets because of exactly that.
# Every launch appends one row here, and the resolved config is snapshotted beside it.
LEDGER="$REPO/experiments/boltzmann-moe/logs/launch_ledger.tsv"
mkdir -p "$(dirname "$LEDGER")" "$REPO/experiments/boltzmann-moe/logs/launch_configs"
[ -s "$LEDGER" ] || printf 'utc	jobid	name	config	gpus	nodes	gpus_per_node	queue	wall	mem	mbs	ga	seq	steps	tokens_per_step	total_tokens	git_commit	config_sha256	cmdline
' > "$LEDGER"
_vals=$(python3 - "$CFG" <<'PYX' 2>/dev/null
import sys, yaml
c = yaml.safe_load(open(sys.argv[1]))
tp = c.get('training_parameters', {})
ds = (c.get('datasets') or [{}])[0].get('class_args', {})
print(tp.get('micro_batch_size',''), tp.get('gradient_accumulation_steps',''),
      ds.get('sequence_length',''), tp.get('num_training_steps',''))
PYX
)
set -- $_vals; _mbs=${1:-}; _ga=${2:-}; _seq=${3:-}; _steps=${4:-}
if [ -n "$_mbs" ] && [ -n "$_ga" ] && [ -n "$_seq" ]; then
    _tps=$(( GPUS * _mbs * _ga * _seq )); _tot=$(( _tps * _steps ))
else _tps=; _tot=; fi
_sha=$(sha256sum "$CFG" 2>/dev/null | cut -c1-16)
_commit=$(git -C "$REPO" rev-parse --short HEAD 2>/dev/null)
cp "$CFG" "$REPO/experiments/boltzmann-moe/logs/launch_configs/${NAME}_${jid:-nojob}.yml" 2>/dev/null
printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
  "$(date -u +%Y-%m-%dT%H:%M:%SZ)" "${jid:-none}" "$NAME" "$CFG" "$GPUS" "${nnodes:-?}" "${gpn:-?}" \
  "$QUEUE" "$WALL" "$MEM" "$_mbs" "$_ga" "$_seq" "$_steps" "${_tps:-?}" "${_tot:-?}" \
  "${_commit:-?}" "${_sha:-?}" "submit_train.sh $NAME $CFG $GPUS $QUEUE $WALL $MEM ${7:-}" >> "$LEDGER"
echo "ledger: $LEDGER  (gpus=$GPUS tok/step=${_tps:-?} total=${_tot:-?})"

[ -n "$jid" ] && { echo; echo "verify in 30-60s:  bjobs -l $jid | grep -E 'Status|PENDING'"; \
  echo "tokens/step from the log: billion_tokens_per_day * 1e9 * step_time / 86400"; }
