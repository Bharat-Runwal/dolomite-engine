#!/bin/bash
# Generic GPU-test submitter. USE THIS instead of running a GPU script directly -- an
# interactive session is often on a CPU-only compute node (checked 2026-09-16: `nvidia-smi` ->
# "No devices were found"), so a "quick test" run in-shell either dies or silently falls back
# to CPU and times the wrong thing.
#
#   bash scripts/bsub/submit_gpu_test.sh <job_name> "<command>" [n_gpus] [walltime] [mem]
#
# Example:
#   bash scripts/bsub/submit_gpu_test.sh sparsebench \
#       "python experiments/boltzmann-moe/scripts/bench_sparse_20260916.py --train"
#
# -q preemptable REQUIRES -G grp_preemptable on this cluster. grp_ebm is capacity-limited
# (`blimits`, not `bjobs`, is the authoritative check) so tests belong on preemptable: they are
# short, and a preempted test costs nothing but a resubmit.
set -euo pipefail
JOB=${1:?job name}
CMD=${2:?command}
GPUS=${3:-1}
WALL=${4:-00:30}
MEM=${5:-32G}
# Machine-specific paths come from experiments/paths.sh (REPO_ROOT is self-located, so this
# works in any clone from any cwd; VENV/DATA_ROOT default to this cluster and are overridable
# with DOLOMITE_VENV / DOLOMITE_DATA_ROOT).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/experiments/paths.sh"
mkdir -p "$HOME/bsub_logs"

# HOST EXCLUSION (added 2026-09-20). Until today this submitter passed NO -R select at all, so
# every eval could land on a host whose CUDA init is broken. That is not a theoretical risk: the
# gsm8k half of the w1w2-sparse eval died TWICE on p5-r09-n1 with
#     RuntimeError: CUDA unknown error ... changing env variable CUDA_VISIBLE_DEVICES
# and BOTH TIMES LSF REPORTED THE JOB AS `DONE`, because the inner script does not propagate the
# python exit status. An eval that "completed" with no results file is the signature -- always
# confirm the output file exists, never trust DONE (HANDOFF 15.2c).
# Same promotion bar as submit_train.sh: TWO faults on one host for BAD_HOSTS, one fault goes to
# SUSPECT_HOSTS for a retry only.
#   p5-r09-n1: two CUDA-init failures, evals 1797724 and 1797745 (2026-09-20).
#   p4-r10-n4: NVLink/NVSwitch fabric faults, error 401 (carried over from submit_train.sh).
BAD_HOSTS="${BAD_HOSTS-p5-r09-n1 p4-r10-n4}"
# p2-r03-n1: ONE CUDA-init failure (job 1796776 -- the trainer silently fell back to a CPU
# DeviceMesh and trained nothing for 20 min). Not yet at the two-fault bar.
SUSPECT_HOSTS="${SUSPECT_HOSTS-p2-r03-n1}"
sel=""
for h in $BAD_HOSTS $SUSPECT_HOSTS; do sel="$sel && hname!='$h'"; done
sel="${sel# && }"
RES=()
[ -n "$sel" ] && RES=(-R "select[$sel]")

bsub \
    -q preemptable -G grp_preemptable \
    -J "$JOB" \
    -gpu "num=${GPUS}/task:mode=exclusive_process" \
    "${RES[@]}" \
    -n 1 -M "$MEM" -W "$WALL" \
    -o "$HOME/bsub_logs/${JOB}_%J.stdout" \
    -e "$HOME/bsub_logs/${JOB}_%J.stderr" \
<<INNER
#!/bin/bash
source ${VENV}/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
cd ${REPO}
${CMD}
INNER

echo "submitted. verify in 30-60s:  bjobs -J $JOB   (STAT must be RUN, not EXIT)"
echo "logs: \$HOME/bsub_logs/${JOB}_*.std{out,err}"
