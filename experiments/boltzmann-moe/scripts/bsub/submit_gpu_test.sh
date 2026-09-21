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
#   p2-r03-n1: THREE CUDA-init faults on 2026-09-20 -- job 1796776 (trainer fell back to a CPU
#     DeviceMesh and trained nothing for 20 min) and evals 1797724 + 1797745 (both
#     'CUDA unknown error ... CUDA_VISIBLE_DEVICES', both reported DONE having written nothing).
#   p4-r10-n4: NVLink/NVSwitch fabric faults, error 401 (carried over from submit_train.sh).
#
# ⚠ HOW THE HOST WAS IDENTIFIED, AND A TRAP THAT GOT IT WRONG FIRST. **LSF RECYCLES JOB IDS**, so
# `bhist -l <jobid>` can return a MONTHS-OLD job: bhist for 1797724/1797745/1797747 all returned
# records dated `Sat Jun 27 2026`, and reading a hostname out of those wrongly accused p5-r09-n1
# (which has no evidence against it at all and is NOT excluded here). Attribute a host from
# `bjobs -o exec_host` WHILE THE JOB IS LIVE, or from the hostname recorded in the job's own
# stdout under $HOME/bsub_logs -- never from bhist on a recycled id. Cross-check the bhist record's
# DATE before believing anything it says.
BAD_HOSTS="${BAD_HOSTS-p2-r03-n1 p4-r10-n4}"
SUSPECT_HOSTS="${SUSPECT_HOSTS-}"
sel=""
for h in $BAD_HOSTS $SUSPECT_HOSTS; do sel="$sel && hname!='$h'"; done
sel="${sel# && }"
RES=()
[ -n "$sel" ] && RES=(-R "select[$sel]")

# QUEUE OVERRIDE (added 2026-09-21). Default stays preemptable -- that is correct while grp_ebm is
# at 32/32, which CLAUDE.md documents as the normal case. But when grp_ebm IS genuinely free the rule
# explicitly allows using it, and preemption is not cheap for evals: a suspended eval RESTARTS FROM
# SCRATCH rather than resuming. Measured 2026-09-21 on abl_B_400M's benchmark -- it was at
# 42,135/138,807 likelihood requests when preempted and came back at 2,284, losing ~30 min. At a
# fairshare priority of 0.0030 that can loop indefinitely and an eval may never finish.
#   EVAL_QUEUE=ebm bash scripts/bsub/submit_gpu_test.sh ...
EVAL_QUEUE="${EVAL_QUEUE:-preemptable}"
if [ "$EVAL_QUEUE" = "ebm" ]; then Q="-q normal -G grp_ebm"; else Q="-q preemptable -G grp_preemptable"; fi

bsub \
    $Q \
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
