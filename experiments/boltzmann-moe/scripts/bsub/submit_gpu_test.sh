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
REPO=/proj/dmfexp/nima/Code/dolomite-engine
mkdir -p "$HOME/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable \
    -J "$JOB" \
    -gpu "num=${GPUS}/task:mode=exclusive_process" \
    -n 1 -M "$MEM" -W "$WALL" \
    -o "$HOME/bsub_logs/${JOB}_%J.stdout" \
    -e "$HOME/bsub_logs/${JOB}_%J.stderr" \
<<INNER
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
cd ${REPO}
${CMD}
INNER

echo "submitted. verify in 30-60s:  bjobs -J $JOB   (STAT must be RUN, not EXIT)"
echo "logs: \$HOME/bsub_logs/${JOB}_*.std{out,err}"
