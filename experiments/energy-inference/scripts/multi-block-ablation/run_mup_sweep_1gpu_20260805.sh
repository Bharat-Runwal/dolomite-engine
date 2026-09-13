#!/bin/bash
# ============================================================================
# Single-GPU launcher for the cheap muP LR-transfer sweep (small proxy widths).
# 1-GPU preemptable; SDPA-fallback training via the nanoGPT venv (same env the
# 1B recovery runs used).  Short runs (no resume/guardian needed).
#   Usage: run_mup_sweep_1gpu_20260805.sh <config.yml> <job_name>
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG="${1:?usage: $0 <config.yml> <job_name>}"
JOB="${2:?usage: $0 <config.yml> <job_name>}"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J ${JOB} \
    -gpu "num=1/task:mode=exclusive_process" -n 1 \
    -M 64G -W 08:00 \
    -o "${HOME}/bsub_logs/${JOB}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
bash ${REPO}/scripts/common/pretrain.sh "${CONFIG}"
BSUB
echo "submitted ${JOB} : ${CONFIG}"
