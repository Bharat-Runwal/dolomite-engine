#!/bin/bash
# Smoke test launcher for psd_anti proj_type.
# Single-shot (no auto-resubmit), 4 GPUs, 30 min walltime.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/smoke_psda_h1_d1280.yml
JOB_NAME=smoke_psda_h1
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 -M 64G -W 00:30 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
bash ${REPO}/scripts/common/pretrain.sh "${CONFIG}"
BSUB
echo "Submitted ${JOB_NAME}"
