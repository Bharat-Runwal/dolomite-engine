#!/bin/bash
# V47: Fine-tune Procrustes 6x2 merged init with original V1 LR (2e-3, cosine decay).
# Reuses v45_merged_procrustes_6x2_init — no new merge step needed.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v47_egpt_6x2_procrustes_merge_finetune_lr2e3_d768.yml
JOB_NAME=egpt_v47_lr2e3_ft

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
bash ${REPO}/scripts/common/pretrain.sh ${CONFIG}
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
