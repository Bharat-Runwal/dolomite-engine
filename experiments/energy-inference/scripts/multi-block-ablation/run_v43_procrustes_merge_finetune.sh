#!/bin/bash
# V43: Merge V1 EGPT 12→6 layers (Procrustes-aligned), then fine-tune 5k steps.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
V1_RESULT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v1_12x1_d768_lr2e3
MERGE_OUT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v43_merged_procrustes_init
CONFIG=${REPO}/configs/multi_block_ablation/v43_egpt_6layer_procrustes_merge_finetune_d768_lr2e3.yml
JOB_NAME=egpt_v43_procrustes_merge_ft

echo "=== Step 1: Merging V1 EGPT 12-layer → 6-layer (Procrustes) ==="
bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME}_merge \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 \
    -M 32G \
    -W 01:00 \
    -o "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_merge_%J.stdout" \
    -e "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_merge_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}

python ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/merge_layers_v42.py \
    --ckpt_path ${V1_RESULT} \
    --out_path  ${MERGE_OUT} \
    --strategy procrustes \
    --out_n_layers 6

echo "Procrustes merge complete. Starting fine-tune job..."

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 02:00 \
    -o "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<INNER
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
bash ${REPO}/scripts/common/pretrain.sh ${CONFIG}
INNER

echo "Fine-tune job submitted."
BSUB_SCRIPT

echo "Submitted merge job (${JOB_NAME}_merge) to preemptable queue."
