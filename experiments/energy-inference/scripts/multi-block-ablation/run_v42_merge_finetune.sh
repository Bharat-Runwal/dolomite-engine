#!/bin/bash
# V42: Merge V1 EGPT 12→6 layers (naive), then fine-tune 5k steps.
# Step 1: run merge_layers_v42.py on a GPU node to save merged HF model
# Step 2: submit fine-tune job from the merged model
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
V1_RESULT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v1_12x1_d768_lr2e3
MERGE_OUT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v42_merged_naive_init
CONFIG=${REPO}/configs/multi_block_ablation/v42_egpt_6layer_naive_merge_finetune_d768_lr2e3.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v42_egpt_6layer_naive_merge_finetune_d768
JOB_NAME=egpt_v42_naive_merge_ft

echo "=== Step 1: Merging V1 EGPT 12-layer → 6-layer ==="
bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME}_merge \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 \
    -M 32G \
    -W 01:00 \
    -o "${HOME}/${JOB_NAME}_merge_%J.stdout" \
    -e "${HOME}/${JOB_NAME}_merge_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}

python ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/merge_layers_v42.py \
    --ckpt_path ${V1_RESULT} \
    --out_path  ${MERGE_OUT} \
    --strategy naive \
    --out_n_layers 6

echo "Merge complete. Starting fine-tune job..."

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 02:00 \
    -o "${HOME}/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/${JOB_NAME}_%J.stderr" \
    <<INNER
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
bash ${REPO}/scripts/common/pretrain.sh ${CONFIG}
INNER

echo "Fine-tune job submitted."
BSUB_SCRIPT

echo "Submitted merge job (${JOB_NAME}_merge) to preemptable queue."
echo "Fine-tune will auto-submit on merge completion."
