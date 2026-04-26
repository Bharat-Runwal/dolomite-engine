#!/bin/bash
# V51: Procrustes 6×2 merge FT, cosine lr=2e-3→2e-4, 10k steps (2× V47).
# V47 (5k steps) got 3.464, still dropping — 10k should push lower.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v51_egpt_6x2_procrustes_merge_finetune_lr2e3_10k_d768.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v51_egpt_6x2_procrustes_merge_finetune_lr2e3_10k_d768
JOB_NAME=egpt_v51_procft_10k

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

LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
if [ -f "\${LATEST_JSON}" ]; then
    echo "Resuming from checkpoint..."
fi

bash ${REPO}/scripts/common/pretrain.sh ${CONFIG}

# Auto-resubmit if not complete
LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
if [ -f "\${LATEST_JSON}" ]; then
    STEP=\$(python -c "import json; d=json.load(open('\${LATEST_JSON}')); print(d.get('iteration', 0))")
    if [ "\${STEP}" -lt 10000 ]; then
        echo "Step \${STEP}/10000 — resubmitting..."
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v51_procrustes_merge_ft_10k.sh
    else
        echo "Training complete at step \${STEP}."
    fi
fi
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
