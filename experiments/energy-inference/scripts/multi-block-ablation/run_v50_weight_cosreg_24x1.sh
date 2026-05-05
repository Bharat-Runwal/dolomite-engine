#!/bin/bash
# V50: EGPT 24×1 d1024 with weight-space cosine similarity regularizer (λ=0.01).
# Scale-up of V48 (12×1 d768 weight cosreg) to 400M. Compare against V40 (act cosreg 24×1).
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v50_egpt_weight_cosreg_24x1_d1024_lr7e4.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v50_egpt_weight_cosreg_24x1_d1024_lr7e4
JOB_NAME=egpt_v50_wcosreg_24x1

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
    if [ "\${STEP}" -lt 30000 ]; then
        echo "Step \${STEP}/30000 — resubmitting..."
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v50_weight_cosreg_24x1.sh
    else
        echo "Training complete at step \${STEP}."
    fi
fi
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
