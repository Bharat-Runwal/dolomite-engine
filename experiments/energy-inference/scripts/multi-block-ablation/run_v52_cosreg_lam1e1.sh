#!/bin/bash
# V52: EGPT 12×1 d768, activation cosreg λ=0.1 (10× V39). 30k steps.
# Tests whether stronger constant regularization actually improves alignment.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3
JOB_NAME=egpt_v52_cosreg_lam1e1

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
RUN_CONFIG="${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    echo "Resuming from step \${LATEST_ITER}"
    TMPCONFIG="/tmp/v52_resume_\${LSB_JOBID}.yml"
    cp "\${RUN_CONFIG}" "\${TMPCONFIG}"
    printf "\nload_args:\n  load_path: %s\n" "${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
fi

bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
[ -f "/tmp/v52_resume_\${LSB_JOBID}.yml" ] && rm -f "/tmp/v52_resume_\${LSB_JOBID}.yml"

LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
if [ -f "\${LATEST_JSON}" ]; then
    STEP=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${STEP}" -lt 30000 ]; then
        echo "Step \${STEP}/30000 — resubmitting..."
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v52_cosreg_lam1e1.sh
    else
        echo "Training complete at step \${STEP}."
    fi
fi
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
