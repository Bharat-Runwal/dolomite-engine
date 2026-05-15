#!/bin/bash
# Submit B1/B2/B3 BoltzmannMoE ablation series.
# B1: baseline (no repulsion, no dropout)
# B2: + stochastic contrastive repulsion (coef=0.01, 4 pairs/step)
# B3: + repulsion + dropout=0.1 + weight_decay=0.3
# All: d=768, 12 blocks, 16 experts × 1024 = ~422M params, preemptable queue.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_DIR=${REPO}/experiments/energy-inference/scripts/multi-block-ablation

submit_boltz() {
    local VERSION=$1
    local CONFIG=$2
    local SAVE_PATH=$3
    local JOB_NAME=egpt_${VERSION}

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
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}
CONFIG=${CONFIG}
SAVE_PATH=${SAVE_PATH}
JOB_NAME=${JOB_NAME}
SCRIPT_PATH=${SCRIPT_DIR}/run_b1_b3_boltz_moe.sh
NUM_TRAINING_STEPS=30000

LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    echo "Resuming from step \${LATEST_ITER}"
    TMPCONFIG="/tmp/\${JOB_NAME}_resume_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    cat >> "\${TMPCONFIG}" <<YAML_APPEND

load_args:
  load_path: \${SAVE_PATH}
YAML_APPEND
    RUN_CONFIG="\${TMPCONFIG}"
fi

bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
[ -f "/tmp/\${JOB_NAME}_resume_\${LSB_JOBID}.yml" ] && rm -f "/tmp/\${JOB_NAME}_resume_\${LSB_JOBID}.yml"

if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        ALREADY=\$(bjobs -J "\${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -cE " RUN | PEND |SSUSP" || echo 0)
        if [ "\${ALREADY}" -gt 0 ]; then
            echo "Job \${JOB_NAME} already has \${ALREADY} running/pending instance(s). Skipping resubmit."
        else
            echo "Training not complete (\${LATEST_ITER}/\${NUM_TRAINING_STEPS}). Resubmitting..."
            bash "\${SCRIPT_PATH}"
        fi
    else
        echo "Training complete at step \${LATEST_ITER}. Unsharding..."
        UNSHARDED_PATH="\${SAVE_PATH}/unsharded"
        UNSHARD_CONFIG="/tmp/\${JOB_NAME}_unshard_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
            "\${SAVE_PATH}" "\${UNSHARDED_PATH}" > "\${UNSHARD_CONFIG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CONFIG}" && rm -f "\${UNSHARD_CONFIG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
            "\${UNSHARDED_PATH}" "eval_\${JOB_NAME}"
    fi
fi
BSUB_SCRIPT
    echo "Submitted ${JOB_NAME}"
}

mkdir -p "${HOME}/bsub_logs"

submit_boltz b1_boltz_moe_16x1024_d768_lr2e3 \
    ${REPO}/configs/multi_block_ablation/b1_boltz_moe_16x1024_d768_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/b1_boltz_moe_16x1024_d768_lr2e3

submit_boltz b2_boltz_moe_repulsion_16x1024_d768_lr2e3 \
    ${REPO}/configs/multi_block_ablation/b2_boltz_moe_repulsion_16x1024_d768_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/b2_boltz_moe_repulsion_16x1024_d768_lr2e3

submit_boltz b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3 \
    ${REPO}/configs/multi_block_ablation/b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3

echo "All 3 BoltzmannMoE jobs submitted to preemptable queue."
echo "Monitor: bjobs | grep egpt_b"
