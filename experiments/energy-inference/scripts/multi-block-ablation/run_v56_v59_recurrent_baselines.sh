#!/bin/bash
# Submit V56/V57 (d=768) and V58/V59 (d=1024) single-block recurrent EGPT baselines.
# V56: 1×12 d=768  V57: 1×6 d=768  V58: 1×24 d=1024  V59: 1×12 d=1024
# These answer: do cosreg models (V52/V53) regress to single-layer recurrent performance?
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_DIR=${REPO}/experiments/energy-inference/scripts/multi-block-ablation

submit_recurrent() {
    local VERSION=$1
    local CONFIG=$2
    local SAVE_PATH=$3
    local JOB_NAME=egpt_${VERSION}
    local NUM_GPUS=$4
    local MEM=$5
    local WALL=$6

    bsub \
        -q preemptable \
        -G grp_preemptable \
        -J ${JOB_NAME} \
        -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" \
        -n 1 \
        -M ${MEM} \
        -W ${WALL} \
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
SCRIPT_PATH=${SCRIPT_DIR}/run_v56_v59_recurrent_baselines.sh
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
        echo "Training complete at step \${LATEST_ITER}. Unsharding and submitting eval..."
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

# d=768 models: 4 GPUs, 48G, 4h chunks
submit_recurrent v56_egpt_1x12_d768_lr2e3 \
    ${REPO}/configs/multi_block_ablation/v56_egpt_1x12_d768_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v56_egpt_1x12_d768_lr2e3 \
    4 48G 04:00

submit_recurrent v57_egpt_1x6_d768_lr2e3 \
    ${REPO}/configs/multi_block_ablation/v57_egpt_1x6_d768_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v57_egpt_1x6_d768_lr2e3 \
    4 48G 04:00

# d=1024 models: 4 GPUs, 64G, 4h chunks
submit_recurrent v58_egpt_1x24_d1024_lr1e3 \
    ${REPO}/configs/multi_block_ablation/v58_egpt_1x24_d1024_lr1e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v58_egpt_1x24_d1024_lr1e3 \
    4 64G 04:00

submit_recurrent v59_egpt_1x12_d1024_lr1e3 \
    ${REPO}/configs/multi_block_ablation/v59_egpt_1x12_d1024_lr1e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v59_egpt_1x12_d1024_lr1e3 \
    4 64G 04:00

echo "All 4 recurrent baseline jobs submitted to preemptable queue."
echo "Monitor: bjobs | grep egpt_v5[6789]"
