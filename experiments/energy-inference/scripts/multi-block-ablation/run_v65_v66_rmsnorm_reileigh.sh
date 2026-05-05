#!/bin/bash
# Submit V65 (EGPT + LayerNorm 1×12 d=768) and V66 (EGPT + RMSNorm + Reileigh 1×12 d=768).
#
# Experiment: Does RMSNorm (vs LayerNorm) matter for recurrent EGPT?
#             Does the Reileigh tangent projection P(v,g)=v-g(g·v)/d improve performance?
#
# Comparison table:
#   V56: EGPT 1×12 d=768 RMSNorm              (already run, 30k steps)
#   V65: EGPT 1×12 d=768 LayerNorm            (this script)
#   V66: EGPT 1×12 d=768 RMSNorm + Reileigh   (this script)
#   V0:  Deep GPT 12×1  d=768 RMSNorm         (already run, 30k steps)
#
# Usage: bash experiments/energy-inference/scripts/multi-block-ablation/run_v65_v66_rmsnorm_reileigh.sh
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v65_v66_rmsnorm_reileigh.sh

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
            bash "${SCRIPT_PATH}"
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

mkdir -p "${HOME}/bsub_logs"

# V65: EGPT 1×12 d=768 LayerNorm — 4 GPUs, 48G, 4h chunks
submit_recurrent v65_egpt_1x12_d768_layernorm_lr2e3 \
    ${REPO}/configs/multi_block_ablation/v65_egpt_1x12_d768_layernorm_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v65_egpt_1x12_d768_layernorm_lr2e3 \
    4 48G 04:00

# V66: EGPT 1×12 d=768 RMSNorm + Reileigh — 4 GPUs, 48G, 4h chunks
submit_recurrent v66_egpt_1x12_d768_rmsnorm_reileigh_lr2e3 \
    ${REPO}/configs/multi_block_ablation/v66_egpt_1x12_d768_rmsnorm_reileigh_lr2e3.yml \
    ${REPO}/experiments/energy-inference/results/multi-block-ablation/v66_egpt_1x12_d768_rmsnorm_reileigh_lr2e3 \
    4 48G 04:00

echo ""
echo "Submitted V65 (LayerNorm) and V66 (RMSNorm+Reileigh). Monitor:"
echo "  bjobs | grep 'egpt_v6[56]'"
echo ""
echo "Reference experiments (already run):"
echo "  V56: EGPT 1×12 RMSNorm (no projection)  → results/multi-block-ablation/v56_egpt_1x12_d768_lr2e3"
echo "  V0:  Deep GPT 12×1    d=768              → results/multi-block-ablation/v0_gpt_baseline_d768"
