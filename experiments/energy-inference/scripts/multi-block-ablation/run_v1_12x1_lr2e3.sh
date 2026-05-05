#!/bin/bash
# V1 LR=2e-3: 12 distinct EGPT blocks, dual unconstrained proj, d=768, lr=2e-3
# ~176M params | 30k steps | 4 GPUs | 7.86B tokens | est. ~3-4h per job chunk
# Preemptable queue with auto-resume: job resubmits itself if training not complete.
#
# Usage: bash run_v1_12x1.sh
# To monitor: bjobs / bpeek <jobid>

set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v1_12x1_dual_unconstrained_d768_lr2e3.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v1_12x1_d768_lr2e3
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v1_12x1_lr2e3.sh
JOB_NAME=egpt_v1_12x1_lr2e3

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}
CONFIG=${CONFIG}
SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}
JOB_NAME=${JOB_NAME}
NUM_TRAINING_STEPS=30000

# Auto-resume: add load_args if checkpoint exists
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    echo "Resuming from step \${LATEST_ITER}"
    # Build a temp config with load_args appended
    TMPCONFIG="/tmp/v1lr2e3_resume_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    cat >> "\${TMPCONFIG}" <<YAML_APPEND

load_args:
  load_path: \${SAVE_PATH}
YAML_APPEND
    RUN_CONFIG="\${TMPCONFIG}"
fi

# Run training
bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
EXIT_CODE=\$?

# Clean up temp config if created
[ -f "/tmp/v1lr2e3_resume_\${LSB_JOBID}.yml" ] && rm -f "/tmp/v1lr2e3_resume_\${LSB_JOBID}.yml"

# Auto-resubmit if training not complete
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        echo "Training not complete (step \${LATEST_ITER}/\${NUM_TRAINING_STEPS}). Resubmitting..."
        bash "\${SCRIPT_PATH}"
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

echo "Submitted ${JOB_NAME} to preemptable queue."
echo "Monitor: bjobs | bpeek <jobid>"
echo "Logs: \$HOME/${JOB_NAME}_<jobid>.stdout"
