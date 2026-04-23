#!/bin/bash
# v36_edesc_12x2_d1024_lr1e3: true EGrad/EDesc run (config bug fixed)

set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=/proj/dmfexp/nima/Code/dolomite-engine/configs/multi_block_ablation/v36_edesc_12x2_d1024_lr1e3.yml
SAVE_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/v36_edesc_12x2_d1024_lr1e3
SCRIPT_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/run_v36_edesc_12x2_d1024_lr1e3.sh
JOB_NAME=egptv36

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o "${HOME}/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:\${PYTHONPATH:-}
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=/proj/dmfexp/nima/Code/dolomite-engine/configs/multi_block_ablation/v36_edesc_12x2_d1024_lr1e3.yml
SAVE_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/v36_edesc_12x2_d1024_lr1e3
SCRIPT_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/run_v36_edesc_12x2_d1024_lr1e3.sh
JOB_NAME=egptv36
NUM_TRAINING_STEPS=30000

LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    echo "Resuming from step \${LATEST_ITER}"
    TMPCONFIG="/tmp/v36_edesc_12x2_d1024_lr1e3_resume_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    cat >> "\${TMPCONFIG}" <<YAML_APPEND

load_args:
  load_path: \${SAVE_PATH}
YAML_APPEND
    RUN_CONFIG="\${TMPCONFIG}"
fi

bash /proj/dmfexp/nima/Code/dolomite-engine/scripts/common/pretrain.sh "\${RUN_CONFIG}"

[ -f "/tmp/v36_edesc_12x2_d1024_lr1e3_resume_\${LSB_JOBID}.yml" ] && rm -f "/tmp/v36_edesc_12x2_d1024_lr1e3_resume_\${LSB_JOBID}.yml"

if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        ALREADY=\$(bjobs -J "\${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -cE " RUN | PEND |SSUSP" || echo 0)
        if [ "\${ALREADY}" -gt 0 ]; then
            echo "Job \${JOB_NAME} already has \${ALREADY} running/pending instance(s). Skipping resubmit."
        else
            echo "Training not complete (step \${LATEST_ITER}/\${NUM_TRAINING_STEPS}). Resubmitting..."
            bash "\${SCRIPT_PATH}"
        fi
    else
        echo "Training complete at step \${LATEST_ITER}. Unsharding and submitting eval..."
        UNSHARDED_PATH="\${SAVE_PATH}/unsharded"
        UNSHARD_CONFIG="/tmp/\${JOB_NAME}_unshard_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
            "\${SAVE_PATH}" "\${UNSHARDED_PATH}" > "\${UNSHARD_CONFIG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CONFIG}" && rm -f "\${UNSHARD_CONFIG}"
        bash /proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
            "\${UNSHARDED_PATH}" "eval_\${JOB_NAME}"
    fi
fi
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
