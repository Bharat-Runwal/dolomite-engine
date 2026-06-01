#!/bin/bash
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/h5_6gpt_1egpt1x_d768_lr2e3.yml
SAVE_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/h5_6gpt_1egpt1x_d768
SCRIPT_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/run_h5_6gpt_1egpt1x_d768.sh
JOB_NAME=h5_6gpt_1egpt1x_d768
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 -M 64G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=30000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    TMPCONFIG="/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
fi
bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
[ -f "/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml" ] && rm -f "/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        bash "\${SCRIPT_PATH}"
    else
        UNSHARDED="\${SAVE_PATH}/unsharded"
        UNSHARD_CFG="/tmp/unshard_\${JOB_NAME}_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh "\${UNSHARDED}" "eval_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME}"
