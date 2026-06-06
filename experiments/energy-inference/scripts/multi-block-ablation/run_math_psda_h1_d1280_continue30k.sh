#!/bin/bash
# Continue from the existing math_psda_h1_6gpt_1egpt6x_d1280 30k checkpoint.
# Loads model weights only (fresh optim, LR scheduler, dataloader, RNG, wandb).
# Trains 30k more steps to step 60000 (single coherent 60k cosine schedule
# starting from a pretrained model).
#
# Self-resubmits via OUR save_path's latest_iter.json — once we've saved at
# least once into math_psda_h1_d1280_continue30k/, future cycles load from
# THAT (the in-progress continuation), not from the original 30k checkpoint.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
BASE_CONFIG=${REPO}/configs/multi_block_ablation/math_psda_h1_d1280_continue30k.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_d1280_continue30k
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_math_psda_h1_d1280_continue30k.sh
JOB_NAME=math_psda_h1_d1280_continue30k
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 -M 64G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; BASE_CONFIG=${BASE_CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=60000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
if [ -f "\${LATEST_JSON}" ]; then
    # Continuation has already started; resume from OUR save_path with full
    # optimizer/LR/dataloader state. Generate a temp config that overrides the
    # base yaml's load_args block.
    TMPCONFIG="/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml"
    grep -v "^load_args:\|^  load_path:\|^  load_optimizer:\|^  load_lr_scheduler:\|^  load_starting_iteration:\|^  load_dataloader_state:\|^  load_experiments_tracker_state:\|^  load_rng_state:" "\${BASE_CONFIG}" > "\${TMPCONFIG}"
    printf "load_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
else
    # First run: use the base config's static load_args (model-only from the
    # original 30k checkpoint).
    RUN_CONFIG="\${BASE_CONFIG}"
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
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh "\${UNSHARDED}" "eval_bbh_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME}"
