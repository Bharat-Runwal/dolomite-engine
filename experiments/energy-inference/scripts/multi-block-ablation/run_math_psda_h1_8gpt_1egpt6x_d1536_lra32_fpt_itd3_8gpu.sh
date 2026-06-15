#!/bin/bash
# Fresh-from-scratch sibling of run_math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_8gpu.sh
# with iter_dropout 6±3 enabled on the EGPT block from step 0. Runs in parallel
# to job 724594 (no-itd) so both basin trajectories are directly comparable.
# Single node, 8 GPU, normal queue / grp_ebm. Same effective batch (524k tok/step),
# same LR (1e-3), same total tokens target (126B).
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_8gpu.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_8gpu
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_8gpu.sh
JOB_NAME=math_h1_d1536_fpt_itd3_8g
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 1 -x -M 64G -W 24:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=240000
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
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh "\${UNSHARDED}" "eval_bbh_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME} (8 GPU normal/grp_ebm, iter_dropout 6±3, fresh from step 0)"
