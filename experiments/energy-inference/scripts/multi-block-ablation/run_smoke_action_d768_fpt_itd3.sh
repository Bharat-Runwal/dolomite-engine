#!/bin/bash
# Smoke test launcher for the energy_action_loss_coef aux loss.
# Usage:
#   bash run_smoke_action_d768_fpt_itd3.sh <lambda_tag>
#   where <lambda_tag> in {lambda0, lambda1e3, lambda1e2, ...}.
# Reads configs/multi_block_ablation/smoke_action_d768_fpt_itd3_<lambda_tag>_2gpu.yml.
# 2 GPU preemptable, 5k steps. Trap-resubmit; no auto-eval (smoke).
set -euo pipefail
TAG="${1:?Usage: bash run_smoke_action_d768_fpt_itd3.sh <lambda_tag>}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/smoke_action_d768_fpt_itd3_${TAG}_2gpu.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/smoke_action_d768_fpt_itd3_${TAG}_2gpu
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_smoke_action_d768_fpt_itd3.sh
JOB_NAME=smoke_action_${TAG}
mkdir -p "${HOME}/bsub_logs"
[ -f "${CONFIG}" ] || { echo "Config not found: ${CONFIG}"; exit 1; }
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 \
    -M 96G -W 06:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
export PYTHONUNBUFFERED=1
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; TAG=${TAG}; NUM_TRAINING_STEPS=5000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    TMPCONFIG="\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
fi
export WANDB_DIR="\${SAVE_PATH}"
mkdir -p "\${WANDB_DIR}"
LATEST_TRACKER=\$(ls -t \${SAVE_PATH}/global_step*/experiments_tracker.json 2>/dev/null | head -1)
if [ -n "\${LATEST_TRACKER}" ] && [ -f "\${LATEST_TRACKER}" ]; then
    SAVED_ID=\$(python3 -c "import json; print(json.load(open('\${LATEST_TRACKER}'))['id'])" 2>/dev/null)
    if [ -n "\${SAVED_ID}" ]; then
        export WANDB_RUN_ID="\${SAVED_ID}"
        export WANDB_RESUME="must"
    fi
fi
if [ -z "\${WANDB_RUN_ID:-}" ]; then
    rm -f "\${WANDB_DIR}/wandb/wandb-resume.json" 2>/dev/null
    export WANDB_RESUME="never"
fi
post_exit() {
    [ -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml" ] && rm -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
    if [ ! -f "\${LATEST_JSON}" ]; then return; fi
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])" 2>/dev/null || echo 0)
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        echo "[trap] step \${LATEST_ITER} < \${NUM_TRAINING_STEPS}; resubmitting via \${SCRIPT_PATH} ${TAG}"
        bash "\${SCRIPT_PATH}" "${TAG}"
    else
        echo "[trap] training complete at step \${LATEST_ITER}; smoke (no auto-eval)."
    fi
}
trap post_exit EXIT
echo "[head=\$(hostname)] LSB_MCPU_HOSTS=\$LSB_MCPU_HOSTS"
bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}" || echo "[main] pretrain returned \$? — trap will still run"
BSUB
echo "Submitted ${JOB_NAME} (2 GPU smoke, lambda_action via config = ${TAG}, 5k steps)"
