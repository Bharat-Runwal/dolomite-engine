#!/bin/bash
# ============================================================================
# 1B EGPT-Dual RESTART launcher — 32 GPU (4 nodes) NORMAL queue / grp_ebm.
# 2026-08-06.  Restarts from another run's checkpoint WEIGHTS at a new (lower)
# LR, then resumes normally from its own checkpoints on guardian resubmits.
#
# First launch (no latest_checkpointed_iteration.json in SAVE_PATH):
#   loads MODEL WEIGHTS ONLY from ${SOURCE_CKPT}/global_step${SOURCE_ITER}
#   (load_optimizer:false => the config's lr becomes the peak LR; fresh Adam +
#   fresh cosine from iteration 0, cushioned by the config's short warmup).
# Resubmit (latest_checkpointed_iteration.json exists): full resume from SAVE_PATH
#   (own optimizer + scheduler + iteration), so training continues seamlessly.
#
#   Usage: run_egptdual_1b_restart_20260806.sh <config.yml> <save_path> <job_name> [num_steps=50000]
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name> [num_steps]}"
SAVE_PATH="${2:?}"; JOB_NAME="${3:?}"; NUM_TRAINING_STEPS="${4:-50000}"
# weights-only source: the lr2e3 run's step-10000 checkpoint
SOURCE_CKPT=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_egptdual_r0_bigbatch_d2048_int8192_126b_32gpu_lr2e3
SOURCE_ITER=10000
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_1b_restart_20260806.sh
mkdir -p "${HOME}/bsub_logs" "${SAVE_PATH}"

bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 4 \
    -R "span[ptile=1] select[hname!='p4-r15-n2']" \
    -M 200G -W 24:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS}
SOURCE_CKPT=${SOURCE_CKPT}; SOURCE_ITER=${SOURCE_ITER}
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
TMPCONFIG="\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
cp "\${CONFIG}" "\${TMPCONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    echo "[head] resubmit: full resume from own checkpoint \${SAVE_PATH}"
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
else
    echo "[head] first launch: WEIGHTS-ONLY load from \${SOURCE_CKPT}/global_step\${SOURCE_ITER}"
    printf "\nload_args:\n  load_path: %s\n  iteration: %s\n  load_optimizer: false\n  load_lr_scheduler: false\n  load_rng_state: false\n  load_starting_iteration: false\n  resume_learning_rate: false\n  load_experiments_tracker_state: false\n" "\${SOURCE_CKPT}" "\${SOURCE_ITER}" >> "\${TMPCONFIG}"
fi
echo "[head] launching pretrain.sh on \$(echo \$LSB_MCPU_HOSTS | tr ' ' '\n' | sed 'n; d' | sort -u | wc -l) nodes via blaunch"
blaunch bash ${REPO}/scripts/common/pretrain.sh "\${TMPCONFIG}"
rm -f "\${TMPCONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        bash "\${SCRIPT_PATH}" "\${CONFIG}" "\${SAVE_PATH}" "\${JOB_NAME}" "\${NUM_TRAINING_STEPS}"
    else
        UNSHARDED="\${SAVE_PATH}/unsharded"
        UNSHARD_CFG="\${SAVE_PATH}/unshard_config_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh "\${UNSHARDED}" "eval_\${JOB_NAME}"
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh "\${UNSHARDED}" "eval_bbh_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME} (32 GPU normal, restart from ${SOURCE_CKPT##*/}/step${SOURCE_ITER}, ${NUM_TRAINING_STEPS} steps): ${CONFIG}"
