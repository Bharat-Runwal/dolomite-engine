#!/bin/bash
# ============================================================================
# 1B EGPT-Dual CONTINUE-FROM-10k launcher — 32 GPU (4 nodes) NORMAL / grp_ebm.
# 2026-08-07.  Loads WEIGHTS from lr2e3/global_step10000 but CONTINUES the step
# count at 10000 (load_starting_iteration:true) with a fresh optimizer at a lower
# peak LR (config), the schedule rebuilt to step 10000 (resume_learning_rate:true)
# so the LR is on the DECAY branch (no fresh from-zero re-peak).
#
# First launch (no ckpt in SAVE_PATH): weights-only + continue-at-10k + fresh
#   wandb run (load_experiments_tracker_state:false so it doesn't clobber lr2e3's
#   wandb run p9o1qty3).
# Resubmit (SAVE_PATH has a ckpt): full resume from SAVE_PATH (own optim+sched+iter).
#
#   Usage: run_egptdual_1b_cont10k_20260807.sh <config.yml> <save_path> <job_name> [num_steps=60000]
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name> [num_steps]}"
SAVE_PATH="${2:?}"; JOB_NAME="${3:?}"; NUM_TRAINING_STEPS="${4:-60000}"
SOURCE_CKPT=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_egptdual_r0_bigbatch_d2048_int8192_126b_32gpu_lr2e3
SOURCE_ITER=10000
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_1b_cont10k_20260807.sh
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
    echo "[head] first launch: WEIGHTS + CONTINUE-AT-\${SOURCE_ITER} from \${SOURCE_CKPT}"
    # NB resume_learning_rate:FALSE — with load_optimizer:false + warmup, the
    # resume_learning_rate path copies grp["lr"](=0 at warmup step 0) into base_lr,
    # zeroing the LR forever (dolomite bug). False => fresh scheduler from config
    # (warmup 200 -> 1e-3 -> cosine over 49800 = the 50k steps global 10k->60k).
    # load_starting_iteration:true still makes the GLOBAL step show 10k+.
    printf "\nload_args:\n  load_path: %s\n  iteration: %s\n  load_optimizer: false\n  load_lr_scheduler: false\n  load_rng_state: false\n  load_starting_iteration: true\n  resume_learning_rate: false\n  load_experiments_tracker_state: false\n" "\${SOURCE_CKPT}" "\${SOURCE_ITER}" >> "\${TMPCONFIG}"
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
echo "Submitted ${JOB_NAME} (32 GPU normal, continue-at-${SOURCE_ITER} from lr2e3 weights, lr per config, -> ${NUM_TRAINING_STEPS}): ${CONFIG}"
