#!/bin/bash
# Goal A SFT: math-CoT continued training from lr2e4 step 10000 → 1000-step run.
# Same 16 GPU multi-node infrastructure as the lr2e4 long run, but:
#   - smaller training budget (1000 steps vs 240000)
#   - lower LR (1e-5 cosine vs 2e-4)
#   - different data mix (90% math reasoning + 10% pretrain holdout)
#   - resumes weights from lr2e4 step 10000, fresh schedule everything else
# Trap pattern auto-resubmits on preempt; final completion chains unshard + eval.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/math_psda_h1_8gpt_1egpt6x_d1536_sft_metamath_20260617.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_8gpt_1egpt6x_d1536_sft_metamath_20260617
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_sft_metamath_20260617.sh
JOB_NAME=sft_metamath_20260617
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 2 -x \
    -R "span[ptile=1] select[hname!='p4-r24-n2']" \
    -M 64G -W 12:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=1000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    # Already-running SFT: load from its own save path, override base config's load_args
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    TMPCONFIG="\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    # Override the load_args block to load from the SFT save_path (not the base lr2e4)
    printf "\nload_args:\n  load_path: %s\n  load_optimizer: true\n  load_lr_scheduler: true\n  load_starting_iteration: true\n  load_dataloader_state: true\n  load_rng_state: true\n  load_experiments_tracker_state: true\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
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
        echo "[trap] step \${LATEST_ITER} < \${NUM_TRAINING_STEPS}; resubmitting via \${SCRIPT_PATH}"
        bash "\${SCRIPT_PATH}"
    else
        echo "[trap] training complete at step \${LATEST_ITER}; chaining unshard + eval"
        UNSHARDED="\${SAVE_PATH}/unsharded"
        UNSHARD_CFG="\${SAVE_PATH}/unshard_config_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
        bash ${REPO}/experiments/eval_scripts/submit_eval.sh "\${UNSHARDED}" "eval_\${JOB_NAME}"
    fi
}
trap post_exit EXIT
echo "[head=\$(hostname)] LSB_MCPU_HOSTS=\$LSB_MCPU_HOSTS"
echo "[head] launching pretrain.sh on \$(echo \$LSB_MCPU_HOSTS | tr ' ' '\n' | sed 'n; d' | sort -u | wc -l) nodes via blaunch"
blaunch bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}" || echo "[main] blaunch returned \$? — trap will still run"
BSUB
echo "Submitted ${JOB_NAME} (SFT, 1000 steps, 16 GPU multi-node, lr=1e-5 cosine, MetaMathQA mix)"
