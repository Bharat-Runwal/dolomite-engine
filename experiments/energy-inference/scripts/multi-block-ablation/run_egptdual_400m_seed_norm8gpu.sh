#!/bin/bash
# ============================================================================
# 400M EGPT-Dual SEED run — 8 H100 GPUs (single node), NORMAL queue / grp_ebm.
# For the 2dK7/itpt seed-variance rebuttal promise: multiple seeds of the paper's
# 400M EGPT-Dual (8 GPT + 4 EGPT, d=1280, iter[3,4,6,9], dual_unconstrained).
# Parametrized so both seeds share ONE launcher.
#
#   Usage: run_egptdual_400m_seed_norm8gpu.sh <config.yml> <save_path> <job_name>
#
# WHY normal (not preemptable): the preemptable queue LIVELOCKED for these seeds
# (2026-07-28..30) — every attempt was preempted at step ~1450-1920, JUST short
# of the save_interval=2000 checkpoint, so nothing was ever saved and each
# guardian resubmit restarted from step 0 (~5 wasted attempts/seed, 0 tokens).
# The `normal` queue RESERVES nodes and is not preempted (only the `priority`
# queue preempts it), so once dispatched it runs stably to the -W limit. Same
# reasoning that moved the 1B recovery arms to normal (see run_egptdual_1b_*).
# ALSO: save_interval was lowered 2000 -> 500 in both configs so a checkpoint
# lands well inside any run window (belt-and-suspenders for the preemptable
# fallback launcher run_egptdual_400m_seed_pmt8gpu.sh).
#
# Batch = mb4 x ga16 x 8 GPU x 4096 = 2.1M tok/step (matches the paper's 512-seq
# global batch, so lr 2e-3 transfers); 30000 steps = ~63B tokens (~few days).
# save_interval 500 => can eval iteration-reduction at ~15k (~31B) / 30k (~63B).
# torch_compile off (avoids the spmd_check gloo deadlock seen on the 1B runs).
# Single node: no multi-node blaunch needed, but we keep span[ptile=1]+blaunch
# for uniformity (fans to the one allocated host). Self-resubmits on death via
# latest_checkpointed_iteration.json; unshards + evals after the final step.
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine

CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name>}"
SAVE_PATH="${2:?usage: $0 <config.yml> <save_path> <job_name>}"
JOB_NAME="${3:?usage: $0 <config.yml> <save_path> <job_name>}"
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_400m_seed_norm8gpu.sh
NUM_TRAINING_STEPS=30000
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 1 \
    -R "span[ptile=1]" \
    -M 64G -W 72:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS}
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    TMPCONFIG="\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
fi
echo "[head] launching pretrain.sh on \$(echo \$LSB_MCPU_HOSTS | tr ' ' '\n' | sed 'n; d' | sort -u | wc -l) node(s) via blaunch"
blaunch bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
[ -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml" ] && rm -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        bash "\${SCRIPT_PATH}" "\${CONFIG}" "\${SAVE_PATH}" "\${JOB_NAME}"
    else
        UNSHARDED="\${SAVE_PATH}/unsharded"
        UNSHARD_CFG="\${SAVE_PATH}/unshard_config_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh "\${UNSHARDED}" "eval_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME} (8 GPU normal/grp_ebm, single node): ${CONFIG}"
