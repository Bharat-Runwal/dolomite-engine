#!/bin/bash
# ============================================================================
# 400M EGPT-Dual SEED run — 4 H100 GPUs (single node), PREEMPTABLE / grp_preemptable.
# ISO-BATCH 4-GPU variant of run_egptdual_400m_seed_pmt8gpu.sh.
# For the 2dK7/itpt seed-variance rebuttal promise: multiple seeds of the paper's
# 400M EGPT-Dual (8 GPT + 4 EGPT, d=1280, iter[3,4,6,9], dual_unconstrained).
#
#   Usage: run_egptdual_400m_seed_pmt4gpu.sh <config.yml> <save_path> <job_name>
#
# WHY 4 GPU: an 8-GPU single-node exclusive reservation could not sustain on the
# packed preemptable queue (2026-07-30/31) — windows collapsed to ~300 steps and
# re-placement kept hard-failing ("no host with 8 free GPUs"). A 4-GPU slot places
# far more readily and re-places faster after preemption. To keep results directly
# comparable to the paper's 400M EGPT-Dual, the configs use ga=32 (was 16) so the
# global batch is UNCHANGED: mb4 x ga32 x 4 GPU x 4096 = 512 seq x 4096 = 2.1M
# tok/step (same as mb4 x ga16 x 8 GPU); lr 2e-3 transfers identically. Cost is
# ~2x wall-clock per step. save_interval=100 so a checkpoint lands well inside any
# preemptable window (progress survives preemption; the 8-GPU/500 combo did not).
# 30000 steps = ~63B tokens. torch_compile on (single node; no spmd_check deadlock).
# Self-resubmits on death via latest_checkpointed_iteration.json; unshards+evals
# after the final step.
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine

CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name>}"
SAVE_PATH="${2:?usage: $0 <config.yml> <save_path> <job_name>}"
JOB_NAME="${3:?usage: $0 <config.yml> <save_path> <job_name>}"
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_400m_seed_pmt4gpu.sh
NUM_TRAINING_STEPS=30000
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 \
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
echo "Submitted ${JOB_NAME} (4 GPU preemptable/grp_preemptable, single node, iso-batch ga32): ${CONFIG}"
