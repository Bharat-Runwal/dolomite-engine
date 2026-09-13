#!/bin/bash
# ============================================================================
# 1B EGPT-Dual register-recovery run — 16 H100 GPUs (2 nodes x 8),
# NORMAL queue / grp_ebm.  Parametrized so BOTH arms (R0 baseline, R128
# softband) share ONE launcher.  Uses the ga4 16-GPU configs =>
# mb4 x ga4 x 16 GPU x 4096 = 1.05M tok/step; 120000 steps = 125.8B tok.
#
#   Usage: run_egptdual_1b_recovery_norm16gpu.sh <config.yml> <save_path> <job_name>
#
# WHY normal (not preemptable): 2026-07-27 the preemptable queue LIVELOCKED —
# on a saturated cluster our jobs were preempted 2-3 min after each dispatch
# (less than torch.compile warmup), so they never reached a training step
# (R128 stuck at ckpt 2000 across many requeues; R0 PEND 5.5h).  The `normal`
# queue can RESERVE nodes and is higher-priority (it preempts, is not
# preempted), so once dispatched it runs stably to the -W limit.  Recipe is
# IDENTICAL to the preemptable variant; jobs resume from existing checkpoints.
#
# Multi-node specifics:
#   * -R "span[ptile=1]" forces one task per host so both nodes are distinct.
#   * blaunch bash pretrain.sh fans the script to every allocated host.
#   * NO -x: num=8/task already grabs all 8 GPUs/host; -x was the PEND blocker
#     (needs a completely empty node; 738 hosts failed it).
# Unshard + eval at the end are head-only (single-process), outside blaunch.
# Self-resubmits via latest_checkpointed_iteration.json; auto-chains
# unshard + LM-harness + BBH eval after the final step.
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine

CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name>}"
SAVE_PATH="${2:?usage: $0 <config.yml> <save_path> <job_name>}"
JOB_NAME="${3:?usage: $0 <config.yml> <save_path> <job_name>}"
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_1b_recovery_norm16gpu.sh
NUM_TRAINING_STEPS=120000
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 2 \
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
echo "[head] launching pretrain.sh on \$(echo \$LSB_MCPU_HOSTS | tr ' ' '\n' | sed 'n; d' | sort -u | wc -l) nodes via blaunch"
echo "[head] LSB_MCPU_HOSTS=\$LSB_MCPU_HOSTS"
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
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh "\${UNSHARDED}" "eval_bbh_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME} (16 GPU normal/grp_ebm, blaunch 2-node): ${CONFIG}"
