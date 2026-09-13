#!/bin/bash
# ============================================================================
# 1B EGPT-Dual BIG-BATCH / TUNED-LR run — 32 H100 GPUs (4 nodes x 8),
# NORMAL queue / grp_ebm.  2026-08-05.
#
# Normal (not preemptable) because: (a) the tuned-LR R0 is the headline run now,
# (b) normal is higher-priority so it can PREEMPT preemptable jobs to assemble
# 4 clean nodes (the 4-node preemptable submit could NOT reserve ngpus_physical
# on the busy cluster), (c) grp_ebm normal cap is 32 GPU and R128 (16) has been
# killed, so the cap has room.  Paired config:
#   math_egptdual_r0_bigbatch_d2048_int8192_126b_32gpu_lr2e3.yml
# (512-seq batch, lr 2e-3 from the muP sweep, grad_clip 1.0).  60000 steps = 126B.
#
# NUM_TRAINING_STEPS is a 4th arg (default 60000): this recipe reaches 126B in
# 60k steps, not the recovery launcher's hardcoded 120k.
#
#   Usage: run_egptdual_1b_bigbatch_norm32gpu_20260805.sh \
#            <config.yml> <save_path> <job_name> [num_training_steps=60000]
#
# Multi-node: -R "span[ptile=1]" = one task/host (4 distinct nodes); blaunch fans
#   pretrain.sh to every host; NO -x (num=8/task already takes all 8 GPUs/host,
#   and -x hurt placement on the packed cluster).  Excludes the known-bad node.
# Self-resubmits via latest_checkpointed_iteration.json (also covers -W wall kills);
# auto-chains unshard + LM-harness + BBH eval after the final step.
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine

CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
SAVE_PATH="${2:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
JOB_NAME="${3:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
NUM_TRAINING_STEPS="${4:-60000}"
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_1b_bigbatch_norm32gpu_20260805.sh
mkdir -p "${HOME}/bsub_logs"

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
echo "Submitted ${JOB_NAME} (32 GPU normal/grp_ebm, blaunch 4-node, ${NUM_TRAINING_STEPS} steps): ${CONFIG}"
