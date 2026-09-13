#!/bin/bash
# ============================================================================
# 1B EGPT-Dual BIG-BATCH / muP-LR run — 32 H100 GPUs (4 nodes x 8),
# PREEMPTABLE queue / grp_preemptable.  2026-08-05.
#
# Unlike run_egptdual_1b_recovery_pmt32gpu.sh (which uses ga2 to KEEP the
# 256-seq batch, just 2x faster), this launcher runs a config with ga4 => the
# 32 GPUs give a 512-seq global batch (2.10M tok/step), matching the 400M seeds.
# Paired config: math_egptdual_r0_bigbatch_d2048_int8192_126b_32gpu_lr1p25e3.yml
# (lr 1.25e-3, grad_clip 1.0).  60000 steps = 125.8B tokens.
#
# NUM_TRAINING_STEPS is a 4th arg (default 60000) because this recipe reaches
# 126B in 60k steps, not 120k — the recovery launcher hardcodes 120000, which
# would resubmit forever here.
#
#   Usage: run_egptdual_1b_bigbatch_pmt32gpu_20260805.sh \
#            <config.yml> <save_path> <job_name> [num_training_steps=60000]
#
# WHY preemptable + 32 GPU now: the normal queue caps grp_ebm at 32 GPU and
# R128 already holds 16 there; preemptable is a separate queue, so this grabs
# the idle nodes immediately without touching the normal cap.  Guardian
# (guardian_egptdual_1b_bigbatch_20260805.sh) + in-job resubmit survive
# preemption via latest_checkpointed_iteration.json.
#
# Multi-node specifics: -R "span[ptile=1]" = one task/host (4 distinct nodes);
#   blaunch fans pretrain.sh to every host; NO -x (num=8/task already takes all
#   8 GPUs/host, and -x was the PEND blocker on a packed queue).
# ============================================================================
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine

CONFIG="${1:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
SAVE_PATH="${2:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
JOB_NAME="${3:?usage: $0 <config.yml> <save_path> <job_name> [num_training_steps]}"
NUM_TRAINING_STEPS="${4:-60000}"
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_egptdual_1b_bigbatch_pmt32gpu_20260805.sh
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 4 \
    -R "span[ptile=1] select[hname!='p4-r15-n2']" \
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
echo "Submitted ${JOB_NAME} (32 GPU preemptable/grp_preemptable, blaunch 4-node, ${NUM_TRAINING_STEPS} steps): ${CONFIG}"
