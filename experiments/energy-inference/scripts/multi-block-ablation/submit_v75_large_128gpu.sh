#!/bin/bash
# Submit V75-large: 128-GPU preemptable run for maximum compute impact.
# 128 GPUs × micro_batch=4 × grad_accum=1 × seq=4096 ≈ 2M tokens/step × 30k = ~60B tokens
# This matches the paper's larger-scale experiments.
# NOTE: grp_preemptable, not grp_ebm

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SAVE=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/v75_large_8gpt_4egpt_rmsray_d1280_reg128
CONFIG=${REPO}/configs/multi_block_ablation/v75_large_8gpt_4egpt_rmsray_d1280_reg128.yml
JOB_NAME=egpt_v75_large_8gpt_4egpt_rmsray_reg128

mkdir -p ~/bsub_logs

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J "${JOB_NAME}" \
    -gpu "num=128/task:mode=exclusive_process" \
    -n 1 \
    -M 2048G \
    -W 06:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
SAVE_PATH=${SAVE}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=30000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
TMPCONFIG="/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml"
cp "${CONFIG}" "\${TMPCONFIG}"
[ -f "\${LATEST_JSON}" ] && printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
bash ${REPO}/scripts/common/pretrain.sh "\${TMPCONFIG}"
[ -f "\${TMPCONFIG}" ] && rm -f "\${TMPCONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        ALREADY=\$(bjobs -J "\${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "\${LSB_JOBID}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
        [ "\${ALREADY}" -gt 0 ] && echo "Already running." || { echo "Resubmitting..."; bash "${BASH_SOURCE[0]}"; }
    fi
fi
BSUB_SCRIPT
echo "Submitted ${JOB_NAME} — 128 GPUs, preemptable"
echo "Check: bjobs | grep v75_large"
echo "WARN: This may stay in PEND for hours/days depending on cluster availability."
