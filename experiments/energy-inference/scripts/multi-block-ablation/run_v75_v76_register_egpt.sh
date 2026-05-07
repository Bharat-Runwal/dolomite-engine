#!/bin/bash
# Register-token EGPT experiments: V75 (400M) and V76 (160M)
# Usage: bash run_v75_v76_register_egpt.sh [v75|v76]

set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_PATH="${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v75_v76_register_egpt.sh"
NUM_TRAINING_STEPS=30000

submit_job() {
    local VERSION=$1 CONFIG=$2 SAVE_PATH=$3 JOB_NAME=$4 NUM_GPUS=$5 MEM=$6 WALL=$7

    ALREADY=$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "${LSB_JOBID:-NOJOBID_SENTINEL}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
    if [ "${ALREADY}" -gt 0 ]; then
        echo "  ${JOB_NAME}: already running/pending — skipping"; return
    fi

    local LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
    if [ -f "${LATEST_JSON}" ]; then
        local ITER=$(python3 -c "import json; print(json.load(open('${LATEST_JSON}'))['latest_checkpointed_iteration'])")
        if [ "${ITER}" -ge "${NUM_TRAINING_STEPS}" ]; then
            echo "  ${JOB_NAME}: complete at step ${ITER} — skipping"; return
        fi
    fi

    bsub -q preemptable -G grp_preemptable -J "${JOB_NAME}" \
        -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" -n 1 -M "${MEM}" -W "${WALL}" \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
SAVE_PATH=${SAVE_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS}
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
        [ "\${ALREADY}" -gt 0 ] && echo "Already running. Skipping." || { echo "Resubmitting..."; bash "${SCRIPT_PATH}" ${VERSION}; }
    else
        echo "Training complete at step \${ITER}."
    fi
fi
BSUB_SCRIPT
    echo "  Submitted ${JOB_NAME}"
}

mkdir -p "${HOME}/bsub_logs"
VERSIONS=("${@}"); [ ${#VERSIONS[@]} -eq 0 ] && VERSIONS=(v75 v76)

for VERSION in "${VERSIONS[@]}"; do
    case "${VERSION}" in
        v75) submit_job v75 \
            "${REPO}/configs/multi_block_ablation/v75_8gpt_4egpt_rmsray_d1280_reg128.yml" \
            "${REPO}/experiments/energy-inference/results/multi-block-ablation/v75_8gpt_4egpt_rmsray_d1280_reg128" \
            "egpt_v75_8gpt_4egpt_rmsray_d1280_reg128" 4 128G 06:00 ;;
        v76) submit_job v76 \
            "${REPO}/configs/multi_block_ablation/v76_4gpt_1egpt6x_rmsray_d1024_reg128.yml" \
            "${REPO}/experiments/energy-inference/results/multi-block-ablation/v76_4gpt_1egpt6x_rmsray_d1024_reg128" \
            "egpt_v76_4gpt_1egpt6x_rmsray_d1024_reg128" 4 128G 06:00 ;;
        *) echo "Unknown: ${VERSION}"; exit 1 ;;
    esac
done
echo "Monitor: bjobs | grep -E 'v7[56]'"
