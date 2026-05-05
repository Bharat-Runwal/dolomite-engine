#!/bin/bash
# Auto-resubmitting submit script for V71/V72/V73 hybrid 400M experiments.
# Each model auto-resubmits itself when preempted or when wall time expires.
#
# Usage:
#   bash run_v71_v72_v73_hybrid400m.sh           # submit all three
#   bash run_v71_v72_v73_hybrid400m.sh v71        # submit just V71
#   bash run_v71_v72_v73_hybrid400m.sh v71 v72    # submit V71 and V72
#
# Models:
#   V71: 8 GPT + 4 EGPT [3,4,6,9], d=1280, RMS+Rayleigh, ~396M  → save_interval=1000
#   V72: 8 GPT + 4 EGPT [3,4,6,9], d=1280, LayerNorm,    ~396M  → save_interval=1000
#   V73: 6 GPT + 1 EGPT×6,         d=1280, RMS+Rayleigh, ~284M  → save_interval=1000

set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_PATH="${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_v71_v72_v73_hybrid400m.sh"
NUM_TRAINING_STEPS=30000

submit_job() {
    local VERSION=$1
    local CONFIG=$2
    local SAVE_PATH=$3
    local JOB_NAME=$4
    local NUM_GPUS=$5
    local MEM=$6
    local WALL=$7

    # Skip if already running or pending
    local ALREADY
    ALREADY=$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "${LSB_JOBID:-NOJOBID_SENTINEL}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
    if [ "${ALREADY}" -gt 0 ]; then
        echo "  ${JOB_NAME}: already has ${ALREADY} running/pending instance(s) — skipping"
        return
    fi

    # Check if already complete
    local LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
    if [ -f "${LATEST_JSON}" ]; then
        local LATEST_ITER
        LATEST_ITER=$(python3 -c "import json; print(json.load(open('${LATEST_JSON}'))['latest_checkpointed_iteration'])")
        if [ "${LATEST_ITER}" -ge "${NUM_TRAINING_STEPS}" ]; then
            echo "  ${JOB_NAME}: training complete at step ${LATEST_ITER} — skipping"
            return
        fi
    fi

    bsub \
        -q preemptable \
        -G grp_preemptable \
        -J "${JOB_NAME}" \
        -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" \
        -n 1 \
        -M "${MEM}" \
        -W "${WALL}" \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}
SAVE_PATH=${SAVE_PATH}
JOB_NAME=${JOB_NAME}
NUM_TRAINING_STEPS=${NUM_TRAINING_STEPS}

LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
TMPCONFIG="/tmp/\${JOB_NAME}_\${LSB_JOBID}.yml"
cp "${CONFIG}" "\${TMPCONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
fi

bash ${REPO}/scripts/common/pretrain.sh "\${TMPCONFIG}"
[ -f "\${TMPCONFIG}" ] && rm -f "\${TMPCONFIG}"

# Auto-resubmit if training not complete
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        ALREADY=\$(bjobs -J "\${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "\${LSB_JOBID}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
        if [ "\${ALREADY}" -gt 0 ]; then
            echo "Job \${JOB_NAME} already has \${ALREADY} running/pending instance(s). Skipping resubmit."
        else
            echo "Training not complete (\${LATEST_ITER}/${NUM_TRAINING_STEPS}). Resubmitting..."
            bash "${SCRIPT_PATH}" ${VERSION}
        fi
    else
        echo "Training complete at step \${LATEST_ITER}."
    fi
fi
BSUB_SCRIPT
    echo "  Submitted ${JOB_NAME}"
}

mkdir -p "${HOME}/bsub_logs"

# Determine which versions to submit
VERSIONS=("${@}")
if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=(v71 v72 v73)
fi

for VERSION in "${VERSIONS[@]}"; do
    case "${VERSION}" in
        v71)
            submit_job v71 \
                "${REPO}/configs/multi_block_ablation/v71_hybrid_8gpt_4egpt_rmsray_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/v71_hybrid_8gpt_4egpt_rmsray_d1280" \
                "egpt_v71_hybrid_8gpt_4egpt_rmsray_d1280" \
                4 128G 06:00
            ;;
        v72)
            submit_job v72 \
                "${REPO}/configs/multi_block_ablation/v72_hybrid_8gpt_4egpt_layernorm_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/v72_hybrid_8gpt_4egpt_layernorm_d1280" \
                "egpt_v72_hybrid_8gpt_4egpt_layernorm_d1280" \
                4 128G 06:00
            ;;
        v73)
            submit_job v73 \
                "${REPO}/configs/multi_block_ablation/v73_6gpt_1egpt6x_rmsray_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/v73_6gpt_1egpt6x_rmsray_d1280" \
                "egpt_v73_6gpt_1egpt6x_rmsray_d1280" \
                4 128G 06:00
            ;;
        *)
            echo "Unknown version: ${VERSION}. Valid: v71 v72 v73"
            exit 1
            ;;
    esac
done

echo ""
echo "Monitor: bjobs | grep -E 'v7[123]'"
