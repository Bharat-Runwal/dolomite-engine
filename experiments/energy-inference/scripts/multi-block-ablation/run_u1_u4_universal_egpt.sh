#!/bin/bash
# Submit U1–U4: Universal-style EGPT experiments.
#
# Architecture pattern: 2 pre-GPT + 4 "middle" blocks × [1-5 random iter] + 1 post-GPT
#   Middle blocks are independent with per-block iter_dropout (approximates universal looping)
#
# U1: 2GPT + 4EGPT×3±2 + 1GPT, d=1280, RMSNorm+Rayleigh  ~289M  (primary)
# U2: 2GPT + 4GPT×3±2  + 1GPT, d=1280, GPT-recurrent      ~275M  (GPT control)
# U3: 2GPT + 4EGPT×3±2 + 1GPT, d=1280, RMSNorm no-Rayleigh ~289M (Rayleigh ablation)
# U4: 2GPT + 4EGPT×3±2 + 1GPT, d=1024, RMSNorm+Rayleigh   ~220M  (narrow width)
#
# Effective passes at base (iter=3 per middle block):
#   2 GPT + 4×3 EGPT + 1 GPT = 15 effective passes
#   With iter_dropout_range=2: each block varies 1–5 → total 7–23 passes
#   Simulates "loop the 4-block group 3–5 times" with random compute budget
#
# Usage:
#   bash run_u1_u4_universal_egpt.sh          # submit all four
#   bash run_u1_u4_universal_egpt.sh u1 u2    # submit specific variants
#
# Analysis: After training, run analyze_layer_cosim_u_series.py to check whether
#   middle blocks converge to similar functions (evidence of shared energy landscape).

set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT_PATH="${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_u1_u4_universal_egpt.sh"
NUM_TRAINING_STEPS=30000

submit_job() {
    local VERSION=$1
    local CONFIG=$2
    local SAVE_PATH=$3
    local JOB_NAME=$4
    local NUM_GPUS=$5
    local MEM=$6
    local WALL=$7

    ALREADY=$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "${LSB_JOBID:-NOJOBID_SENTINEL}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
    if [ "${ALREADY}" -gt 0 ]; then
        echo "  ${JOB_NAME}: already has ${ALREADY} running/pending — skipping"
        return
    fi

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

if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        ALREADY=\$(bjobs -J "\${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -v "\${LSB_JOBID}" | grep -E " RUN | PEND |SSUSP" | wc -l) || true
        if [ "\${ALREADY}" -gt 0 ]; then
            echo "Already has \${ALREADY} instance(s). Skipping resubmit."
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

VERSIONS=("${@}")
if [ ${#VERSIONS[@]} -eq 0 ]; then
    VERSIONS=(u1 u2 u3 u4)
fi

for VERSION in "${VERSIONS[@]}"; do
    case "${VERSION}" in
        u1)
            submit_job u1 \
                "${REPO}/configs/multi_block_ablation/u1_2gpt_4egpt3x_rmsray_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/u1_2gpt_4egpt3x_rmsray_d1280" \
                "egpt_u1_2gpt_4egpt3x_rmsray_d1280" \
                4 128G 06:00
            ;;
        u2)
            submit_job u2 \
                "${REPO}/configs/multi_block_ablation/u2_2gpt_4gptrec3x_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/u2_2gpt_4gptrec3x_d1280" \
                "egpt_u2_2gpt_4gptrec3x_d1280" \
                4 128G 06:00
            ;;
        u3)
            submit_job u3 \
                "${REPO}/configs/multi_block_ablation/u3_2gpt_4egpt3x_rmsnorm_d1280.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/u3_2gpt_4egpt3x_rmsnorm_d1280" \
                "egpt_u3_2gpt_4egpt3x_rmsnorm_d1280" \
                4 128G 06:00
            ;;
        u4)
            submit_job u4 \
                "${REPO}/configs/multi_block_ablation/u4_2gpt_4egpt3x_rmsray_d1024.yml" \
                "${REPO}/experiments/energy-inference/results/multi-block-ablation/u4_2gpt_4egpt3x_rmsray_d1024" \
                "egpt_u4_2gpt_4egpt3x_rmsray_d1024" \
                4 128G 06:00
            ;;
        *)
            echo "Unknown: ${VERSION}. Valid: u1 u2 u3 u4"
            exit 1
            ;;
    esac
done

echo ""
echo "Monitor: bjobs | grep -E 'egpt_u[1-4]'"
