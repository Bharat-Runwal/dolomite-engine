#!/bin/bash
# bench_checkpoint.sh — one-shot end-to-end benchmark of a sharded checkpoint.
# Unshards (if needed), then submits LM-harness + BBH eval bsubs.
#
# Usage:
#   bash bench_checkpoint.sh <save_path> <step> <run_name> [--no-bbh]
#
# Example:
#   bash bench_checkpoint.sh \
#     /proj/.../math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_16gpu \
#     34000  bench_step34k_itd3
#
# Outputs:
#   <save_path>/unsharded_step<N>/                          — HF-format weights
#   <save_path>/unsharded_step<N>/harness_results.json      — 13-task LM
#   <save_path>/unsharded_step<N>/harness_bbh_results_*.json — BBH (if --with-bbh)

set -euo pipefail
SAVE_PATH="${1:?Usage: bench_checkpoint.sh <save_path> <step> <run_name> [--no-bbh]}"
STEP="${2:?step required}"
RUN_NAME="${3:?run_name required}"
BBH_FLAG="${4:---with-bbh}"

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPTS_DIR="${REPO}/experiments/eval_scripts"
UNSHARD_DIR="${SAVE_PATH}/unsharded_step${STEP}"
JOB_NAME="bench_${RUN_NAME}_step${STEP}"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J "${JOB_NAME}_unshard" \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 32G -W 01:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_unshard_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_unshard_%J.stderr" \
    <<EOF
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
if [ ! -f "${UNSHARD_DIR}/model.safetensors" ]; then
    UNSHARD_CFG="${SAVE_PATH}/unshard_step${STEP}_\${LSB_JOBID}.yml"
    printf "load_args:\n  load_path: %s\n  iteration: ${STEP}\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE_PATH}" "${UNSHARD_DIR}" > "\${UNSHARD_CFG}"
    python -m lm_engine.unshard --config "\${UNSHARD_CFG}"
    rm -f "\${UNSHARD_CFG}"
fi
echo "[unshard] ready at ${UNSHARD_DIR}"
bash ${SCRIPTS_DIR}/submit_eval.sh "${UNSHARD_DIR}" "${JOB_NAME}_lm" "${BBH_FLAG}"
EOF
echo "Submitted ${JOB_NAME}_unshard (chains LM + BBH after unshard)"
