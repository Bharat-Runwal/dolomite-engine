#!/bin/bash
# Unshard a specific global_step ckpt and run lm-evaluation-harness on it.
# Used to capture progression snapshots before they rotate out under max_to_keep=3.
#
# Usage: bash unshard_eval_progression.sh <save_path> <step> <job_name>
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SAVE_PATH="${1:?Usage: unshard_eval_progression.sh <save_path> <step> <job_name>}"
STEP="${2:?need step}"
JOB_NAME="${3:?need job_name}"

UNSHARDED="${SAVE_PATH}/unsharded_step${STEP}"

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J "${JOB_NAME}" \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 03:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval -q

TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"

if [ ! -f "${UNSHARDED}/model.safetensors" ] && [ ! -f "${UNSHARDED}/pytorch_model.bin" ]; then
    echo "=== Unsharding ${SAVE_PATH} step ${STEP} ==="
    UNSHARD_CFG="/tmp/unshard_${STEP}_\$\$.yml"
    printf "load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${SAVE_PATH}" "${STEP}" "${UNSHARDED}" > "\${UNSHARD_CFG}"
    python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
    echo "Unsharded → ${UNSHARDED}"
else
    echo "Skipping unshard (already unsharded)"
fi

echo "=== Running eval harness ==="
cd ${REPO}
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \
    --model hf \
    --model_args "pretrained=${UNSHARDED},dtype=bfloat16,trust_remote_code=True" \
    --tasks \${TASKS} \
    --device cuda:0 \
    --batch_size 4 \
    --trust_remote_code \
    --output_path "${UNSHARDED}/harness_results.json"
echo "Eval done: ${UNSHARDED}/harness_results.json"
BSUB_SCRIPT

echo "Submitted: ${JOB_NAME}  →  ${UNSHARDED}"
