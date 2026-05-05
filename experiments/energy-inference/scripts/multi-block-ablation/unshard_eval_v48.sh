#!/bin/bash
# Unshard V48 (weight cosreg 12×1 d768, step 30k) and run eval harness.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=unshard_eval_v48

bsub \
    -q normal \
    -G grp_ebm \
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

SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/v48_egpt_weight_cosreg_12x1_d768_lr2e3
UNSHARDED=\${SAVE_PATH}/unsharded
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"

echo "=== Unsharding V48 ==="
UNSHARD_CFG="/tmp/v48_unshard_\$\$.yml"
printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
    "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
echo "Unsharded to \${UNSHARDED}"

echo "=== Running eval harness ==="
cd ${REPO}
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \
    --model hf \
    --model_args "pretrained=\${UNSHARDED},dtype=bfloat16,trust_remote_code=True" \
    --tasks \${TASKS} \
    --device cuda:0 \
    --batch_size 4 \
    --trust_remote_code \
    --output_path "\${UNSHARDED}/harness_results.json"
echo "Eval done: \${UNSHARDED}"
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to normal queue."
