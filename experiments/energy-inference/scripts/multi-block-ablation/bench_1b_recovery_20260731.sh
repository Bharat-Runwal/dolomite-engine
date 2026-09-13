#!/bin/bash
# Interim trajectory benchmark for a 1B register-recovery arm (R0 or R128-softband).
# Unshards the LATEST checkpoint (read-only; does not disturb the active training job)
# and runs the LM-harness suite incl. gsm8k + gsm8k_cot. Single 1-GPU preemptable job.
#   Usage: bench_1b_recovery_20260731.sh <run_name> <job_name>
# NB. these are MID-training checkpoints at different % (R0 ~73%, R128 ~47%), so this
# is a trajectory read, not an iso-step head-to-head. Report gsm8k_cot flexible-extract.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
RUN_NAME="${1:?usage: $0 <run_name> <job_name>}"
JOB_NAME="${2:?usage: $0 <run_name> <job_name>}"
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/${RUN_NAME}
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
VENV=/proj/dmfexp/nima/Code/nanoGPT-og/.venv
source "\${VENV}/bin/activate"
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
PY="\${VENV}/bin/python"

STEP=\$("\${PY}" -c "import json;print(json.load(open('${SAVE_PATH}/latest_checkpointed_iteration.json'))['latest_checkpointed_iteration'])")
echo "[bench] ${RUN_NAME} latest step=\${STEP}"
UNSHARD_DIR="${SAVE_PATH}/unsharded_step\${STEP}"

# 1. Unshard the latest checkpoint (pin iteration so it's a consistent read)
if [ ! -f "\${UNSHARD_DIR}/model.safetensors" ]; then
    UNSHARD_CFG="${SAVE_PATH}/unshard_bench_\${LSB_JOBID}.yml"
    printf "load_args:\n  load_path: %s\n  iteration: \${STEP}\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE_PATH}" "\${UNSHARD_DIR}" > "\${UNSHARD_CFG}"
    "\${PY}" -m lm_engine.unshard --config "\${UNSHARD_CFG}"
    rm -f "\${UNSHARD_CFG}"
    echo "[unshard] done at \${UNSHARD_DIR}"
else
    echo "[unshard] already exists, skipping"
fi
ls -la "\${UNSHARD_DIR}/" | head

# 2. LM-harness benchmark (gsm8k_cot flexible-extract is the headline recovery metric)
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd ${REPO}
"\${PY}" experiments/energy-inference/scripts/structured-proj/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=\${UNSHARD_DIR},dtype=bfloat16,trust_remote_code=True" \\
    --tasks "\${TASKS}" \\
    --device cuda:0 \\
    --batch_size 8 \\
    --output_path "\${UNSHARD_DIR}/harness_results.json" \\
  || echo "[lm_harness] FAILED"
echo "===ALL DONE=== \${UNSHARD_DIR}/harness_results.json"
BSUB
echo "Submitted ${JOB_NAME} for ${RUN_NAME} (latest ckpt, 1-GPU preemptable)"
