#!/bin/bash
# Unshard step34000 (17.8B tokens) of math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_16gpu
# and run LM-harness benchmark + fresh energy stratification.
# Single 1-GPU preemptable job; doesn't compete with training (757175) on grp_ebm.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
RUN_NAME=math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_16gpu
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/${RUN_NAME}
STEP=34000
JOB_NAME=bench_step${STEP}_itd3
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

UNSHARD_DIR="${SAVE_PATH}/unsharded_step${STEP}"

# 1. Unshard
if [ ! -f "\${UNSHARD_DIR}/model.safetensors" ]; then
    UNSHARD_CFG="${SAVE_PATH}/unshard_step${STEP}_\${LSB_JOBID}.yml"
    printf "load_args:\n  load_path: %s\n  iteration: ${STEP}\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE_PATH}" "\${UNSHARD_DIR}" > "\${UNSHARD_CFG}"
    "\${PY}" -m lm_engine.unshard --config "\${UNSHARD_CFG}"
    rm -f "\${UNSHARD_CFG}"
    echo "[unshard] done at \${UNSHARD_DIR}"
else
    echo "[unshard] already exists, skipping"
fi
ls -la "\${UNSHARD_DIR}/" | head

# 2. Energy stratification — runs in <1 min
cd "${SAVE_PATH}"
[ -L unsharded ] && rm unsharded
[ -e unsharded ] && [ ! -L unsharded ] && mv unsharded unsharded_BACKUP_\${LSB_JOBID}
ln -s unsharded_step${STEP} unsharded
cd /proj/dmfexp/nima/Code/GPT-experiments
MODEL=${RUN_NAME} \\
  "\${PY}" projects/EGPT-RL/scripts/plot_energy_vs_correctness.py \\
  --n-docs 500 --seq-len 256 --batch-size 4 || echo "[energy_strat] FAILED"
# rename outputs to include step number
ENERGY_NPZ_SRC="/proj/dmfexp/nima/Code/GPT-experiments/projects/EGPT-RL/results/energy_vs_correctness_${RUN_NAME}.npz"
ENERGY_NPZ_DST="/proj/dmfexp/nima/Code/GPT-experiments/projects/EGPT-RL/results/energy_vs_correctness_${RUN_NAME}_step${STEP}.npz"
[ -f "\${ENERGY_NPZ_SRC}" ] && cp "\${ENERGY_NPZ_SRC}" "\${ENERGY_NPZ_DST}" && echo "[energy_strat] saved \${ENERGY_NPZ_DST}"
echo "[energy_strat] done"

# Restore symlink immediately
cd "${SAVE_PATH}"
[ -L unsharded ] && rm unsharded
[ -d unsharded_BACKUP_\${LSB_JOBID} ] && mv unsharded_BACKUP_\${LSB_JOBID} unsharded
echo "[energy_strat] symlink restored"

# 3. LM-harness benchmark — the slow one (~1-2 hours)
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
echo "[lm_harness] done"
echo "===ALL DONE==="
BSUB
echo "Submitted ${JOB_NAME}"
