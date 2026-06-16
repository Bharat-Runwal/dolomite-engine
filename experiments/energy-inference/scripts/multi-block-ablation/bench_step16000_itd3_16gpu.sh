#!/bin/bash
# Unshard step16000 of math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_16gpu and
# run the LM-harness benchmark + fresh energy stratification.
# Single 1-GPU preemptable job; doesn't compete with the running training (752128)
# on grp_ebm.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
RUN_NAME=math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_16gpu
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/${RUN_NAME}
JOB_NAME=bench_step16k_itd3
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

# 1. Unshard global_step16000 -> unsharded_step16000
UNSHARD_DIR="${SAVE_PATH}/unsharded_step16000"
if [ ! -f "\${UNSHARD_DIR}/model.safetensors" ]; then
    UNSHARD_CFG="${SAVE_PATH}/unshard_step16k_\${LSB_JOBID}.yml"
    printf "load_args:\n  load_path: %s\n  iteration: 16000\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE_PATH}" "\${UNSHARD_DIR}" > "\${UNSHARD_CFG}"
    "\${PY}" -m lm_engine.unshard --config "\${UNSHARD_CFG}"
    rm -f "\${UNSHARD_CFG}"
    echo "[unshard] done at \${UNSHARD_DIR}"
else
    echo "[unshard] already exists, skipping"
fi
ls -la "\${UNSHARD_DIR}/"

# 2. Energy stratification (5-bucket gap) — fast, do first
cd "${SAVE_PATH}"
[ -L unsharded ] && rm unsharded
[ -d unsharded ] && ! [ -L unsharded ] && mv unsharded unsharded_BACKUP_\${LSB_JOBID}
ln -s unsharded_step16000 unsharded
cd /proj/dmfexp/nima/Code/GPT-experiments
MODEL=${RUN_NAME}_step16k_DUMMY  # not used, RESULTS_BASE expects model dir
# Hack: temporarily symlink RESULTS dir name to point at our run
RBASE=${REPO}/experiments/energy-inference/results/multi-block-ablation
[ -L "\${RBASE}/${RUN_NAME}_step16k_DUMMY" ] && rm "\${RBASE}/${RUN_NAME}_step16k_DUMMY"
ln -s ${RUN_NAME} "\${RBASE}/${RUN_NAME}_step16k_DUMMY"
MODEL=${RUN_NAME}_step16k_DUMMY \\
  "\${PY}" projects/EGPT-RL/scripts/plot_energy_vs_correctness.py \\
  --n-docs 500 --seq-len 256 --batch-size 4 || echo "[energy_strat] FAILED"
rm -f "\${RBASE}/${RUN_NAME}_step16k_DUMMY"
echo "[energy_strat] done"

# 3. LM-harness benchmark (the slow one — runs 13 tasks)
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
cd ${REPO}
"\${PY}" experiments/energy-inference/scripts/structured-proj/eval_harness.py \\
    --pretrained "\${UNSHARD_DIR}" \\
    --tasks "\${TASKS}" \\
    --batch_size 8 \\
    --output_path "\${UNSHARD_DIR}/harness_results.json" \\
    --device cuda \\
  || echo "[lm_harness] FAILED"
echo "[lm_harness] done"

# 4. Restore symlink
cd "${SAVE_PATH}"
[ -L unsharded ] && rm unsharded
[ -d unsharded_BACKUP_\${LSB_JOBID} ] && mv unsharded_BACKUP_\${LSB_JOBID} unsharded
echo "[restore] symlink restored"
echo "===ALL DONE==="
BSUB
echo "Submitted ${JOB_NAME}"
