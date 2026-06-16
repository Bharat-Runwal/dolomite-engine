#!/bin/bash
# Re-run of energy stratification on already-unsharded step6000 (no-itd 8-GPU FPT).
# Bypasses uv (which switched to wrong venv) — calls python directly from
# nanoGPT-og .venv where dolomite + xma + matching transformers live.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_8gpu
JOB_NAME=energy_strat_step6k
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 32G -W 02:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
VENV=/proj/dmfexp/nima/Code/nanoGPT-og/.venv
source "\${VENV}/bin/activate"
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
PY="\${VENV}/bin/python"

# Symlink unsharded -> unsharded_step6000 so the script (which hardcodes "unsharded")
# loads the step6000 weights.
CD_DIR=${SAVE_PATH}
cd "\$CD_DIR"
[ -L unsharded ] && rm unsharded
[ -d unsharded ] && ! [ -L unsharded ] && mv unsharded unsharded_BACKUP_\${LSB_JOBID}
ln -s unsharded_step6000 unsharded
ls -la unsharded
echo

cd /proj/dmfexp/nima/Code/GPT-experiments
MODEL=math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_8gpu \\
  "\${PY}" projects/EGPT-RL/scripts/plot_energy_vs_correctness.py \\
  --n-docs 200 --seq-len 256 --batch-size 4
echo "[analysis] done"

# Restore
cd "\$CD_DIR"
[ -L unsharded ] && rm unsharded
[ -d unsharded_BACKUP_\${LSB_JOBID} ] && mv unsharded_BACKUP_\${LSB_JOBID} unsharded
echo "[restore] symlink restored"
BSUB
echo "Submitted ${JOB_NAME}"
