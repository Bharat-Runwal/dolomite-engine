#!/bin/bash
# One-shot unshard of step6000 from the no-itd 8-GPU FPT run, then run the
# energy stratification (bucket-gap) analysis on that early checkpoint.
# Single GPU, preemptable / grp_preemptable to avoid competing with the
# 16-GPU itd3 production run on grp_ebm.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_8gpu
JOB_NAME=unshard_step6k_noitd
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 32G -W 02:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}

# 1. Unshard global_step6000 -> unsharded_step6000/
UNSHARD_DIR="${SAVE_PATH}/unsharded_step6000"
UNSHARD_CFG="/tmp/unshard_step6k_\${LSB_JOBID}.yml"
printf "load_args:\n  load_path: %s\n  iteration: 6000\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE_PATH}" "\${UNSHARD_DIR}" > "\${UNSHARD_CFG}"
echo "[unshard] config:"; cat "\${UNSHARD_CFG}"
python -m lm_engine.unshard --config "\${UNSHARD_CFG}"
rm -f "\${UNSHARD_CFG}"
echo "[unshard] done — unsharded ckpt at \${UNSHARD_DIR}"
ls -la "\${UNSHARD_DIR}/" | head

# 2. Run energy-vs-correctness analysis (the 5-bucket gap signal).
# plot_energy_vs_correctness.py expects RESULTS_BASE / model / unsharded.
# We point it at unsharded_step6000 by symlinking unsharded -> unsharded_step6000
# (since the script hardcodes 'unsharded'). After analysis we restore.
CD_DIR=${SAVE_PATH}
cd "\$CD_DIR"
[ -L unsharded ] && rm unsharded
[ -d unsharded ] && mv unsharded unsharded_BACKUP_\${LSB_JOBID}
ln -s unsharded_step6000 unsharded
cd /proj/dmfexp/nima/Code/GPT-experiments
MODEL=math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_8gpu \
  uv run python projects/EGPT-RL/scripts/plot_energy_vs_correctness.py \
  --n-docs 200 --seq-len 256 --batch-size 4
echo "[analysis] done"
# Restore — undo the symlink hack
cd "\$CD_DIR"
[ -L unsharded ] && rm unsharded
[ -d unsharded_BACKUP_\${LSB_JOBID} ] && mv unsharded_BACKUP_\${LSB_JOBID} unsharded
echo "[restore] symlink restored"
BSUB
echo "Submitted ${JOB_NAME}"
