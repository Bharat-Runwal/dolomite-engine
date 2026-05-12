#!/bin/bash
# Submit post-hoc surrogate router training on B5 checkpoint.
# Loads B5, freezes all weights, trains only linear surrogate routers (d→16).
# 5k steps on 1 GPU — fast (~15-30 min).

set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
B5_CKPT=${REPO}/experiments/energy-inference/results/multi-block-ablation/b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3/unsharded
OUTPUT=${REPO}/experiments/boltzmann-moe/results/b5_with_surrogate_router
JOB_NAME=b5_surrogate_router

mkdir -p ${HOME}/bsub_logs

bsub \
    -q normal \
    -G grp_ebm \
    -J ${JOB_NAME} \
    -gpu "num=1" \
    -n 1 \
    -M 32G \
    -W 01:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}

python ${REPO}/experiments/boltzmann-moe/scripts/train_surrogate_router_b5.py \
    --checkpoint ${B5_CKPT} \
    --steps 5000 \
    --lr 1e-3 \
    --batch_size 8 \
    --seq_len 1024 \
    --temperature 1.0 \
    --output ${OUTPUT} \
    --log_interval 100

echo "Surrogate router training complete. Model saved to ${OUTPUT}"
BSUB_SCRIPT

echo "Submitted ${JOB_NAME}"
echo "Monitor: bjobs | bpeek <jobid>"
echo "Output:  ${OUTPUT}"
