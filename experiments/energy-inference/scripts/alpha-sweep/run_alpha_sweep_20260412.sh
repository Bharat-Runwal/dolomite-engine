#!/bin/bash
# Submit alpha-sweep inference job to bsub.
# Run from dolomite-engine repo root.
#
# Usage: bash experiments/energy-inference/scripts/run_alpha_sweep_20260412.sh
#
# Or for an interactive session first:
#   bsub -Is -n 1 -G grp_ebm -gpu "num=1/task:mode=exclusive_process" /bin/bash
#   cd /proj/dmfexp/nima/Code/dolomite-engine && pip install -e . -q
#   python experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py

set -e

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CKPT_BASE=/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m

# Checkpoints to sweep (comment out unwanted ones)
CHECKPOINTS=(
    "${CKPT_BASE}/410m_hybrid_s8e4_lr1e3_60k"
    "${CKPT_BASE}/410m_hybrid_s8e4_lr2e3_60k"   # scale_ff ~0 — interesting control
    "${CKPT_BASE}/400m_s1_e6_e6_e6_e6_e6_e6_s1_lr3e4_30k"
)

for CKPT in "${CHECKPOINTS[@]}"; do
    JOB_NAME="alpha_sweep_$(basename $CKPT)"
    echo "Submitting: $JOB_NAME"

    bsub \
        -q standard \
        -G grp_ebm \
        -J "$JOB_NAME" \
        -gpu "num=1/task:mode=exclusive_process" \
        -n 1 \
        -M 32G \
        -W 01:00 \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate -q
cd $REPO
python experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py \
    --checkpoint "$CKPT" \
    --alphas 0.0 0.1 0.25 0.5 0.75 0.9 1.0 \
    --max_new_tokens 80
EOF

done

echo "All jobs submitted. Monitor with: bjobs"
