#!/bin/bash
# Variant: PSD-only (J=0, train only R=LL^T) + warm-start + alpha noise.
#
# Goal: verify that pure gradient descent (no curl) can match unconstrained sym.
# Also checks whether lower energy after recurrence correlates with lower perplexity
# (since R=LL^T guarantees the update is always a descent step).
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_psd_only.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_psd_only \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_psd_only_%J.stdout" \
    -e "${HOME}/struct_proj_psd_only_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py \
    --steps 3000 \
    --batch_size 8 \
    --grad_accum 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --alpha_min 0.35 \
    --alpha_max 0.65 \
    --residual \
    --psd_only \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_psd_only"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
