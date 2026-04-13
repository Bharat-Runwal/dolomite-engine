#!/bin/bash
# Variant: Two-LR training (backbone lr=1e-5, proj lr=3e-4) + warm-start + alpha noise.
#
# Unfreezes the backbone layers alongside the structured proj so the full
# model can co-adapt, but at a much lower LR to prevent catastrophic
# forgetting of the pretrained knowledge.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_two_lr.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_two_lr \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 02:00 \
    -o "${HOME}/struct_proj_two_lr_%J.stdout" \
    -e "${HOME}/struct_proj_two_lr_%J.stderr" \
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
    --residual \
    --backbone_lr 1e-5 \
    --alpha_min 0.35 \
    --alpha_max 0.65 \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_two_lr"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
