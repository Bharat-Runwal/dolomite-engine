#!/bin/bash
# Variant: Learnable alpha (per-stream sigmoid scalar) + warm-start.
#
# Each energy block learns its own alpha_attn and alpha_ff, converging
# toward the natural descent/curl balance the landscape prefers.
# Logs alpha_{attn,ff} per layer to wandb to see if early/late layers
# diverge in their preferred balance (descent-first vs rotation-first).
#
# Alpha noise is disabled when --learnable_alpha is active.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_learnable_alpha.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_learnable_alpha \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_learnable_alpha_%J.stdout" \
    -e "${HOME}/struct_proj_learnable_alpha_%J.stderr" \
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
    --learnable_alpha \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_learnable_alpha"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
