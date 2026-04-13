#!/bin/bash
# Per-layer focus: train only the last energy block (h.11), freeze h.8-h.10.
#
# Hypothesis: h.11 is the decoding block (bridges latent space → token space).
# Retraining only it with learnable alpha tests whether the alpha balance matters
# most at the final latent→token transition, or earlier in the recurrence.
#
# h.11 has 9 recurrence iterations in this checkpoint (s8e4 config).
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_last_block.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_last_block \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_last_block_%J.stdout" \
    -e "${HOME}/struct_proj_last_block_%J.stderr" \
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
    --train_layers 11 \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_last_block_h11"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
