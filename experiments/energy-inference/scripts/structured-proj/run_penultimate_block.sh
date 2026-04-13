#!/bin/bash
# Per-layer focus: train only the penultimate energy block (h.10), freeze h.8,h.9,h.11.
#
# Hypothesis: latent-space "thinking" happens in intermediate energy blocks,
# not the final decoding step. h.10 (6 recurrence iterations) may be where
# the curl/descent balance matters most for representation quality.
# If h.10-only outperforms h.11-only, it supports the latent-thinking view.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_penultimate_block.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_penult_block \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_penult_block_%J.stdout" \
    -e "${HOME}/struct_proj_penult_block_%J.stderr" \
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
    --train_layers 10 \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_penult_block_h10_20260413"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
