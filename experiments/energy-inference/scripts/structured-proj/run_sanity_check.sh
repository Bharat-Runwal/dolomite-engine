#!/bin/bash
# Sanity check: train only unconstrained proj.weight (no arch change).
# Initial loss should be ~2.4 nats/token (PPL ~10.83).
# Also runs a frozen eval pass to confirm the checkpoint itself is correct.

REPO=/proj/dmfexp/nima/Code/dolomite-engine

# Job 1: frozen eval (pure eval loss, zero gradients)
bsub \
    -q normal \
    -G grp_ebm \
    -J sanity_frozen_eval \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 00:20 \
    -o "${HOME}/sanity_frozen_%J.stdout" \
    -e "${HOME}/sanity_frozen_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/structured-proj/train_unconstrained_sanity_20260413.py \
    --steps 50 \
    --batch_size 8 \
    --seq_len 512 \
    --grad_accum 1 \
    --log_interval 5 \
    --frozen \
    --wandb_name "410m_frozen_eval_sanity"
EOF

# Job 2: train unconstrained proj (should start at ~2.4, decrease)
bsub \
    -q normal \
    -G grp_ebm \
    -J sanity_unconstrained_train \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 00:30 \
    -o "${HOME}/sanity_train_%J.stdout" \
    -e "${HOME}/sanity_train_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/structured-proj/train_unconstrained_sanity_20260413.py \
    --steps 500 \
    --batch_size 8 \
    --seq_len 512 \
    --grad_accum 8 \
    --log_interval 5 \
    --wandb_name "410m_unconstrained_sanity_train"
EOF

echo "Submitted sanity jobs. Monitor: bjobs"
