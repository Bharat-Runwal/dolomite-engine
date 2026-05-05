#!/bin/bash
# Variant: Random init (no warm-start) + alpha noise.
# Tests whether the structured J-R parametrisation can converge from scratch.
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_no_warmstart.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_no_warmstart \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/bsub_logs/struct_proj_no_warmstart_%J.stdout" \
    -e "${HOME}/bsub_logs/struct_proj_no_warmstart_%J.stderr" \
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
    --no_warmstart \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_no_warmstart"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
