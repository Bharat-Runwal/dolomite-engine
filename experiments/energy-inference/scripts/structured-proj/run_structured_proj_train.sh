#!/bin/bash
# Retrain energy block projections with dual_low_rank_port_hamiltonian + alpha noise.
# Only trains ~816k params (4 proj blocks), everything else frozen.
# Expected runtime: ~10-15 min on 1x H100.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_structured_proj_train.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_retrain \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 02:00 \
    -o "${HOME}/struct_proj_retrain_%J.stdout" \
    -e "${HOME}/struct_proj_retrain_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py \
    --steps 5000 \
    --batch_size 8 \
    --grad_accum 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --alpha_min 0.35 \
    --alpha_max 0.65 \
    --residual \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_residual_alpha"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
echo "wandb: https://wandb.ai/home -> project energy-gpt"
