#!/bin/bash
# Variant: No alpha noise (structured proj, warm-start, fixed alpha=0.5).
#
# Control for the alpha-noise experiments. Shows whether the structured
# parametrisation alone (without trajectory diversity training) improves
# over the unconstrained proj. Expected to match the alpha=0.51 NP baseline.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_no_noise_baseline.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_no_noise \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_no_noise_%J.stdout" \
    -e "${HOME}/struct_proj_no_noise_%J.stderr" \
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
    --no_alpha_noise \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_no_noise_baseline"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
