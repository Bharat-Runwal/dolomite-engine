#!/bin/bash
# MAB only (no Langevin) — ablation to isolate MAB contribution.
# Usage: bash experiments/energy-inference/scripts/mab-langevin/run_mab_only.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_mab_only \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 02:00 \
    -o "${HOME}/bsub_logs/struct_mab_only_%J.stdout" \
    -e "${HOME}/bsub_logs/struct_mab_only_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/mab-langevin/train_mab_proj_20260413.py \
    --steps 5000 \
    --batch_size 8 \
    --grad_accum 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --n_arms 10 \
    --alpha_min 0.30 \
    --alpha_max 0.70 \
    --bandit_lr 0.1 \
    --no_langevin \
    --save_interval 2000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_mab_only"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
