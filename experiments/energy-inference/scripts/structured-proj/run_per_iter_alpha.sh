#!/bin/bash
# Per-iteration alpha experiment: structured proj on h.10 only (penultimate block, 6 iters).
#
# Usage: bash run_per_iter_alpha.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME="per_iter_alpha_h10"

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J "${JOB_NAME}" \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 \
    -M 48G \
    -W 06:00 \
    -o "${HOME}/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/${JOB_NAME}_%J.stderr" \
    <<'EOF'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd /proj/dmfexp/nima/Code/dolomite-engine
python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260413.py \
    --target_block 10 \
    --num_iters 6 \
    --rank 256 \
    --dissipation_rank 128 \
    --lr 3e-4 \
    --steps 10000 \
    --batch_size 2 \
    --grad_accum 8 \
    --norm_log_interval 100 \
    --wandb_name "410m_per_iter_alpha_h10"
EOF

echo "Submitted: ${JOB_NAME}"
