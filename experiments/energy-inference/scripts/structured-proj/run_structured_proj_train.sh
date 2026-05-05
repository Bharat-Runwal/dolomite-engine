#!/bin/bash
# Settings: seq_len=4096, batch=2, grad_accum=8 (65536 tok/step), rank=256/128
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_structured_proj_train.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_retrain \
    -gpu "num=1" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o "${HOME}/bsub_logs/struct_proj_retrain_%J.stdout" \
    -e "${HOME}/bsub_logs/struct_proj_retrain_%J.stderr" \
    <<'BSUB_SCRIPT'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
REPO=/proj/dmfexp/nima/Code/dolomite-engine
export PYTHONPATH=$REPO:$PYTHONPATH
uv pip install accelerate datasets wandb tqdm lm-eval -q
cd $REPO

TMPLOG="/tmp/train_${LSB_JOBID}.log"
python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py \
    --seq_len 4096 \
    --batch_size 2 \
    --grad_accum 8 \
    --rank 256 \
    --dissipation_rank 128 \
    --lr 3e-4 \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 200 \
    --steps 5000 \
    --alpha_min 0.35 \
    --alpha_max 0.65 \
    --wandb_name "410m_struct_residual_alpha_20260413" \
    2>&1 | tee "$TMPLOG"

FINAL_CKPT=$(grep '^FINAL_CKPT=' "$TMPLOG" | tail -1 | cut -d= -f2-)
rm -f "$TMPLOG"
if [ -n "$FINAL_CKPT" ] && [ -d "$FINAL_CKPT" ]; then
    bash $REPO/experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
        "$FINAL_CKPT" "eval_struct_proj_retrain"
else
    echo "WARNING: FINAL_CKPT not found; run submit_eval.sh manually."
fi
BSUB_SCRIPT

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
