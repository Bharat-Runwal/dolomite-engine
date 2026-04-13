#!/bin/bash
# Per-layer focus: train only the last energy block (h.11), freeze h.8-h.10.
# Tests whether h.11 as the decoding block is where alpha balance matters most.
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
    <<'BSUB_SCRIPT'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
REPO=/proj/dmfexp/nima/Code/dolomite-engine
export PYTHONPATH=$REPO:$PYTHONPATH
uv pip install accelerate datasets wandb tqdm lm-eval -q
cd $REPO

# Run training; tee output so we can extract the final checkpoint path
TMPLOG="/tmp/train_${LSB_JOBID}.log"
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
    --wandb_name "410m_struct_last_block_h11_20260413" \
    2>&1 | tee "$TMPLOG"

# Auto-submit harness eval on the saved final checkpoint
FINAL_CKPT=$(grep '^FINAL_CKPT=' "$TMPLOG" | tail -1 | cut -d= -f2-)
rm -f "$TMPLOG"
if [ -n "$FINAL_CKPT" ] && [ -d "$FINAL_CKPT" ]; then
    bash $REPO/experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
        "$FINAL_CKPT" "eval_struct_proj_last_block"
else
    echo "WARNING: FINAL_CKPT not found in training output; run submit_eval.sh manually."
fi
BSUB_SCRIPT

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
