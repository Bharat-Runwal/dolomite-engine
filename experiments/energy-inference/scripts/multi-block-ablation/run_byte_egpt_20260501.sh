#!/bin/bash
# Submit byte-EGPT experiments (w4s2 tokenizer, WikiText-103, energy-inference project).
#
# Two variants:
#   layernorm:        EGPT + LayerNorm, standard update rule
#   rmsnorm_reileigh: EGPT + RMSNorm + Reileigh tangent projection
#
# Both: D=1280, 20 heads, 2 pre-GPT + 1 EGPT×10 + 1 post-GPT, batch=32, 60k steps, 1 GPU
# Total params ~84M: embedding 0.35M (0.4%) + transformer 83M (99.6%)
# Data: WikiText-103 raw text (byte-level, ~530M bytes), 2× steps vs BPE baseline
#
# Logs to wandb project energy-inference (not energy-inference-large).
#
# Usage:
#   bash experiments/energy-inference/scripts/multi-block-ablation/run_byte_egpt_20260501.sh
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/train_byte_egpt_20260501.py
EVAL_SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/eval_byte_egpt_20260501.py
RESULTS=${REPO}/experiments/energy-inference/results/multi-block-ablation
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,winogrande,wikitext,mmlu"

submit_byte_egpt() {
    local VARIANT=$1
    local JOB_NAME=byte_egpt_${VARIANT}

    bsub \
        -q preemptable \
        -G grp_preemptable \
        -J ${JOB_NAME} \
        -gpu "num=1/task:mode=exclusive_process" \
        -n 1 \
        -M 64G \
        -W 08:00 \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
uv pip install datasets -q 2>/dev/null || pip install datasets -q

SAVE_DIR=${RESULTS}/byte_egpt_w4s2_${VARIANT}_D1280_10iter
CKPT="\${SAVE_DIR}/ckpt_030000.pt"

# If final checkpoint exists, skip (already done)
if [ -f "\${CKPT}" ]; then
    echo "Training already complete. Skipping."
    exit 0
fi

# Find latest checkpoint for resume
LATEST_STEP=0
if ls "\${SAVE_DIR}"/ckpt_*.pt 2>/dev/null | tail -1 | grep -q ckpt; then
    LATEST_STEP=\$(ls "\${SAVE_DIR}"/ckpt_*.pt 2>/dev/null | tail -1 | grep -oP '\d+(?=\.pt)' || echo 0)
fi

# Simple resume: pass --resume flag (model loads from latest ckpt in save_dir)
python "${SCRIPT}" \
    --variant ${VARIANT} \
    --d_model 1280 \
    --d_local 64 \
    --n_head 20 \
    --n_pre 2 \
    --n_post 1 \
    --n_egpt_iter 10 \
    --window_size 4 \
    --stride 2 \
    --block_size 1024 \
    --batch_size 64 \
    --grad_accum 1 \
    --lr 2e-3 \
    --min_lr 2e-4 \
    --steps 30000 \
    --warmup_steps 2000 \
    --eval_interval 1000 \
    --log_interval 50 \
    --save_interval 10000 \
    --dataset megatron \
    --save_dir "\${SAVE_DIR}" \
    --wandb_project energy-inference-large

# Auto-resubmit if not complete; run eval if complete
if [ ! -f "\${CKPT}" ]; then
    RUNNING=\$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -cE " RUN | PEND " || echo 0)
    if [ "\${RUNNING}" -eq 0 ]; then
        echo "Not complete. Resubmitting..."
        bash "${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_byte_egpt_20260501.sh"
    fi
else
    # Training complete — run lm-eval harness inline before job closes
    BEST_CKPT="\${SAVE_DIR}/best.pt"
    EVAL_OUT="\${SAVE_DIR}/harness_results.json"
    if [ -f "\${BEST_CKPT}" ] && [ ! -f "\${EVAL_OUT}" ]; then
        echo "Training done. Running lm-eval harness on best checkpoint..."
        uv pip install lm-eval -q 2>/dev/null || pip install lm-eval -q
        python "${EVAL_SCRIPT}" \
            --checkpoint "\${BEST_CKPT}" \
            --output_path "\${EVAL_OUT}" \
            --tasks "${TASKS}" \
            --device cuda \
            --batch_size 4
        echo "Eval complete: \${EVAL_OUT}"
    else
        [ -f "\${EVAL_OUT}" ] && echo "Eval already exists: \${EVAL_OUT}" || echo "No best.pt found."
    fi
fi
BSUB_SCRIPT
    echo "Submitted ${JOB_NAME}"
}

submit_byte_egpt_custom() {
    # Like submit_byte_egpt but with custom d_model/n_head/n_pre/n_post/n_egpt_iter
    local VARIANT=$1
    local D_MODEL=$2
    local N_HEAD=$3
    local N_PRE=$4
    local N_POST=$5
    local N_ITER=$6
    local JOB_NAME=byte_egpt_${VARIANT}

    bsub \
        -q preemptable \
        -G grp_preemptable \
        -J ${JOB_NAME} \
        -gpu "num=1/task:mode=exclusive_process" \
        -n 1 \
        -M 64G \
        -W 08:00 \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1

SAVE_DIR=${RESULTS}/byte_${VARIANT}_w4s2_D${D_MODEL}_${N_ITER}iter
CKPT="\${SAVE_DIR}/ckpt_030000.pt"

if [ -f "\${CKPT}" ]; then
    echo "Training already complete."
    BEST_CKPT="\${SAVE_DIR}/best.pt"
    EVAL_OUT="\${SAVE_DIR}/harness_results.json"
    if [ -f "\${BEST_CKPT}" ] && [ ! -f "\${EVAL_OUT}" ]; then
        uv pip install lm-eval -q 2>/dev/null || true
        python "${EVAL_SCRIPT}" --checkpoint "\${BEST_CKPT}" --output_path "\${EVAL_OUT}" --tasks "${TASKS}" --device cuda --batch_size 4
    fi
    exit 0
fi

python "${SCRIPT}" \
    --variant ${VARIANT} \
    --d_model ${D_MODEL} \
    --d_local 64 \
    --n_head ${N_HEAD} \
    --n_pre ${N_PRE} \
    --n_post ${N_POST} \
    --n_egpt_iter ${N_ITER} \
    --window_size 4 \
    --stride 2 \
    --block_size 1024 \
    --batch_size 64 \
    --grad_accum 1 \
    --lr 2e-3 \
    --min_lr 2e-4 \
    --steps 30000 \
    --warmup_steps 2000 \
    --eval_interval 1000 \
    --log_interval 50 \
    --save_interval 10000 \
    --dataset megatron \
    --save_dir "\${SAVE_DIR}" \
    --wandb_project energy-inference-large

if [ ! -f "\${CKPT}" ]; then
    RUNNING=\$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -cE " RUN | PEND " || echo 0)
    [ "\${RUNNING}" -eq 0 ] && bash "${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_byte_egpt_20260501.sh"
else
    BEST_CKPT="\${SAVE_DIR}/best.pt"
    EVAL_OUT="\${SAVE_DIR}/harness_results.json"
    if [ -f "\${BEST_CKPT}" ] && [ ! -f "\${EVAL_OUT}" ]; then
        echo "Running lm-eval harness..."
        uv pip install lm-eval -q 2>/dev/null || true
        python "${EVAL_SCRIPT}" --checkpoint "\${BEST_CKPT}" --output_path "\${EVAL_OUT}" --tasks "${TASKS}" --device cuda --batch_size 4
    fi
fi
BSUB_SCRIPT
    echo "Submitted ${JOB_NAME}"
}

mkdir -p "${HOME}/bsub_logs"

# ── Shared-center EGPT variants (D=1280, 2+1×10+1 sandwich, 13 passes, ~87M) ─
submit_byte_egpt layernorm
submit_byte_egpt rmsnorm_rayleigh

# ── Recurrent GPT (same arch as EGPT, GPT update rule instead) ───────────────
submit_byte_egpt_custom rec_gpt           1280 20  2 1 10

# ── Deep GPT (12 separate blocks, D=768, ~87M) ───────────────────────────────
submit_byte_egpt_custom deep_gpt           768 12 12 0  0

# ── Deep EGPT + Rayleigh (12 separate EGPT blocks, D=768, ~87M) ──────────────
# Ablates weight-sharing: same energy+Rayleigh update but independent block weights
submit_byte_egpt_custom deep_egpt_rayleigh 768 12  2 1 10

echo ""
echo "Submitted 4 byte-level experiments to energy-inference-large:"
echo "  byte_layernorm:        EGPT + LayerNorm        (D=1280, ~87M)"
echo "  byte_rmsnorm_reileigh: EGPT + RMSNorm+Reileigh (D=1280, ~87M)"
echo "  byte_rec_gpt:          Recurrent GPT           (D=1280, ~83M, same passes)"
echo "  byte_deep_gpt:         Deep GPT 12L            (D=768,  ~87M, no sharing)"
echo ""
echo "Monitor: bjobs | grep byte_egpt"
