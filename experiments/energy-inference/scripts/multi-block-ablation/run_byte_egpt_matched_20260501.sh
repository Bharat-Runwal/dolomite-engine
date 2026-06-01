#!/bin/bash
# Data-matched byte-EGPT experiments: equal effective token budget as Vxx 30k-step runs.
#
# Vxx data budget: 4 GPUs × batch=4 × grad_accum=4 × seq=4096 × 30k = 7.86B BPE tokens
#                = 7.86B × 4 bytes/token ≈ 31.4B bytes
#
# This script: 4 GPUs × batch=64 × grad_accum=4 × seq=1024 × 30k = 31.5B bytes ✓
#
# With DataParallel (automatic from train script when n_gpu>1):
#   effective_batch = 4 × 64 = 256 per step × grad_accum=4 × seq=1024 = 1,048,576 bytes/step
#   30k steps = 31.5B bytes ≈ Vxx token-matched
#
# LR scaling: batch 64→256 (4×) → sqrt(4) × 2e-3 = 4e-3 (or keep 2e-3 as conservative)
# We use lr=3e-3 (sqrt scaling) as compromise.
#
# Usage:
#   bash run_byte_egpt_matched_20260501.sh [variant]
#   bash run_byte_egpt_matched_20260501.sh  # submits all matched variants

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/train_byte_egpt_20260501.py
EVAL_SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/eval_byte_egpt_20260501.py
RESULTS=${REPO}/experiments/energy-inference/results/multi-block-ablation
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,winogrande,wikitext,mmlu"

submit_matched() {
    local VARIANT=$1
    local D_MODEL=${2:-1280}
    local N_HEAD=${3:-20}
    local N_PRE=${4:-2}
    local N_POST=${5:-1}
    local N_ITER=${6:-10}
    local IDROP=${7:-0}
    local JOB_NAME=byte_egpt_${VARIANT}_matched

    local IDROP_SUFFIX=""
    [ "$IDROP" -gt 0 ] && IDROP_SUFFIX="_idrop${IDROP}"
    local SAVE_DIR=${RESULTS}/byte_${VARIANT}${IDROP_SUFFIX}_matched_w4s2_D${D_MODEL}_${N_ITER}iter

    bsub \
        -q preemptable \
        -G grp_preemptable \
        -J ${JOB_NAME} \
        -gpu "num=4/task:mode=exclusive_process" \
        -n 1 \
        -M 128G \
        -W 08:00 \
        -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
        -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
        <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
SAVE_DIR=${SAVE_DIR}
CKPT="\${SAVE_DIR}/ckpt_030000.pt"
[ -f "\${CKPT}" ] && exit 0

torchrun --nproc_per_node=4 --standalone "${SCRIPT}" \
    --variant ${VARIANT} \
    --d_model ${D_MODEL} --d_local 64 --n_head ${N_HEAD} \
    --n_pre ${N_PRE} --n_post ${N_POST} --n_egpt_iter ${N_ITER} \
    --iter_dropout_range ${IDROP} \
    --window_size 4 --stride 2 --block_size 1024 \
    --batch_size 256 \
    --grad_accum 1 \
    --lr 3e-3 --min_lr 3e-4 \
    --steps 30000 --warmup_steps 2000 \
    --eval_interval 1000 --log_interval 50 --save_interval 10000 \
    --dataset megatron \
    --save_dir "\${SAVE_DIR}" \
    --wandb_project energy-inference-large

# Eval inline
if [ -f "\${CKPT}" ]; then
    BEST="\${SAVE_DIR}/best.pt"
    [ -f "\${BEST}" ] && python "${EVAL_SCRIPT}" \
        --checkpoint "\${BEST}" \
        --output_path "\${SAVE_DIR}/harness_results.json" \
        --tasks "${TASKS}" --device cuda --batch_size 4 --iter_sweep
else
    RUNNING=\$(bjobs -J "${JOB_NAME}" 2>/dev/null | tail -n +2 | grep -cE " RUN | PEND " || echo 0)
    [ "\${RUNNING}" -eq 0 ] && bash "${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_byte_egpt_matched_20260501.sh"
fi
BSUB_SCRIPT
    echo "Submitted ${JOB_NAME} (4 GPUs, grad_accum=4, ~31.5B bytes)"
}

# Submit variants requested by user (or all if no args)
VARIANTS="${1:-all}"

if [ "$VARIANTS" = "all" ] || echo "$VARIANTS" | grep -q "layernorm"; then
    submit_matched layernorm          1280 20 2 1 10 0
fi
if [ "$VARIANTS" = "all" ] || echo "$VARIANTS" | grep -q "rmsnorm_rayleigh"; then
    submit_matched rmsnorm_rayleigh   1280 20 2 1 10 0
fi
if [ "$VARIANTS" = "all" ] || echo "$VARIANTS" | grep -q "rec_gpt"; then
    submit_matched rec_gpt            1280 20 2 1 10 0
fi
if [ "$VARIANTS" = "all" ] || echo "$VARIANTS" | grep -q "deep_gpt"; then
    submit_matched deep_gpt            768 12 12 0  0 0
fi
if [ "$VARIANTS" = "all" ] || echo "$VARIANTS" | grep -q "rec_gpt_idrop"; then
    submit_matched rec_gpt            1280 20 2 1 10 3   # with iter_dropout
fi

echo ""
echo "Data-matched byte experiments: 4×64×4×1024×30k = 31.5B bytes ≈ Vxx 7.86B token budget"
echo "Monitor: bjobs | grep byte_egpt.*matched"
