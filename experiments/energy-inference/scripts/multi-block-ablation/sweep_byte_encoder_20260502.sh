#!/bin/bash
# Sweep: byte encoder architecture for 145M RMSNorm-Rayleigh model
# Varies window_size, stride, d_local (input channels) at 1k steps each.
#
# Question: does a larger window (8 or 16 bytes) + smaller d_local (32) improve
# over the current w4s2/d_local=64 baseline for the 145M model?
#
# All 7 configs run SEQUENTIALLY in one job (~1.5h total).
# --no_compile avoids 15-min compile warmup dominating 1k-step runs.
#
# Metrics logged to wandb energy-inference-large with names:
#   sweep_145M_w{W}s{S}_dl{DL}
#
# Submit:
#   bash sweep_byte_encoder_20260502.sh
#
# Or directly:
#   bsub ... < sweep_byte_encoder_20260502.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/train_byte_egpt_20260501.py
RESULTS=${REPO}/experiments/energy-inference/results/multi-block-ablation/sweep_byte_encoder

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J byte_encoder_sweep \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 128G \
    -W 03:00 \
    -R "select[hname!='p5-r16-n2' && hname!='p1-r06-n1' && hname!='p6-r15-n2']" \
    -o "${HOME}/bsub_logs/byte_encoder_sweep_%J.stdout" \
    -e "${HOME}/bsub_logs/byte_encoder_sweep_%J.stderr" \
    <<'BSUB_EOF'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
export PYTHONUNBUFFERED=1
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

SCRIPT=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/train_byte_egpt_20260501.py
RESULTS=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/sweep_byte_encoder
mkdir -p "${RESULTS}"

# 145M architecture: n_pre=5, n_post=1, n_egpt_iter=10, D=1280, n_head=20
# --no_compile: skip compile warmup (would dominate 1k-step runs)
# eval_interval=500: get 2 validation points per run
# save_interval=99999: no checkpoint saves, just metrics

run_config() {
    local W=$1 S=$2 DL=$3
    local NAME="sweep_145M_w${W}s${S}_dl${DL}"
    local SAVE_DIR="${RESULTS}/${NAME}"

    echo ""
    echo "========================================================"
    echo "Config: w=${W} s=${S} d_local=${DL}  →  ${NAME}"
    echo "  compressed_len=$(( 1024 / S )), context/token=${W} bytes"
    echo "========================================================"

    torchrun --nproc_per_node=4 --standalone \
        --master_port $(shuf -i 20000-65000 -n 1) \
        "${SCRIPT}" \
        --variant rmsnorm_rayleigh \
        --d_model 1280 --d_local ${DL} --n_head 20 \
        --n_pre 5 --n_post 1 --n_egpt_iter 10 \
        --window_size ${W} --stride ${S} \
        --block_size 1024 \
        --batch_size 64 --grad_accum 1 \
        --lr 3e-3 --min_lr 3e-4 \
        --steps 1000 --warmup_steps 100 \
        --eval_interval 500 --log_interval 50 \
        --save_interval 99999 \
        --dataset megatron \
        --save_dir "${SAVE_DIR}" \
        --wandb_project energy-inference-large \
        --no_compile

    echo "Done: ${NAME}"
}

# ── 7 configs ────────────────────────────────────────────────────────────────
# w=4 (baseline family)
run_config 4 2 32     # smaller d_local, same w4s2
run_config 4 2 64     # current baseline

# w=8 family
run_config 8 4 32     # 4× compression, 8-byte context, small d_local
run_config 8 4 64     # 4× compression, 8-byte context, larger d_local
run_config 8 2 32     # 8-byte context, same seq length as baseline (512)

# w=16 family
run_config 16 8 32    # 8× compression, 16-byte context
run_config 16 4 32    # 4× compression, 16-byte context (overlap)

echo ""
echo "========================================================"
echo "All 7 configs complete."
echo "Results in: ${RESULTS}/"
echo "Compare on wandb: energy-inference-large project, filter 'sweep_145M'"
echo "========================================================"
BSUB_EOF

echo "Submitted byte encoder sweep (7 configs × 1k steps, sequential)"
echo "Monitor: bjobs | grep byte_encoder_sweep"
echo "Estimated total: ~1.5h"
