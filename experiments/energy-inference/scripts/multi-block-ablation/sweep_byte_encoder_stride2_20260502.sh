#!/bin/bash
# Additional sweep: stride=2 only, varying window and d_local.
#
# Key insight: with stride=2 fixed, larger window is FREE in effective tokens —
# same 512 compressed tokens, same 1024 predictions per step, just more context
# per token. The only cost is encoder params: pool_proj = W × d_local × d_model.
#
# Parameter budget comparison (pool_proj only, d_model=1280):
#   w=4, s=2, dl=64: 4×64×1280 = 327K  ← current baseline
#   w=8, s=2, dl=32: 8×32×1280 = 327K  ← same params, 2× receptive field!
#   w=8, s=2, dl=64: 8×64×1280 = 655K  ← 2× params, 2× receptive field
#   w=16,s=2, dl=16: 16×16×1280= 327K  ← same params, 4× receptive field
#   w=16,s=2, dl=32: 16×32×1280= 655K  ← 2× params, 4× receptive field
#   w=16,s=2, dl=64: 16×64×1280=1310K  ← 4× params, 4× receptive field
#
# Complement to sweep_byte_encoder_20260502.sh which covers stride != 2 configs.
# Configs already in first sweep (skip here): w4s2dl32, w4s2dl64, w8s2dl32.
#
# Run sequentially in one job, ~1.2h for 5 configs × 1k steps.

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPT=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/train_byte_egpt_20260501.py
RESULTS=${REPO}/experiments/energy-inference/results/multi-block-ablation/sweep_byte_encoder

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J byte_enc_sweep_s2 \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 128G \
    -W 03:00 \
    -R "select[hname!='p5-r16-n2' && hname!='p1-r06-n1' && hname!='p6-r15-n2']" \
    -o "${HOME}/bsub_logs/byte_enc_sweep_s2_%J.stdout" \
    -e "${HOME}/bsub_logs/byte_enc_sweep_s2_%J.stderr" \
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

run_config() {
    local W=$1 S=$2 DL=$3
    local NAME="sweep_145M_w${W}s${S}_dl${DL}"
    local SAVE_DIR="${RESULTS}/${NAME}"
    local ENC_PARAMS=$(( W * DL * 1280 ))

    echo ""
    echo "========================================================"
    echo "Config: w=${W} s=${S} d_local=${DL}  →  ${NAME}"
    echo "  pool_proj params: ${W}×${DL}×1280 = ${ENC_PARAMS}"
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

# ── Stride=2 configs NOT in first sweep ──────────────────────────────────────
# (w4s2dl32, w4s2dl64, w8s2dl32 are already covered by sweep 90791)

# w=8, stride=2 — same sequence length as baseline, 2× receptive field
run_config 8 2 64    # same receptive field as above, larger encoder (2× baseline params)

# w=16, stride=2 — same sequence length as baseline, 4× receptive field
run_config 16 2 16   # iso-param to baseline (16×16=256 = 4×64), smallest d_local
run_config 16 2 32   # 2× baseline params, moderate d_local
run_config 16 2 64   # 4× baseline params, full d_local

# w=32, stride=2 — extreme: 32-byte receptive field (covers a short word)
run_config 32 2 16   # 32-byte window, very small d_local

echo ""
echo "========================================================"
echo "Stride=2 sweep complete. Key iso-param comparisons:"
echo "  w=4,s=2,dl=64 (baseline): 4×64×1280 = 327K pool params"
echo "  w=8,s=2,dl=32:             8×32×1280 = 327K pool params ← 2× receptive field, FREE"
echo "  w=16,s=2,dl=16:           16×16×1280 = 327K pool params ← 4× receptive field, FREE"
echo "  w=32,s=2,dl=16:           32×16×1280 = 655K pool params ← 8× receptive field"
echo ""
echo "Check wandb energy-inference-large, filter 'sweep_145M'"
echo "========================================================"
BSUB_EOF

echo "Submitted stride=2 sweep (5 configs × 1k steps, sequential)"
echo "Monitor: bjobs | grep byte_enc_sweep_s2"