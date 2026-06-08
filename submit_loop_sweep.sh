#!/bin/bash

# ============================================================
# 400M-class LR sweep: 9 jobs, 8 nodes x 8 GPUs = 64 GPUs each.
#
# Architecture: hidden=1280, 20 heads (head_dim=64), int_sm=4352,
#   12 layers. No looping. Egrad-wide-extra uses int_sm=4768
#   so total params match baseline (~407.2M).
#
# Runs (3 baseline + 3 egrad normal + 3 egrad wide-extra, LRs: 1e-3, 3e-3, 4e-3):
#  1. gpt_baseline_12b_lr1e3_30k_gl4bl4_400m           ~407.7M
#  2. gpt_baseline_12b_lr3e3_30k_gl4bl4_400m           ~407.7M
#  3. gpt_baseline_12b_lr4e3_30k_gl4bl4_400m           ~407.7M
#  4. egrad_12b_lr1e3_30k_gl4bl4_400m                  ~388.0M
#  5. egrad_12b_lr3e3_30k_gl4bl4_400m                  ~388.0M
#  6. egrad_12b_lr4e3_30k_gl4bl4_400m                  ~388.0M
#  7. egrad_12b_lr1e3_30k_gl4bl4_400m_wide       ~407.2M (int_sm=4768)
#  8. egrad_12b_lr3e3_30k_gl4bl4_400m_wide       ~407.2M
#  9. egrad_12b_lr4e3_30k_gl4bl4_400m_wide       ~407.2M
# ============================================================
# Usage: bash submit_loop_sweep.sh

BASE_CONFIG="configs/energy/energy_test_bs_12b_160.yml"
LOG_DIR="/proj/dmfexp/energy-gpt/logs/grouploop_400m"
GEN_DIR="/proj/dmfexp/energy-gpt/bs-configs-generated-grouploop_400m"
CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/grouploop_400m"

mkdir -p "$LOG_DIR" "$GEN_DIR"

NSTEPS=30000
STEP_K=$((NSTEPS / 1000))
DECAY_STEPS=$((NSTEPS - 2000))

HIDDEN=1280
HEADS=20
HEAD_DIM=$((HIDDEN / HEADS))   # 64
D_INT=4352
D_INT_EGRAD_WIDE=4768   # widens MLP to compensate for missing V+c_proj rows in egrad -> matches baseline ~407.2M

# Attention multiplier = 1/sqrt(head_dim) = 1/8 = 0.125 for head_dim=64
ATTN_MULT="0.125"

NNODES=8
GPUS_PER_NODE=8

submit_job() {
  local NAME="$1"
  local LR="$2"
  local PATCH_CODE="$3"

  PATCHED_CONFIG="${GEN_DIR}/${NAME}_$$.yml"
  cp "$BASE_CONFIG" "$PATCHED_CONFIG"

  python3 -c "
import yaml
with open('${PATCHED_CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
pc = cfg['model_args']['pretrained_config']
pc['hidden_size'] = ${HIDDEN}
pc['rope_dim'] = ${HEAD_DIM}
cfg['training_parameters']['micro_batch_size'] = 2
${PATCH_CODE}
cfg['optimizer_args']['class_args']['lr'] = float('${LR}')
cfg['training_parameters']['num_training_steps'] = ${NSTEPS}
cfg['lr_scheduler_args']['num_decay_steps'] = ${DECAY_STEPS}
cfg['logging_args']['wandb_args']['name'] = '${NAME}'
cfg['save_args']['save_path'] = '${CKPT_BASE}/${NAME}'
cfg['load_args'] = {'load_path': '${CKPT_BASE}/${NAME}'}
with open('${PATCHED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

  echo "Submitting ${NAME} (lr=${LR}, 8 nodes x 8 GPUs = 64 GPUs) -> ${PATCHED_CONFIG}"

  bsub \
    -r \
    -q preemptable \
    -G grp_preemptable \
    -M 2000G \
    -hl \
    -n ${NNODES} \
    -J "bs-${NAME}" \
    -gpu "num=${GPUS_PER_NODE}/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/bs-${NAME}.out" \
    -eo "${LOG_DIR}/bs-${NAME}.err" \
    blaunch bash launch-scripts/pretrain.sh "$PATCHED_CONFIG"

  echo "  -> Submitted ${NAME}"
  COUNT=$((COUNT + 1))
}

COUNT=0

# Helper mixer/MLP dicts (20 heads, head_dim=64 @ hidden=1280)
SOFT_SM="{'sequence_mixer_type': 'softmax_attention', 'num_attention_heads': ${HEADS}, 'num_key_value_heads': ${HEADS}, 'add_bias': False, 'attention_multiplier': ${ATTN_MULT}}"
EGRAD_SM="{'sequence_mixer_type': 'egrad_attention', 'num_attention_heads': ${HEADS}, 'num_key_value_heads': ${HEADS}, 'num_energy_heads': $((HEADS / 2)), 'energy_head_placement': 'interleaved', 'add_bias': False, 'attention_multiplier': ${ATTN_MULT}}"
MLP_SWIGLU="{'mlp_type': 'MLP', 'activation_function': 'swiglu', 'intermediate_size': ${D_INT}, 'add_bias': False}"
MLP_SWIGLU_WIDE="{'mlp_type': 'MLP', 'activation_function': 'swiglu', 'intermediate_size': ${D_INT_EGRAD_WIDE}, 'add_bias': False}"

BASELINE_PATCH="
soft_sm = ${SOFT_SM}
mlp_s = ${MLP_SWIGLU}
pc['num_layers'] = 12
pc['sequence_mixer_blocks'] = [dict(soft_sm) for _ in range(12)]
pc['mlp_blocks'] = [dict(mlp_s) for _ in range(12)]
pc.pop('layer_loop_groups', None)
pc['layer_iterations'] = [1]*12
pc['iter_dropout_range'] = 0
pc['proj_mode'] = 'riemannian_split'
"

EGRAD_PATCH="
egrad_sm = ${EGRAD_SM}
mlp_s = ${MLP_SWIGLU}
pc['num_layers'] = 12
pc['sequence_mixer_blocks'] = [dict(egrad_sm) for _ in range(12)]
pc['mlp_blocks'] = [dict(mlp_s) for _ in range(12)]
pc.pop('layer_loop_groups', None)
pc['layer_iterations'] = [1]*12
pc['iter_dropout_range'] = 0
pc['proj_mode'] = 'riemannian_split'
"

EGRAD_WIDE_PATCH="
egrad_sm = ${EGRAD_SM}
mlp_s = ${MLP_SWIGLU_WIDE}
pc['num_layers'] = 12
pc['sequence_mixer_blocks'] = [dict(egrad_sm) for _ in range(12)]
pc['mlp_blocks'] = [dict(mlp_s) for _ in range(12)]
pc.pop('layer_loop_groups', None)
pc['layer_iterations'] = [1]*12
pc['iter_dropout_range'] = 0
pc['proj_mode'] = 'riemannian_split'
"

# ==================================================================
# 1-3. gpt_baseline_12b_400m at lr 1e-3, 3e-3, 4e-3
# ==================================================================
submit_job "gpt_baseline_12b_lr1e3_${STEP_K}k_gl4bl4_400m" "1e-3" "${BASELINE_PATCH}"
submit_job "gpt_baseline_12b_lr3e3_${STEP_K}k_gl4bl4_400m" "3e-3" "${BASELINE_PATCH}"
submit_job "gpt_baseline_12b_lr4e3_${STEP_K}k_gl4bl4_400m" "4e-3" "${BASELINE_PATCH}"

# ==================================================================
# 4-6. egrad_12b_400m (10 energy + 10 softmax heads, interleaved) at lr 1e-3, 3e-3, 4e-3
# ==================================================================
submit_job "egrad_12b_lr1e3_${STEP_K}k_gl4bl4_400m" "1e-3" "${EGRAD_PATCH}"
submit_job "egrad_12b_lr3e3_${STEP_K}k_gl4bl4_400m" "3e-3" "${EGRAD_PATCH}"
submit_job "egrad_12b_lr4e3_${STEP_K}k_gl4bl4_400m" "4e-3" "${EGRAD_PATCH}"

# ==================================================================
# 7-9. egrad_12b_400m_wide (MLP int=4768, matches baseline ~407.2M) at lr 1e-3, 3e-3, 4e-3
# ==================================================================
submit_job "egrad_12b_lr1e3_${STEP_K}k_gl4bl4_400m_wide" "1e-3" "${EGRAD_WIDE_PATCH}"
submit_job "egrad_12b_lr3e3_${STEP_K}k_gl4bl4_400m_wide" "3e-3" "${EGRAD_WIDE_PATCH}"
submit_job "egrad_12b_lr4e3_${STEP_K}k_gl4bl4_400m_wide" "4e-3" "${EGRAD_WIDE_PATCH}"

echo ""
echo "All ${COUNT} 400M-class jobs submitted (8 nodes x 8 GPUs = 64 GPUs each)."
echo "Monitor with:"
echo "  bjobs -w | grep bs-"
echo "  tail -f ${LOG_DIR}/bs-*.out"
