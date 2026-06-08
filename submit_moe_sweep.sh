#!/bin/bash

# ============================================================
# egpt_s8e4 + Boltzmann MoE (aux-loss-stabilized): 2 configs, single-stage
#
# Variants (Boltzmann MoE in last 4 layers):
#   1) dense MLP   (layers 0-7) + Boltzmann MoE (layers 8-11)
#   2) standard MoE(layers 0-7) + Boltzmann MoE (layers 8-11)
#
# Per-Boltzmann-block:
#   num_experts=8, top-k=2, intermediate=1024
#   boltzmann_temperature=4.0     → β_r init = 0.25 (soft)
#   learnable_temperature=True    → β_r adapts on its own
#   aux_loss_stable=True          → Fix B + Fix C:
#     • detach expert_energies in router_logits_for_aux
#     • z_loss coefficient 0.1 → 0.0
#   Both eliminate the squared-logsumexp gradient-explosion vector that
#   drove early-training spikes.
#
# LR=8e-4, 30k steps, single training run per variant. Two-stage warmup
# is no longer needed — the spikes were aux-loss driven, not routing-saturation.
#
# Run name suffix `_auxstable` differentiates from previous _2sw / non-suffix
# runs.
# ============================================================

BASE_CONFIG="configs/energy/moe_test_bs.yml"
LOG_DIR="/proj/dmfexp/energy-gpt/logs/moe_s8e4_compare"
GEN_DIR="/proj/dmfexp/energy-gpt/bs-configs-generated-moe_s8e4_compare"
CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/moe_s8e4_compare"

mkdir -p "$LOG_DIR" "$GEN_DIR"

NSTEPS=30000
STEP_K=$((NSTEPS / 1000))
DECAY_STEPS=$((NSTEPS - 2000))
LR="8e-4"
NUM_EXPERTS=8
TOPK=2
MOE_INTERMEDIATE=1024
DENSE_INTERMEDIATE=2048
SAVE_INTERVAL=2000

# Scale-up patch: H=768 -> 1600, heads=12 -> 25
SCALE_PATCH="
pc['hidden_size'] = 1600
pc['rope_dim'] = 64
for sm in pc['sequence_mixer_blocks']:
    sm['num_attention_heads'] = 25
    sm['num_key_value_heads'] = 25
    sm['attention_multiplier'] = 1.0 / (64 ** 0.5)
"

# Args:
#   $1 NAME
#   $2 FIRST8_MODE   (dense | stdmoe)
#   $3 LAST4_MODE    (boltz)
submit_job() {
  local NAME="$1"
  local FIRST8_MODE="$2"
  local LAST4_MODE="$3"

  PATCHED_CONFIG="${GEN_DIR}/${NAME}_$$.yml"
  cp "$BASE_CONFIG" "$PATCHED_CONFIG"

  python3 -c "
import yaml
with open('${PATCHED_CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
pc = cfg['model_args']['pretrained_config']
cfg['training_parameters']['micro_batch_size'] = 1
cfg['training_parameters']['gradient_accumulation_steps'] = 8
${SCALE_PATCH}

dense_block = {
    'mlp_type': 'MLP',
    'activation_function': 'swiglu',
    'intermediate_size': ${DENSE_INTERMEDIATE},
    'add_bias': False,
}
stdmoe_block = {
    'mlp_type': 'MoE',
    'activation_function': 'swiglu',
    'intermediate_size': ${MOE_INTERMEDIATE},
    'add_bias': False,
    'num_experts': ${NUM_EXPERTS},
    'num_experts_per_tok': ${TOPK},
    'normalized_topk': True,
    'shared_expert_gating': False,
    'use_interleaved_weights': False,
}
boltz_block = {
    'mlp_type': 'MoE_Energy_F5',
    'intermediate_size': ${MOE_INTERMEDIATE},
    'add_bias': False,
    'dropout': 0.1,
    'num_experts': ${NUM_EXPERTS},
    'num_experts_per_tok': ${TOPK},
    'normalized_topk': True,
    'boltzmann_temperature': 4.0,        # soft init: β_r = 0.25
    'learnable_temperature': True,        # adapts during training
    'distillation_weight': 0.01,
    'expert_repulsion_lambda': 0.01,
    'aux_loss_stable': True,              # Fix B + Fix C
}
first8_mode = '${FIRST8_MODE}'
last4_mode  = '${LAST4_MODE}'
def pick(mode):
    if mode == 'dense':  return dict(dense_block)
    if mode == 'stdmoe': return dict(stdmoe_block)
    if mode == 'boltz':  return dict(boltz_block)
    raise ValueError(f'unknown mode: {mode}')
first8 = pick(first8_mode)
last4  = pick(last4_mode)
for i in range(8):
    pc['mlp_blocks'][i] = dict(first8)
for i in range(8, 12):
    pc['mlp_blocks'][i] = dict(last4)

cfg['optimizer_args']['class_args']['weight_decay'] = 0.1
cfg['optimizer_args']['class_args']['lr'] = float('${LR}')
cfg['training_parameters']['num_training_steps'] = ${NSTEPS}
cfg['lr_scheduler_args']['num_decay_steps'] = ${DECAY_STEPS}
cfg['save_args']['save_path'] = '${CKPT_BASE}/${NAME}'
cfg['save_args']['save_interval'] = ${SAVE_INTERVAL}
cfg['logging_args']['wandb_args']['name'] = '${NAME}'
cfg['load_args'] = {'load_path': '${CKPT_BASE}/${NAME}'}  # auto-resume self if preempted

with open('${PATCHED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

  echo "Submitting ${NAME} → ${PATCHED_CONFIG}"

  bsub \
    -r \
    -q preemptable \
    -G grp_preemptable \
    -M 2000G \
    -hl \
    -n 4 \
    -J "bs-${NAME}" \
    -gpu "num=8/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/bs-${NAME}.out" \
    -eo "${LOG_DIR}/bs-${NAME}.err" \
    blaunch bash launch-scripts/pretrain.sh "$PATCHED_CONFIG"

  echo "  → Submitted ${NAME}"
  COUNT=$((COUNT + 1))
}

COUNT=0
SUFFIX="_auxstable"
TAG="egpt_s8e4_E${NUM_EXPERTS}k${TOPK}_lr${LR}_${STEP_K}k${SUFFIX}"

# Variant 1: dense MLP (0-7) + Boltzmann MoE (8-11)
submit_job "${TAG}_dense_boltz"  "dense"  "boltz"

# Variant 2: standard MoE (0-7) + Boltzmann MoE (8-11)
submit_job "${TAG}_stdmoe_boltz" "stdmoe" "boltz"

echo ""
echo "=========================================="
echo "Aux-stable Boltzmann runs submitted (${COUNT} jobs)."
echo "=========================================="
echo "Monitor with:"
echo "  bjobs -w | grep ${TAG}"
echo "  tail -f ${LOG_DIR}/bs-${TAG}*.out"
