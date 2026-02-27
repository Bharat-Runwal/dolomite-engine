#!/bin/bash

# ============================================================
# MoE_Energy_F5 Hyperparameter Sweep
# ============================================================
# Usage: bash submit_f5_sweep.sh
#
# Submits 8 jobs sweeping the most impactful HPs for MoE_Energy_F5.
# Each job patches the base config with one HP change.
#
# Switch variant by changing VARIANT below:
#   E2: intermediate_size (4096, 1024, 4096), num_experts=8
#   E6: intermediate_size (4096, 2048, 4096), num_experts=32
#   E9: intermediate_size (512,  1024, 512),  num_experts=32

BASE_CONFIG="configs/workshop_nfam/f5_energy_32x1024_slim_bs.yml"
LOG_DIR="/proj/dmfexp/energy-gpt/logs"
GEN_DIR="/proj/dmfexp/energy-gpt/bs-configs-generated"

# ── Pick MLP type: MoE_Energy_F5 (FF2W) or MoE_Energy_F6 (FF1W) ──
MLP_TYPE="MoE_Energy_F5"

# ── Pick variant: E2, E6, or E9 ───────────────────────────
VARIANT="E6"

# Derive short label from MLP_TYPE (e.g. MoE_Energy_F5 -> F5, MoE_Energy_F6 -> F6)
MLP_LABEL="${MLP_TYPE##*_}"

if [ "$VARIANT" = "E2" ]; then
  CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/workshop_nfam/E2_${MLP_LABEL}_Energy"
  TAG="8x1024_6iter"
  ARCH_PATCH="mlps = cfg['model_args']['pretrained_config']['mlp_blocks']; mlps[0]['intermediate_size'] = 4096; mlps[1]['intermediate_size'] = 1024; mlps[1]['num_experts'] = 8; mlps[1]['mlp_type'] = '${MLP_TYPE}'; mlps[2]['intermediate_size'] = 4096"
elif [ "$VARIANT" = "E6" ]; then
  CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/workshop_nfam/E6_${MLP_LABEL}_Energy"
  TAG="32x2048_6iter"
  ARCH_PATCH="mlps = cfg['model_args']['pretrained_config']['mlp_blocks']; mlps[0]['intermediate_size'] = 4096; mlps[1]['intermediate_size'] = 2048; mlps[1]['num_experts'] = 32; mlps[1]['mlp_type'] = '${MLP_TYPE}'; mlps[2]['intermediate_size'] = 4096"
elif [ "$VARIANT" = "E9" ]; then
  CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/workshop_nfam/E9_${MLP_LABEL}_Energy"
  TAG="32x1024_6iter_slim"
  ARCH_PATCH="mlps = cfg['model_args']['pretrained_config']['mlp_blocks']; mlps[1]['mlp_type'] = '${MLP_TYPE}'"
else
  echo "ERROR: VARIANT must be E2, E6, or E9"; exit 1
fi

mkdir -p "$LOG_DIR" "$GEN_DIR"

# ── Sweep definitions ──────────────────────────────────────
# Format: "name|python_patch_code"
# The python patch code modifies the 'cfg' dict in-place.
ML="${MLP_LABEL,,}"  # lowercase: f5 or f6
SWEEPS=(
  # "${ML}_baseline|pass"
  # "${ML}_temp_low|cfg['model_args']['pretrained_config']['mlp_blocks'][1]['boltzmann_temperature'] = 0.5"
  # "${ML}_temp_high|cfg['model_args']['pretrained_config']['mlp_blocks'][1]['boltzmann_temperature'] = 2.0"
  "${ML}_distill_high|cfg['model_args']['pretrained_config']['mlp_blocks'][1]['distillation_weight'] = 0.1"
  "${ML}_distill_low|cfg['model_args']['pretrained_config']['mlp_blocks'][1]['distillation_weight'] = 0.001"
  # "${ML}_topk4|cfg['model_args']['pretrained_config']['mlp_blocks'][1]['num_experts_per_tok'] = 4"
  "${ML}_lr_high|cfg['optimizer_args']['class_args']['lr'] = 6e-4"
  # "${ML}_aux_high|cfg['model_args']['pretrained_config']['router_aux_loss_coef'] = 0.01"
)

for entry in "${SWEEPS[@]}"; do
  IFS='|' read -r NAME PATCH <<< "$entry"

  PATCHED_CONFIG="${GEN_DIR}/${VARIANT}_${NAME}_$$.yml"

  cp "$BASE_CONFIG" "$PATCHED_CONFIG"

  python3 -c "
import yaml
with open('${PATCHED_CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
${ARCH_PATCH}
${PATCH}
cfg['logging_args']['wandb_args']['name'] = '${VARIANT}_${NAME}_${TAG}'
cfg['save_args']['save_path'] = '${CKPT_BASE}/${VARIANT}_${NAME}_${TAG}'
with open('${PATCHED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

  echo "Submitting ${VARIANT}_${NAME} → ${PATCHED_CONFIG}"

  bsub \
    -q preemptable \
    -G grp_preemptable \
    -M 2000G \
    -hl \
    -n 1 \
    -J "bs-${VARIANT}-${NAME}" \
    -gpu "num=4/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/bs-${VARIANT}-${NAME}.out" \
    -eo "${LOG_DIR}/bs-${VARIANT}-${NAME}.err" \
    blaunch bash launch-scripts/pretrain.sh "$PATCHED_CONFIG"

  echo "  → Submitted ${VARIANT}_${NAME}"
done

# echo ""
# echo "All ${VARIANT} F5 sweep jobs submitted. Monitor with:"
# echo "  bjobs -w | grep bs-${VARIANT}-f5"
# echo "  tail -f ${LOG_DIR}/bs-${VARIANT}-f5-*.out"

# -q normal \