#!/bin/bash
# Unshard all recently trained models in batch.
# Usage: bash unshard_all.sh

set -e

UNSHARD_CFG="configs/unshard_v2.yml"
UNSHARD_SCRIPT="scripts/common/unshard.sh"
UNSHARD_DIR="/proj/checkpoints/bharat/personal/energy_gpt/model_unshard_FINAL_30k"
ITERATION=30000

declare -A MODELS
MODELS=(
  ["vk_residual_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/vk_residual_400M_lr2e3"
  ["projected_softmax_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/projected_softmax_400M_lr2e3"
  ["4b_projsoftmax_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/4b_projsoftmax_400M_lr2e3"
  # ["egrad_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/egrad_400M_lr2e3"  # only at step 20k, skipping
  ["edesc_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/edesc_400M_lr2e3"
  ["egrad_T1_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/egrad_T1_400M_lr2e3"
  ["mixedhead_T1_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/mixedhead_T1_400M_lr2e3"
  ["projsoftmax_energymlp_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/projsoftmax_energymlp_400M_lr2e3"
  ["vk_residual_energymlp_400M_lr2e3"]="/proj/checkpoints/bharat/personal/energy_gpt/new/vk_residual_energymlp_400M_lr2e3"
)

echo "============================================"
echo "Unsharding ${#MODELS[@]} models"
echo "============================================"
echo ""

SUMMARY=""

for name in $(echo "${!MODELS[@]}" | tr ' ' '\n' | sort); do
  load_path="${MODELS[$name]}"
  unshard_path="${UNSHARD_DIR}/${name}_unshard"

  echo "--------------------------------------------"
  echo "Model: $name"
  echo "  Load:    $load_path"
  echo "  Output:  $unshard_path"
  echo "--------------------------------------------"

  if [ ! -d "$load_path" ]; then
    echo "  SKIP: load_path does not exist"
    SUMMARY="${SUMMARY}${name}: SKIPPED (no checkpoint)\n"
    echo ""
    continue
  fi

  cat > "$UNSHARD_CFG" <<EOF
load_args:
  load_path: ${load_path}
  iteration: ${ITERATION}

mixed_precision_args:
  dtype: bf16

unsharded_path: ${unshard_path}
EOF

  OUTPUT=$(bash "$UNSHARD_SCRIPT" "$UNSHARD_CFG" 2>&1)
  echo "$OUTPUT" | tail -5

  PARAMS=$(echo "$OUTPUT" | grep -o "num parameters in the model = [0-9,]*" | head -1)
  if [ -z "$PARAMS" ]; then
    PARAMS="(param count not found)"
  fi

  SUMMARY="${SUMMARY}${name}: ${PARAMS} -> ${unshard_path}\n"
  echo ""
done

echo ""
echo "============================================"
echo "UNSHARD SUMMARY"
echo "============================================"
echo -e "$SUMMARY"
