#!/bin/bash

# ============================================================
# Energy Loop Test Sweep
# ============================================================
# Usage: bash submit_loop_sweep.sh
#
# 4-block model with non-uniform iterations (shallow-deep-deep-shallow).
# Edit LAYER_ITERS and NUM_LAYERS below to adjust.

BASE_CONFIG="configs/energy/energy_test_bs.yml"
LOG_DIR="/proj/dmfexp/energy-gpt/logs/egpt_test"
GEN_DIR="/proj/dmfexp/energy-gpt/bs-configs-generated-egpt_test"
CKPT_BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/egpt_test"

mkdir -p "$LOG_DIR" "$GEN_DIR"

# ── Sweep definitions ──────────────────────────────────────
# Number of physical layers
NUM_LAYERS=4
# Layer iterations: shallow-deep-deep-shallow pattern
LAYER_ITERS="14, 4, 4, 14"

NAME="EGPT_4_blocks_ml_14_4_4_14"

for _RUN in 1; do
  PATCH="cfg['model_args']['pretrained_config']['layer_iterations'] = [${LAYER_ITERS}]"

  PATCHED_CONFIG="${GEN_DIR}/${NAME}_$$.yml"

  cp "$BASE_CONFIG" "$PATCHED_CONFIG"

  python3 -c "
import yaml
with open('${PATCHED_CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
${PATCH}
cfg['logging_args']['wandb_args']['name'] = '${NAME}'
cfg['save_args']['save_path'] = '${CKPT_BASE}/${NAME}'
cfg['load_args'] = {'load_path': '${CKPT_BASE}/${NAME}'}
with open('${PATCHED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

  echo "Submitting ${NAME} → ${PATCHED_CONFIG}"

  bsub \
    -q preemptable \
    -G grp_preemptable \
    -M 2000G \
    -hl \
    -n 1 \
    -J "bs-${NAME}" \
    -gpu "num=8/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/bs-${NAME}.out" \
    -eo "${LOG_DIR}/bs-${NAME}.err" \
    blaunch bash launch-scripts/pretrain.sh "$PATCHED_CONFIG"

  echo "  → Submitted ${NAME}"
done

# echo ""
# echo "All sweep jobs submitted. Monitor with:"
# echo "  bjobs -w | grep bs-all_"
# echo "  tail -f ${LOG_DIR}/bs-all_*.out"