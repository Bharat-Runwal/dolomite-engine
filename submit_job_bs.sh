#!/bin/bash

# Energy-GPT Training Job Submission
# Usage: bash submit_train.sh [config_file]. 

# CONFIG=${1:-"configs/energy/energy_bs.yml"}
# CONFIG=${1:-"configs/energy/energy_loop_rec_egpt_nemo_bs.yml"}
# CONFIG=${1:-"configs/workshop_nfam/e9f_gaussian_boltz_32x1024_slim_bs.yml"}

# LOG_DIR="/proj/dmfexp/energy-gpt/logs"

# mkdir -p "$LOG_DIR"

# bsub \
#   -q preemptable \
#   -G grp_preemptable \
#   -M 2000G \
#   -hl \
#   -n 4 \
#   -J bs-energy-nemo-dl \
#   -gpu "num=8/task:mode=exclusive_process" \
#   -oo "${LOG_DIR}/bs-energy-nemo-gmm-lrd.out" \
#   -eo "${LOG_DIR}/bs-energy-nemo-gmm-lrd.err" \
#   blaunch bash launch-scripts/pretrain.sh "$CONFIG"

CONFIG=${1:-"configs/workshop_nfam/e9f_gaussian_boltz_32x1024_slim_bs.yml"}

LR=3e-4  # ← change this directly

LOG_DIR="/proj/dmfexp/energy-gpt/logs"
mkdir -p "$LOG_DIR"

PATCHED_CONFIG="/proj/dmfexp/energy-gpt/bs-configs-generated/config_lr${LR}_$$.yml"
mkdir -p /proj/dmfexp/energy-gpt/bs-configs-generated

cp "$CONFIG" "$PATCHED_CONFIG"
python3 -c "
import yaml
with open('${PATCHED_CONFIG}', 'r') as f:
    cfg = yaml.safe_load(f)
cfg['optimizer_args']['class_args']['lr'] = float('${LR}')
wandb_name = cfg['logging_args']['wandb_args']['name'] + '_lr${LR}'
cfg['logging_args']['wandb_args']['name'] = wandb_name
cfg['save_args']['save_path'] = '/proj/dmfexp/energy-gpt/checkpoints-bsaha/workshop_nfam/' + wandb_name
with open('${PATCHED_CONFIG}', 'w') as f:
    yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
"

bsub \
  -q preemptable \
  -G grp_preemptable \
  -M 2000G \
  -hl \
  -n 4 \
   -J "bs-energy-nemo-lrd-${LR}" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/bs-energy-nemo-gmm-pr2-${LR}-%J.out" \
  -eo "${LOG_DIR}/bs-energy-nemo-gmm-pr2-${LR}-%J.err" \
  blaunch bash launch-scripts/pretrain.sh "$PATCHED_CONFIG"


#   -oo "${LOG_DIR}/bs-energy-train-%J.out" \
#   -eo "${LOG_DIR}/bs-energy-train-%J.err" \