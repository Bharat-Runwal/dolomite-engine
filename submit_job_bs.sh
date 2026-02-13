#!/bin/bash

# Energy-GPT Training Job Submission
# Usage: bash submit_train.sh [config_file]

# CONFIG=${1:-"configs/energy/energy_bs.yml"}
CONFIG=${1:-"configs/energy/energy_loop_rec_egpt_nemo_bs.yml"}

LOG_DIR="/proj/dmfexp/energy-gpt/logs"

mkdir -p "$LOG_DIR"

bsub \
  -q preemptable \
  -G grp_preemptable \
  -M 2000G \
  -hl \
  -n 4 \
  -J bs-energy-nemo \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/bs-energy-nemo-%J.out" \
  -eo "${LOG_DIR}/bs-energy-nemo-%J.err" \
  blaunch bash launch-scripts/pretrain.sh "$CONFIG"


#   -oo "${LOG_DIR}/bs-energy-train-%J.out" \
#   -eo "${LOG_DIR}/bs-energy-train-%J.err" \