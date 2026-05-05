#!/bin/bash
# Template: submit an energy-inference dolomite job to bsub
# Usage: ./run_energy_inference.sh <config.yml> [job_name] [num_gpus] [wall_hours]
#
# Example:
#   ./run_energy_inference.sh experiments/energy-inference/configs/alpha_sweep_20260411.yml energy_alpha 2 4

CONFIG=${1:-experiments/energy-inference/configs/pretrain.yml}
JOB_NAME=${2:-energy_exp}
NUM_GPUS=${3:-2}
WALL_HOURS=${4:-4}

WALL=$(printf "%02d:00" $WALL_HOURS)

bsub \
    -q standard \
    -J $JOB_NAME \
    -gpu "num=${NUM_GPUS}/task:mode=exclusive_process" \
    -n 1 \
    -M 32G \
    -W $WALL \
    -o $HOME/${JOB_NAME}_%J.stdout \
    -e $HOME/${JOB_NAME}_%J.stderr \
    scripts/common/pretrain.sh $CONFIG
