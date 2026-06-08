#!/bin/bash
# Submit all six 7B energy-MoE configs (see experiments/boltzmann-moe/SEVEN_B_CONFIGS.md).
# 4 nodes x 8 GPUs = 32 GPUs each, preemptable queue. Launcher activates .venv-boltz
# (transformers 4.57.1); override with PRETRAIN_VENV for a kernel-ready 7B env.
# Pass config basenames as args to submit a subset, e.g.:
#   bash submit_7b_configs.sh s7b_egpt_boltz_h3_32gpt8egpt_d1536

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"
NODES=4

ALL=(
  s7b_egpt_boltz_rec_20gpt4egpt6699_d1536
  s7b_egpt_boltz_h3_32gpt8egpt_d1536
  s7b_egpt_stdmoe_rec_20gpt4egpt6699_d1536
  s7b_egpt_stdmoe_h3_32gpt8egpt_d1536
  s7b_recgpt_rec_20l4rec6699_d1536
  s7b_recgpt_h3_40l_d1536
)

TARGETS=("$@"); [ ${#TARGETS[@]} -eq 0 ] && TARGETS=("${ALL[@]}")

for NAME in "${TARGETS[@]}"; do
  CFG="configs/boltzmann_moe/${NAME}.yml"
  if [ ! -f "$CFG" ]; then echo "SKIP: $CFG not found"; continue; fi
  echo "Submitting ${NAME} (nodes=${NODES} -> 32 GPUs)"
  bsub \
    -r \
    -q preemptable \
    -G grp_preemptable \
    -M 2000G \
    -hl \
    -n ${NODES} \
    -J "${NAME}" \
    -gpu "num=8/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/${NAME}.out" \
    -eo "${LOG_DIR}/${NAME}.err" \
    blaunch bash launch-scripts/pretrain.sh "$CFG"
done

echo "Done. Watch with: bjobs -w | grep s7b_"
