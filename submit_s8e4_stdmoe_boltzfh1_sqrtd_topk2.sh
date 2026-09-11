#!/bin/bash
# One-off submit for the boltzfh1 sqrt(1/d)-scaling variant.
# IDENTICAL to s8e4_stdmoe_boltzfh1_topk2 except the Boltzmann router scales
# energies by a FIXED sqrt(1/intermediate_size) instead of a learnable
# temperature T (energy_scale_mode: sqrt_inv_d, learnable_temperature: false).
# Mentor suggestion: fixed sqrt(1/d) scaling beats a learnable T.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="s8e4_stdmoe_boltzfh1_sqrtd_topk2_400m_30k"
CFG="configs/boltzman_moe_all/s8e4_stdmoe_boltzfh1_sqrtd_topk2.yml"
NODES=8

echo "Submitting ${NAME} (config=${CFG}, nodes=${NODES})"
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

echo "Submitted. Watch with: bjobs -w | grep ${NAME}"
echo "Log: ${LOG_DIR}/${NAME}.err"
