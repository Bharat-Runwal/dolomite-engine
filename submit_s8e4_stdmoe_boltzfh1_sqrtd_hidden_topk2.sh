#!/bin/bash
# One-off submit for the boltzfh1 sqrt(1/d)-scaling variant with d = hidden_size.
# IDENTICAL to s8e4_stdmoe_boltzfh1_sqrtd_topk2 EXCEPT the sqrt(1/d) dimension:
# this run uses d = hidden_size (1024) instead of intermediate_size (9536),
# i.e. routing scale = sqrt(1/1024) (sqrt_inv_d_dim: hidden). Fixed, no learnable T.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="s8e4_stdmoe_boltzfh1_sqrtd_hidden_topk2_400m_30k"
CFG="configs/boltzman_moe_all/s8e4_stdmoe_boltzfh1_sqrtd_hidden_topk2.yml"
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
