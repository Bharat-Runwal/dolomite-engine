#!/bin/bash
# Submit s8e4_stdmoe_fh2_boltz_topk2 — same as s8e4_stdmoe_boltzfh1_sqrtd_topk2
# EXCEPT the 4 energy MLP blocks use Nima's BoltzmannMoE_Energy_MLP (fh2_boltz)
# instead of your BoltzRouter_TopK_Energy_MoE_MLP. Iso-param: 387M active / 1101M total.
# Differs from boltzfh1_sqrtd in: router (no-renorm top-k truncation), aux loss
# (cosine repulsion vs Switch load-balance), both use 1/sqrt(expert_I=9536) scaling.
# Recipe identical: 30k steps, 2e-3 LR, 8 nodes x 8 GPUs.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="s8e4_stdmoe_fh2_boltz_topk2_400m_30k"
CFG="configs/boltzmann_moe/s8e4_stdmoe_fh2_boltz_topk2.yml"
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
