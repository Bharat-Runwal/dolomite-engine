#!/bin/bash
# Submit h1_boltz_680m_mathmix — Nima's EXACT h1_boltz_moe_580m_8x4096_d1536
# (11 GPT + 1 EGPT x6, 8 experts, d=1536, BoltzmannMoE_Energy_MLP) with ONLY
# the data mix changed: nemotron-cc-hq + megamath-web-pro + finemath-3plus.
# Goal: test whether GSM8k improves over Nima's nemotron-only run.
# 32 GPUs (4 nodes x 8) to match his exact setup: 124k steps x mb4 x accum4 x seq4096.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="h1_boltz_680m_8x4096_d1536_mathmix"
CFG="configs/boltzmann_moe/h1_boltz_680m_8x4096_d1536_mathmix.yml"
NODES=4   # 4 nodes x 8 GPUs = 32 GPUs (matches Nima's setup exactly)

echo "Submitting ${NAME} (config=${CFG}, nodes=${NODES} -> 32 GPUs)"
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
