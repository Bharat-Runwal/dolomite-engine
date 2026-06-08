#!/bin/bash
# Submit fh2_boltz_680m — EXACT rerun of Nima's h1_boltz_moe_580m_8x4096_d1536.
# 32 GPUs (4 nodes x 8) to match his exact token budget: 124k steps x mb4 x accum4
# x seq4096 x 32 = ~260B tokens. Soft (dense) Boltzmann routing, 8 experts, d=1536.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="s11e1_mlp_fh2_boltz_600m_124k_1e-3_680m_680m"
CFG="configs/boltzmann_moe/fh2_boltz_680m_8x4096_d1536.yml"
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
