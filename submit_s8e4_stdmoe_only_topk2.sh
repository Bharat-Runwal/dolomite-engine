#!/bin/bash
# One-off submit for the missing baseline: s8e4_stdmoe_only_topk2.
# This config (8 softmax + 4 energy-attn, 12 std-MoE blocks at ipe=3200 top-2)
# was NOT in the original 400M-30k sweep. It is the cleanest matched-active
# baseline to compare against s8e4_stdmoe_fh1_topk2 — same attention layout,
# same MoE expert count/top-k, only the last-4 MLPs differ (std MoE here vs
# fh1 there).
#
# Param match against the other two configs in this directory:
#   s8e4_stdmoe_fh1_topk2     active=391.2M / total=1105.3M
#   s12_stdmoe_only_topk2     active=389.1M / total=1096.9M
#   s8e4_stdmoe_only_topk2    active=389.1M / total=1096.9M  <- this run

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

NAME="s8e4_stdmoe_only_topk2_400m_30k"
CFG="configs/boltzman_moe_all/s8e4_stdmoe_only_topk2.yml"
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
