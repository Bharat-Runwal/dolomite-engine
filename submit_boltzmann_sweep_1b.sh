#!/bin/bash
# 1.7B-active / 9B-total head-to-head at 16 experts top-2.
# d=2048, L=24, 16 heads x 128 head_dim, lr=2e-3, ga=4, 30k steps, 16 nodes each.
# Per-step tokens: 1 x 4 x 4096 x (16 nodes x 8 GPUs) = 2.1M.
# Total tokens per run: 30000 x 2.1M = ~63B.
#
# Configs:
#   1. s8e4_stdmoe_fh1_topk2_1b_30k     (Boltzmann TopK_Energy_MoE hybrid)
#   2. s12_stdmoe_only_topk2_1b_30k     (vanilla MoE, all 24 softmax)
#   3. s8e4_stdmoe_only_topk2_1b_30k    (vanilla MoE, s16e8 attention layout)

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

submit_variant() {
  local NAME="$1"
  local CFG="$2"
  local NODES="$3"

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
  echo "  -> Submitted ${NAME}"
}

submit_variant "s8e4_stdmoe_fh1_topk2_1b_30k" \
  "configs/boltzmann_moe/s8e4_stdmoe_fh1_topk2_1b_30k.yml" 16

submit_variant "s8e4_stdmoe_only_topk2_1b_30k" \
  "configs/boltzmann_moe/s8e4_stdmoe_only_topk2_1b_30k.yml" 16

submit_variant "s12_stdmoe_only_topk2_1b_30k" \
  "configs/boltzmann_moe/s12_stdmoe_only_topk2_1b_30k.yml" 16

echo ""
echo "All 3 jobs submitted. Watch with: bjobs -w | grep _1b_30k"
echo "Logs: ${LOG_DIR}/{s8e4_stdmoe_fh1_topk2,s12_stdmoe_only_topk2,s8e4_stdmoe_only_topk2}_1b_30k.{out,err}"
