#!/bin/bash
# Submits the three head-to-head configs:
#   1. s8e4_stdmoe_fh1_topk2     (Boltzmann TopK_Energy_MoE hybrid, no looping)
#   2. s12_stdmoe_only_topk2     (vanilla MoE, ALL 12 softmax — reproduces 1B baseline)
#   3. s8e4_stdmoe_only_topk2    (vanilla MoE, s8e4 attention — cleanest ablation)
#
# All three: ~390M active / ~1.1B total, d=1024, 12 layers, 30k steps.
# fh1 wins on Avg, MMLU, gsm8k, Wiki PPL vs both std-MoE baselines — see README.
#
# To scale up to 9B-active, edit the YAMLs:
#   - hidden_size:          1024 -> 4096   (or whatever target d gives 9B-active)
#   - num_layers:           12   -> 32+
#   - num_attention_heads:  16   -> 32+
#   - intermediate_size in MoE blocks: scale with hidden_size to keep top-2/8 sparsity
#   - num_training_steps:   30000 -> longer for 9B-active (e.g. 100k+)
#   - micro_batch_size / gradient_accumulation_steps / NODES: tune for memory.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"

submit_variant() {
  # Args: $1 NAME, $2 CONFIG_PATH, $3 NODES
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

# At ~1.1B total / mb=1 ga=4 the top-2 group needs 8 nodes for memory headroom.
# Effective batch = 1 x 4 x 64 = 256 = 1.05M tok/step.
submit_variant "s8e4_stdmoe_fh1_topk2" \
  "configs/boltzmann_moe/s8e4_stdmoe_fh1_topk2.yml" 8

submit_variant "s12_stdmoe_only_topk2" \
  "configs/boltzmann_moe/s12_stdmoe_only_topk2.yml" 8

submit_variant "s8e4_stdmoe_only_topk2" \
  "configs/boltzmann_moe/s8e4_stdmoe_only_topk2.yml" 8

echo ""
echo "All 3 jobs submitted. Watch with: bjobs -w | grep -E 'stdmoe_fh1_topk2|stdmoe_only_topk2'"
echo "Logs: ${LOG_DIR}/{s8e4_stdmoe_fh1_topk2,s12_stdmoe_only_topk2,s8e4_stdmoe_only_topk2}.{out,err}"
