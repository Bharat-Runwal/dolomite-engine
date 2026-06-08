#!/bin/bash
# Submit BOTH h1_boltz_680m runs as a head-to-head pair:
#   1. _nemo    — faithful reproduction of Nima's best run (nemotron-cc-hq only)
#   2. _mathmix — same model, math data mix added (nemotron + megamath + finemath)
# Both are Nima's EXACT h1_boltz_moe_580m_8x4096_d1536 architecture (679M):
#   11 GPT + 1 EGPT x6, 8 experts x 4096, d=1536, BoltzmannMoE_Energy_MLP.
#
# Each: 32 GPUs (4 nodes x 8), 124k steps, mb4 x accum4 x seq4096
#   -> tokens/step = 4 * 4 * 4096 * 32 = 2,097,152
#   -> total = 124,000 * 2,097,152 = ~260B tokens

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"
NODES=4   # 4 nodes x 8 GPUs = 32 GPUs (matches Nima's setup exactly)

submit_one () {
  local NAME="$1" CFG="$2"
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
}

submit_one "h1_boltz_680m_8x4096_d1536_nemo"    "configs/boltzmann_moe/h1_boltz_680m_8x4096_d1536_nemo.yml"
submit_one "h1_boltz_680m_8x4096_d1536_mathmix" "configs/boltzmann_moe/h1_boltz_680m_8x4096_d1536_mathmix.yml"

echo "Submitted both. Watch with: bjobs -w | grep h1_boltz_680m"
