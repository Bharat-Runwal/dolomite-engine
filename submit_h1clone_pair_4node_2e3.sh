#!/bin/bash
# Submit the lr=2e-3 variants of the 4-node (32-GPU) / 31k-step h1_boltz clones.
# Identical to the lr=1e-3 4-node runs except optimizer lr: 2e-3. preemptable queue.

LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"
NODES=4   # 4 nodes x 8 GPUs = 32 GPUs

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

submit_one "s11e1_egpt6_boltz_nemo_680m_31k_2e-3_679m_679m_4node" "configs/boltzman_moe_all/s11e1_egpt6_boltz_nemo_680m_31k_2e-3_679m_679m_4node.yml"
submit_one "s11e1_egpt6_boltz_math_680m_31k_2e-3_679m_679m_4node" "configs/boltzman_moe_all/s11e1_egpt6_boltz_math_680m_31k_2e-3_679m_679m_4node.yml"

echo "Submitted both. Watch with: bjobs -w | grep 2e-3"
