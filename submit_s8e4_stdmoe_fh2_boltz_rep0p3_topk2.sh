#!/bin/bash
# fh2_boltz tuning variant: rep0p3. Identical to s8e4_stdmoe_fh2_boltz_topk2
# except one knob on the 4 energy blocks. Iso-param 387M/1101M, 30k/2e-3, 8 nodes.
LOG_DIR="/proj/dmfexp/energy-gpt/logs/boltzmann_sweep"
mkdir -p "$LOG_DIR"
NAME="s8e4_stdmoe_fh2_boltz_rep0p3_topk2_400m_30k"
CFG="configs/boltzmann_moe/s8e4_stdmoe_fh2_boltz_rep0p3_topk2.yml"
NODES=8
echo "Submitting ${NAME}"
bsub -r -q preemptable -G grp_preemptable -M 2000G -hl -n ${NODES} -J "${NAME}" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/${NAME}.out" -eo "${LOG_DIR}/${NAME}.err" \
  blaunch bash launch-scripts/pretrain.sh "$CFG"
echo "Submitted. Log: ${LOG_DIR}/${NAME}.err"
