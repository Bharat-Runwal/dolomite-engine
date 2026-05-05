#!/bin/bash
# V49: Procrustes 6×2 merge, constant lr=2e-3 (no decay), 5k steps.
# V47 (cosine 2e-3→2e-4) got 3.464 vs V46 (const 2e-4) 3.560. Model still recovering.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/v49_egpt_6x2_procrustes_merge_finetune_constlr2e3_d768.yml
JOB_NAME=egpt_v49_constlr2e3_ft

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
bash ${REPO}/scripts/common/pretrain.sh ${CONFIG}
BSUB_SCRIPT

echo "Submitted ${JOB_NAME}."
