#!/bin/bash
# Submit 400M deep-EGPT layer-cosim analysis to bsub.
# Models: V9, V1_400m, V19, V20, V31, V32, V40.
# Produces full i,j heatmaps + consecutive-pair line plots — extends the
# existing 400M analysis (V9/V19/V20 only) to include V1 EGPT 400M, V31, V32, V40.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=cosim_400m_20260426

bsub \
    -q normal \
    -G grp_ebm \
    -J "${JOB_NAME}" \
    -gpu "num=1" \
    -n 1 \
    -M 64G \
    -W 03:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install matplotlib numpy torch transformers -q

cd ${REPO}
python experiments/energy-inference/scripts/multi-block-ablation/analyze_layer_cosim_400m_20260426.py
echo "Done: 400M layer-cosim analysis"
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to normal queue (1 GPU, 3h, 64G)."
echo "Logs: \$HOME/bsub_logs/${JOB_NAME}_<jobid>.stdout"
