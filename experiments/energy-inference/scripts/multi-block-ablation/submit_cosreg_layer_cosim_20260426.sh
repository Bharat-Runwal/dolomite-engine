#!/bin/bash
# Submit cosreg layer-cosim analysis (V0, V1, V39, V52, V53, V48) to bsub.
# Produces full i,j heatmaps + consecutive-pair line plots that complement
# analyze_v52v53_cosim_20260425e.py (which only computes summary metrics).
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=cosim_cosreg_20260426

bsub \
    -q normal \
    -G grp_ebm \
    -J "${JOB_NAME}" \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 02:00 \
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
python experiments/energy-inference/scripts/multi-block-ablation/analyze_cosreg_layer_cosim_20260426.py
echo "Done: cosreg layer-cosim analysis"
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to normal queue (1 GPU, 2h)."
echo "Logs: \$HOME/bsub_logs/${JOB_NAME}_<jobid>.stdout"
