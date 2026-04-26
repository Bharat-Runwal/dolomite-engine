#!/bin/bash
# Run same-input delta analysis with attn/FFN breakdown (20260425b).
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=analyze_layer_activations_b

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 01:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<'BSUB_SCRIPT'
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}

python /proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/analyze_layer_activations_20260425b.py

echo "=== Activation analysis b complete ==="
BSUB_SCRIPT

echo "Submitted ${JOB_NAME}."
