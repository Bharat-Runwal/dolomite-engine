#!/bin/bash
# Run activation-space layer similarity analysis across V0/V1/V39/V41/V42/V43.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=analyze_layer_activations

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

python /proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/analyze_layer_activations_20260425.py

echo "=== Activation analysis complete ==="
BSUB_SCRIPT

echo "Submitted ${JOB_NAME}."
