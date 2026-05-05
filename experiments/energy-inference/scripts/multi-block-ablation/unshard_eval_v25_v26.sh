#!/bin/bash
# Unshard and eval V25 and V26 (W_O→W_O_gpt compat fix applied to checkpointing/__init__.py)
set -euo pipefail

source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}

REPO=/proj/dmfexp/nima/Code/dolomite-engine
EVAL_SCRIPT=${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh

for V in v25_energy_grad_6x4_d1024_lr1e3 v26_energy_grad_6x_ramp_d1024_lr1e3; do
    SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/${V}
    UNSHARDED=${SAVE_PATH}/unsharded

    echo "=== Unsharding ${V} ==="
    UNSHARD_CFG="/tmp/${V}_unshard_$$.yml"
    printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${SAVE_PATH}" "${UNSHARDED}" > "${UNSHARD_CFG}"

    python -m lm_engine.unshard --config "${UNSHARD_CFG}" && rm -f "${UNSHARD_CFG}"
    echo "Unsharded to ${UNSHARDED}"

    EVAL_NAME="eval_egpt_${V}"
    bash "${EVAL_SCRIPT}" "${UNSHARDED}" "${EVAL_NAME}"
    echo "Eval submitted: ${EVAL_NAME}"
done
