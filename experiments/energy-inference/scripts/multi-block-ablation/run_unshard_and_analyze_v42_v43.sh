#!/bin/bash
# Unshard V42/V43 fine-tuned models (DCP format) and run layer merger analysis.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=unshard_analyze_v42_v43

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J ${JOB_NAME} \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 02:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<'BSUB_SCRIPT'
#!/bin/bash
set -euo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}

REPO=/proj/dmfexp/nima/Code/dolomite-engine

# Unshard V42 FT (naive merge fine-tuned)
V42_FT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v42_egpt_6layer_naive_merge_finetune_d768
V42_UNSHARDED=${V42_FT}/unsharded
if [ ! -d "${V42_UNSHARDED}" ]; then
    echo "=== Unsharding V42 FT ==="
    CFG="/tmp/unshard_v42_$$.yml"
    printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${V42_FT}" "${V42_UNSHARDED}" > "${CFG}"
    python -m lm_engine.unshard --config "${CFG}" && rm -f "${CFG}"
    echo "V42 FT unsharded to ${V42_UNSHARDED}"
else
    echo "V42 FT already unsharded, skipping."
fi

# Unshard V43 FT (Procrustes merge fine-tuned)
V43_FT=${REPO}/experiments/energy-inference/results/multi-block-ablation/v43_egpt_6layer_procrustes_merge_finetune_d768
V43_UNSHARDED=${V43_FT}/unsharded
if [ ! -d "${V43_UNSHARDED}" ]; then
    echo "=== Unsharding V43 FT ==="
    CFG="/tmp/unshard_v43_$$.yml"
    printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${V43_FT}" "${V43_UNSHARDED}" > "${CFG}"
    python -m lm_engine.unshard --config "${CFG}" && rm -f "${CFG}"
    echo "V43 FT unsharded to ${V43_UNSHARDED}"
else
    echo "V43 FT already unsharded, skipping."
fi

# Run analysis
echo "=== Running layer merger analysis ==="
python ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/analyze_layer_merger_20260425.py

echo "=== Done ==="
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} to preemptable queue."
