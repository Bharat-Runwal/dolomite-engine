#!/bin/bash
# Run visualize.py on all 6-block models
set -e

BASE="/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_test"
OUT="tools/bs/energy_landscape"

MODELS=(
    "EGPT_6_blocks_1_13_13_13_13_1_30k"
    "EGPT_6_blocks_15_6_6_6_6_15_30k"
    "EGPT_6_blocks_2_12_13_13_12_2_30k"
    "EGPT_6_blocks_3_12_12_12_12_3_30k"
    "EGPT_6_blocks_3_12_12_9_3_15_30k"
    "EGPT_6_blocks_4_11_12_12_11_4_30k"
    "EGPT_6_blocks_9_9_9_9_3_15_30k"
    "EGPT_6_blocks_9_9_9_9_9_9_30k"
)

for model in "${MODELS[@]}"; do
    echo "============================================"
    echo "Processing: $model"
    echo "============================================"
    uv run python visualize.py "$BASE/$model" --output_dir "$OUT/$model" --num_samples 30
    echo ""
done

echo "All done! Results in $OUT/"
