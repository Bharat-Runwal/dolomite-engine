#!/bin/bash
# Mother script: submit cosreg + 400M layer-cosim analyses in parallel via bsub.
# Each child does its own bsub; this just kicks both off so the user runs one command.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== Submitting cosreg layer-cosim (V0/V1/V39/V52/V53/V48) ==="
bash "${HERE}/submit_cosreg_layer_cosim_20260426.sh"
echo

echo "=== Submitting 400M layer-cosim (V9/V1_400m/V19/V20/V31/V32/V40) ==="
bash "${HERE}/submit_layer_cosim_400m_20260426.sh"
echo

echo "Both jobs submitted. Check bjobs status with:  bjobs -J 'cosim_*_20260426'"
