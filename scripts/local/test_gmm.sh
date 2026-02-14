#!/usr/bin/env bash
set -euo pipefail

# ============================================================
# Local GMM Test: Train GaussianBoltzmannMoE on WikiText-2
# ============================================================
# Usage:
#   ./scripts/local/test_gmm.sh              # full: prepare data + train
#   ./scripts/local/test_gmm.sh --train-only  # skip data prep (if already done)
#   ./scripts/local/test_gmm.sh --prepare-only # only prepare data, don't train
#   ./scripts/local/test_gmm.sh --no-wandb     # disable wandb logging
#
# Prerequisites: pip install datasets transformers
# Hardware: Single GPU (tested on RTX 4090)

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="$REPO_DIR/configs/local/gmm_wikitext_test.yml"
DATA_DIR="$REPO_DIR/data/wikitext2"

# Use the project conda env which has wandb, transformers, etc.
CONDA_DIR="/home/nima/__work/LLM/.conda"
if [ -x "$CONDA_DIR/bin/python" ]; then
    export PATH="$CONDA_DIR/bin:$PATH"
    echo "Using Python: $(which python) ($(python --version 2>&1))"
fi

TRAIN_ONLY=false
PREPARE_ONLY=false
NO_WANDB=false

for arg in "$@"; do
    case "$arg" in
        --train-only)  TRAIN_ONLY=true ;;
        --prepare-only) PREPARE_ONLY=true ;;
        --no-wandb)    NO_WANDB=true ;;
        *) echo "Unknown arg: $arg"; exit 1 ;;
    esac
done

cd "$REPO_DIR"

# ── Step 1: Prepare data ──
if [ "$TRAIN_ONLY" = false ]; then
    if [ -f "$DATA_DIR/wikitext2_text.bin" ] && [ -f "$DATA_DIR/wikitext2_text.idx" ]; then
        echo "✓ Data already prepared at $DATA_DIR"
    else
        echo "═══ Preparing WikiText-2 data ═══"
        python scripts/local/prepare_wikitext.py --output-dir "$DATA_DIR"
        echo "✓ Data prepared"
    fi
fi

if [ "$PREPARE_ONLY" = true ]; then
    echo "Done (prepare only)."
    exit 0
fi

# ── Step 2: Train ──
echo "═══ Training GaussianBoltzmannMoE on WikiText-2 ═══"
echo "Config: $CONFIG"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'unknown')"

EXTRA_ENV="GIT_PYTHON_GIT_EXECUTABLE=/usr/bin/git"
if [ "$NO_WANDB" = true ]; then
    EXTRA_ENV="$EXTRA_ENV WANDB_MODE=disabled"
fi

# Single-GPU training via torchrun
env $EXTRA_ENV TOKENIZERS_PARALLELISM=false \
    torchrun --nproc_per_node=1 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:0 \
    -m lm_engine.pretrain \
    --config "$CONFIG"
