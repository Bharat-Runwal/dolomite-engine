#!/bin/bash
# Single-node (8 GPU) Slurm launcher for lm_engine pretraining. LSF was retired 2026-11-04;
# scripts/common/pretrain.sh reads LSB_* vars, so this calls torchrun --standalone directly.
#   sbatch --job-name=<name> experiments/boltzmann-moe/scripts/slurm/train_1node.sbatch <config.yml>
#SBATCH --account=slurm_grp_ebm
#SBATCH --partition=gpu-high
#SBATCH --qos=slurm_grp_ebm_high_1n
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=64
#SBATCH --mem=256G
#SBATCH --time=10:00:00
#SBATCH --output=/proj/dmfexp/energy-gpt/logs/rebuttal/%x-%j.out
#SBATCH --error=/proj/dmfexp/energy-gpt/logs/rebuttal/%x-%j.err
set -euo pipefail
CFG=${1:?config}
REPO=${REPO:-/proj/dmfexp/bishwajit/Code/de-reb}
VENV=${DOLOMITE_VENV:-/proj/dmfexp/bishwajit/Code/dolomite-engine/.venv-nima}
unset TMPDIR TEMP TMP
source ${VENV}/bin/activate
cd ${REPO}
export PYTHONPATH=${REPO}:${PYTHONPATH:-}
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export NCCL_NVLS_ENABLE=0 NCCL_DEBUG=WARN TOKENIZERS_PARALLELISM=false
export DOLOMITE_LAUNCH_CMD="sbatch train_1node.sbatch ${CFG} (job ${SLURM_JOB_ID})"
[ -f "$CFG" ] || CFG="$REPO/$CFG"
SP=$(grep -E "^\s*save_path:" "$CFG" | head -1 | sed 's/.*save_path:\s*//')
mkdir -p "$SP"
if [ -f "$SP/latest_checkpointed_iteration.json" ] && ! grep -qE "^load_args:" "$CFG"; then
    RCFG="$SP/runtime_resume_${SLURM_JOB_ID}.yml"; cp "$CFG" "$RCFG"
    printf "\nload_args:\n  load_path: %s\n" "$SP" >> "$RCFG"; CFG="$RCFG"
    echo "RESUME from $(cat $SP/latest_checkpointed_iteration.json)"
else
    echo "fresh start"
fi
nvidia-smi -L
echo "host $(hostname)  commit $(git -C $REPO rev-parse --short HEAD)  cfg $CFG"
torchrun --standalone --nproc_per_node=8 -m lm_engine.pretrain --config "$CFG"
