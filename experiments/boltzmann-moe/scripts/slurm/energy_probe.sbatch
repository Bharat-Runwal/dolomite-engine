#!/bin/bash
#SBATCH --account=slurm_grp_ebm
#SBATCH --partition=gpu-high
#SBATCH --qos=slurm_grp_ebm_high_1n
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --job-name=energy_probe
#SBATCH --output=/proj/dmfexp/energy-gpt/logs/rebuttal/eval/%x-%j.out
#SBATCH --error=/proj/dmfexp/energy-gpt/logs/rebuttal/eval/%x-%j.err
TREE=/proj/dmfexp/bishwajit/Code/de-reb; PY=/proj/dmfexp/bishwajit/Code/dolomite-engine/.venv-nima/bin/python
export PYTHONPATH=$TREE/accelerated-model-architectures:$TREE HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
cd $TREE; O=/proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/energy_probe; mkdir -p $O
R=/proj/dmfexp/energy-gpt/checkpoints-bsaha
for spec in "8S4E $R/rebuttal/reb_8S4E_isoact_d768/sparseeval_nodrop_step7629" "6S6E $R/fineweb/fineweb_G6_6S6E_isoact_d768_s7/sparseeval_nodrop_step7629" "12E $R/rebuttal/reb_12E_isoact_d768/sparseeval_nodrop_step7629"; do
  set -- $spec; $PY experiments/boltzmann-moe/scripts/rebuttal/energy_probe.py $2 $O/$1.json 32 1024 || echo "FAILED $1"
done
