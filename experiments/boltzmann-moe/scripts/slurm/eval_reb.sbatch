#!/bin/bash
# Unshard + evaluate one rebuttal checkpoint with the PAPER's protocols (lm-eval v0.4.9.2, 0-shot,
# 11 Avg11 tasks + wikitext word-PPL):
#   exact : unsharded checkpoint as-is -> energy layers use exact all-K routing (sparse_start_step>0 and
#           no step at inference). bs=1. = Table 1 protocol (eval_milestones.sh).
#   proxy : same weights (hard links), config patched sparse_start_step=0, sparse_capacity_factor=K/k
#           (no drop) -> deployed proxy router. bs=16. = Table 7 / G6-seed protocol (eval_g6_seeds.sh).
#   sbatch --job-name=ev_<name> eval_reb.sbatch <ckpt_dir> <iteration> <exact|exact,proxy>
#SBATCH --account=slurm_grp_ebm
#SBATCH --partition=gpu-high
#SBATCH --qos=slurm_grp_ebm_high_1n
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=08:00:00
#SBATCH --requeue
#SBATCH --output=/proj/dmfexp/energy-gpt/logs/rebuttal/eval/%x-%j.out
#SBATCH --error=/proj/dmfexp/energy-gpt/logs/rebuttal/eval/%x-%j.err
set -uo pipefail
D=${1:?ckpt dir}; IT=${2:?iteration}; MODES=${3:-exact}
TREE=/proj/dmfexp/bishwajit/Code/de-reb
PY=/proj/dmfexp/bishwajit/Code/dolomite-engine/.venv-nima/bin/python
export PYTHONPATH=/proj/dmfexp/energy-gpt/lm-evaluation-harness:$TREE/accelerated-model-architectures:$TREE
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 \
       HF_HUB_DISABLE_TELEMETRY=1 HF_DATASETS_TRUST_REMOTE_CODE=1 TOKENIZERS_PARALLELISM=false
TASKS=arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,wikitext
cd $TREE; echo "host=$(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader) D=$D IT=$IT MODES=$MODES $(date)"
U=$D/unsharded_step$IT
if [ ! -f $U/model.safetensors ]; then
  T=$U.tmp.$$; rm -rf $T; C=$(mktemp /tmp/unsh_XXXX.yml)
  printf 'load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n' $D $IT $T > $C
  $PY -m lm_engine.unshard --config $C || { echo "UNSHARD FAILED"; exit 1; }
  mv $T $U; rm -f $C
fi
run() { # dir out bs
  [ -n "$(ls $2/harness_results_*.json 2>/dev/null)" ] && { echo "skip $2"; return; }
  mkdir -p $2
  $PY experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$1,dtype=bfloat16,trust_remote_code=True \
    --tasks $TASKS --device cuda:0 --batch_size $3 --trust_remote_code --output_path $2/harness_results.json > $2/eval.log 2>&1
  echo "$2 rc=$? $(date +%T)"
}
[[ $MODES == *exact* ]] && run $U $U/eval_exact_bs1 1
if [[ $MODES == *proxy* ]]; then
  P=$D/sparseeval_nodrop_step$IT
  if [ ! -f $P/config.json ]; then
    mkdir -p $P; for f in $U/*; do [ -f $f ] && [ $(basename $f) != config.json ] && ln -f $f $P/; done
    $PY - $U/config.json $P/config.json <<'PYX'
import json,sys
c=json.load(open(sys.argv[1])); n=0
for b in c['mlp_blocks']:
    if 'sparse_start_step' in b:
        b['sparse_start_step']=0; b['sparse_capacity_factor']=float(b['n_experts'])/b['top_k']; n+=1
json.dump(c,open(sys.argv[2],'w'),indent=2); print('patched',n,'energy mlp blocks')
PYX
  fi
  run $P $P/eval_bs16 16
fi
echo DONE
