#!/bin/bash
# The exact submission used for the 2026-09-16 sparsity wall-clock numbers in HANDOFF 12.9.
# Sweeps eager/compiled x fwd/fwd+bwd, then repeats fwd+bwd at the pure arm's real per-call
# token count (micro_batch_size 4 x sequence_length 4096 = 16384; the hybrids use 4096).
# Reproduce with: bash scripts/bsub/bench_sparse.sh
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
mkdir -p "$HOME/bsub_logs"
bsub \
    -q preemptable -G grp_preemptable \
    -J bench_sparse \
    -gpu "num=1/task:mode=exclusive_process" \
    -n 1 -M 32G -W 00:40 \
    -o "$HOME/bsub_logs/bench_sparse_%J.stdout" \
    -e "$HOME/bsub_logs/bench_sparse_%J.stderr" \
<<'INNER'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
cd /proj/dmfexp/nima/Code/dolomite-engine
S=experiments/boltzmann-moe/scripts/bench_sparse_20260916.py
for a in "" "--train" "--compile" "--compile --train"; do
  echo "################ python $S --tokens 4096 $a"
  python $S --tokens 4096 $a
  echo
done
echo "################ larger batch, fwd+bwd eager"
python $S --tokens 16384 --train
INNER
