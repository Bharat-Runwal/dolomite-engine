#!/bin/bash
# Submit a BBH-only lm-evaluation-harness job for a given checkpoint.
# Runs bbh_fewshot (27 subtasks, 3-shot, exact match) and writes results to
# <checkpoint_path>/harness_bbh_results_<timestamp>.json
#
# Usage: bash submit_bbh_eval.sh <checkpoint_path> <job_name>

CHECKPOINT="${1:?Usage: submit_bbh_eval.sh <checkpoint_path> <job_name>}"
JOB_NAME="${2:-eval_bbh}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J "${JOB_NAME}" \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 03:00 \
    -o "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval -q
cd $REPO
TIMESTAMP=\$(date +%Y-%m-%dT%H-%M-%S.%6N)
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=$CHECKPOINT,dtype=bfloat16,trust_remote_code=True" \\
    --tasks bbh_fewshot \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "$CHECKPOINT/harness_bbh_results_\${TIMESTAMP}.json"
echo "BBH eval done: $CHECKPOINT"
EOF

echo "BBH eval job submitted: ${JOB_NAME}  checkpoint: ${CHECKPOINT}"
