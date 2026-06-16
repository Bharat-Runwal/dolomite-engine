#!/bin/bash
# submit_bbh_eval.sh — Big-Bench Hard (3-shot, 27 subtasks, exact match) eval.
# Heavier than the main 13-task LM-harness suite (~3 hours typical).
#
# Usage:
#   bash submit_bbh_eval.sh <unsharded_ckpt> <job_name>
#
# Output: <ckpt>/harness_bbh_results_<UTC-timestamp>.json
#
# Notes:
# - lm-eval-harness BBH requires the SaylorTwift/bbh dataset to be cached
#   under $HF_HOME. If it isn't, run a one-shot online dataset fetch first
#   (drop the HF_HUB_OFFLINE=1 line below).
# - Above-random performance is expected at ≥400M params on math+web pretraining;
#   <random ⇒ check tokenizer / model registration.

set -euo pipefail
CHECKPOINT="${1:?Usage: submit_bbh_eval.sh <unsharded_ckpt> <job_name>}"
JOB_NAME="${2:-eval_bbh}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPTS_DIR="${REPO}/experiments/eval_scripts"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J "${JOB_NAME}" \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<EOF
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval -q
cd ${REPO}
TIMESTAMP=\$(date -u +%Y-%m-%dT%H-%M-%SZ)
python ${SCRIPTS_DIR}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${CHECKPOINT},dtype=bfloat16,trust_remote_code=True" \\
    --tasks bbh_fewshot \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "${CHECKPOINT}/harness_bbh_results_\${TIMESTAMP}.json"
echo "[bbh] done"
EOF
echo "BBH eval submitted: ${JOB_NAME}  checkpoint: ${CHECKPOINT}"
