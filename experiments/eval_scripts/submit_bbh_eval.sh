#!/bin/bash
# submit_bbh_eval.sh — Big-Bench Hard (3-shot, 27 subtasks, exact match) eval.
# Heavier than the main 13-task LM-harness suite (~3 hours typical).
#
# Usage:
#   bash submit_bbh_eval.sh <unsharded_ckpt> <job_name>
#
# Output:
#   <ckpt>/harness_bbh_results_<UTC>_<timestamp>.json          — raw lm-eval output (em=0 broken)
#   <ckpt>/bbh_samples_<UTC>/samples_bbh_fewshot_*.jsonl        — per-sample log
#   <ckpt>/bbh_samples_<UTC>/bbh_rescore_summary.json           — per-subtask rescored em
#   <ckpt>/harness_bbh_results_<UTC>_<timestamp>.rescored.json  — patched lm-eval JSON
#
# Notes:
# - lm-eval-harness BBH `bbh_fewshot` task has NO filter_list on its exact_match
#   metric, so the raw model continuation is compared verbatim against the gold
#   string. Small models produce continuations like " False" (leading space) or
#   "False." (trailing punctuation), neither of which == "False" → em=0 on all
#   27 subtasks. We now run with `--log_samples` and post-process with
#   bbh_rescore.py which applies the per-subtask flexible-extract regex from
#   lm-eval's `bbh_zeroshot` variant (yes/no, true/false, "(A)..(R)", integer).
# - lm-eval-harness BBH requires the SaylorTwift/bbh dataset to be cached
#   under \$HF_HOME. If it isn't, run a one-shot online dataset fetch first
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
SAMPLES_DIR="${CHECKPOINT}/bbh_samples_\${TIMESTAMP}"
mkdir -p "\${SAMPLES_DIR}"
python ${SCRIPTS_DIR}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${CHECKPOINT},dtype=bfloat16,trust_remote_code=True" \\
    --tasks bbh_fewshot \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --log_samples \\
    --output_path "\${SAMPLES_DIR}/harness_bbh_results_\${TIMESTAMP}.json"
echo "[bbh] lm-eval done — running rescorer"
# lm_eval writes samples files into a model-sanitized subdir under --output_path.
# Locate them and rescore.
SAMPLES_LEAF=\$(find "\${SAMPLES_DIR}" -maxdepth 3 -name "samples_bbh_fewshot_*.jsonl" -printf "%h\n" | sort -u | head -n 1)
if [ -z "\${SAMPLES_LEAF}" ]; then
    echo "[bbh_rescore] WARNING: no samples files found under \${SAMPLES_DIR}"
else
    echo "[bbh_rescore] samples dir: \${SAMPLES_LEAF}"
    # Copy the harness JSON next to samples so rescorer can patch it.
    HARNESS_JSON=\$(find "\${SAMPLES_DIR}" -maxdepth 3 -name "results_*.json" | head -n 1)
    if [ -n "\${HARNESS_JSON}" ]; then
        cp "\${HARNESS_JSON}" "\${SAMPLES_LEAF}/harness_bbh_results_\${TIMESTAMP}.json"
        # Also drop a copy at the checkpoint root for backward compatibility.
        cp "\${HARNESS_JSON}" "${CHECKPOINT}/harness_bbh_results_\${TIMESTAMP}.json"
    fi
    python ${SCRIPTS_DIR}/bbh_rescore.py "\${SAMPLES_LEAF}" --show-examples 3
    # Place rescored JSON at checkpoint root for easy discovery.
    cp "\${SAMPLES_LEAF}"/harness_bbh_results_*.rescored.json "${CHECKPOINT}/" 2>/dev/null || true
    cp "\${SAMPLES_LEAF}"/bbh_rescore_summary.json "${CHECKPOINT}/bbh_rescore_summary_\${TIMESTAMP}.json" 2>/dev/null || true
fi
echo "[bbh] done"
EOF
echo "BBH eval submitted: ${JOB_NAME}  checkpoint: ${CHECKPOINT}"
