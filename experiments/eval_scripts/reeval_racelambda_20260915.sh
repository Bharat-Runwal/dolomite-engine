#!/bin/bash
# reeval_racelambda_20260915.sh <ckpt_dir> <job_name>
#
# Submit a 1-GPU preemptable job that evaluates ONLY race + lambada_openai on an
# unsharded HF checkpoint, then merges the two tasks into that checkpoint's
# existing 9-task harness_results JSON to produce a COMPLETE 11-task Avg11 file
# (harness_results_<UTC>_avg11reeval.json) — non-destructively (the original JSON
# is untouched).
#
# Why only 2 tasks: every scale/headline/MoE checkpoint already has the other 9
# Avg11 tasks + mmlu + gsm8k + wikitext scored; only race+lambada were missing
# (pre-pyarrow>=20). Re-running just these two is ~15 min vs ~2 h for a full
# 15-task re-eval, and does not overwrite a colleague's harness_results.json.
set -uo pipefail
CKPT="${1:?Usage: reeval_racelambda_20260915.sh <unsharded_ckpt_dir> <job_name>}"
JOB="${2:-rl_eval}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPTS_DIR="${REPO}/experiments/eval_scripts"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J "${JOB}" \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 01:00 \
    -o "${HOME}/bsub_logs/${JOB}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB}_%J.stderr" \
    <<EOF
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate -q
source ${SCRIPTS_DIR}/eval_tasks.sh
eval_env
eval_assert_pin
cd ${REPO}
python ${SCRIPTS_DIR}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${CKPT},dtype=bfloat16,trust_remote_code=True" \\
    --tasks race,lambada_openai \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "${CKPT}/racelambda.json"
echo "[racelambda] eval done for ${CKPT}"
python ${SCRIPTS_DIR}/merge_racelambda_into_avg11_20260915.py "${CKPT}"
echo "[racelambda] merge done"
python ${SCRIPTS_DIR}/compute_avg11.py "${CKPT}" || true
EOF
echo "submitted ${JOB}  ->  ${CKPT}"
