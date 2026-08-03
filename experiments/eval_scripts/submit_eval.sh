#!/bin/bash
# submit_eval.sh — submit an LM-evaluation-harness job (and optionally BBH) for
# an unsharded checkpoint. Canonical entry point used by all run_*.sh launchers.
#
# Usage:
#   bash submit_eval.sh <unsharded_ckpt> <job_name> [--with-bbh|--no-bbh]
#
# Default: --with-bbh (BBH submitted as separate bsub job).
#
# Outputs (next to checkpoint):
#   <ckpt>/harness_results.json                            — 13-task LM suite
#   <ckpt>/harness_bbh_results_<UTC-timestamp>.json        — BBH (if enabled)
#
# Tasks (13): arc_challenge, arc_easy, boolq, copa, hellaswag, openbookqa,
#             piqa, sciq, wikitext, winogrande, mmlu, gsm8k, gsm8k_cot.
# Excluded by default (cluster-specific):
#   race, lambada_openai — re-enabled 2026-08-03 via pyarrow>=20 pin (was a pyarrow<20 reader bug)
#   bbh_fewshot           — separate path, run via --with-bbh
#
# Aggregates: after results land, run
#   python experiments/eval_scripts/compute_aggregates.py <ckpt>/harness_results.json
# to print avg10, avg10_norm, WikiText-PPL, GSM8K, etc.

set -euo pipefail
CHECKPOINT="${1:?Usage: submit_eval.sh <unsharded_ckpt> <job_name> [--with-bbh|--no-bbh]}"
JOB_NAME="${2:-eval}"
BBH_FLAG="${3:---with-bbh}"
case "${BBH_FLAG}" in
  --with-bbh|--bbh)        WITH_BBH=1 ;;
  --no-bbh|--without-bbh)  WITH_BBH=0 ;;
  *) echo "Unknown flag: ${BBH_FLAG}"; exit 2 ;;
esac

REPO=/proj/dmfexp/nima/Code/dolomite-engine
SCRIPTS_DIR="${REPO}/experiments/eval_scripts"
mkdir -p "${HOME}/bsub_logs"

# 1. Main 13-task LM-harness job
TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,race,sciq,wikitext,winogrande,lambada_openai,mmlu,gsm8k,gsm8k_cot"
bsub \
    -q preemptable -G grp_preemptable -J "${JOB_NAME}" \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 02:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<EOF
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval 'pyarrow>=20' -q   # pyarrow>=20 required to read race/lambada parquet (else 'Repetition level histogram size mismatch')
cd ${REPO}
python ${SCRIPTS_DIR}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${CHECKPOINT},dtype=bfloat16,trust_remote_code=True" \\
    --tasks ${TASKS} \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "${CHECKPOINT}/harness_results.json"
echo "[lm_harness] done"
echo "Aggregates:"
python ${SCRIPTS_DIR}/compute_aggregates.py "${CHECKPOINT}/harness_results.json" || true
EOF
echo "LM-harness eval submitted: ${JOB_NAME}  checkpoint: ${CHECKPOINT}"

# 2. BBH job (separate bsub, in parallel with LM-harness)
if [ "${WITH_BBH}" -eq 1 ]; then
  bash "${SCRIPTS_DIR}/submit_bbh_eval.sh" "${CHECKPOINT}" "${JOB_NAME}_bbh"
fi
