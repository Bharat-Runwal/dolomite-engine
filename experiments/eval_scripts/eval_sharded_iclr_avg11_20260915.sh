#!/bin/bash
# eval_sharded_iclr_avg11_20260915.sh {verify|submit|resubmit|status|report} [name ...]
#
# WHY THIS EXISTS.  run_avg11_sweep_20260915.sh + reeval_racelambda_20260915.sh handle
# the 55 checkpoints that ALREADY have a 9-task base harness_results json and only need
# race + lambada topped up.  Six ICLR arms have NO eval at all: they exist only as
# `global_step*/model/*.distcp` shards, with no `unsharded*` dir and no
# `harness_results_*.json`.  Those need the full chain, which this script runs as ONE
# self-contained 1-GPU bsub job per arm:
#
#   1. python -m lm_engine.unshard   ->  <unsharded_dir>   (config.json + safetensors + tokenizer)
#   2. eval_harness.py over the FULL canonical task list ($EVAL_TASKS from eval_tasks.sh:
#      the 11 Avg11 tasks + mmlu + gsm8k + gsm8k_cot + wikitext)  ->  harness_results_<ts>.json
#   3. compute_avg11.py                                    ->  printed Avg11 / MMLU / GSM8K / PPL
#
# Because step 2 scores race + lambada_openai directly, the output json is a COMPLETE
# 11-task Avg11 file from the start -- no merge_racelambda step is needed here.
#
# ALL SIX ARMS ARE PARTIALLY TRAINED (see STEP/TARGET in the table below).  Any number
# produced here MUST be quoted with its step count; none of them is a finished run.
#
# ---------------------------------------------------------------------------------
# THE TWO iclr_scale ARMS ARE READ FROM A STAGED COPY, NOT FROM THE LIVE RUN DIR.
# ---------------------------------------------------------------------------------
# LSF 1647293 (scale32B_boltz_hop) and 1647503 (scale32B_gptswitch) were TRAINING while
# this script was written, writing into
#   results/iclr_scale/<arm>/global_step*   with save_interval 1000 and max_to_keep 2.
# gptswitch lands a checkpoint every ~23 min, so the "latest" shard is deleted by the
# trainer's own pruning within ~45 min of appearing -- far too short to survive queueing
# plus a 4.5 GB unshard.  So on 2026-09-15 02:12 UTC the COMPLETED iteration named by each
# arm's latest_checkpointed_iteration.json was COPIED (pure read; nothing in the live dir
# was created, moved or deleted) to
#   results/iclr_scale_eval_staging/<arm>/global_step<N>/{model,metadata.json,training_config.yml}
#   results/iclr_scale_eval_staging/<arm>/latest_checkpointed_iteration.json
# which is all load_checkpoint_and_unshard() reads (async_checkpointing: false => it reads
# only model/, not optimizer/).  The jobs below point at the staging dir, so the live
# training dirs are never opened by an eval job at all.
set -uo pipefail
MODE="${1:-verify}"; shift || true
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SD="${REPO}/experiments/eval_scripts"
RES="${REPO}/experiments/boltzmann-moe/results"
STG="${RES}/iclr_scale_eval_staging"
mkdir -p "${HOME}/bsub_logs"

# name | load_path (dir holding global_step<N>) | step | unsharded_dir
# Priority order: w1w2_K32_top2 first (the paper's "in progress" arm).
CKPTS=(
  "w1w2_K32_top2|${RES}/iclr_decide/w1w2_K32_top2|10000|${RES}/iclr_decide/w1w2_K32_top2/unsharded_step10000"
  "slope90k_1blk|${RES}/iclr_slope/slope90k_1blk|40000|${RES}/iclr_slope/slope90k_1blk/unsharded_step40000"
  "big_hop_sandwich|${RES}/iclr_big/iclr_big_hop_sandwich|4000|${RES}/iclr_big/iclr_big_hop_sandwich/unsharded_step4000"
  "pure_hop_isoP_bal|${RES}/iclr_balance/pure_hop_isoP_bal|16000|${RES}/iclr_balance/pure_hop_isoP_bal/unsharded_step16000"
  "scale32B_boltz_hop|${STG}/scale32B_boltz_hop|6000|${STG}/scale32B_boltz_hop/unsharded_step6000"
  "scale32B_gptswitch|${STG}/scale32B_gptswitch|58000|${STG}/scale32B_gptswitch/unsharded_step58000"
  # 2026-09-15 03:45 UTC: scale32B_gptswitch FINISHED TRAINING (LSF 1647503 DONE) at its
  # full 61035-step / 32.00B-token target, so its run dir is no longer live and is read
  # AND written directly -- no staging needed.  This is the FINAL checkpoint of the arm and
  # supersedes the step-58000 staged row above (which was taken mid-run and is kept only as
  # an intermediate data point).  It is safe from the trainer's max_to_keep pruning because
  # watchdog_loop.sh:249 marks an arm DONE once cur_step >= num_training_steps and never
  # resubmits it, and because pruning only ever removes dirs named global_step*.
  "scale32B_gptswitch_final|${RES}/iclr_scale/scale32B_gptswitch|61035|${RES}/iclr_scale/scale32B_gptswitch/unsharded_step61035"
)
# DELIBERATELY EXCLUDED: iclr_balance/pure_hop_isoP_bal_DIVERGED_unclamped_bias_20260914
# -- that arm diverged (unclamped load-balance bias); it is not a valid result.

FILTER=("$@")   # optional list of arm names; empty => all arms
selected() {
  local name="$1" f
  [ "${#FILTER[@]}" -eq 0 ] && return 0
  for f in ${FILTER[@]+"${FILTER[@]}"}; do [ "$f" = "$name" ] && return 0; done
  return 1
}

has_results() { ls "$1"/harness_results_*.json >/dev/null 2>&1; }
is_unsharded() { [ -f "$1/model.safetensors" ] || [ -f "$1/pytorch_model.bin" ]; }
results_json() { ls -t "$1"/harness_results_*.json 2>/dev/null | grep -v harness_bbh | head -1; }

submit_one() {
  local name="$1" lp="$2" step="$3" ud="$4"
  local job="av11_${name}"
  local js; js=$(mktemp "/tmp/${job}_XXXXXX.sh")
  cat > "${js}" <<JOBEOF
#!/bin/bash
set -uo pipefail
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate -q
source ${SD}/eval_tasks.sh
eval_env
eval_assert_pin
cd ${REPO}
echo "=== [${name}] host=\$(hostname) step=${step} load_path=${lp}"

# ---- 1. unshard (idempotent) --------------------------------------------------
if [ ! -f "${ud}/model.safetensors" ] && [ ! -f "${ud}/pytorch_model.bin" ]; then
    echo "=== [${name}] unsharding global_step${step} -> ${ud}"
    UCFG="/tmp/unshard_${name}_${step}_\$\$.yml"
    printf "load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${lp}" "${step}" "${ud}" > "\${UCFG}"
    python -m lm_engine.unshard --config "\${UCFG}" || { echo "FATAL [${name}]: unshard failed"; exit 1; }
    rm -f "\${UCFG}"
    echo "=== [${name}] unshard done"
else
    echo "=== [${name}] already unsharded"
fi

# ---- 2. FULL canonical harness eval (11 Avg11 tasks + mmlu + gsm8k + gsm8k_cot + wikitext)
echo "=== [${name}] tasks: \${EVAL_TASKS}"
python ${SD}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${ud},dtype=bfloat16,trust_remote_code=True" \\
    --tasks "\${EVAL_TASKS}" \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "${ud}/harness_results.json" || { echo "FATAL [${name}]: harness eval failed"; exit 1; }
echo "=== [${name}] harness eval done"

# ---- 3. canonical Avg11 -------------------------------------------------------
python ${SD}/compute_avg11.py "${ud}" || true
echo "=== [${name}] COMPLETE"
JOBEOF
  bsub -q preemptable -G grp_preemptable -J "${job}" \
       -gpu "num=1/task:mode=exclusive_process" -n 1 -M 64G -W 04:00 \
       -o "${HOME}/bsub_logs/${job}_%J.stdout" \
       -e "${HOME}/bsub_logs/${job}_%J.stderr" < "${js}"
  rm -f "${js}"
  echo "submitted ${job}  step=${step}  -> ${ud}"
}

case "${MODE}" in
  verify)
    for row in "${CKPTS[@]}"; do
      IFS='|' read -r name lp step ud <<< "${row}"
      selected "${name}" || continue
      sh="no"; [ -d "${lp}/global_step${step}/model" ] && sh="yes"
      un="no"; is_unsharded "${ud}" && un="yes"
      hr="no"; has_results "${ud}" && hr="yes"
      printf "  %-20s shard=%-3s unsharded=%-3s results=%-3s  step=%-6s %s\n" \
             "${name}" "${sh}" "${un}" "${hr}" "${step}" "${lp}"
    done
    ;;
  submit|resubmit)
    n=0
    for row in "${CKPTS[@]}"; do
      IFS='|' read -r name lp step ud <<< "${row}"
      selected "${name}" || continue
      if has_results "${ud}"; then echo "skip ${name}: already has harness_results_*.json"; continue; fi
      if [ ! -d "${lp}/global_step${step}/model" ]; then
        echo "SKIP ${name}: no shard at ${lp}/global_step${step}/model" >&2; continue
      fi
      submit_one "${name}" "${lp}" "${step}" "${ud}"; n=$((n+1))
    done
    echo "=== submitted ${n} job(s) (mode=${MODE}) ==="
    ;;
  status)
    bjobs -w 2>/dev/null | grep -E 'JOBID|av11_' || echo "(no av11_* jobs in queue)"
    echo "---"
    for row in "${CKPTS[@]}"; do
      IFS='|' read -r name lp step ud <<< "${row}"
      j=$(results_json "${ud}")
      printf "  %-20s %s\n" "${name}" "${j:-<no results yet>}"
    done
    ;;
  report)
    for row in "${CKPTS[@]}"; do
      IFS='|' read -r name lp step ud <<< "${row}"
      selected "${name}" || continue
      j=$(results_json "${ud}")
      echo "################ ${name}  (global_step${step})"
      if [ -z "${j}" ]; then echo "  NO RESULTS -- not evaluated"; continue; fi
      python "${SD}/compute_avg11.py" "${j}"
      echo "  json: ${j}"
      echo
    done
    ;;
  *) echo "usage: $0 {verify|submit|resubmit|status|report} [name ...]"; exit 2 ;;
esac
