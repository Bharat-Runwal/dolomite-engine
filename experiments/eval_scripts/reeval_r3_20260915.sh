#!/bin/bash
# reeval_r3_20260915.sh
#
# R3 (paper tab:headline row) is the ONE cited checkpoint stored SHARDED
# (global_step36000/model/*.distcp) with no unsharded HF copy, and whose base
# eval is named harness_final_36k.json (not harness_results_*.json). Its base
# eval already contains all 9 Avg11 base tasks + mmlu + gsm8k + wikitext,
# validly scored (likelihood tasks are unaffected by the register-decode bug),
# and is MISSING only race + lambada_openai — exactly like the other 55.
#
# This single job chains, on one GPU:
#   1. unshard global_step36000 -> unsharded_step36000 (HF: config + safetensors)
#   2. build a slim base harness_results_36k.json inside the unsharded dir
#      (= harness_final_36k.json minus the 173 MB `samples` blob) so the standard
#      merge/compute/restate tooling (which globs harness_results_*.json) sees it
#   3. eval ONLY race + lambada_openai on the unsharded checkpoint
#   4. merge the two into the slim base -> harness_results_<ts>_avg11reeval.json
#   5. compute_avg11
set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SD="${REPO}/experiments/eval_scripts"
SAVE_PATH="${REPO}/experiments/energy-inference/results/multi-block-ablation/r3_11gpt_1egpt6x_rmsray_d1280"
STEP=36000
UNSHARDED="${SAVE_PATH}/unsharded_step${STEP}"
BASE_FULL="${SAVE_PATH}/harness_final_36k.json"
JOB="rl_r3"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J "${JOB}" \
    -gpu "num=1/task:mode=exclusive_process" -n 1 -M 64G -W 03:00 \
    -o "${HOME}/bsub_logs/${JOB}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB}_%J.stderr" \
    <<EOF
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

# 1. unshard (skip if already done)
if [ ! -f "${UNSHARDED}/model.safetensors" ] && [ ! -f "${UNSHARDED}/pytorch_model.bin" ]; then
    echo "=== [r3] unsharding step ${STEP} ==="
    UCFG="/tmp/unshard_r3_${STEP}_\$\$.yml"
    printf "load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
        "${SAVE_PATH}" "${STEP}" "${UNSHARDED}" > "\${UCFG}"
    python -m lm_engine.unshard --config "\${UCFG}" && rm -f "\${UCFG}"
    echo "=== [r3] unsharded -> ${UNSHARDED} ==="
else
    echo "=== [r3] already unsharded ==="
fi

# 2. slim base (strip the big per-sample 'samples' blob, keep all aggregate sections)
python3 - "${BASE_FULL}" "${UNSHARDED}/harness_results_36k.json" <<'PY'
import json,sys
src,dst=sys.argv[1],sys.argv[2]
d=json.load(open(src))
d.pop("samples",None)
json.dump(d,open(dst,"w"),indent=2)
r=d.get("results",{})
print("[r3] slim base written:",dst,"n_result_keys=",len(r),
      "has_race=",("race" in r),"has_lambada=",("lambada_openai" in r))
PY

# 3. race + lambada only
python ${SD}/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=${UNSHARDED},dtype=bfloat16,trust_remote_code=True" \\
    --tasks race,lambada_openai \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "${UNSHARDED}/racelambda.json"
echo "=== [r3] race+lambada eval done ==="

# 4. merge + 5. compute
python ${SD}/merge_racelambda_into_avg11_20260915.py "${UNSHARDED}"
echo "=== [r3] merge done ==="
python ${SD}/compute_avg11.py "${UNSHARDED}" || true
EOF
echo "submitted ${JOB}  ->  ${UNSHARDED}"
