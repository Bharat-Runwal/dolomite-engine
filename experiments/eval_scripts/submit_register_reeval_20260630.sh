#!/bin/bash
# submit_register_reeval_20260630.sh
#
# Re-runs BBH + main eval for all register checkpoints with the decode bug fixed.
# Bug: cached-decode used position_id = R+T+k instead of T+k (off-by-R RoPE).
# Fix: set register_generation_mode = "no_cache" in config.json.
# See: lm_engine/hf_models/models/register_energy/REGISTER_DECODE_BUG.md
#
# Wall times are doubled vs defaults because no_cache re-prefills R+T tokens
# at each decode step instead of appending one token to the KV cache.
# Main eval: 04:00 (was 02:00). BBH: 08:00 (was 04:00).

set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
BASE="${REPO}/experiments/energy-inference/results/multi-block-ablation"
SCRIPTS="${REPO}/experiments/eval_scripts"
mkdir -p "${HOME}/bsub_logs"

# All unsharded register checkpoints
CKPTS=(
    "reg_h1_6gpt_1egpt6x_d768_r128"
    "reg_h1_6gpt_1egpt6x_d768_r256"
    "reg_v0_gpt_12x1_d768_r128"
    "reg_v0_gpt_12x1_d768_r256"
    "reg_v1_400m_d1024_r128"
    "reg_v1_400m_d1024_r256"
    "reg_v1_egpt_12x1_d768_r128"
    "reg_v1_egpt_12x1_d768_r16"
    "reg_v1_egpt_12x1_d768_r256"
    "reg_v41_sandwich_2g8e2g_d768_r128"
    "reg_v56_1x12_d768_r128"
    "reg_v73_6gpt_1egpt6x_d1280_r128"
    "reg_v73_6gpt_1egpt6x_d1280_r256"
    "h1_sel_reg_128_d768"
    "v76_4gpt_1egpt6x_rmsray_d1024_reg128"
    "math_fet_hopfield_mean_r256_8gpt_1egpt6x_d1536_int8k_lra32_itd3_lr1p5e3_33b_16gpu"
    "scale_reg_v1_egpt_d768_r128_126b"
)

patch_config() {
    local cfg="$1"
    python3 - <<PYEOF
import json, sys
p = "$cfg"
c = json.load(open(p))
if c.get("register_generation_mode") != "no_cache":
    c["register_generation_mode"] = "no_cache"
    json.dump(c, open(p, "w"), indent=2)
    print(f"  patched: {p}")
else:
    print(f"  already no_cache: {p}")
PYEOF
}

# Submit main eval (13 tasks incl gsm8k) with longer wall time
submit_main() {
    local ckpt_dir="$1"
    local jname="$2"
    local tasks="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"

    bsub \
        -q preemptable -G grp_preemptable -J "${jname}" \
        -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 06:00 \
        -o "${HOME}/bsub_logs/${jname}_%J.stdout" \
        -e "${HOME}/bsub_logs/${jname}_%J.stderr" \
        <<EOF
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval -q
cd ${REPO}
python ${SCRIPTS}/eval_harness.py \
    --model hf \
    --model_args "pretrained=${ckpt_dir},dtype=bfloat16,trust_remote_code=True" \
    --tasks ${tasks} \
    --device cuda:0 \
    --batch_size 4 \
    --trust_remote_code \
    --output_path "${ckpt_dir}/harness_results_nocache.json"
echo "[main-eval] done"
EOF
    echo "  main-eval submitted: ${jname}"
}

# Submit BBH eval with longer wall time
submit_bbh() {
    local ckpt_dir="$1"
    local jname="$2"

    bsub \
        -q preemptable -G grp_preemptable -J "${jname}" \
        -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 12:00 \
        -o "${HOME}/bsub_logs/${jname}_%J.stdout" \
        -e "${HOME}/bsub_logs/${jname}_%J.stderr" \
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
SAMPLES_DIR="${ckpt_dir}/bbh_nocache_\${TIMESTAMP}"
mkdir -p "\${SAMPLES_DIR}"
python ${SCRIPTS}/eval_harness.py \
    --model hf \
    --model_args "pretrained=${ckpt_dir},dtype=bfloat16,trust_remote_code=True" \
    --tasks bbh_fewshot \
    --device cuda:0 \
    --batch_size 2 \
    --trust_remote_code \
    --log_samples \
    --output_path "\${SAMPLES_DIR}/harness_bbh_results_\${TIMESTAMP}.json"
echo "[bbh] lm-eval done — running rescorer"
SAMPLES_LEAF=\$(find "\${SAMPLES_DIR}" -maxdepth 3 -name "samples_bbh_fewshot_*.jsonl" -printf "%h\n" | sort -u | head -n 1)
if [ -z "\${SAMPLES_LEAF}" ]; then
    echo "[bbh_rescore] WARNING: no samples files found"
else
    HARNESS_JSON=\$(find "\${SAMPLES_DIR}" -maxdepth 3 -name "results_*.json" | head -n 1)
    [ -n "\${HARNESS_JSON}" ] && cp "\${HARNESS_JSON}" "\${SAMPLES_LEAF}/harness_bbh_results_\${TIMESTAMP}.json"
    [ -n "\${HARNESS_JSON}" ] && cp "\${HARNESS_JSON}" "${ckpt_dir}/harness_bbh_results_nocache_\${TIMESTAMP}.json"
    python ${SCRIPTS}/bbh_rescore.py "\${SAMPLES_LEAF}" --show-examples 3
    cp "\${SAMPLES_LEAF}"/harness_bbh_results_*.rescored.json "${ckpt_dir}/" 2>/dev/null || true
    cp "\${SAMPLES_LEAF}"/bbh_rescore_summary.json "${ckpt_dir}/bbh_rescore_summary_nocache_\${TIMESTAMP}.json" 2>/dev/null || true
fi
echo "[bbh] done"
EOF
    echo "  bbh submitted: ${jname}"
}

echo "=== Patching configs: bypass → no_cache ==="
for name in "${CKPTS[@]}"; do
    cfg="${BASE}/${name}/unsharded/config.json"
    if [ ! -f "$cfg" ]; then
        echo "  SKIP (no config): $name"
        continue
    fi
    patch_config "$cfg"
done

echo ""
echo "=== Submitting main evals ==="
for name in "${CKPTS[@]}"; do
    ckpt="${BASE}/${name}/unsharded"
    [ ! -d "$ckpt" ] && echo "  SKIP: $name" && continue
    # Shorten job name for LSF (max 30 chars)
    jname="nocache_$(echo $name | sed 's/reg_//;s/_6gpt.*//;s/_12x1.*//;s/_d[0-9]*.*//;s/_egpt//;s/_gpt//' | cut -c1-20)"
    submit_main "$ckpt" "${jname}_main"
done

echo ""
echo "=== Submitting BBH evals ==="
for name in "${CKPTS[@]}"; do
    ckpt="${BASE}/${name}/unsharded"
    [ ! -d "$ckpt" ] && echo "  SKIP: $name" && continue
    jname="nocache_$(echo $name | sed 's/reg_//;s/_6gpt.*//;s/_12x1.*//;s/_d[0-9]*.*//;s/_egpt//;s/_gpt//' | cut -c1-20)"
    submit_bbh "$ckpt" "${jname}_bbh"
done

echo ""
echo "=== Done. $(bjobs 2>/dev/null | grep -c nocache || echo 0) nocache jobs submitted ==="
echo "Monitor with: bjobs | grep nocache"
echo "Results at: <ckpt>/harness_results_nocache.json and bbh_rescore_summary_nocache_*.json"
