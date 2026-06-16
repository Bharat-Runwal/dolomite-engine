#!/bin/bash
# 24-GPU multi-node launcher for math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3.
# 3 nodes × 8 H100, normal queue / grp_ebm, blaunch + span[ptile=1] + exclude
# the broken host p4-r24-n2.
#
# Effective batch: mb=4 × ga=1 × 24 GPU = 393k tok/step (vs boltz 524k — 75%).
# LR 1e-3 transfers fine (sqrt scaling: optimal would be 0.87e-3, within noise).
# Total tokens: 320k steps × 393k tok/step = 126B target.
# iter_dropout 6±3 enabled on the EGPT block from step 0.
#
# DCP checkpoints are world-size-agnostic; can later upgrade to 32 GPU via the
# *_32gpu launcher pointed at the same save_path. Effective batch then jumps to
# 524k tok/step (boltz match) — a small mid-training batch shift but tolerable.
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_24gpu.yml
SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_24gpu
SCRIPT_PATH=${REPO}/experiments/energy-inference/scripts/multi-block-ablation/run_math_psda_h1_8gpt_1egpt6x_d1536_lra32_fpt_itd3_24gpu.sh
JOB_NAME=math_h1_d1536_fpt_itd3_24g
mkdir -p "${HOME}/bsub_logs"
bsub \
    -q normal -G grp_ebm -J ${JOB_NAME} \
    -gpu "num=8/task:mode=exclusive_process" -n 3 -x \
    -R "span[ptile=1] select[hname!='p4-r24-n2']" \
    -M 64G -W 24:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
unset TMPDIR TEMP TMP
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM_TRAINING_STEPS=320000
LATEST_JSON="\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CONFIG="\${CONFIG}"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    TMPCONFIG="\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
    cp "\${CONFIG}" "\${TMPCONFIG}"
    printf "\nload_args:\n  load_path: %s\n" "\${SAVE_PATH}" >> "\${TMPCONFIG}"
    RUN_CONFIG="\${TMPCONFIG}"
fi
echo "[head=\$(hostname)] LSB_MCPU_HOSTS=\$LSB_MCPU_HOSTS"
echo "[head] launching pretrain.sh on \$(echo \$LSB_MCPU_HOSTS | tr ' ' '\n' | sed 'n; d' | sort -u | wc -l) nodes via blaunch"
blaunch bash ${REPO}/scripts/common/pretrain.sh "\${RUN_CONFIG}"
[ -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml" ] && rm -f "\${SAVE_PATH}/runtime_config_\${LSB_JOBID}.yml"
if [ -f "\${LATEST_JSON}" ]; then
    LATEST_ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST_JSON}'))['latest_checkpointed_iteration'])")
    if [ "\${LATEST_ITER}" -lt "\${NUM_TRAINING_STEPS}" ]; then
        bash "\${SCRIPT_PATH}"
    else
        UNSHARDED="\${SAVE_PATH}/unsharded"
        UNSHARD_CFG="\${SAVE_PATH}/unshard_config_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\${SAVE_PATH}" "\${UNSHARDED}" > "\${UNSHARD_CFG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CFG}" && rm -f "\${UNSHARD_CFG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh "\${UNSHARDED}" "eval_\${JOB_NAME}"
        bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh "\${UNSHARDED}" "eval_bbh_\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME} (24 GPU multi-node, blaunch, iter_dropout 6±3, xma fused kernels)"
