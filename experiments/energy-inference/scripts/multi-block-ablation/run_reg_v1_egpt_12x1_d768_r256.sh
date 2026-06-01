#!/bin/bash
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
CONFIG=${REPO}/configs/multi_block_ablation/reg_v1_egpt_12x1_d768_r256_lr2e3.yml
SAVE_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/reg_v1_egpt_12x1_d768_r256
SCRIPT_PATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/run_reg_v1_egpt_12x1_d768_r256.sh
JOB_NAME=reg_v1_egpt_12x1_d768_r256
bsub -q preemptable -G grp_preemptable -J ${JOB_NAME} \
    -gpu "num=4/task:mode=exclusive_process" -n 1 -M 64G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\\${PYTHONPATH:-}
REPO=${REPO}; CONFIG=${CONFIG}; SAVE_PATH=${SAVE_PATH}
SCRIPT_PATH=${SCRIPT_PATH}; JOB_NAME=${JOB_NAME}; NUM=30000
LATEST="\\${SAVE_PATH}/latest_checkpointed_iteration.json"
RUN_CFG="\\${CONFIG}"
if [ -f "\\${LATEST}" ]; then
    TMP="/tmp/\\${JOB_NAME}_\\${LSB_JOBID}.yml"
    cp "\\${CONFIG}" "\\${TMP}"; printf "\nload_args:\n  load_path: %s\n" "\\${SAVE_PATH}" >> "\\${TMP}"
    RUN_CFG="\\${TMP}"
fi
bash ${REPO}/scripts/common/pretrain.sh "\\${RUN_CFG}"
[ -f "/tmp/\\${JOB_NAME}_\\${LSB_JOBID}.yml" ] && rm -f "/tmp/\\${JOB_NAME}_\\${LSB_JOBID}.yml"
if [ -f "\\${LATEST}" ]; then
    STEP=\\$(python3 -c "import json; print(json.load(open('\\${LATEST}'))['latest_checkpointed_iteration'])")
    if [ "\\${STEP}" -lt "\\${NUM}" ]; then bash "\\${SCRIPT_PATH}"; else
        UNSHARDED="\\${SAVE_PATH}/unsharded"
        UCFG="/tmp/u_\\${JOB_NAME}_\\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "\\${SAVE_PATH}" "\\${UNSHARDED}" > "\\${UCFG}"
        python -m lm_engine.unshard --config "\\${UCFG}" && rm -f "\\${UCFG}"
        bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh "\\${UNSHARDED}" "eval_\\${JOB_NAME}"
    fi
fi
BSUB
echo "Submitted ${JOB_NAME}"
