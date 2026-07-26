#!/bin/bash
# resume_v78_6gpt_1gptrec6x_d1280_20260726.sh
# v78 (RecGPT counterpart to V73, 6GPT+1GPTrec x6, d1280) stalled at step 23000/30000:
# its original auto-resubmit chain called the wrong script (run_u1_u4_universal_egpt.sh,
# a copy-paste bug) instead of resubmitting itself. This script resumes it correctly,
# auto-resubmitting on preemption until step 30000, then unshards + evals (main+BBH).
set -euo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SAVE="${REPO}/experiments/energy-inference/results/multi-block-ablation/v78_6gpt_1gptrec6x_d1280"
CONFIG="${REPO}/configs/multi_block_ablation/v78_6gpt_1gptrec6x_d1280.yml"
JOB_NAME="egpt_v78_resume"
mkdir -p "${HOME}/bsub_logs"

bsub \
    -q preemptable -G grp_preemptable -J "${JOB_NAME}" \
    -gpu "num=4/task:mode=exclusive_process" -n 1 -M 128G -W 04:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
LATEST="${SAVE}/latest_checkpointed_iteration.json"
TMPCONFIG="/tmp/v78_resume_\${LSB_JOBID}.yml"
cp "${CONFIG}" "\${TMPCONFIG}"
[ -f "\${LATEST}" ] && printf "\nload_args:\n  load_path: %s\n" "${SAVE}" >> "\${TMPCONFIG}"
bash ${REPO}/scripts/common/pretrain.sh "\${TMPCONFIG}"
rm -f "\${TMPCONFIG}"

ITER=\$(python3 -c "import json; print(json.load(open('\${LATEST}'))['latest_checkpointed_iteration'])")
if [ "\${ITER}" -lt 30000 ]; then
    bash ${REPO}/experiments/energy-inference/scripts/multi-block-ablation/resume_v78_6gpt_1gptrec6x_d1280_20260726.sh
else
    UNSHARDED="${SAVE}/unsharded"
    UCFG="/tmp/u_v78_\${LSB_JOBID}.yml"
    printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" "${SAVE}" "\${UNSHARDED}" > "\${UCFG}"
    python -m lm_engine.unshard --config "\${UCFG}" && rm -f "\${UCFG}"
    bash ${REPO}/experiments/eval_scripts/submit_eval.sh "\${UNSHARDED}" "v78_final_main"
    bash ${REPO}/experiments/eval_scripts/submit_bbh_eval.sh "\${UNSHARDED}" "v78_final_bbh"
fi
EOF
echo "v78 resume submitted."
