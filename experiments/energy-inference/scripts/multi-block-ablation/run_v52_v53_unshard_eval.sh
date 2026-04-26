#!/bin/bash
# Unshard V52 and V53 EGPT cosreg models then submit harness eval jobs.
set -euo pipefail

REPO=/proj/dmfexp/nima/Code/dolomite-engine
JOB_NAME=egpt_v52v53_unshard

bsub \
    -q normal \
    -G grp_ebm \
    -J ${JOB_NAME} \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<BSUB_SCRIPT
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}

for VERSION in v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3 v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3; do
    SAVE_PATH=${REPO}/experiments/energy-inference/results/multi-block-ablation/\${VERSION}
    UNSHARDED_PATH=\${SAVE_PATH}/unsharded

    if [ -d "\${UNSHARDED_PATH}" ] && [ -f "\${UNSHARDED_PATH}/model.safetensors" ]; then
        echo "\${VERSION}: unsharded already exists, skipping"
    else
        echo "Unsharding \${VERSION} ..."
        UNSHARD_CONFIG="/tmp/\${VERSION}_unshard_\${LSB_JOBID}.yml"
        printf "load_args:\n  load_path: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
            "\${SAVE_PATH}" "\${UNSHARDED_PATH}" > "\${UNSHARD_CONFIG}"
        python -m lm_engine.unshard --config "\${UNSHARD_CONFIG}"
        rm -f "\${UNSHARD_CONFIG}"
        echo "\${VERSION}: unsharded to \${UNSHARDED_PATH}"
    fi

    echo "Submitting eval for \${VERSION} ..."
    bash ${REPO}/experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
        "\${UNSHARDED_PATH}" "eval_\${VERSION}"
done
BSUB_SCRIPT

echo "Submitted ${JOB_NAME} (unshard+eval for V52 and V53)."
echo "Logs: \$HOME/bsub_logs/${JOB_NAME}_<jobid>.stdout"
