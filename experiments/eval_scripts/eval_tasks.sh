#!/bin/bash
# eval_tasks.sh — SINGLE SOURCE OF TRUTH for our evaluation setup.
# Source this from every eval launcher:  source .../eval_scripts/eval_tasks.sh
#
# Before 2026-09-12 three scripts carried three different task lists, so "Avg" was
# not always over the same tasks. Do not hardcode a task list anywhere else.

REPO=/proj/dmfexp/nima/Code/dolomite-engine
EVAL_HARNESS="$REPO/experiments/eval_scripts/lm-evaluation-harness"

# 10 accuracy tasks that form avg10 / avg10_norm (see compute_aggregates.py),
# + wikitext (word perplexity), race and lambada_openai (reported, NOT in the avg),
# + gsm8k and gsm8k_cot (generative).
EVAL_TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,race,sciq,wikitext,winogrande,lambada_openai,mmlu,gsm8k,gsm8k_cot"

# Pinned harness at commit ad3f4d0c / v0.4.9.2 (see VENDORED_FROM.txt). PYTHONPATH
# must come FIRST so it shadows any lm_eval in site-packages. Do NOT pip install
# lm-eval — that pulls a different version and silently changes the pin.
eval_env() {
    export PYTHONPATH="$EVAL_HARNESS:${PYTHONPATH:-}"
    export HF_DATASETS_OFFLINE=1
    export HF_HUB_OFFLINE=1
    export TMPDIR="${TMPDIR:-/proj/dmfexp/nima/.cache/tmp}"; mkdir -p "$TMPDIR"
}

# PIN CHECK IS FATAL (2026-09-13). This used to sys.exit("WRONG HARNESS: ..."),
# returning 1 -- but every caller runs `set -uo pipefail` WITHOUT `-e`, so the message
# was printed and the eval continued against whatever lm_eval site-packages provided.
# The guard existed and was INEFFECTIVE. Now it kills the shell explicitly.
eval_assert_pin() {
    python - <<'PY'
import lm_eval, os, sys
want = "experiments/eval_scripts/lm-evaluation-harness"
got = os.path.dirname(os.path.dirname(lm_eval.__file__))
if want not in got:
    sys.exit(f"WRONG HARNESS: lm_eval resolved to {got}, expected the vendored {want}")
print(f"  harness OK: {got}")
PY
    local rc=$?
    if [ "$rc" -ne 0 ]; then
        echo "FATAL: harness pin check failed (rc=$rc); refusing to eval against the wrong lm_eval." >&2
        exit 1
    fi
}
