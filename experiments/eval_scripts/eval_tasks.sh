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

# ---------------------------------------------------------------------------
# THE AGGREGATION CONVENTION (fixed 2026-09-13, on the user's instruction)
# ---------------------------------------------------------------------------
# avg10_norm = mean over the 10 tasks in EVAL_TASKS, using acc_norm for EVERY task that
# reports one and acc otherwise. Six tasks report acc_norm: arc_challenge, arc_easy,
# hellaswag, openbookqa, piqa, sciq. Four use acc: boolq, copa, winogrande, mmlu.
#
# WHY THIS NEEDED FIXING. Three conventions were in circulation and sciq was the only task
# they disagreed on (acc 0.762 vs acc_norm 0.680; spread over 10 tasks that is the ~0.8pp
# gap between our scripts and our tables). compute_aggregates.py's docstring said "acc_norm
# whenever the task reports it" while its code listed only five, omitting sciq.
#
# IMPORTANT: the vendored lm-evaluation-harness does NOT define an aggregate. It emits
# per-task acc and acc_norm only. Pinning it fixes the harness version and per-task scoring,
# NOT the averaging rule -- so there is no upstream convention to defer to. The rule lives
# in compute_aggregates.py (ACC_NORM_TASKS) and is mirrored by
# collect_flops_wave_20260912.sh's `table` mode. Change both together or neither.
#
# Verified 2026-09-13: all eight avg10 figures printed in the appendix reproduce exactly
# under this rule, so no published number changes.
