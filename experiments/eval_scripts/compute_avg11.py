#!/usr/bin/env python3
"""compute_avg11.py — CANONICAL headline aggregator, consistent with colleagues.

This reproduces the **paper `tab:scaling` Avg11 recipe** used by the EGPT-RL /
FET series, so our numbers are directly comparable to theirs (no avg9/avg10 vs
avg11 apples-to-oranges — see the warning that used to live in the boltzmann-moe
CLAUDE.md).

Recipe (source of truth: ~/Code/GPT-experiments/projects/EGPT-RL/RESULTS.md:247-249):

    Avg11 = unweighted mean of ELEVEN tasks
        acc_norm : arc_challenge, arc_easy, hellaswag, openbookqa, piqa, sciq
        acc      : boolq, copa, winogrande, race, lambada_openai
    MMLU (acc) and GSM8K-CoT (flexible-extract) are reported SEPARATELY,
    NEVER folded into the average. WikiText word-PPL likewise separate.

How this differs from our old `compute_aggregates.py` avg10:
    - avg10 INCLUDED mmlu in the mean and EXCLUDED race + lambada_openai.
    - Avg11 EXCLUDES mmlu from the mean and INCLUDES race + lambada_openai.
    Net: drop mmlu, add race + lambada. Because race (~0.28) and lambada (~0.23)
    sit near chance at our scale, Avg11 runs ~3pp below the old avg10 — that gap
    is a scoring-convention artifact, not a model difference. Do NOT put avg10
    and Avg11 in the same table.

VALIDATION / PROVENANCE (2026-09-14): the recipe here (task list + metric-per-task)
matches the EGPT-RL colleague's exactly. On the numbers, be careful — there are
TWO distinct sets for the shared math_egptdual seeds, and an earlier draft of this
docstring conflated them (it claimed a +0.004pp reproduction of 49.35/49.67; that
claim was WRONG and has been removed):
  - The colleague's stored Avg11 (49.35 seed42, 49.67 seed1234) came from THEIR OWN
    complete-task eval (all 11 tasks incl. race+lambada). Those eval JSONs are NOT
    in this repo, so 49.35/49.67 are not reproducible here.
  - The in-repo step-dirs one naturally reaches for (seed42@16100, seed1234@16200)
    are 9/11 INCOMPLETE (race+lambada absent); this script correctly FLAGS them as
    "INCOMPLETE (9/11)" rather than averaging (their partial 9-task means are 53.14
    / 53.56 — NOT an Avg11, do not quote).
  - The COMPLETE final unsharded dirs for those seeds DO carry all 11 tasks and
    score, under this recipe, Avg11 = 51.88 (seed42) / 50.75 (seed1234). Those are
    the reproducible IN-REPO Avg11 values; use them for any in-repo cross-check.

Missing-task policy: if any of the 11 tasks is absent from the results file,
this script REFUSES to print a plain "Avg11". It prints "Avg11 INCOMPLETE (k/11)"
and names the missing tasks, so a partial average can never be mistaken for a
real one. (race + lambada_openai were unrunnable before the pyarrow>=20 fix of
2026-08-03; re-run those checkpoints if you see them flagged missing.)

Usage:
    python compute_avg11.py <harness_results.json | run_dir> [--json]

If given a directory, the latest harness_results_*.json in it (recursively) is
used, ignoring harness_bbh_results_*.json.
"""
import sys
import json
import glob
import argparse
from pathlib import Path

# --- The Avg11 task suite (metric per task matches lm-eval key convention) ---
ACC_NORM_TASKS = ["arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa", "sciq"]
ACC_TASKS      = ["boolq", "copa", "winogrande", "race", "lambada_openai"]
AVG11_TASKS    = [(t, "acc_norm") for t in ACC_NORM_TASKS] + [(t, "acc") for t in ACC_TASKS]
assert len(AVG11_TASKS) == 11


def metric(task_dict, key):
    """lm-eval stores metrics as 'acc,none' / 'acc_norm,none' etc. Accept either."""
    if task_dict is None:
        return None
    if f"{key},none" in task_dict:
        return task_dict[f"{key},none"]
    if key in task_dict:
        return task_dict[key]
    for full in task_dict:                      # any '<key>,<filter>' variant
        if full.startswith(f"{key},"):
            return task_dict[full]
    return None


def resolve_results_path(p: Path) -> Path:
    if p.is_dir():
        cands = [f for f in glob.glob(str(p / "**" / "harness_results_*.json"), recursive=True)
                 if "harness_bbh_results" not in f]
        if not cands:
            print(f"no harness_results_*.json under {p}", file=sys.stderr)
            sys.exit(1)
        # A merged *_avg11reeval.json (base 9 tasks + race + lambada) is the
        # canonical COMPLETE Avg11 file and MUST win over any sibling base eval,
        # regardless of lexicographic order (a base named e.g. harness_results_
        # nogen_* or harness_results_36k.json otherwise sorts after 2026-09-*).
        cands.sort(key=lambda f: (1 if "avg11reeval" in Path(f).name else 0, Path(f).name))
        return Path(cands[-1])
    return p


def compute(results: dict):
    per_task, present, missing = {}, [], []
    for task, key in AVG11_TASKS:
        v = metric(results.get(task), key)
        per_task[task] = (key, v)
        (present if v is not None else missing).append(task)

    vals = [per_task[t][1] for t in present]
    avg11 = sum(vals) / len(vals) if vals else float("nan")

    mmlu = metric(results.get("mmlu"), "acc")
    ppl = metric(results.get("wikitext"), "word_perplexity")
    gsm_cot = metric(results.get("gsm8k_cot"), "exact_match")  # flexible-extract variant caught by prefix match
    # prefer the explicit flexible-extract filter when present
    gsm_cot_flex = metric(results.get("gsm8k_cot"), "exact_match,flexible-extract") or gsm_cot

    return {
        "avg11": avg11,
        "avg11_complete": len(missing) == 0,
        "n_present": len(present),
        "missing": missing,
        "mmlu": mmlu,
        "wikitext_word_ppl": ppl,
        "gsm8k_cot_flex": gsm_cot_flex,
        "per_task": {t: {"metric": per_task[t][0], "value": per_task[t][1]} for t in per_task},
    }


def main():
    ap = argparse.ArgumentParser(description="Canonical Avg11 aggregator (colleague-consistent).")
    ap.add_argument("path", help="harness_results.json OR a run directory")
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    args = ap.parse_args()

    p = resolve_results_path(Path(args.path))
    if not p.exists():
        print(f"file not found: {p}", file=sys.stderr)
        sys.exit(1)
    blob = json.load(open(p))
    results = blob.get("results", blob)
    out = compute(results)

    if args.json:
        print(json.dumps(out, indent=2))
        return

    print(f"== Avg11 aggregate from {p} ==")
    if out["avg11_complete"]:
        print(f"  Avg11      = {100*out['avg11']:.2f}   (11-task mean, paper tab:scaling recipe)")
    else:
        print(f"  Avg11      = INCOMPLETE ({out['n_present']}/11) — DO NOT quote as Avg11")
        print(f"               partial mean = {100*out['avg11']:.2f} over {out['n_present']} tasks")
        print(f"               MISSING: {', '.join(out['missing'])}")
        print(f"               (race/lambada need pyarrow>=20; re-run this checkpoint's eval)")
    mmlu = out["mmlu"]; ppl = out["wikitext_word_ppl"]; gsm = out["gsm8k_cot_flex"]
    print(f"  MMLU       = {100*mmlu:.2f}   (separate, acc)" if mmlu is not None else "  MMLU       = (not reported)")
    print(f"  GSM8K-CoT  = {100*gsm:.2f}   (separate, flexible-extract)" if gsm is not None else "  GSM8K-CoT  = (not reported)")
    print(f"  WikiText   = {ppl:.2f} word-PPL" if ppl is not None else "  WikiText   = (not reported)")
    print()
    print(f"  {'task':<16s} {'metric':<9s} {'value':>7s}")
    for t, key in AVG11_TASKS:
        v = out["per_task"][t]["value"]
        vs = f"{100*v:.2f}" if v is not None else "  --  "
        print(f"  {t:<16s} {key:<9s} {vs:>7s}")


if __name__ == "__main__":
    main()
