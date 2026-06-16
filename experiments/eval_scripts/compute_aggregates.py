#!/usr/bin/env python3
"""compute_aggregates.py — print headline numbers from a harness_results.json.

Outputs:
  - avg10:        mean primary acc over the 10-task suite (uses `acc` for all)
  - avg10_norm:   mean over same 10 tasks but uses `acc_norm` whenever the task
                  reports it (arc_challenge/easy, hellaswag, openbookqa, piqa);
                  falls back to `acc` for tasks that only report `acc`
                  (boolq, copa, sciq, winogrande, mmlu).
  - WikiText word-perplexity
  - GSM8K (strict-match) and GSM8K-CoT (flexible-extract)
  - per-task acc + acc_norm (where available)

Usage:
  python compute_aggregates.py <path/to/harness_results.json> [--json]

If --json is passed, emits a machine-readable JSON dict instead of the
pretty-printed table.
"""
import sys
import json
import argparse
from pathlib import Path

# 10-task LM-harness suite (canonical)
TEN_TASKS = [
    "arc_challenge", "arc_easy", "boolq", "copa", "hellaswag",
    "openbookqa", "piqa", "sciq", "winogrande", "mmlu",
]

# Tasks where `acc_norm` is the standard normalized metric
ACC_NORM_TASKS = {"arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa"}


def first_metric(d, candidates):
    """Return value of the first key in `candidates` that exists in `d`, else None."""
    for k in candidates:
        if k in d:
            return d[k]
        # lm-eval sometimes appends ',none' or ',strict-match' etc.
        for full_key in d:
            if full_key.startswith(f"{k},"):
                return d[full_key]
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path", help="path to harness_results.json")
    ap.add_argument("--json", action="store_true", help="machine-readable JSON output")
    args = ap.parse_args()

    p = Path(args.path)
    if not p.exists():
        print(f"file not found: {p}", file=sys.stderr)
        sys.exit(1)
    blob = json.load(open(p))
    results = blob.get("results", blob)  # accept either format

    rows = []
    avg10_acc = []
    avg10_norm = []
    for task in TEN_TASKS:
        if task not in results:
            print(f"WARN: {task} not in results — skipped from avg", file=sys.stderr)
            continue
        d = results[task]
        acc = first_metric(d, ["acc"])
        acc_norm = first_metric(d, ["acc_norm"])
        rows.append((task, acc, acc_norm))
        if acc is not None:
            avg10_acc.append(acc)
            avg10_norm.append(acc_norm if (task in ACC_NORM_TASKS and acc_norm is not None) else acc)

    avg10 = sum(avg10_acc) / len(avg10_acc) if avg10_acc else float("nan")
    avg10n = sum(avg10_norm) / len(avg10_norm) if avg10_norm else float("nan")

    wt = results.get("wikitext", {})
    ppl = first_metric(wt, ["word_perplexity"])

    g8k = results.get("gsm8k", {})
    g8k_strict = first_metric(g8k, ["exact_match,strict-match", "exact_match"])
    g8k_cot = results.get("gsm8k_cot", {})
    g8k_flex = first_metric(g8k_cot, ["exact_match,flexible-extract", "exact_match"])

    if args.json:
        out = {
            "avg10": avg10,
            "avg10_norm": avg10n,
            "wikitext_word_ppl": ppl,
            "gsm8k_strict": g8k_strict,
            "gsm8k_cot_flex": g8k_flex,
            "per_task": {t: {"acc": a, "acc_norm": an} for (t, a, an) in rows},
        }
        print(json.dumps(out, indent=2))
        return

    # pretty-print
    print(f"== Aggregates from {p} ==")
    print(f"  avg10      = {avg10:.4f}  (using `acc` for all 10 tasks)")
    print(f"  avg10_norm = {avg10n:.4f}  (using `acc_norm` where available)")
    print(f"  WikiText   = {ppl:.2f} word-PPL" if ppl is not None else "  WikiText   = (not reported)")
    print(f"  GSM8K      = {g8k_strict:.4f} (strict)" if g8k_strict is not None else "  GSM8K      = (not reported)")
    print(f"  GSM8K-CoT  = {g8k_flex:.4f} (flexible-extract)" if g8k_flex is not None else "  GSM8K-CoT  = (not reported)")
    print()
    print(f"  {'task':<18s} {'acc':>8s} {'acc_norm':>10s}")
    for (t, a, an) in rows:
        a_s = f"{a:.4f}" if a is not None else "  --  "
        an_s = f"{an:.4f}" if an is not None else "    --    "
        print(f"  {t:<18s} {a_s:>8s} {an_s:>10s}")


if __name__ == "__main__":
    main()
