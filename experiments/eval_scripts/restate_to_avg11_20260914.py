#!/usr/bin/env python3
"""restate_to_avg11_20260914.py — restate every stored eval onto the CANONICAL Avg11.

This is the Avg11 successor to `restate_avg9_to_avg10_20260912.py`. Where that
script measured the avg9->avg10 gap, this one recomputes the colleague-consistent
**Avg11** (paper tab:scaling recipe) for every stored `harness_results_*.json`, so
the docs can be standardised on one number that is directly comparable to the
EGPT-RL / FET series.

Avg11 recipe (imported verbatim from compute_avg11.py so it can never drift):
    acc_norm : arc_challenge, arc_easy, hellaswag, openbookqa, piqa, sciq
    acc      : boolq, copa, winogrande, race, lambada_openai
    MMLU (acc), GSM8K-CoT (flexible-extract) and WikiText word-PPL reported
    SEPARATELY, never folded into the mean.

MISSING-TASK POLICY (identical to compute_avg11.py): a run that is missing any of
the 11 tasks is printed as `INCOMPLETE (k/11)` with the missing tasks named. It is
NEVER given a plain Avg11 number. race + lambada_openai were unrunnable before the
pyarrow>=20 fix of 2026-08-03, so most pre-August evals (the h1_/b*/c*/d* series)
come back INCOMPLETE and must be re-evaluated before an Avg11 can be quoted.

One row per checkpoint directory (the newest harness_results_*.json in it); this
keeps distinct step-XXXX checkpoints separate. harness_bbh_results_*.json ignored.

Pure stdlib (json/glob/argparse/pathlib) — safe to run directly on a compute node.

Usage:
    python experiments/eval_scripts/restate_to_avg11_20260914.py            # text table
    python experiments/eval_scripts/restate_to_avg11_20260914.py --md       # markdown
    python experiments/eval_scripts/restate_to_avg11_20260914.py --only h1 iclr
    python experiments/eval_scripts/restate_to_avg11_20260914.py --complete-only
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compute_avg11 import compute  # noqa: E402  (reuse the validated recipe)

REPO = Path("/proj/dmfexp/nima/Code/dolomite-engine")
SEARCH_ROOT = REPO / "experiments"


def _rank(p: Path):
    """Selection key: a merged *_avg11reeval.json (the canonical COMPLETE Avg11
    file = base 9 tasks + race + lambada) always wins over a sibling base eval;
    within a category, newest by filename (timestamps sort lexicographically).
    Without the avg11reeval priority, a base named harness_results_nogen_* or
    harness_results_36k.json sorts after the 2026-09-* merged file and shadows it."""
    return (1 if "avg11reeval" in p.name else 0, p.name)


def find_results() -> dict[Path, Path]:
    """Map each checkpoint dir -> its canonical harness_results_*.json."""
    best: dict[Path, Path] = {}
    for f in glob.glob(str(SEARCH_ROOT / "**" / "harness_results_*.json"), recursive=True):
        if "harness_bbh_results" in f:
            continue
        p = Path(f)
        d = p.parent
        if d not in best or _rank(p) > _rank(best[d]):
            best[d] = p
    return best


def run_label(json_path: Path) -> str:
    """Human-readable run id from the path, relative to the results roots."""
    rel = json_path.parent
    try:
        rel = rel.relative_to(SEARCH_ROOT)
    except ValueError:
        pass
    s = str(rel)
    for chop in ("boltzmann-moe/results/", "energy-inference/results/multi-block-ablation/",
                 "energy-inference/results/"):
        s = s.replace(chop, "")
    return s


def gather() -> list[dict]:
    rows = []
    for d, jf in find_results().items():
        try:
            blob = json.load(open(jf))
        except Exception as e:  # noqa: BLE001
            print(f"skip {jf}: {e}", file=sys.stderr)
            continue
        results = blob.get("results", blob)
        out = compute(results)
        out["label"] = run_label(jf)
        out["json"] = str(jf)
        rows.append(out)
    return rows


def fmt_pct(v):
    return f"{100*v:.2f}" if v is not None else "-"


def main() -> None:
    ap = argparse.ArgumentParser(description="Restate every stored eval onto canonical Avg11.")
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    ap.add_argument("--only", nargs="+", default=None, help="substring filter on run label")
    ap.add_argument("--complete-only", action="store_true",
                    help="only rows whose Avg11 is complete (11/11)")
    args = ap.parse_args()

    rows = gather()
    if args.only:
        rows = [r for r in rows if any(k in r["label"] for k in args.only)]
    if args.complete_only:
        rows = [r for r in rows if r["avg11_complete"]]

    # complete rows first (sorted by Avg11 desc), then incomplete (alpha)
    complete = sorted([r for r in rows if r["avg11_complete"]],
                      key=lambda r: -r["avg11"])
    incomplete = sorted([r for r in rows if not r["avg11_complete"]],
                        key=lambda r: r["label"])
    ordered = complete + incomplete

    def avg_cell(r):
        if r["avg11_complete"]:
            return f"{100*r['avg11']:.2f}"
        return f"INCOMPLETE ({r['n_present']}/11)"

    if args.md:
        print("| Run | Avg11 | MMLU | GSM8K-CoT | WikiPPL | missing |")
        print("|---|---:|---:|---:|---:|---|")
        for r in ordered:
            miss = ", ".join(r["missing"]) if r["missing"] else ""
            print(f"| `{r['label'][:64]}` | {avg_cell(r)} | {fmt_pct(r['mmlu'])} | "
                  f"{fmt_pct(r['gsm8k_cot_flex'])} | "
                  f"{r['wikitext_word_ppl']:.2f} | {miss} |"
                  if r["wikitext_word_ppl"] is not None else
                  f"| `{r['label'][:64]}` | {avg_cell(r)} | {fmt_pct(r['mmlu'])} | "
                  f"{fmt_pct(r['gsm8k_cot_flex'])} | - | {miss} |")
        print(f"\n_{len(complete)} complete / {len(incomplete)} incomplete "
              f"({len(ordered)} checkpoints)._")
        return

    print(f"{'run':70s} {'Avg11':>18s} {'MMLU':>6s} {'GSM8K':>6s} {'wikiPPL':>8s}  missing")
    print("-" * 130)
    for r in ordered:
        ppl = f"{r['wikitext_word_ppl']:.2f}" if r["wikitext_word_ppl"] is not None else "-"
        miss = ", ".join(r["missing"]) if r["missing"] else ""
        print(f"{r['label'][:70]:70s} {avg_cell(r):>18s} {fmt_pct(r['mmlu']):>6s} "
              f"{fmt_pct(r['gsm8k_cot_flex']):>6s} {ppl:>8s}  {miss}")
    print()
    print(f"{len(complete)} complete / {len(incomplete)} incomplete ({len(ordered)} checkpoints).")
    print("Avg11 = 11-task mean (acc_norm x6, acc x5); MMLU/GSM8K-CoT/WikiPPL separate.")
    print("INCOMPLETE rows are missing race and/or lambada_openai (pre-pyarrow>=20 evals):")
    print("re-run their harness eval before any Avg11 can be quoted.")


if __name__ == "__main__":
    main()
