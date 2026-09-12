"""Restate every stored eval onto ONE aggregate convention.

WHY THIS EXISTS
---------------
Two different averages are in circulation in this project and the documents do not
consistently say which they use:

  avg9   (OLD, pre-2026-06)  mean over 9 tasks, MMLU **EXCLUDED**, and acc_norm used
                             on all SIX tasks that report it (incl. sciq).
                             Used by: PROGRESS.md "Avg acc", CLAUDE.md, and
                             BOLTZ_MOE_BEST.md (which at least labels it `avg9` and
                             breaks MMLU out into its own column).
  avg10  (CURRENT)           compute_aggregates.py: mean over 10 tasks, MMLU
                             **INCLUDED**, acc_norm on FIVE tasks (sciq uses acc).
                             Used by: EGPT-action/RESULTS.md (the FET series).

Reverse-engineered from the five published h1 numbers simultaneously; the avg9
formula above reproduces all five to a worst-case error of 0.0004.

Consequence: h1_* / B-series / 585M / 680M figures are ~1.4-1.8pp HIGHER than they
would be under avg10, purely because MMLU at these scales sits near chance (~0.245)
and dragging it into the mean lowers it. **Any table mixing pre- and post-2026-06
numbers overstates the older runs by ~1.5pp.**

Usage:
  python experiments/eval_scripts/restate_avg9_to_avg10_20260912.py
  python experiments/eval_scripts/restate_avg9_to_avg10_20260912.py --md
"""

from __future__ import annotations

import argparse
import glob
import json
from pathlib import Path

REPO = Path("/proj/dmfexp/nima/Code/dolomite-engine")
SEARCH = [
    REPO / "experiments/boltzmann-moe/results",
    REPO / "experiments/energy-inference/results/multi-block-ablation",
]

TEN = ["arc_challenge", "arc_easy", "boolq", "copa", "hellaswag",
       "openbookqa", "piqa", "sciq", "winogrande", "mmlu"]
NINE = [t for t in TEN if t != "mmlu"]
NORM_NEW = {"arc_challenge", "arc_easy", "hellaswag", "openbookqa", "piqa"}
NORM_OLD = NORM_NEW | {"sciq"}          # the old convention also normalised sciq


def pick(res: dict, task: str, norm_set: set) -> float | None:
    d = res.get(task)
    if not d:
        return None
    if task in norm_set and d.get("acc_norm,none") is not None:
        return d["acc_norm,none"]
    return d.get("acc,none")


def aggregates(res: dict) -> dict:
    out = {}
    for label, tasks, ns in (("avg9_old", NINE, NORM_OLD), ("avg10_new", TEN, NORM_NEW)):
        v = [pick(res, t, ns) for t in tasks]
        have = [x for x in v if x is not None]
        out[label] = sum(have) / len(have) if len(have) == len(tasks) else None
        out[label + "_n"] = len(have)
    wt = res.get("wikitext", {})
    out["ppl"] = wt.get("word_perplexity,none")
    out["mmlu"] = (res.get("mmlu") or {}).get("acc,none")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--md", action="store_true", help="emit a markdown table")
    ap.add_argument("--only", nargs="+", default=None,
                    help="substring filter on run name")
    args = ap.parse_args()

    rows = []
    for base in SEARCH:
        for j in sorted(glob.glob(str(base / "*/unsharded/harness_results*.json"))):
            run = Path(j).parts[-3]
            try:
                d = json.load(open(j))
            except Exception:
                continue
            res = d.get("results", d)
            a = aggregates(res)
            if a["avg10_new"] is None:
                continue
            # one row per run: keep the newest harness json
            rows.append((run, a, j))
    best = {}
    for run, a, j in rows:
        if run not in best or j > best[run][1]:
            best[run] = (a, j)
    rows = [(r, v[0]) for r, v in best.items()]
    rows.sort(key=lambda r: -r[1]["avg10_new"])
    if args.only:
        rows = [r for r in rows if any(k in r[0] for k in args.only)]

    if args.md:
        print("| Run | avg9 (old, no MMLU) | avg10 (current) | Δ | MMLU | WikiPPL |")
        print("|---|---:|---:|---:|---:|---:|")
        for run, a in rows:
            d9 = f"{100*a['avg9_old']:.2f}" if a["avg9_old"] else "-"
            dl = (f"{100*(a['avg10_new']-a['avg9_old']):+.2f}" if a["avg9_old"] else "-")
            mm = f"{100*a['mmlu']:.2f}" if a["mmlu"] else "-"
            pp = f"{a['ppl']:.2f}" if a["ppl"] else "-"
            print(f"| `{run[:52]}` | {d9} | **{100*a['avg10_new']:.2f}** | {dl} | {mm} | {pp} |")
        return

    print(f"{'run':56s} {'avg9_old':>9s} {'avg10_new':>10s} {'delta':>7s} "
          f"{'MMLU':>6s} {'wikiPPL':>8s}")
    print("-" * 100)
    for run, a in rows:
        d9 = 100 * a["avg9_old"] if a["avg9_old"] else float("nan")
        d10 = 100 * a["avg10_new"]
        print(f"{run[:56]:56s} {d9:9.2f} {d10:10.2f} {d10-d9:+7.2f} "
              f"{100*(a['mmlu'] or 0):6.2f} {(a['ppl'] or 0):8.2f}")
    print()
    print("avg9_old  = 9 tasks, MMLU EXCLUDED, acc_norm on 6 (incl. sciq)  <- PROGRESS.md, CLAUDE.md,")
    print("            BOLTZ_MOE_BEST.md. Reproduces the 5 published h1 numbers to 0.0004.")
    print("avg10_new = 10 tasks, MMLU INCLUDED, acc_norm on 5 (sciq uses acc)  <- compute_aggregates.py,")
    print("            EGPT-action/RESULTS.md (FET series). THIS is the convention to standardise on.")
    print()
    print("The delta is negative because MMLU is near chance (~24.5%) at these scales, so")
    print("including it pulls the mean down. Mixing the two conventions in one table")
    print("overstates every pre-2026-06 run by roughly 1.5pp.")


if __name__ == "__main__":
    main()
