r"""
bbh_rescore.py — fix BBH exact_match=0 caused by missing extractor on bbh_fewshot.

Background
----------
The upstream lm-evaluation-harness `bbh_fewshot` task config has **no
`filter_list`** on its `exact_match` metric. The metric is therefore a
verbatim Python `==` between the model's raw continuation and the gold
target string. With `doc_to_text: "Q: {{input}}\nA:"` and `target_delimiter: " "`,
a model that copies the few-shot demo format produces text like
`" False"` (note the leading space), trailing punctuation, or a longer
continuation before the stop sequence. None of these match the gold
string `"False"` byte-for-byte ⇒ em=0 on **all** 27 subtasks for every
small model we evaluate.

The parallel `bbh_zeroshot` task DOES wire up a per-subtask `flexible-extract`
filter that picks the right answer pattern (`\b(True|False)\b`,
`\(([A-Z])\)`, an integer, etc.) and trims punctuation. We port those
per-subtask regexes here and apply them to the **raw 3-shot generations**
saved via `--log_samples`. This preserves the 3-shot prompting (which
small models need) while fixing the extraction so em is comparable.

Usage
-----
    python bbh_rescore.py <samples_dir>

    samples_dir should contain files of the form
    `samples_bbh_fewshot_<subtask>_<UTC>.jsonl` produced by
    `lm_eval --log_samples --output_path <samples_dir>/...`.

Writes
------
- `<samples_dir>/bbh_rescore_summary.json` — per-subtask em (rescored).
- `<samples_dir>/<orig_harness_bbh_results>.rescored.json` — copy of the
  original LM-harness JSON with `results[*]['exact_match,none']` patched
  to the rescored values.

If the LM-harness JSON for the same run is in the same directory as the
samples (the convention `submit_bbh_eval.sh` uses), it is auto-detected.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

# ── Per-subtask flexible-extract regex (ported from
# lm_eval/tasks/bbh/zeroshot/<subtask>.yaml, `flexible-extract` filter,
# with `group_select` semantics. We just re.findall and take the last
# (or first, per group_select) match.

# Each entry: (regex_pattern, group_select). group_select=-1 means last match,
# 0 means first match, None means take_first (string).
_BBH_REGEX: dict[str, tuple[str, int]] = {
    # Yes/No/True/False/valid-invalid (case-sensitive boundary match).
    "boolean_expressions": (r"\b(True|False)\b", 0),
    "web_of_lies": (r"\b(Yes|No|yes|no)\b", 0),
    "navigate": (r"\b(Yes|No|yes|no)\b", 0),
    "sports_understanding": (r"\b(yes|no)\b", 0),
    "formal_fallacies": (r"\b(valid|invalid)\b", 0),
    "causal_judgement": (r"\b(Yes|No|yes|no)\b", 0),
    # Letter multiple-choice. Use [A-Z] uniformly to match upstream lm_eval
    # bbh_zeroshot regex. The capture group strips the parens so we compare
    # the bare letter to the bare letter (gold "(A)" → "A"; pred "...is (A)" → "A").
    "date_understanding": (r"\(([A-Z])\)", 0),
    "disambiguation_qa": (r"\(([A-Z])\)", 0),
    "geometric_shapes": (r"\(([A-Z])\)", 0),
    "hyperbaton": (r"\(([A-Z])\)", 0),
    "logical_deduction_five_objects": (r"\(([A-Z])\)", 0),
    "logical_deduction_seven_objects": (r"\(([A-Z])\)", 0),
    "logical_deduction_three_objects": (r"\(([A-Z])\)", 0),
    "movie_recommendation": (r"\(([A-Z])\)", 0),
    "penguins_in_a_table": (r"\(([A-Z])\)", 0),
    "reasoning_about_colored_objects": (r"\(([A-Z])\)", 0),
    "ruin_names": (r"\(([A-Z])\)", 0),
    "salient_translation_error_detection": (r"\(([A-Z])\)", 0),
    "snarks": (r"\(([A-Z])\)", 0),
    "temporal_sequences": (r"\(([A-Z])\)", 0),
    "tracking_shuffled_objects_three_objects": (r"\(([A-Z])\)", 0),
    "tracking_shuffled_objects_five_objects": (r"\(([A-Z])\)", 0),
    "tracking_shuffled_objects_seven_objects": (r"\(([A-Z])\)", 0),
    # Integer-answer subtasks. matches "-?\d+" (object_counting has only non-negative).
    "multistep_arithmetic_two": (r"-?\d+", 0),
    "object_counting": (r"-?\d+", 0),
    # dyck_languages — closing-bracket sequence ("] ]", "} ]"). Generic compare.
    # word_sorting — space-separated sorted words. Generic compare.
}

# Subtasks whose target uses different framing than the flexible-extract regex.
# For each, we need to know how to *also* normalize the gold target so we can
# compare apples-to-apples after extraction. Convention: for "letter" subtasks
# the gold is like "(A)" with the parens — we extract the bare letter from
# both sides. For True/False the gold is "True"/"False" verbatim. For the
# yes/no ones the gold is typically "Yes"/"No" verbatim.

# For all subtasks not in _BBH_REGEX, fall through to "first non-empty line,
# strip trailing punct" matching after case-folding.


# Per-subtask target normalizer. If we extracted "A" from "(A)", normalize the
# gold "(A)" to "A" before comparison. Default: identity (case-fold + strip).
def _normalize_pair(subtask: str, pred: str, gold: str) -> tuple[str, str]:
    """Return (pred_norm, gold_norm) ready for == comparison.

    Strategy:
      1. If subtask has a known regex in _BBH_REGEX, use it on both pred
         and gold (the gold for "(A)" subtasks is literally "(A)"; the
         regex extracts the letter "A" from both — fair comparison).
      2. Otherwise, generic normalize: case-fold, strip whitespace, strip
         trailing punctuation, take first non-empty line.
    """
    if subtask in _BBH_REGEX:
        pattern, group_select = _BBH_REGEX[subtask]
        m_pred = re.findall(pattern, pred)
        m_gold = re.findall(pattern, gold)
        if not m_pred:
            return ("", _take_last_or_first(m_gold, group_select))
        if not m_gold:
            # gold should always match its own regex; if it doesn't, fall
            # through to generic
            return _generic_normalize_pair(pred, gold)
        return (
            _take_last_or_first(m_pred, group_select),
            _take_last_or_first(m_gold, group_select),
        )
    # Generic fallback for dyck_languages, word_sorting, anything not listed.
    return _generic_normalize_pair(pred, gold)


def _take_last_or_first(matches: list[Any], group_select: int) -> str:
    if not matches:
        return ""
    idx = group_select if group_select >= 0 else len(matches) + group_select
    # findall returns either str (single group) or tuple (multiple groups)
    item = matches[idx if 0 <= idx < len(matches) else 0]
    if isinstance(item, tuple):
        # take the first non-empty group
        for s in item:
            if s:
                return s
        return ""
    return item


def _generic_normalize_pair(pred: str, gold: str) -> tuple[str, str]:
    def norm(s: str) -> str:
        s = s.strip()
        # first non-empty line
        for line in s.splitlines():
            line = line.strip()
            if line:
                s = line
                break
        # strip trailing punctuation
        s = re.sub(r"[.\,;:\!\?\"\']+$", "", s)
        return s.casefold()

    return (norm(pred), norm(gold))


def _resp_to_text(resp: Any) -> str:
    """Extract the generated string from a samples_*.jsonl `resps` entry.

    The format is `resps: [[ "<text>" ]]` (list-of-list-of-str) for
    generate_until tasks. `filtered_resps: [ "<text>" ]` for tasks with
    no filter chain (bbh_fewshot) post-filter is identical to raw.
    """
    if isinstance(resp, str):
        return resp
    if isinstance(resp, list):
        if not resp:
            return ""
        if isinstance(resp[0], str):
            return resp[0]
        # list of lists
        return _resp_to_text(resp[0])
    return str(resp)


def rescore_subtask(samples_path: Path, subtask: str) -> tuple[float, int, int]:
    """Return (rescored_em, n_correct, n_total)."""
    n_total = 0
    n_correct = 0
    with samples_path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            target = str(row.get("target", ""))
            # prefer filtered_resps (post any filter), fall back to resps
            pred_field = row.get("filtered_resps") or row.get("resps") or [""]
            pred = _resp_to_text(pred_field)
            pred_n, gold_n = _normalize_pair(subtask, pred, target)
            n_total += 1
            if pred_n == gold_n and gold_n != "":
                n_correct += 1
    em = n_correct / n_total if n_total else 0.0
    return em, n_correct, n_total


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("samples_dir", type=Path,
                    help="Directory containing samples_bbh_fewshot_<subtask>_*.jsonl files")
    ap.add_argument("--show-examples", type=int, default=3,
                    help="Print this many raw pred/target/extracted examples per subtask")
    args = ap.parse_args()

    samples_dir: Path = args.samples_dir
    if not samples_dir.is_dir():
        print(f"[bbh_rescore] not a directory: {samples_dir}", file=sys.stderr)
        return 2

    sample_files = sorted(samples_dir.glob("samples_bbh_fewshot_*.jsonl"))
    if not sample_files:
        print(f"[bbh_rescore] no samples_bbh_fewshot_*.jsonl in {samples_dir}",
              file=sys.stderr)
        return 3

    print(f"[bbh_rescore] found {len(sample_files)} samples files")

    summary: dict[str, dict[str, float]] = {}
    for sf in sample_files:
        # samples_bbh_fewshot_<subtask>_<UTC>.jsonl
        stem = sf.stem  # samples_bbh_fewshot_boolean_expressions_2026-06-...
        name = stem[len("samples_bbh_fewshot_") :] if stem.startswith("samples_bbh_fewshot_") else stem
        # subtask = first underscore-separated tokens up to the timestamp.
        # The timestamp begins with 4-digit year; cut there.
        m = re.match(r"^(.*?)_(\d{4}-\d{2}-\d{2}T.*)$", name)
        subtask = m.group(1) if m else name
        try:
            em, n_correct, n_total = rescore_subtask(sf, subtask)
        except Exception as e:
            print(f"[bbh_rescore] {subtask}: ERROR {e}")
            continue
        summary[subtask] = {
            "exact_match,none": em,
            "n_correct": n_correct,
            "n_total": n_total,
        }
        marker = "  " if em == 0 else "✓ "
        print(f"  {marker}{subtask:50s} em={em:.4f}  ({n_correct}/{n_total})")

        if args.show_examples > 0:
            with sf.open() as f:
                shown = 0
                for line in f:
                    if shown >= args.show_examples:
                        break
                    row = json.loads(line)
                    target = str(row.get("target", ""))
                    pred = _resp_to_text(row.get("filtered_resps") or row.get("resps") or [""])
                    pred_n, gold_n = _normalize_pair(subtask, pred, target)
                    print(f"      ex{shown}: raw_pred={pred!r:80s} "
                          f"target={target!r} ext_pred={pred_n!r} ext_gold={gold_n!r} "
                          f"match={pred_n == gold_n and gold_n != ''}")
                    shown += 1

    # Aggregate "bbh_fewshot" group em across the per-subtask em (equal-weighted,
    # matching lm_eval's `weight_by_size: true` ≈ mean when all subtasks have
    # similar size).
    if summary:
        agg_em = sum(s["exact_match,none"] for s in summary.values()) / len(summary)
        summary["__bbh_fewshot_macro_avg"] = {
            "exact_match,none": agg_em,
            "n_subtasks": len(summary),
        }
        print(f"\n[bbh_rescore] macro-avg em across {len(summary)-1} subtasks: {agg_em:.4f}")

    # Write rescored summary
    out_path = samples_dir / "bbh_rescore_summary.json"
    with out_path.open("w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[bbh_rescore] wrote {out_path}")

    # If a harness_bbh_results_*.json sits next to samples, patch a copy.
    harness_jsons = sorted(samples_dir.glob("harness_bbh_results_*.json"))
    # Skip already-rescored copies
    harness_jsons = [p for p in harness_jsons if not p.name.endswith(".rescored.json")]
    if harness_jsons:
        src = harness_jsons[-1]  # most recent
        with src.open() as f:
            data = json.load(f)
        # Patch results[bbh_fewshot_<subtask>]
        for subtask, vals in summary.items():
            if subtask.startswith("__"):
                continue
            task_name = f"bbh_fewshot_{subtask}"
            if task_name in data.get("results", {}):
                data["results"][task_name]["exact_match,none"] = vals["exact_match,none"]
        # Patch group-level bbh_fewshot
        if "bbh_fewshot" in data.get("results", {}):
            macro = summary.get("__bbh_fewshot_macro_avg", {}).get("exact_match,none", 0.0)
            data["results"]["bbh_fewshot"]["exact_match,none"] = macro
        rescored_path = src.with_suffix(".rescored.json")
        with rescored_path.open("w") as f:
            json.dump(data, f, indent=2)
        print(f"[bbh_rescore] wrote {rescored_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
