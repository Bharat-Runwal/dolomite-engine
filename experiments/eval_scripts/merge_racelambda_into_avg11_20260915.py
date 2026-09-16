#!/usr/bin/env python3
"""merge_racelambda_into_avg11_20260915.py

Non-destructively merge a 2-task race+lambada_openai eval into an existing 9-task
harness_results JSON so the result is a COMPLETE 11-task Avg11 file, WITHOUT
re-running the other 9 tasks and WITHOUT overwriting the original JSON (which a
colleague's pipeline may depend on).

Context: every scale/headline/MoE checkpoint on disk has a stored
`harness_results_<ts>.json` containing the 9 Avg11 base tasks + mmlu + gsm8k +
wikitext, but MISSING race + lambada_openai (they were unrunnable before the
pyarrow>=20 fix of 2026-08-03). Re-running only the two missing tasks (~15 min on
1 GPU) and merging is far cheaper than a full 15-task re-eval (~2 h) and does not
clobber the colleague's file.

Given a checkpoint dir:
  base = newest harness_results_*.json  (EXCLUDING *avg11reeval* and harness_bbh*)
  rl   = newest racelambda_*.json       (the 2-task eval produced by the launcher)
  out  = harness_results_<UTC-now>_avg11reeval.json   (base + race + lambada)

The output filename carries a current timestamp, so compute_avg11.py /
restate_to_avg11_20260914.py (which take the newest harness_results_*.json per dir)
pick it up automatically, and an explicit `_avg11reeval` marker so a human can see
at a glance that it is a merge, not a fresh 15-task eval.

Merges the two tasks into: results, versions, configs, n-samples, n-shot,
higher_is_better (all keyed by task name), and records provenance under
`_avg11_merge`. Everything else (model config, git_hash, the other tasks) is kept
verbatim from the base file.

Idempotent: safe to re-run. If the base already contains race+lambada it still
writes a fresh merged file (same values). Refuses only if rl lacks either task.

Usage:
    python merge_racelambda_into_avg11_20260915.py <checkpoint_dir> [--quiet]
    python merge_racelambda_into_avg11_20260915.py <checkpoint_dir> --check   # dry run
"""
import sys, os, json, glob, argparse, datetime
from pathlib import Path

RL_TASKS = ("race", "lambada_openai")
# per-task sections to carry across if present in the rl file
TASK_KEYED_SECTIONS = ("results", "versions", "configs", "n-samples", "n-shot",
                       "higher_is_better", "group_subtasks", "task_hashes")


def newest(paths):
    return sorted(paths)[-1] if paths else None


def find_base(ckpt: Path):
    cands = [f for f in glob.glob(str(ckpt / "harness_results_*.json"))
             if "harness_bbh_results" not in f and "avg11reeval" not in os.path.basename(f)]
    return newest(cands)


def find_rl(ckpt: Path):
    return newest(glob.glob(str(ckpt / "racelambda_*.json")))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", help="unsharded checkpoint dir")
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--check", action="store_true", help="report only, write nothing")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    if not ckpt.is_dir():
        print(f"NOT A DIR: {ckpt}", file=sys.stderr); sys.exit(2)

    base_p = find_base(ckpt)
    rl_p = find_rl(ckpt)
    if base_p is None:
        print(f"NO BASE harness_results_*.json in {ckpt}", file=sys.stderr); sys.exit(3)
    if rl_p is None:
        print(f"NO racelambda_*.json in {ckpt} (run the eval first)", file=sys.stderr); sys.exit(4)

    base = json.load(open(base_p))
    rl = json.load(open(rl_p))
    rl_res = rl.get("results", {})
    have = [t for t in RL_TASKS if t in rl_res and (f"acc,none" in rl_res[t])]
    if sorted(have) != sorted(RL_TASKS):
        print(f"racelambda file {rl_p} is missing a task's acc,none: has {have}", file=sys.stderr)
        sys.exit(5)

    if not args.quiet:
        print(f"  base : {os.path.basename(base_p)}")
        print(f"  rl   : {os.path.basename(rl_p)}")
        for t in RL_TASKS:
            print(f"    {t:16s} acc={rl_res[t].get('acc,none'):.4f}")

    if args.check:
        print("  [--check] no file written")
        return

    # merge the two tasks into every per-task-keyed section that exists in rl
    for section in TASK_KEYED_SECTIONS:
        if section in rl and isinstance(rl[section], dict):
            base.setdefault(section, {})
            for t in RL_TASKS:
                if t in rl[section]:
                    base[section][t] = rl[section][t]

    base["_avg11_merge"] = {
        "merged_tasks": list(RL_TASKS),
        "base_file": os.path.basename(base_p),
        "racelambda_file": os.path.basename(rl_p),
        "merged_utc": datetime.datetime.utcnow().isoformat(),
        "lm_eval_version_rl": rl.get("lm_eval_version"),
        "note": "race+lambda merged into a pre-pyarrow>=20 9-task eval; other tasks verbatim from base_file",
    }

    ts = datetime.datetime.utcnow().strftime("%Y-%m-%dT%H-%M-%S.%f")
    out_p = ckpt / f"harness_results_{ts}_avg11reeval.json"
    with open(out_p, "w") as fh:
        json.dump(base, fh, indent=2)
    if not args.quiet:
        print(f"  wrote: {out_p.name}")
    else:
        print(str(out_p))


if __name__ == "__main__":
    main()
