#!/usr/bin/env python3
"""Merge a split eval (likelihood tasks + gsm8k_cot) into one harness_results file.

WHY THIS EXISTS: gsm8k_cot generation is ~55-85 min of a >1.5 h 14-task eval, and on the
preemptable queue that whole job kept dying inside it (ev_cmix_134M_pure lost 51 min, then
another 90 min). Splitting gsm8k into its own job means a preemption costs one task.

But compute_avg11.py resolves a run dir to a SINGLE newest harness_results_*.json -- it does
not union across files. So the split must be re-joined here, into a file that IS the newest
under the main dir, while the gsm8k raw output is kept OUTSIDE that dir so it can never be
selected on its own (which would look like a complete eval missing 11 of 14 tasks).

Idempotent, and refuses to write a partial merge: if either side is missing it exits 0 having
done nothing, so whichever job finishes last performs the merge.
"""
import json, glob, sys, os, time

def newest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None

def main():
    if len(sys.argv) != 3:
        print("usage: merge_eval_results.py <main_unsharded_dir> <gsm_raw_dir>", file=sys.stderr)
        return 2
    main_dir, gsm_dir = sys.argv[1], sys.argv[2]
    a = newest(f'{main_dir}/harness_results_*.json')
    # gsm8k raw is named gsm8k_raw_*.json, NOT harness_results_*.json, so that
    # compute_avg11.py's recursive glob can never select a gsm8k-only file and read it as a
    # complete eval that is missing 11 of 14 tasks.
    b = newest(f'{gsm_dir}/gsm8k_raw_*.json') or newest(f'{gsm_dir}/harness_results_*.json')
    if not a or not b:
        print(f"merge deferred (likelihood={'ok' if a else 'missing'}, gsm8k={'ok' if b else 'missing'})")
        return 0
    # already merged and up to date?
    m = newest(f'{main_dir}/harness_results_merged_*.json')
    if m and os.path.getmtime(m) >= max(os.path.getmtime(a), os.path.getmtime(b)):
        print(f"merge up to date: {os.path.basename(m)}")
        return 0
    A, B = json.load(open(a)), json.load(open(b))
    # a merged file is itself a valid input; drop it from consideration by rebuilding from parts
    out = dict(A)
    ra, rb = A.get("results", {}), B.get("results", {})
    merged = dict(ra); merged.update(rb)
    out["results"] = merged
    for k in ("configs", "versions", "n-samples", "task_hashes", "higher_is_better"):
        if isinstance(A.get(k), dict) and isinstance(B.get(k), dict):
            d = dict(A[k]); d.update(B[k]); out[k] = d
    out["_merged_from"] = [os.path.basename(a), os.path.basename(b)]
    ts = time.strftime('%Y-%m-%dT%H-%M-%S')
    dst = f'{main_dir}/harness_results_merged_{ts}.json'
    json.dump(out, open(dst, 'w'), indent=1, default=str)
    print(f"merged {len(ra)} + {len(rb)} tasks -> {len(merged)} in {os.path.basename(dst)}")
    return 0

if __name__ == '__main__':
    sys.exit(main())
