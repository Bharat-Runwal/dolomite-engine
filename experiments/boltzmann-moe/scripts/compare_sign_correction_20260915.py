#!/usr/bin/env python3
"""compare_sign_correction_20260915.py -- pair each corrected-sign + Sinkhorn rerun arm
against the ORIGINAL (inverted-sign) arm it replaces, and report the Avg11 delta.

Why this exists: the ICLR tables currently hold Avg11 from inverted-sign checkpoints. The
paper will report the corrected model, so every energy-routed row is being re-measured.
What the write-up needs is not two independent tables but the PAIRED delta -- how far the
sign correction moved each row -- because that is the number a reviewer will ask about and
the number that decides whether any qualitative claim in the paper changes.

The metric is NOT reimplemented here. It is imported from the canonical
experiments/eval_scripts/compute_avg11.py, so this script cannot drift from the convention
the colleagues headline. MMLU and GSM8K stay OUT of the mean and are reported separately.

An arm with no harness_results yet prints PENDING rather than being silently dropped, so a
half-finished batch can never masquerade as a complete comparison.
"""
import sys, json, glob
from pathlib import Path

REPO = Path("/proj/dmfexp/nima/Code/dolomite-engine")
sys.path.insert(0, str(REPO / "experiments" / "eval_scripts"))
from compute_avg11 import AVG11_TASKS, metric  # canonical definition, do not redefine

RESULTS = REPO / "experiments/boltzmann-moe/results"
# rerun arm -> the published arm it replaces
PAIRS = {
    "iclr_sink/iclr_hop_K16_top2_sink":            "iclr_flops/iclr_hop_K16_top2",
    "iclr_sink/iclr_hop_K16_dense_sink":           "iclr_flops/iclr_hop_K16_dense",
    "iclr_sink/iclr_hop_K32_top2_sink":            "iclr_flops/iclr_hop_K32_top2",
    "iclr_sink/iclr_hop_K32_top1_sink":            "iclr_flops/iclr_hop_K32_top1",
    "iclr_sink/iclr_hop_K16_top2_nofix_sink":      "iclr_flops/iclr_hop_K16_top2_nofix",
    "iclr_sink/iclr_pure_hop_K16_top2_1blk_sink":  "iclr_flops/iclr_pure_hop_K16_top2_1blk",
    "iclr_sink/iclr_hop_K16_top2_renorm_sink":     "iclr_ctrl/iclr_hop_K16_top2_renorm",
    "iclr_sink/iclr_pure_hop_isoP_sink":           "iclr_1blk/iclr_pure_hop_isoP",
    "iclr_sink/pure_hop_isoP_bal_corr":            "iclr_balance/pure_hop_isoP_bal",
    "iclr_sink/iclr_big_hop_pure_sink":            "iclr_big/iclr_big_hop_pure",
    "iclr_sink/iclr_big_hop_sandwich_sink":        "iclr_big/iclr_big_hop_sandwich",
    "iclr_sink/pure_hop_T12_sink":                 "iclr_gptmoe/pure_hop_T12",
    "iclr_sink/slope90k_1blk_sink":                "iclr_slope/slope90k_1blk",
    "iclr_sink/slope90k_hyb_sink":                 "iclr_slope/slope90k_hyb",
}
# arms whose definition CHANGED in the rerun -- the delta is not like-for-like
CAVEAT = {
    "iclr_sink/iclr_hop_K16_top2_nofix_sink":
        "tau normalised to 1.0 removed one of the 4 reverted knobs; 3-knob revert at matched tau",
    "iclr_sink/pure_hop_isoP_bal_corr":
        "CONTROL arm: clamped proportional control, sinkhorn OFF (balance_rate conflict)",
}

def load(rel):
    js = sorted(glob.glob(str(RESULTS / rel / "unsharded" / "harness_results*.json")))
    if not js:
        return None
    d = json.load(open(js[-1]))
    return d.get("results", d)

def avg11(res):
    vals = []
    for task, key in AVG11_TASKS:
        if task not in res:
            return None, f"missing {task}"
        v = metric(res[task], key)
        if v is None:
            return None, f"no {key} for {task}"
        vals.append(v)
    return 100.0 * sum(vals) / len(vals), None

def sep(res, task, key):
    if task not in res: return None
    v = metric(res[task], key)
    return None if v is None else 100.0 * v

# --selftest: prove the metric path reproduces the PUBLISHED table before trusting any
# delta. Without this the script's old-arm path is only exercised once reruns exist, i.e.
# exactly when a silent metric drift would be hardest to notice. Values are from
# sec/experiments.tex tab:frontier (all Avg11).
PAPER_TRUTH = {
    "iclr_flops/iclr_hop_K16_top2":       43.91,
    "iclr_flops/iclr_hop_K32_top2":       44.38,
    "iclr_flops/iclr_hop_K32_top1":       44.12,
    "iclr_flops/iclr_hop_K16_dense":      44.78,
    "iclr_ctrl/iclr_hop_K16_top2_renorm": 43.74,
}
if "--selftest" in sys.argv:
    bad = 0
    print(f"{'published arm':40s} {'paper':>7} {'recomputed':>11} {'diff':>7}  verdict")
    for arm, want in PAPER_TRUTH.items():
        r = load(arm)
        got, err = (avg11(r) if r else (None, "no results"))
        if got is None:
            print(f"{arm:40s} {want:7.2f} {'FAIL':>11}  {err}"); bad += 1; continue
        d = got - want; ok = abs(d) < 0.02
        bad += (not ok)
        print(f"{arm:40s} {want:7.2f} {got:11.2f} {d:+7.2f}  {'MATCH' if ok else '*** MISMATCH ***'}")
    print(f"\n  {len(PAPER_TRUTH)-bad}/{len(PAPER_TRUTH)} reproduce the published Avg11 to <0.02pp")
    sys.exit(1 if bad else 0)

print(f"{'arm':34s} {'published':>10} {'corrected':>10} {'delta':>8}   {'MMLU':>6} {'GSM8K':>6}")
print("-" * 88)
rows, pending = [], []
for new, old in sorted(PAIRS.items()):
    rn, ro = load(new), load(old)
    name = new.split("/")[-1]
    if rn is None:
        pending.append((name, "rerun not evaluated yet")); continue
    an, en = avg11(rn)
    ao, eo = (avg11(ro) if ro else (None, "published arm has no results"))
    if an is None:
        pending.append((name, f"rerun INCOMPLETE: {en}")); continue
    d = None if ao is None else an - ao
    rows.append((name, ao, an, d, sep(rn, "mmlu", "acc"), sep(rn, "gsm8k_cot", "acc"), new))
    print(f"{name:34s} {('%.2f'%ao) if ao is not None else '   n/a':>10} {an:10.2f} "
          f"{('%+.2f'%d) if d is not None else '   n/a':>8}   "
          f"{('%.2f'%(rows[-1][4] or 0)) if rows[-1][4] is not None else '  n/a':>6} "
          f"{('%.2f'%(rows[-1][5] or 0)) if rows[-1][5] is not None else '  n/a':>6}")
print()
if rows:
    ds = [r[3] for r in rows if r[3] is not None]
    if ds:
        print(f"  {len(ds)} paired arms: mean delta {sum(ds)/len(ds):+.2f}pp, "
              f"range {min(ds):+.2f} .. {max(ds):+.2f}")
        print(f"  NOTE the replicate noise floor is ~0.038 in lm_loss; on Avg11 treat any")
        print(f"  |delta| under ~0.5pp as indistinguishable from seed noise at this scale.")
for n, why in pending:
    print(f"  PENDING {n}: {why}")
if CAVEAT:
    print("\n  NOT like-for-like:")
    for k, v in CAVEAT.items():
        print(f"    {k.split('/')[-1]}: {v}")
