#!/usr/bin/env python3
"""Parse the boltz-accel A/B logs: steady-state s/step + loss curve per arm.

Startup (torch.compile, dataloader warm-up, the first few steps) is EXCLUDED --
the question is steady-state throughput, so we drop everything before
--skip-steps and report the median of what remains (median, not mean: a
preemptable host can stall one step and blow up an average).
"""
import argparse, glob, os, re, statistics as st
from collections import defaultdict

ARMS = ["accel_A_base", "accel_B_fused", "accel_C_fused_rep10", "accel_D_fused_rep10_proxy8"]
STEP = re.compile(r"step = (\d+),")
def num(key, line):
    m = re.search(re.escape(key) + r" = (-?[\d.]+(?:e-?\d+)?)", line)
    return float(m.group(1)) if m else None

def parse(arm):
    logs = sorted(glob.glob(os.path.expanduser(f"~/bsub_logs/{arm}_*.stderr")),
                  key=os.path.getmtime)
    if not logs:
        return None
    rows = {}
    for lg in logs:                      # later logs (resubmits) win per step
        with open(lg, errors="ignore") as f:
            for line in f:
                m = STEP.search(line)
                if not m or "train-loss" not in line:
                    continue
                rows[int(m.group(1))] = {
                    "t":     num("train-step_time (sec)", line),
                    "loss":  num("train-loss", line),
                    "aux":   num("train-aux_loss", line),
                    "gnorm": num("train-grad_norm", line),
                    "btpd":  num("train-billion_tokens_per_day", line),
                    "eff":   num("ffwd.load_effective_n_experts", line),
                    "agree": num("ffwd.proxy_topk_agree", line),
                }
    return dict(sorted(rows.items())) or None

ap = argparse.ArgumentParser()
ap.add_argument("--skip-steps", type=int, default=60,
                help="drop steps <= this (compile + warm-up) before timing")
a = ap.parse_args()

data = {arm: parse(arm) for arm in ARMS}
base_t = None
print(f"{'arm':30s} {'steps':>7s} {'s/step':>8s} {'vs A':>7s} {'Btok/d':>8s} "
      f"{'loss@last':>10s} {'aux':>8s} {'effK':>6s} {'proxy':>6s}")
print("-" * 100)
for arm in ARMS:
    d = data[arm]
    if not d:
        print(f"{arm:30s} {'--- no log yet ---':>40s}"); continue
    steady = [v["t"] for k, v in d.items() if k > a.skip_steps and v["t"]]
    med = st.median(steady) if steady else float("nan")
    if arm == ARMS[0] and steady:
        base_t = med
    ratio = f"{base_t/med:.2f}x" if (base_t and steady and med) else "-"
    last = max(d)
    v = d[last]
    ag = f"{v['agree']:.3f}" if v.get("agree") is not None else "-"
    print(f"{arm:30s} {last:>7d} {med:>8.3f} {ratio:>7s} {v['btpd'] or 0:>8.3f} "
          f"{v['loss']:>10.4f} {v['aux'] or 0:>8.4f} {v['eff'] or 0:>6.2f} {ag:>6s}")

# loss-curve comparison against arm A at shared steps
print("\n=== loss vs arm A at shared steps (|dloss|; B must be ~0 = fusion exact) ===")
A = data[ARMS[0]]
if A:
    for arm in ARMS[1:]:
        d = data[arm]
        if not d:
            print(f"  {arm:30s} no log"); continue
        common = [k for k in d if k in A]
        if not common:
            print(f"  {arm:30s} no overlapping steps"); continue
        diffs = [abs(d[k]["loss"] - A[k]["loss"]) for k in common]
        print(f"  {arm:30s} n={len(common):3d}  mean|d|={st.mean(diffs):.5f}  "
              f"max|d|={max(diffs):.5f}  last d={d[max(common)]['loss']-A[max(common)]['loss']:+.5f}")
