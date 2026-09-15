#!/usr/bin/env python3
"""Parse the boltz-accel A/B logs: steady-state s/step + loss curve per arm.

Startup (torch.compile, dataloader warm-up, the first few steps) is EXCLUDED --
the question is steady-state throughput, so we drop everything before
--skip-steps and report the median of what remains (median, not mean: a
preemptable host can stall one step and blow up an average).
"""
import argparse, glob, os, re, statistics as st
from collections import defaultdict

ARMS = ["accel_A_base", "accel_A2_base_replicate", "accel_B_fused",
        "accel_C_fused_rep10", "accel_D_fused_rep10_proxy8"]
STEP = re.compile(r"step = (\d+),")
def num(key, line):
    m = re.search(re.escape(key) + r" = (-?[\d.]+(?:e-?\d+)?)", line)
    return float(m.group(1)) if m else None

def parse(arm):
    logs = sorted(glob.glob(os.path.expanduser(f"~/bsub_logs/{arm}_*.stderr")),
                  key=os.path.getmtime)
    if not logs:
        return None
    # NEWEST LOG ONLY. Merging every log per arm silently mixes CODE BUILDS: arm D
    # was restarted after a bug fix, and the merged view kept serving the old
    # build's numbers (aux 19.77) as if current. Restarts also replay from step 0
    # here (checkpointing is off during the probe), so the newest log is complete
    # on its own. The job id is printed so the reader can tell which build it is.
    rows = {}
    for lg in logs[-1:]:
        with open(lg, errors="ignore") as f:
            for line in f:
                m = STEP.search(line)
                if not m or "train-loss" not in line:
                    continue
                rows[int(m.group(1))] = {
                    "t":     num("train-step_time (sec)", line),
                    "loss":  num("train-loss", line),
                    "lm":    num("train-lm_loss", line),
                    "aux":   num("train-aux_loss", line),
                    "gnorm": num("train-grad_norm", line),
                    "btpd":  num("train-billion_tokens_per_day", line),
                    "eff":   num("ffwd.load_effective_n_experts", line),
                    "agree": num("ffwd.proxy_topk_agree", line),
                }
    return (dict(sorted(rows.items())), os.path.basename(logs[-1])) if rows else None

ap = argparse.ArgumentParser()
ap.add_argument("--skip-steps", type=int, default=60,
                help="drop steps <= this (compile + warm-up) before timing")
a = ap.parse_args()

parsed = {arm: parse(arm) for arm in ARMS}
data = {a: (v[0] if v else None) for a, v in parsed.items()}
srcs = {a: (v[1] if v else "-") for a, v in parsed.items()}
print("logs read (newest per arm):")
for a_ in ARMS:
    print(f"  {a_:30s} {srcs[a_]}")
print()
base_t = None
print(f"{'arm':30s} {'steps':>6s} {'s/step':>8s} {'vs A':>7s} {'Btok/d':>8s} "
      f"{'lm_loss':>9s} {'mean aux':>9s} {'effK':>6s} {'proxy':>6s}")
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
    # MEAN aux over steady steps, not the last value: with repulsion_scale_comp the
    # coefficient is multiplied by `interval` on firing steps, so any single step is
    # either ~0 or ~interval x the baseline. Only the MEAN is comparable, and matching
    # it is the whole claim of the intermittent design.
    auxs = [x["aux"] for k, x in d.items() if k > a.skip_steps and x["aux"] is not None]
    maux = st.mean(auxs) if auxs else float("nan")
    print(f"{arm:30s} {last:>6d} {med:>8.3f} {ratio:>7s} {v['btpd'] or 0:>8.3f} "
          f"{v['lm'] or 0:>9.4f} {maux:>9.4f} {v['eff'] or 0:>6.2f} {ag:>6s}")

# loss-curve comparison against arm A at shared steps
print("\n=== lm_loss vs arm A at shared steps (B must be ~0: the fusion is exact) ===")
print("    (lm_loss, NOT total loss: scale-compensated repulsion inflates total on firing steps)")
A = data[ARMS[0]]
if A:
    for arm in ARMS[1:]:
        d = data[arm]
        if not d:
            print(f"  {arm:30s} no log"); continue
        common = [k for k in d if k in A]
        if not common:
            print(f"  {arm:30s} no overlapping steps"); continue
        diffs = [abs((d[k]["lm"] or 0) - (A[k]["lm"] or 0)) for k in common]
        print(f"  {arm:30s} n={len(common):3d}  mean|d|={st.mean(diffs):.5f}  "
              f"max|d|={max(diffs):.5f}  "
              f"last d={(d[max(common)]['lm'] or 0)-(A[max(common)]['lm'] or 0):+.5f}")

# Does intermittent repulsion preserve E[repulsion]? That is the design claim.
print("\n=== mean aux_loss over steady steps (intermittent must MATCH arm A) ===")
for arm in ARMS:
    dd = data[arm]
    if not dd:
        continue
    auxs = [v["aux"] for k, v in dd.items() if k > a.skip_steps and v["aux"] is not None]
    if not auxs:
        print(f"  {arm:30s} no steady steps yet"); continue
    n_fire = sum(1 for x in auxs if x > 1e-6)
    print(f"  {arm:30s} mean={st.mean(auxs):.5f}  n={len(auxs):3d}  "
          f"nonzero={n_fire:3d} ({100*n_fire/len(auxs):.0f}% of steps fired)")


# Routing behaviour must not move: the fusion is exact, so effective_n_experts
# should track arm A step-for-step. Comparing last-values is invalid when arms sit
# at different steps (effK evolves fast early), so compare only shared steps.
print("\n=== load_effective_n_experts vs arm A at shared steps ===")
if A:
    for arm in ARMS[1:]:
        d = data[arm]
        if not d:
            print(f"  {arm:30s} no log"); continue
        common = [k for k in d if k in A and d[k]["eff"] is not None and A[k]["eff"] is not None]
        if not common:
            print(f"  {arm:30s} no overlapping steps"); continue
        diffs = [abs(d[k]["eff"] - A[k]["eff"]) for k in common]
        print(f"  {arm:30s} n={len(common):3d}  mean|d effK|={st.mean(diffs):.3f}  "
              f"max={max(diffs):.3f}   (A={A[max(common)]['eff']:.2f} vs {d[max(common)]['eff']:.2f} "
              f"@step {max(common)})")


# ---------------------------------------------------------------------------
# NOISE FLOOR. accel_A2_base_replicate is byte-identical to arm A, so whatever it
# diverges by is pure run-to-run nondeterminism (FSDP reduction order,
# non-deterministic kernels, torch.compile) -- NOT an effect of any knob. The
# fused path is exact in float64 to 1.2e-15, so in bf16 its divergence from A
# should sit at or below this floor. Judging "close enough" by eye is exactly the
# mistake this replicate exists to prevent.
# ---------------------------------------------------------------------------
def diverge(arm, field):
    d, ref = data[arm], data[ARMS[0]]
    if not d or not ref:
        return None
    common = [k for k in d if k in ref
              and d[k].get(field) is not None and ref[k].get(field) is not None]
    if not common:
        return None
    return st.mean([abs(d[k][field] - ref[k][field]) for k in common]), len(common)

print("\n=== divergence from arm A, against the A-vs-A' nondeterminism floor ===")
floor = {f: diverge("accel_A2_base_replicate", f) for f in ("lm", "eff")}
for f, label in (("lm", "lm_loss"), ("eff", "effK")):
    fl = floor[f]
    if not fl:
        print(f"  {label}: A' replicate has no overlapping steps yet -- floor unknown, "
              f"so no verdict can be drawn on the other arms for this metric.")
        continue
    print(f"  {label}: noise floor (A vs A', n={fl[1]}) = {fl[0]:.5f}")
    for arm in ARMS[2:]:
        r = diverge(arm, f)
        if not r:
            print(f"      {arm:30s} n/a"); continue
        verdict = "within noise" if r[0] <= fl[0] * 1.5 else "ABOVE noise -- investigate"
        print(f"      {arm:30s} {r[0]:.5f} (n={r[1]:3d})  {verdict}")
