#!/usr/bin/env python3
"""Compare the live 400M arm before vs after enabling fused_experts.

This is the tightest test of the fusion available: the arm resumed from
global_step8000, so post-flip steps re-run the SAME weights on the SAME data. Loss
should continue the pre-flip trajectory, and step_time should drop ~1.6x.

The 4-GPU probe's noise floor (lm_loss 0.038) does NOT transfer here -- that was at
131k tokens/step against this run's 524k, so this run is intrinsically quieter. The
local scale used instead is the pre-flip run's OWN step-to-step variability, which is
the right baseline for "did the loss jump".
"""
import re, glob, os, sys, statistics as st
REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
PRE  = f"{REPO}/experiments/boltzmann-moe/results/router_analysis/preflip_boltz_hop_trajectory.txt"
PREK = f"{REPO}/experiments/boltzmann-moe/results/router_analysis/preflip_boltz_hop_effK.txt"

pre  = {int(l.split()[0]): tuple(float(x) for x in l.split()[1:])
        for l in open(PRE) if l.strip()}                     # step -> (tot, lm, aux, t)
prek = {int(l.split()[0]): float(l.split()[1]) for l in open(PREK) if l.strip()}

jid = sys.argv[1] if len(sys.argv) > 1 else None
logs = sorted(glob.glob(os.path.expanduser(f"~/bsub_logs/scale32B_boltz_hop_{jid or '*'}.stderr")),
              key=os.path.getmtime)
post, postk = {}, {}
for l in open(logs[-1], errors="ignore"):
    m = re.search(r"step = (\d+),", l)
    if not m or "train-lm_loss" not in l: continue
    g = lambda k: (lambda x: float(x.group(1)) if x else None)(
        re.search(re.escape(k) + r" = ([\d.eE+-]+)", l))
    s = int(m.group(1))
    post[s]  = (g("train-loss"), g("train-lm_loss"), g("train-aux_loss"), g("train-step_time (sec)"))
    v = g("ffwd.load_effective_n_experts")
    if v is not None: postk[s] = v
print(f"post-flip log: {os.path.basename(logs[-1])}   steps logged: {len(post)}"
      + (f" ({min(post)}..{max(post)})" if post else ""))
if not post:
    print("no steps yet"); raise SystemExit(0)

# local noise scale from the pre-flip run itself
ks = sorted(pre)
d_lm = [abs(pre[b][1] - pre[a][1]) for a, b in zip(ks, ks[1:]) if b - a == 10]
d_k  = [abs(prek[b] - prek[a]) for a, b in zip(sorted(prek), sorted(prek)[1:]) if b - a == 10]
print(f"\npre-flip local step-to-step variability (|delta| over 10 steps):")
print(f"  lm_loss  median {st.median(d_lm):.4f}   p90 {sorted(d_lm)[int(.9*len(d_lm))]:.4f}")
print(f"  effK     median {st.median(d_k):.4f}   p90 {sorted(d_k)[int(.9*len(d_k))]:.4f}")

steady_pre  = [pre[s][3] for s in pre if s >= 2000]
steady_post = [post[s][3] for s in post if s > min(post) + 30 and post[s][3]]
print(f"\n=== SPEED ===")
print(f"  pre-flip  median s/step : {st.median(steady_pre):.4f}  (n={len(steady_pre)})")
if steady_post:
    mp = st.median(steady_post)
    print(f"  post-flip median s/step : {mp:.4f}  (n={len(steady_post)}, first 30 steps dropped)")
    print(f"  SPEEDUP                 : {st.median(steady_pre)/mp:.3f}x")
    days = (61035 - max(post)) * mp / 86400
    print(f"  remaining to 61,035     : {days:.2f} days  (was "
          f"{(61035-max(post))*st.median(steady_pre)/86400:.2f} at the old rate)")
else:
    print("  post-flip: not enough steady steps yet")

shared = sorted(s for s in post if s in pre)
print(f"\n=== CONTINUITY at re-run steps (n={len(shared)}) ===")
if shared:
    print(f"  {'step':>6s} {'lm pre':>8s} {'lm post':>8s} {'d lm':>8s} "
          f"{'effK pre':>9s} {'effK post':>9s} {'d effK':>8s}")
    for s in shared[:14]:
        kp = prek.get(s); kq = postk.get(s)
        print(f"  {s:6d} {pre[s][1]:8.4f} {post[s][1]:8.4f} {post[s][1]-pre[s][1]:+8.4f} "
              + (f"{kp:9.3f} {kq:9.3f} {kq-kp:+8.3f}" if kp and kq else f"{'--':>9s} {'--':>9s} {'--':>8s}"))
    dl = [post[s][1]-pre[s][1] for s in shared]
    print(f"\n  mean d lm_loss = {st.mean(dl):+.4f}  (pre-flip local median |d| = {st.median(d_lm):.4f})")
    verdict = "WITHIN the run's own step-to-step variability -- continuous" \
        if abs(st.mean(dl)) <= st.median(d_lm) else "LARGER than local variability -- investigate"
    print(f"  -> {verdict}")
    kk = [postk[s]-prek[s] for s in shared if s in prek and s in postk]
    if kk:
        print(f"  mean d effK    = {st.mean(kk):+.3f}  (pre-flip local median |d| = {st.median(d_k):.3f})")
        print(f"  -> {'WITHIN local variability' if abs(st.mean(kk)) <= st.median(d_k) else 'ABOVE local variability -- the open effK question'}")
else:
    print("  no overlapping steps yet (post-flip has not reached the pre-flip range)")
