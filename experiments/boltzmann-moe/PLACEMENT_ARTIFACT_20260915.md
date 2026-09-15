# ⚠ THE "5x SLOWER THAN GPTSWITCH" FIGURE IS SUBSTANTIALLY A HOST-PLACEMENT ARTIFACT

Found 2026-09-15 by accident, while verifying an unrelated revert.

## The observation

The SAME arm, SAME code (unfused), SAME config, resumed from the SAME checkpoint,
on a DIFFERENT host pair:

| | hosts | s/step | n readings |
|---|---|---:|---:|
| before (job 1647293) | `p3-r20-n3 : p2-r17-n3` | **6.72-6.81** (median 6.76) | ~700 |
| after (job 1664787) | `p5-r24-n3 : p3-r29-n4` | **2.45-2.53** (median 2.49) | 12+ |

**2.7x, from placement alone.**

## Alternatives ruled out

- **Sibling-job contention.** `scale32B_gptswitch`, the other 16-GPU 2-node job,
  finished at 03:05:13. The boltz arm's median was **6.739 s/step before** that and
  **6.784 after** (n=611 / n=89). Removing the neighbour changed nothing.
- **Faster GPUs.** Identical on both pairs:
  `NVIDIAH10080GBH`, driver `590.48.01`, `gpu_factor 9.0`, `mig N`.
- **Code.** Both measurements are the unfused path. (`fused_experts` was enabled
  briefly in between, wedged, and was reverted — see ACCEL_FINDINGS.)

## Likely mechanism: dataloader starvation, not compute

GPUs are `exclusive_process` but **CPU slots are shared**. Per-rank GPU-busy time is
1.71 s/step (Phase-B trace), so:

    at 6.76 s/step wall -> 25% GPU utilisation
    at 2.49 s/step wall -> 68% GPU utilisation

That is a STALL signature, not a compute signature. The project has hit this exact
failure before: commit c41ea310 *"reserve 16 CPU slots per host: 1 slot for 8 GPUs
was starving the dataloader"*, later reverted (a17f13f2) because the 16-slot ask
would not schedule. A CPU-heavy neighbour on the old hosts fits every observation.

## What this invalidates

1. **The "~5x slower per step than gptswitch" claim** (in
   `configs/iclr_scale/scale32B_boltz_hop.yml:37` and echoed through TODO.md as
   "the 5x gap"). It compares two runs **on different host pairs**, and placement
   alone swings 2.7x. On comparable placement the ratio is
   **2.49 / 1.34 = 1.9x**. Note that is *still* not placement-controlled — gptswitch
   ran on `p3-r31-n3 : p2-r20-n4`, a third pair.
2. **The "launch-overhead bound" inference.** It rested on *GPU-busy 1.71 s much less
   than 6.75 s wall*, i.e. 25% utilisation, and concluded the 79.5k kernel launches
   per step were the dominant problem. At 68% utilisation that conclusion is largely
   void: the step is much closer to compute-bound than to launch-bound.
3. Any conclusion of the form "the recurrence x dense-32-expert design costs 5x"
   needs re-deriving. The design is more expensive than a top-1 sparse Switch — the
   FLOP accounting is not in question — but the measured *magnitude* was inflated.

## What survives

- The fused-GEMM speedup (1.61x) was measured on **4-GPU single-node** probes, where
  cross-node placement effects are absent, and against an A' replicate that landed
  within 0.6% — so that number is placement-controlled and stands.
- Every repulsion / sign / proxy result today was likewise 4-GPU single-node.
- The FLOP and elementwise-vs-GEMM *proportions* from the trace (61% / 25%) are
  per-rank kernel-time ratios and are unaffected by wall-clock stalls.

## Rule going forward

**No s/step claim from a multi-node run is admissible unless it is
placement-controlled** — either both arms on the same hosts, or several placements
per arm with the spread reported. A single 2-node measurement can be off by 2.7x.
