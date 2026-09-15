# boltz-accel: measured results (2026-09-15)

Branch `boltz-accel`, isolated worktree `/proj/dmfexp/nima/Code/dolomite-accel`.
Motivation: the 400M Boltzmann arm runs **6.75 s/step** against gptswitch's
**1.34 s** (5x), which sets the 32B-token run at ~4.8 days.

Probe: 4 arms x 4 H100, 300 steps, identical seed/data, checkpointing disabled so
I/O stays out of the timing. **4 GPU = 131,072 tok/step vs production's 524,288 at
16 GPU, so only the RATIO between arms transfers, not absolute s/step.**
Cluster GPUs verified homogeneous (4928 x H100 80GB), so cross-host ratios are valid.

## Speed (median s/step, first 60 steps dropped for compile + warm-up)

| arm | knobs | s/step | vs base |
|---|---|---:|---:|
| A base | current `main` | 2.356 | 1.00x |
| **B fused** | `fused_experts` | **1.463** | **1.61x** |
| C fused+rep10 | `+ repulsion_interval=10` | 1.212 | 1.94x |
| D fused+rep10+proxy8 | `+ proxy_rank=8` | 1.253 | 1.88x |

- **Fusion alone buys 1.61x** and is exact arithmetic (float64: max rel diff
  1.227e-15 vs `main`). On the production 16-GPU shape that projects 6.75 ->
  ~4.2 s/step, i.e. 32B tokens in ~3.0 days instead of ~4.8.
- The rank-8 proxy costs ~3% (1.212 -> 1.253), matching the predicted K*d*r.

## FINDING: intermittent repulsion is a WEAKER REGULARISER, not a free win

The drafted patch argued that multiplying the coefficient by `interval` keeps the
time-averaged repulsion pressure unchanged. **That holds instantaneously at a
fixed weight state, and fails over a trajectory**, because the penalty is
state-dependent. Measured repulsion aux loss:

| step | A (every step) | C (1-in-10, 10x coef) | ratio |
|---:|---:|---:|---:|
| 20 | 0.1849 | 0.2212 | 1.2x |
| 40 | 0.0697 | 0.0803 | 1.2x |
| 70 | 0.0213 | 0.0821 | 3.9x |
| 100 | 0.0106 | 0.0672 | 6.3x |
| 130 | 0.0093 | 0.0546 | 5.9x |

A's penalty **decays** as the regulariser decorrelates the experts. C's
**plateaus around 0.05-0.08 and stops decaying**: applying 10x force one step in
ten reaches a different, less-decorrelated equilibrium than 1x force every step.
The experts in C are measurably less repelled.

`lm_loss` still tracked arm A closely over the probe (mean |d| = 0.015), so this
is not visibly hurting the LM at 300 steps -- but repulsion is a quality lever we
have measured before (`repulsion_form` abs vs signed is worth +0.69pp / -1.83
ppl), so a weaker regulariser is not something to ship on a 300-step probe.

**Recommendation: do not ship `repulsion_interval` for the headline run.** The
bench already found the better option: **weight-space repulsion is 3.5x cheaper
than output repulsion AND sparse-compatible**. Applied EVERY step it keeps the
regulariser's every-step character while still removing most of the cost. That is
the change worth making instead.

## Proxy router: measurable, not yet useful

`proxy_topk_agree` = fraction of the exact router's top-k set the rank-r proxy
also picks (genuine set overlap; HANDOFF 7.9 records that `torch.isin` fakes this).
Chance at k=2, K=32 is 0.0625.

- Buggy build (KL inflated ~4096x by the `batchmean` bug): **0.800** by step 120.
- Fixed build (correctly per-token KL, coef 0.01): **0.170** by step 100.

The inflated loss was accidentally acting as a much larger learning signal for
the proxy. With a correctly scaled KL at coef 0.01 the proxy learns far too
slowly to be usable. **Next: raise `proxy_loss_coef` (0.1-1.0)** -- it is now
correctly normalised and the input is detached, so a larger coefficient is safe
provided the global grad norm is watched (see the clipping note below).

Note this only ever buys INFERENCE time. Skipping the forward projection for
unselected experts needs the sparse-dispatch kernel, and the earlier bench found
the grouped-loop form 0.59x (i.e. slower) in training at N=4096.

## Bugs found (chronological), and what each one teaches

1. **`ModuleNotFoundError: No module named 'helpers'`** -- PRE-EXISTING first-run
   bootstrap bug. `data/megatron/utils/__init__.py` registers
   `sys.modules["helpers"]` at import time only if `build/helpers.so` already
   exists; on a fresh checkout it does not, `compile_helpers()` builds it but
   never registers it. An old checkout hides this because a previous run left the
   .so behind. Fixed: register after the build barrier.

2. **`CheckpointError: saved [7,1280,1536] vs recomputed [8,1280,1536]`** -- mine.
   The fused repulsion gathered the UNIQUE expert set of the sampled pairs, whose
   SIZE varies with the draw. Activation checkpointing re-runs the forward in
   backward, `random.sample` redrew, the shape changed. Fixed by indexing
   `i_idx`/`j_idx` directly so shapes are `(n_pairs, ...)` always -- which is
   exactly why the pre-existing looped path never hit it.
   **Lesson: the float64 equivalence test passed throughout.** It compares
   arithmetic; this was a shape contract with a training wrapper. Exactness tests
   do not substitute for running under the production wrapper. Guard added:
   `scripts/ckpt_test_accel_20260915.py`, mutation-checked (restoring the bug
   makes it fail at iteration 1 with the exact cluster error).

3. **Gate could disagree between forward and recompute** -- mine.
   `random.random()` is not restored by `torch.utils.checkpoint` (torch's RNG is,
   Python's is not), so the gate could fire in the forward and not the recompute,
   changing the saved-tensor count. Now `torch.randint` on the CPU generator.

4. **Proxy KL inflated by the sequence length** -- mine.
   `F.kl_div(reduction="batchmean")` divides by dim 0 only, so on `(batch, seq, K)`
   it inflates by seq. Logged `aux_loss = 29.47` against `lm_loss = 7.52`.
   **And my safety claim was wrong**: I said enabling the proxy "cannot degrade
   training" because its input is detached, evidenced by bit-identical `grad_x`
   and `grad_W`. Both true, conclusion invalid -- `gradient_clipping: 1` clips on
   the GLOBAL grad norm and the proxy's parameters are in `model.parameters()`, so
   an inflated proxy gradient scales the BACKBONE gradients down. Detaching blocks
   the graph path; it does not decouple the optimizer. A per-parameter gradient
   check cannot see a coupling that lives in the optimizer.

5. **Measurement bugs in my own report**, each of which nearly published a false
   number: it merged logs across code builds (so a restarted arm kept serving the
   old build's numbers), and it compared TOTAL loss, which scale-compensated
   repulsion inflates ~10x on firing steps. Now newest-log-only with the job id
   printed, comparisons on `lm_loss` at SHARED steps, and mean aux reported
   separately as the direct test of E[repulsion].

## Open at time of writing

- `accel_A2_base_replicate`: byte-identical replicate of arm A, running to
  establish the run-to-run **nondeterminism floor**. Needed because `effK`
  diverges between A and B by ~2.8 (10.37 vs 8.03 at step 190). The fusion is
  exact in float64, and bf16 reduction order feeding a DISCRETE top-k selection
  plausibly amplifies that -- but "plausibly" is not a measurement. If A-vs-A'
  diverges comparably, the fusion is clean; if A-vs-B sits clearly above the
  floor, something in the fused path moves routing and must be understood before
  a restart is staked on it.
- Whether to restart the live 32B arm. 1.61x is real but is not the 4x that would
  close the gap to gptswitch, and a restart discards ~7000 banked steps.
