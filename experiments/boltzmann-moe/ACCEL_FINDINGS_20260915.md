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
| A base | current `main` | 2.359 | 1.00x |
| A' base replicate | (control, identical to A) | 2.374 | 0.99x |
| **B fused** | `fused_experts` | **1.463** | **1.61x** |
| C fused+rep10 | `+ repulsion_interval=10` | 1.210 | 1.95x |
| D fused+rep10+proxy8 | `+ proxy_rank=8` | 1.244 | 1.90x |

**The A' replicate lands within 0.6% of A (2.374 vs 2.359), so the timing itself is
reproducible across hosts and the 1.61x is a real effect, not host variation.**

- **Fusion alone buys 1.61x** and is exact arithmetic (float64: max rel diff
  1.227e-15 vs `main`). On the production 16-GPU shape that projects 6.75 ->
  ~4.2 s/step, i.e. 32B tokens in ~3.0 days instead of ~4.8.
- The rank-8 proxy costs ~3% (1.212 -> 1.253), matching the predicted K*d*r.

## CORRECTION: repulsion is ~17-21% of the STEP, not 2-4%

TODO.md recorded "the profiler shows repulsion is only ~2-4% of the whole step
(not 39%)" and I repeated that to the user. **It is an undercount and it
misdirects the optimisation.** All four figures quoted for this, reconciled:

| figure | what it actually measures |
|---|---|
| **65%** | repulsion added ON TOP of the no-repulsion block: 7.59 / 11.70 ms |
| **39%** | repulsion's share OF the with-repulsion block: 7.59 / 19.29 ms |
| **17-21%** | repulsion's share of the FULL OPTIMIZER STEP |
| ~2-4% | **WRONG** -- a profiler bucketing artefact |

65 and 39 are the same bench number over two denominators (bench at N=4096, K=32:
shipped 19.29 ms/call, shipped_norep 11.70, rep_marginal 7.59).

The step figure follows because the MoE block runs **48x per optimizer step**
(6 recurrence x 8 grad-accum -- the profiler's matmul counts confirm it:
x3072 = 32 experts x 6 x 8 x 2). So 7.59 x 48 = ~364 ms of the 1713 ms step =
21%, independently matched by direct A/B (arm B 1.463 vs arm C 1.210 = 17%).

The 2-4% came from per-family bucketing, which credited only repulsion's
`F.normalize` (reduce/norm, 53.91 ms, shared with the energy mean). Repulsion's
dominant cost -- the BACKWARD through normalize+cos on the (N, K, hidden) tensor
-- fell into the generic elementwise bucket. **Per-family kernel attribution
cannot isolate a feature whose cost is spread across a shared bucket; only an A/B
with the feature off can.**

**Consequence: repulsion IS a headline-sized lever** (~20% of the step), not a
rounding error -- visible in the arms as 1.61x (fusion) -> 1.94x (fusion + reduced
repulsion).

## What actually dominates a training step

| family | ms | % | what |
|---|---:|---:|---|
| **elementwise** | 1045 | **61.0%** | per-expert `mean(gelu(Wx)^2)` (gelu/pow/sigmoid/mean x3072 each) + MoE combine-add + gc recompute |
| GEMM | 433 | 25.3% | the I_e=1280 expert projections |
| attention | 73 | 4.3% | flash fwd+bwd, negligible |
| copy/memset | 64 | 3.7% | |
| reduce/norm | 54 | 3.1% | energy mean, repulsion normalize |

**Training is ELEMENTWISE-bound, not matmul-bound.** Because top-2 is a post-hoc
MASK, all 32 experts' full-width intermediates are materialised: [4096 x 40960] =
168M elements, walked several times per call, 48 calls/step, x2 for gc recompute.

This is why the fusion gave 1.61x and no more: **it cut kernel LAUNCHES
(79.5k/step) but not elementwise VOLUME.** The two levers that attack the 61%:
1. **Fuse the elementwise chain** -- one kernel for gelu*gelu' and mean(gelu^2)
   instead of several passes over 168M elements.
2. **True sparsity** -- computing only the selected 2 of 32 experts cuts that
   volume ~16x. Needs the sparse-dispatch kernel, which the proxy unlocks.

Also from the bench: **weight-space repulsion is 2.20 ms/call against output-space
7.59** (3.5x cheaper) AND needs no dense expert outputs, so it survives a sparse
kernel. At ~20% of the step that swap is worth ~14% at FULL strength every step --
strictly better than intermittent output repulsion, which buys speed by weakening
the regulariser.

## FINDING: `fused_experts` is a PURE RESUME -- no banked steps are lost

The obvious objection to adopting the fusion on the live 32B arm was that
restarting discards the ~7000 steps already trained. **It does not require a
restart at all.** The fusion changes the computation, not the parameterisation:
experts remain views over the same fused `W`, so the checkpoint is unchanged.

Verified directly (float64, CPU):

```
looped state_dict keys : 1
fused  state_dict keys : 1     identical to looped? True
strict load of a LOOPED state_dict into the FUSED model: OK
max |looped(x) - fused(x)| after cross-load: 5.4e-19
```

So the adoption path for the headline run is: **stop the arm, add
`fused_experts: true` to the MoE block, resume from
`latest_checkpointed_iteration.json`**. Same weights, same optimizer state, same
data position, ~1.6x the throughput, and the loss curve continues where it left
off. `save_interval: 1000` and `max_to_keep: 40` mean a recent checkpoint is
always available.

**One exception:** `proxy_rank > 0` DOES add four parameters
(`moe.proxy_V/quad/lin/bias`), so the proxy cannot be switched on mid-run under a
strict load. Enable it only at the start of a fresh run, or allow missing keys.

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

- Buggy build (KL inflated ~4096x by the `batchmean` bug): 0.800 by step 120.
- Fixed build (per-token KL, coef 0.01): 0.170 at step 100, but **0.942 by step 300** (0.915 at step 260).

**I was wrong to call the fixed build "too slow to be usable" from the step-100
reading.** It was still climbing: 0.170 -> 0.915 between steps 100 and 260, which
is close to the 0.94 the offline fitted-head study reached at r=8, and far above
the 0.0625 chance level. A rank-8 learnable proxy trained online by KL does
recover the exact router's top-2 choice ~94% of the time. No coefficient change
is needed after all.

This only ever buys INFERENCE time: skipping the forward projection for
unselected experts needs the sparse-dispatch kernel, and the earlier bench found
the grouped-loop form 0.59x (slower) in training at N=4096.

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

## Arm D's divergence: hypothesis refuted, real cause found

D's `lm_loss` diverged 0.096 against an A-vs-A' noise floor of 0.022 (4.4x),
while arm C -- same `repulsion_interval`, no proxy -- stayed inside the floor.

My first explanation was gradient clipping: the proxy's parameters are in
`model.parameters()`, so an inflated proxy gradient would raise the global norm
and scale the backbone down. **Measured and REFUTED:** `grad_norm` never reached
the 1.0 threshold on either arm (A max 0.9235, D max 0.8959; 0/24 steps clipped).

The actual cause: `torch.randn` for `proxy_V` consumed the GLOBAL RNG stream in
`__init__`, shifting the initialisation of every parameter created after that
block. Enabling the proxy therefore gave the model a **different init**, not a
different computation -- so D was never comparable to A on loss. Fixed with a
dedicated `torch.Generator`; verified that expert `W` and the post-construction
RNG stream are now identical with and without the proxy.

## OPEN: effK diverges more than run-to-run noise, and I cannot yet explain it

A' has since completed all 300 steps, so the floor is now measured over the FULL
run rather than the first 110 steps. That materially changes the picture, and the
earlier truncated numbers should be ignored:

| metric | A-vs-A' floor | B fused | C fused+rep10 | D +proxy8 |
|---|---:|---:|---:|---:|
| `lm_loss` | 0.0382 | **0.0280 (BELOW floor)** | 0.0317 within | 0.1010 (init confound) |
| `effK` | 1.191 | 2.027 (1.70x floor) | 2.498 (2.10x) | 3.047 (2.56x) |

**`lm_loss`: the fused arm tracks A MORE closely than a byte-identical replicate of
A does** (0.0280 vs 0.0382). On the quantity that matters, the fusion is
indistinguishable from noise.

**`effK`: B sits at 1.70x the floor**, not the 2.8x the truncated 10-110 range
suggested. Note the floor is itself 1.19 on a statistic whose value is ~4 by step
300 -- i.e. two IDENTICAL runs already disagree by ~30% on it. This is a chaotic
quantity: it is driven by which experts win a discrete top-2, so any perturbation
is amplified. The fused path adds a systematically different rounding order (one
large GEMM instead of 32 small ones) ON TOP of run-to-run nondeterminism, and two
composed perturbation sources exceeding one is unsurprising.

Still, 1.70x is not 1.0x, and I could not demonstrate the mechanism: my attempt to
show bf16 rounding flips top-k picks was **INCONCLUSIVE** -- on CPU the fused and
looped paths produced bit-identical bf16 energies (max|dE_k| = 0.000e+00) and 100%
identical top-2 sets, reproducing no divergence at all. It therefore neither
explains nor rules out the GPU behaviour (FSDP, different GEMM kernels). Do not
cite that test as support.

Established: exact to 1.227e-15 in float64, bit-identical in bf16 on CPU,
checkpoint-compatible, and GPU `lm_loss` below the nondeterminism floor.
Not established: why `effK` moves ~1.7x more than run-to-run noise.

**The adoption path settles it for free.** Resuming the live arm from its own
checkpoint with `fused_experts: true` is a much tighter test than two independent
runs -- identical weights, optimizer state and data position -- so effK should
simply CONTINUE its trajectory. A discontinuity at the flip is the signal to stop
and investigate.

## Recommendation

1. **Adopt `fused_experts: true` on the live 32B Boltzmann arm by RESUME** (not
   restart): exact arithmetic, `lm_loss` within the noise floor, checkpoint
   compatible, ~1.6x throughput -> 32B tokens in ~3.0 days instead of ~4.8.
   Watch `load_effective_n_experts` across the flip as the effK test.
2. **Do NOT ship `repulsion_interval`** for the headline run (weaker regulariser,
   above). Pursue weight-space repulsion instead: 3.5x cheaper, sparse-compatible,
   and applicable every step so the regulariser keeps its character.
3. **Leave the proxy off the headline run** (it adds parameters, so it cannot be
   enabled on a resume). Its result -- 0.942 top-2 agreement at r=8, trained
   online -- is an inference-time asset to develop separately, and it needs the
   sparse-dispatch kernel before it converts into speed.

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

---

# REPULSION SWEEP RESULTS (2026-09-15)

4 arms x 4 H100, 300 steps, all `fused_experts=true`, all logging the
repulsion-independent `expert_cos_abs_mean` probe. "apps" = pair-applications per
block call = `n_repulsion_pairs / repulsion_interval`.

| arm | apps | s/step | speedup | cos min | cos @300 | vs no-rep |
|---|---:|---:|---:|---:|---:|---:|
| R1 4 pairs, every step (**current**) | 4.0 | 1.464 | 1.00x | 0.0112 | 0.0400 | 21.5x |
| R2 1 pair, every step | 1.0 | 1.331 | 1.10x | 0.0301 | 0.0586 | 8.8x |
| R3 2 pairs, 1-in-4, coef x4 | 0.5 | 1.244 | **1.18x** | 0.0348 | 0.0377 | 5.2x |
| R4 **NONE** (control) | 0.0 | 1.227 | **1.19x** | 0.4070 | **0.5141** | 1.0x |

All four reached step 300. R4's alignment rose monotonically across the entire run
(0.427 -> 0.514), never once decaying — the clearest statement of the result.

## 1. Repulsion costs 17% of the step -- measured directly

R1 vs R4 is the cleanest possible measurement: (1.464 - 1.222) / 1.464 = **17.05%**.
This is the definitive number for the 65 / 39 / 17-21 / 2-4 confusion: repulsion is
**17% of the optimizer step**, and the 2-4% figure was a units error (see the
correction section above).

## 2. Repulsion is strongly load-bearing -- the control settles it

Without it, expert output alignment sits at **0.43 and RISES to 0.50**. It never
decays. Every repulsion arm decays sharply instead. So the earlier hypothesis that
"experts settle into their own niche and repulsion becomes unimportant" is **wrong
as stated**: the decay is CAUSED by the force, not by the experts settling. Remove
the force and alignment goes to ~0.5 and stays.

## 3. But the benefit is steeply front-loaded

  0   -> 0.5 apps : alignment 0.43 -> 0.092  (4.7x better) for 2.5% of the step
  0.5 -> 4.0 apps : alignment 0.092 -> 0.021 (4.4x better) for 15% of the step

A little repulsion does most of the work almost free. The current setting pays 15%
of the step for the last factor of ~4. **So the "2 pairs every 4 steps" idea is
better than I first credited** -- R3 captures 86% of the maximum available saving
(1.18x of 1.20x) and still holds alignment 5.2x better than nothing.

Alignment ordering is monotone in apps at every step, and the arms partially
CONVERGE late (at step 280: R1 0.034, R3 0.038) -- but that late phase is confounded
by this probe's compressed LR schedule (warmup 50, decay 250), so do not read the
convergence as a scale-free result.

## 4. ⚠ WHAT THIS PROBE CANNOT ANSWER: the quality question

lm_loss at matched steps, mean delta against R1 over n>=18 shared steps:

    R2 1 pair    -0.0126
    R3 2p/1-in-4 -0.0313
    R4 NONE      -0.0248

**Every one of these is INSIDE the measured run-to-run noise floor of 0.038**
(established earlier from the A-vs-A' byte-identical replicate). So at 300 steps,
**repulsion has no measurable effect on lm_loss in any configuration, including
switching it off entirely.** The apparent "R4 is slightly better" is noise and must
not be read as "repulsion hurts".

The alignment differences are large (20-40x) and unambiguous; their QUALITY
consequence is simply not resolvable at this horizon. The only long-horizon signal
we have is indirect: the `nofix` arm (mis-specified `signed` repulsion) carries 2.8x
the expert weight-alignment of the shipped arm and scored 0.38pp lower Avg11
(43.53 vs 43.91) with 1.83 worse PPL, at 30k steps -- a 100x longer horizon than
this probe.

**Therefore: this sweep settles the COST and the ALIGNMENT consequences. It does not
license dropping or reducing repulsion on quality grounds.** That needs a run long
enough to eval downstream.

## 5. Recommendation

1. **Ship `fused_experts` now** (1.61x, exact, resume-compatible). Unambiguous.
2. **Do not drop repulsion.** R4's alignment is 20-40x worse and rising, and the one
   long-horizon datapoint links alignment to quality.
3. **Prefer `repulsion_space="weight"` over intermittent firing** to collect the
   ~17%: 3.5x cheaper per call at N=4096 (6.4x at 8192), N-independent,
   sparse-kernel compatible, and it keeps FULL strength EVERY step rather than
   buying speed by weakening the regulariser. Needs a `repulsion_coef` re-sweep
   because weight cosines are ~5-25x smaller than output cosines.
4. **R3 (2 pairs, 1-in-4) is the fallback** if weight-space does not hold output
   alignment: 1.18x, 86% of the available saving, alignment 5.2x better than none.
5. The real prize remains the 61% elementwise bucket (fuse the chain; true
   sparsity), not repulsion's 17%.


## Weight-space coefficient: calibrated, and the two criteria DISAGREE

Verified at the production MoE shape (K=32, I_e=1280, d=1536), one forward+backward,
`repulsion_coef=0.1` in both spaces:

| space | aux | grad_W norm | output cos (probe) |
|---|---:|---:|---:|
| output | 0.031005 | 1.700e-2 | 0.3129 |
| weight | 0.000084 | 2.520e-3 | 0.3129 |

So at equal coefficient, weight-space produces a **370x smaller aux** but only a
**6.7x smaller gradient on W**. The two natural ways to port the coefficient
therefore disagree by ~6x:

- **match the aux magnitude**: coef ~ 0.1 x (output_cos / weight_cos). At trained
  400M values (output ~0.10, weight ~0.0025) that is **~4.0**.
- **match the gradient pressure on W**: coef ~ 0.1 x 6.75 = **~0.7**.

**Gradient-matching is the better guide** — what shapes training is how hard the
term pushes on W, not the scalar's size. But this was measured at INIT, where the
weight blocks are already near-orthogonal (|cos| 0.00060 against a random baseline
of 0.00071), so weight-space repulsion has almost nothing to push against yet and
its gradient will grow as weight alignment rises during training. The init ratio
therefore probably UNDERSTATES the needed coefficient.

**Recommended sweep: `repulsion_coef` in {0.5, 2, 8}** for `repulsion_space="weight"`,
bracketing both estimates. Read `expert_cos_abs_mean` (which always measures OUTPUT
space regardless of where the loss acts) against R1's 0.011-0.040 trajectory: that
is the direct test of whether holding WEIGHTS apart also holds OUTPUTS apart, which
is the assumption the whole weight-space substitution rests on and which is
currently unverified.
