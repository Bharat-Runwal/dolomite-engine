# BoltzmannMoE Experiments — Progress & Results

> ## ⚠ METRIC CONVENTION — READ BEFORE QUOTING ANY "Avg" IN THIS FILE
>
> **CANONICAL (2026-09-14 onward): `Avg11`** — the paper `tab:scaling` recipe and
> the SAME headline the EGPT-RL / FET colleagues use, so ours are directly
> comparable. Compute with `experiments/eval_scripts/compute_avg11.py` (recipe
> matches the colleagues' exactly; the COMPLETE in-repo `math_egptdual` seeds
> reproduce as Avg11 51.88 / 50.75 — see the provenance note in that script, and
> the correction below: the earlier "+0.004pp vs 49.35/49.67" claim was on the
> 9/11-INCOMPLETE step-dirs and has been retracted). Recipe: 11-task unweighted mean — `acc_norm` on
> {arc_challenge, arc_easy, hellaswag, openbookqa, piqa, sciq}, `acc` on
> {boolq, copa, winogrande, **race**, **lambada_openai**}; **MMLU (acc) and
> GSM8K-CoT (flex) reported SEPARATELY**. Source: `EGPT-RL/RESULTS.md:247-249`.
>
> **Legacy conventions in old tables — never mix with Avg11:**
> - **`avg9`**: 9 tasks, MMLU excluded, race/lambada absent (pre-`pyarrow>=20` bug,
>   fixed 2026-08-03). `acc_norm` on all six that report it.
> - **`avg10`** (`compute_aggregates.py`, now DEPRECATED for headlines): 10 tasks,
>   MMLU included, race/lambada excluded.
>
> **Avg11 ≈ 3pp BELOW avg10** (race ~0.28 + lambada ~0.23 near chance at our scale);
> avg10 is 1.2–2.7pp below avg9. **Do not put avg9 / avg10 / Avg11 in one table.**
>
> Legacy avg9→avg10: `restate_avg9_to_avg10_20260912.py --md` / `AVG10_RESTATED.md`.

## 2026-09-16 — TRUE SPARSITY: 4.69x measured, and the proxy router diagnosed

Full detail in `HANDOFF.md` §12.9-12.11.

**Speedup** (H100, compiled, forward+backward, at each arm's REAL tokens/call). `sparse_forward`
(skip both projections for unselected experts) = **4.69x** on the pure_T12 block shape,
2.09-2.37x on the hybrids. That is past the `1/2(1+k/K)` = **1.73x** floor which binds any EXACT
router, and breaking it is the whole point of the proxy. `sparse_backproj` alone = 1.18-1.35x.
**The speedup is batch-size dependent and the sign FLIPS**: at 4096 tokens/call `sparse_forward`
LOSES on both hybrid shapes (0.41-0.49x). Never quote a sparsity number without the per-call
token count.

**Correctness.** With an oracle proxy, and independently with `sparse_candidates = n_experts`, the
sparse path reproduces the dense output to 4-8e-16 in every routing configuration. So the proxy's
prediction error is the ENTIRE approximation -- there is nothing to audit in the dispatch.

**Quality** on `pure_hop_T12_sink`, wikitext bits/byte, dense = 1.0996:

| configuration | bpb | attributable to |
|---|---:|---|
| + proxy SELECTION, exact energy weights | 1.1161 | **+0.0165** -- selection is cheap |
| + `renormalize_topk: true` | 1.5996 | +0.483 -- a MODEL change, avoid it |
| `sparse_forward` + renorm | 1.9177 | +0.318 sparse machinery |
| `sparse_forward` as-trained | 3.4346 | proxy-filled denominator, -1.52 nats |

Proxy selection is nearly free; the damage was the softmax DENOMINATOR. The proxy ranks well and
sums badly. Fix: `sparse_candidates = p` over-selects, and the exact energies of the p candidates
both re-rank to the final top-k and supply p exact denominator terms.

**The proxy.** §7.6's "0.90 top-2" was for the exact energy restricted to a rank-r subspace, NOT
for the diagonal-quadratic head it recommended; fitted post-hoc that head gives 0.348 against a
`k/K` = 0.125 chance floor. Two corrections: `proxy_kind="subspace"` keeps the true `gelu(.)^2`,
and training on the routing **KL** rather than energy MSE. Closed-form least squares gives
0.33-0.60; 400 KL steps on the SAME parameterisation gives **0.82-0.88** on a held-out split.
Per-iteration heads add ~+0.05 -- agreement ranged 0.080 (BELOW chance, i.e. anti-correlated) to
0.592 across the 12 iterations of the shared block. `proxy_out_dim=512` is free in accuracy at
0.15% of dense MACs, and is required: the full-`I_e` form holds the same `(T,K,I_e)` activations
as the dense path.

**Ops.** `t90k_pure_T12` had been restarting from step 0 on every preemption (~2670 steps lost):
the watchdog resolved `load_args` at SUBMIT time while LSF requeues the original command. Fixed in
both the configs and the watchdog (now resolved at run time). `slope90k_1blk_sink` retired.

**Three silent no-ops found and closed:** `proxy_route` was set and never read; `sparse_backproj`
never reached the pydantic config so was unusable from YAML; and `proxy_rank > 0` with
`fused_experts: false` trains nothing, because `_forward_looped` never calls `_proxy_step`.

## 2026-09-14 — MoE training-step benchmark → repulsion is the cheap win; two-stage is not

Goal: find where the Boltzmann arm's ~6.75 s/step goes vs gptswitch's ~1.34 s/step
(5.0×), and whether the intermittent-repulsion / proxy-router tricks pay off.

**Benchmark** `scripts/bench_moe_train_step_20260914.py` — fwd+BWD at the exact
`scale32B_boltz_hop` MoE-block shape (d=1536, K=32 hopfield, I_e=1280, top-2,
τ=0.35, zscore), 1 GPU, results in `results/router_analysis/bench_moe_train_step_20260914.json`.
At **N=4096** (production per-GPU tokens = mbs1×seq4096), ms/call fwd+bwd:

| path | ms/call | vs shipped | note |
|------|---------|-----------|------|
| shipped (all-K + top-k mask + output repulsion) | 19.29 | 1.00× | as-trained |
| shipped_norep | 11.70 | **1.65×** | repulsion = **+65%** of the block |
| two_stage (all-K energy + grouped top-k 2nd matmul) | 32.64 | **0.59× (SLOWER)** | grouped-loop backward |
| proxy (linear router → top-k, selected-only both matmuls) | 12.65 | 1.52× | no 16× FLOP saving at N=4096 |
| repulsion marginal | +7.59 | — | single largest MoE-block cost |
| output repulsion **1-in-10** (amortized) | ~12.46 | **1.55× block** | with magnitude comp |
| **weight-space** repulsion marginal | +2.20 | — | **3.5× cheaper AND sparse-compatible** |

Findings:
- **Two-stage is not the lever — it is actively SLOWER in training** (0.48–0.69×
  across N), because the grouped per-expert loop's backward dominates. Consistent
  with the existing `project_sparse_boltz_perf` finding. Do not pursue it for the
  dense arm.
- **Repulsion is the surprise cheap win.** It is +65% of the MoE block; running it
  1-in-10 (with magnitude compensation) gives a **1.55× block speedup** at the same
  time-averaged pressure. This is your "repulsion once in 10 steps, bump magnitude"
  idea, and the bench confirms it lands.
- **Proxy needs a fused grouped-GEMM kernel** to matter: at the production token
  count (4096/call, fixed by mbs1×seq4096; recurrence is sequential) the Python
  per-expert loop's fixed overhead erases the 16× FLOP saving (only 2.99× at N=8192).
- **The 5× boltz-vs-gptswitch gap is probably NOT MoE density.** boltz_hop runs
  block-7 recurrence 6× + energy attention; gptswitch has NO recurrence. Estimated
  MoE share ≈14% of the step (48 MoE calls/step = 6 recurrence × 8 grad-accum ×
  19.3ms). **This was an INFERENCE — and Phase B (below) REFUTED it: the dense MoE
  is not a 14% slice, it dominates. `recurrence × dense-32-expert` compounds.**

**Phase B (a) — profiler config submitted.** `configs/iclr_scale/scale32B_boltz_hop_PROFILE.yml`
(throwaway: scratch save/wandb, fresh init, 15 steps) enables the engine's built-in
`TorchProfiler` via `logging_args.torch_profiler_trace_path`; trace →
`results/profiler/boltz_hop_trace`. Submitted job **1658324** on 4 preemptable GPUs.
It does NOT touch the headline pair (1647293 boltz_hop / 1647503 gptswitch). The
active-step trace splits recurrence vs energy-attention vs dense-MoE to confirm the
~14% MoE estimate before we commit to any kernel.

**Phase B RESULT (2026-09-15).** Job 1658324 completed clean (15 steps, 379 MB
trace). Parsed with `scripts/parse_profiler_trace_20260915.py` (pure-stdlib, runs on
the compute node). One optimizer step = **1712.99 ms of GPU kernel time**, one rank.
By kernel family:

| family | ms | % | what it is |
|--------|----|----|-----------|
| **elementwise** | 1045.16 | **61.0%** | per-expert energy `mean(gelu(Wx)²)` (gelu/pow/sigmoid/mean ×3072 each) + MoE combine-add (add ×9385, unary ×14064) + gc-recompute |
| **GEMM** | 433.42 | **25.3%** | dominated by the I_e=1280 expert projections (see shapes) |
| attention | 73.43 | 4.3% | cudnn flash fprop 27.6 + bprop 36.5 ms — negligible |
| copy/memset | 63.50 | 3.7% | cat / fills |
| reduce/norm | 53.91 | 3.1% | energy `mean`, repulsion `F.normalize` |
| FSDP_comms | 6.24 | 0.4% | negligible on this shape |
| triton_fused | 4.44 | 0.3% | |

Matmul shape attribution (top 5, all the **I_e=1280 dense-expert projections**):
`[4096,1536]@[1536,1280]` 131.5 ms ×4608 · `[1,4096,1536]@[1536,1280]` 134.7 ms
×3072 · `[4096,1280]@[1280,1536]` 119.8 ms ×4608 · `[1,4096,1280]@[1280,1536]`
124.0 ms ×3072 · `[1280,4096]@[4096,1536]` 62.9 ms ×3072. The prefix FFN
(3072/4608/8192-wide) is single-digit ms each, ×96. **The counts decode the cost
structure exactly:** ×3072 = `32 experts × 6 recurrence × 8 grad-accum × 2` (fwd +
gc recompute); ×4608 = `…× 3` (fwd + recompute + weight-grad bwd). So the expert
projections fire **1536× per optimizer step** (32×6×8) before recompute/bwd — the
dense-32-expert MoE run 6× via recurrence, with gc doubling the forward.

**Verdict — the 5× gap is `recurrence × dense-32-expert`, not repulsion/attn/comms.**
The ~14% MoE estimate was wrong: both the 61% elementwise AND the 25% GEMM are
overwhelmingly the dense MoE. gptswitch is cheap because it is top-1 *sparse* with no
recurrence; boltz computes **all 32 experts' fwd projection + energy every one of the
6 iterations**, then combines top-2. Attention (4.3%), comms (0.4%), and — at
whole-step scale — repulsion (~2–4%) are all small.

**Reconciling with the microbench — ⚠ THE ORIGINAL RECONCILIATION BELOW WAS WRONG,
corrected 2026-09-15.** It read: *"the expert compute fires 1536× while repulsion
fires ~48×, so repulsion is only ~2–4% of the whole step."* That divides
48 / 1536 = 3.1%, but the two counts are in **different units**: 1536 counts
*per-expert projections* (32 experts × 6 recurrence × 8 grad-accum) while 48 counts
*block calls*, and the bench's 7.59 ms marginal was measured **per block call,
already covering all 32 experts**. Dividing one by the other double-counts the 32×.

**Correct arithmetic:** 7.59 ms/call × 48 calls = **~364 ms of the 1713 ms step =
21%**. Independently confirmed by a direct A/B on 4×H100 (arm B, 4 pairs every step,
1.463 s/step vs arm C, 1-in-10, 1.210 s/step = **17%**). Two methods, 17–21%.

So repulsion IS a headline-sized lever, ~20% of the step — not 2–4%. See
`ACCEL_FINDINGS_20260915.md` on the `boltz-accel` branch. Generalisable lesson:
when reconciling a microbench against a whole-step profile, check that the event
counts are in the same units before taking a ratio; and prefer an A/B with the
feature disabled, since per-family kernel attribution cannot isolate a cost that is
spread across a shared bucket (repulsion's `F.normalize` landed in reduce/norm while
its dominant backward landed in the generic elementwise bucket).

**But intermittent firing is still not the right way to collect it** — measured
2026-09-15, it buys speed by weakening the regulariser (expert alignment rises
2.7–4.4×). Prefer **weight-space repulsion**: 2.20 vs 7.59 ms/call in the same
bench, sparse-compatible, and applicable at full strength every step.

**Also notable:** ~79.5 k kernel launches in a single step (unary ×14064, add ×9385)
— the dense per-expert path emits a torrent of tiny kernels, so the step is partly
launch-overhead bound (GPU-busy 1.71 s ≪ the 6.75 s production wall). **Fusing the
per-expert energy loop into one grouped kernel would cut both the elementwise time
and the launch overhead** — plausibly a bigger win than any FLOP cut.

**Ranked levers to actually close the gap (biggest first):**
1. **Sparsify the back-projection.** `[·,1280]@[1280,1536]` (119.8+124.0 ≈ 244 ms
   GEMM + matched elementwise) is computed for all 32 experts but only top-2 used —
   15/16 waste. A fused top-k kernel removes most of it. The fwd projection + energy
   for all 32 stays (routing needs it).
2. **Rank-r proxy router** (TODO line 28; Hopfield ceiling 96.7% top-1 @ r=16)
   replaces the all-32 fwd projection `[·,1536]@[1536,1280]` (131.5+134.7 ≈ 266 ms)
   with an ~80× cheaper low-rank score, then full fwd only for top-2. Together (1)+(2)
   attack the whole ~510 ms of expert GEMM + its elementwise tail.
3. **Fuse the per-expert energy loop** (grouped-GEMM + fused gelu²-mean) → cuts
   elementwise share and the launch-overhead gap.
4. **Reduce recurrence-× on the MoE / route-once-reuse** — only if routing is stable
   across the 6 iterations for THIS Hopfield config (TODO line 54 saw 0.87–0.90
   iter-argmax agreement on the Hopfield line, but flagged it as a degenerate-uniform
   artifact — must re-verify on the trained scale32B config, not assume).
5. **Intermittent repulsion (patch b)** — ~2–3% of step; ship it, but it is a
   rounding error against 1–4.

**Intermittent-repulsion patch (b) — DRAFTED, not applied.**
`results/router_analysis/intermittent_repulsion_20260914.patch`. Adds config fields
`repulsion_interval` (default 1 = current behavior, byte-identical) and
`repulsion_scale_comp` (default true = coef × interval on firing steps) to
`BoltzmannMoEFFEnergy`; the fire decision is a Bernoulli(1/interval) gate inside
`_add_repulsion_loss`. Opt-in per config; zero effect on existing/running runs.
Needs sign-off before it touches model source. Weight-space repulsion is offered as
the sparse-compatible alternative (cheaper) for a future fused kernel.

## 2026-09-14 — adopted colleague-consistent `Avg11` headline metric

Switched the headline eval metric to **`Avg11`** to match the EGPT-RL / FET
colleagues (their `tab:scaling` recipe). Why: our old `avg10` averaged MMLU *in*
and dropped `race`+`lambada_openai`; theirs does the opposite. Averaging 10 while
they average 11 (different composition) made our numbers look ~3pp better than
theirs for a pure scoring-convention reason — the same footgun as the documented
avg9/avg10 gap.

- Nothing was broken: `race`+`lambada` had failed on a pre-`pyarrow>=20`
  Arrow/parquet bug (fixed 2026-08-03; eval venv now has pyarrow 25.0.0). All 22
  `iclr_*` runs already have both scored — the deficit was purely in the
  aggregation script, not the eval run.
- New canonical aggregator: **`experiments/eval_scripts/compute_avg11.py`**.
  RECIPE validated against EGPT-RL (task list + metric-per-task identical). NUMBER
  provenance corrected 2026-09-14: the earlier "reproduces 49.35/49.67 to +0.004pp"
  was WRONG — those came from the colleague's OWN complete-task eval (not in repo),
  while the in-repo step-dirs (seed42@16100, seed1234@16200) are 9/11 INCOMPLETE
  and the script correctly FLAGS them. The COMPLETE final unsharded dirs reproduce
  in-repo as **Avg11 51.88 (seed42) / 50.75 (seed1234)**. `compute_aggregates.py`
  is now deprecated for headlines.
  The script REFUSES to emit a plain "Avg11" if any of the 11 tasks is missing
  (prints `INCOMPLETE k/11`) so a partial mean can't be mistaken for a real one.
- Recipe: 11-task unweighted mean — `acc_norm` {arc_challenge, arc_easy,
  hellaswag, openbookqa, piqa, sciq}, `acc` {boolq, copa, winogrande, race,
  lambada_openai}; MMLU (acc) + GSM8K-CoT (flex) reported separately.

**ICLR grid restated in Avg11** (latest eval per run; all 22 COMPLETE):

| run | Avg11 | MMLU | GSM_cot | PPL |
|-----|------:|-----:|--------:|----:|
| iclr_moebase/iclr_switch_K16_top2 | 44.83 | 24.18 | 1.90 | 40.02 |
| iclr_flops/iclr_hop_K16_dense | 44.78 | 24.64 | 2.20 | 40.73 |
| iclr_ctrl/iclr_learn_K16_dense | 44.70 | 24.74 | 2.05 | 39.60 |
| iclr_gptmoe/gptmoe_last_isoP | 44.43 | 25.17 | 1.90 | 40.76 |
| iclr_flops/iclr_hop_K32_top2 | 44.38 | 25.17 | 1.82 | 40.59 |
| iclr_slope/slope90k_hyb | 44.32 | 25.66 | 1.97 | 39.89 |
| iclr_gptmoe/gptmoe_last_3x | 44.20 | 24.51 | 1.97 | 38.98 |
| iclr_flops/iclr_hop_K32_top1 | 44.12 | 24.33 | 2.20 | 40.77 |
| iclr_moebase/iclr_switch_K16_top2_shared | 43.96 | 24.94 | 1.82 | 40.11 |
| iclr_flops/iclr_hop_K16_top2 | 43.91 | 26.55 | 2.43 | 40.49 |
| iclr_flops/iclr_learn_K16_top2 | 43.83 | 24.57 | 2.35 | 40.13 |
| iclr_flops/iclr_w1w2_K16_top2 | 43.83 | 25.49 | 2.27 | 40.00 |
| iclr_ctrl/iclr_hop_K16_top2_renorm | 43.74 | 24.55 | 2.05 | 40.36 |
| iclr_flops/iclr_hop_K16_top2_nofix | 43.53 | 25.09 | 1.59 | 42.32 |
| iclr_moebase/iclr_learn_K16_top2_noLB | 43.12 | 25.07 | 1.67 | 40.08 |
| iclr_gptmoe/gptmoe_all_isoP | 43.05 | 23.91 | 2.05 | 44.03 |
| iclr_1blk/iclr_pure_learn_isoP | 41.58 | 24.84 | 1.90 | 57.08 |
| iclr_gptmoe/pure_hop_T12 | 41.19 | 26.42 | 1.67 | 57.68 |
| iclr_flops/iclr_pure_hop_K16_top2_1blk | 40.21 | 24.73 | 1.82 | 69.66 |
| iclr_1blk/iclr_pure_hop_isoP | 40.14 | 25.39 | 2.27 | 61.45 |
| iclr_big/iclr_big_hop_pure | 40.02 | 24.53 | 1.36 | 69.42 |
| iclr_big/iclr_big_learn_pure_1node | 39.83 | 24.98 | 1.67 | 61.87 |

(These are mid-run checkpoints; treat as relative ranking, not final.)

---

## Overview

This series tests Mixture-of-Experts (MoE) routing inside the Energy GPT (EGPT)
framework. Three distinct design axes have been explored:

1. **Where to apply MoE**: FFN only (B/C-series), attention only (C3), or
   joint (attn+FFN) paired units (C4), or as the FFN of one recurrent EGPT
   block in a GPT+EGPT hybrid (H-series).
2. **How to route**: Boltzmann energy-based (B/C2), top-k sparse (C1/H1-topk),
   surrogate linear approximation (C2), or attention-alignment (C3/C4).
3. **Anti-collapse**: stochastic contrastive repulsion, dropout, high WD.

---

## Architecture variants

### B-series: Deep EGPT + BoltzmannMoE FFN (iso-param)

12 distinct deep EGPT blocks, each with BoltzmannMoE FFN.
`intermediate_size = n_experts × per_expert_I` (same total params as V1 Energy_MLP).

```
E_moe(h) = log(Σᵢ exp(Eᵢ(h)))   pᵢ = softmax(Eᵢ/τ)
∂E_moe/∂h = Σᵢ pᵢ · ∂Eᵢ/∂h
```

**Critical flaw**: iso-param with `intermediate_size=16384` gives FFN:Attn ≈ 21:1.
Only 14M of 407M params are in attention. V1 EGPT d=768 (143M) beats all B variants.

| Run | Anti-collapse | Avg acc | WikiPPL | Notes |
|-----|--------------|---------|---------|-------|
| B1 (baseline) | none | 0.474 | 51.9 | best MoE variant, hard routing by step 500 |
| B2 | rep λ=0.01 | 0.462 | 52.5 | |
| B3 | rep+drop+WD=0.3 | 0.450 | 58.0 | WD hurts LM quality |
| B4 | rep λ=0.1 | 0.466 | 51.9 | best load balance (max_load 0.37) |
| B5 | rep+drop+WD | 0.471 | 58.7 | |

**Routing findings**: Hard routing (eff_n≈1 per token) emerges by step 500 in all
variants. All 16 experts are used across the batch but only 2–3 dominate.
Semantic specialisation confirmed: COPA uniquely isolated (Mahalanobis distance >7
from all MMLU/BoolQ/GSM8k), GSM8k and MMLU occupy distinct PCA clusters.
B4 (rep 0.1) increases COPA's isolation to d_M>8 vs 7 in B1.

### C-series: Design fixes for the B-series FFN-heaviness problem

These target the root cause: the MoE FFN should not dwarf attention.

**C1 — TopK_Energy_MoE (non-iso-param, d=768)**
- 4 full-size experts (int=2048 each, same as V1 single Energy_MLP), top-2 routing
- Linear gate router, load-balance loss, dropout=0.1
- 4× more FFN params than B-series per active expert, but proper capacity per expert
- FFN:Attn ~ 5:1 (much better than 21:1 of B-series)
- avg=0.474, ppl=47.3 — same as B1. Non-iso-param doesn't help if routing still collapses.
- Status: **done**

**C2 — SurrogateBoltzmannMoE (iso-param as B5 + learned linear router)**
- Same B5 architecture + linear layer (d→16) trained to mimic Boltzmann weights (KL loss)
- At inference: cheap surrogate router (O(d·K) vs O(d·I) for Boltzmann)
- Tests the surrogate routing hypothesis: can a linear approximation replace energy routing?
- Status: **running** (job 254254) — preempted at step 20k, resubmitted

**C3 — BoltzmannMoE on Attention (2 energy-attn experts)**
- Normal Energy_MLP FFN (int=2048, same as V1)
- 2 independent EnergyAttention_QK modules per block, Boltzmann-mixed
- Routing: alignment score x·attn_out_i / d → softmax weights
- Addresses FFN-heaviness by adding capacity on the attention side
- FFN:Attn ratio is *reduced* not increased
- **avg=0.393, ppl=1383** — catastrophic generalization failure. Training loss 3.29 is
  *better* than C1 (3.33), suggesting overfitting rather than architectural failure.
  Investigation needed: likely memorizes training distribution but doesn't generalize.

**C4 — PairedUnitMoE (2 joint attn+FFN expert units)**
- 2 full paired units (EnergyAttention_QK + Energy_MLP) per block
- Joint routing: E_i = x·(attn_out_i + ffn_out_i) / d
- FFN:Attn ratio preserved at V1's ~2.7:1 regardless of n_units
- Most architecturally balanced MoE design
- **avg=0.370, ppl=5637** — worse generalization than C3. Training loss 3.23 is
  excellent but eval PPL is catastrophically high. Joint routing over (attn+FFN) may
  encourage mode-collapse to one expert unit that memorizes training patterns.

### H-series: Hybrid GPT+EGPT-MoE **(currently best results)**

Architecture: 6 standard GPT layers + 1 recurrent EGPT block (×6 iterations).
Only the final EGPT block uses MoE for its FFN. The GPT prefix builds rich
representations, giving the MoE router a meaningful signal.

**H1 baseline** (6 GPT + 1 EGPT×6, Energy_MLP FFN, no MoE): PPL≈41.35, avg≈0.469

**h1_boltz_egpt_moe** — BoltzmannMoE in EGPT block (4 experts × int=512, iso-param)
- Iso-param with H1 → same capacity problem as B-series (tiny per-expert capacity)
- avg=0.464, ppl=46.13 — *worse* than H1 baseline
- Confirmed: iso-param MoE with too-small experts fails even in hybrid setting

**h1_topk_egpt_moe** — TopK MoE in EGPT block (4 full experts × int=2048, top-2)
- NOT iso-param: 4× more FFN in EGPT block, ~2× FLOPs for that block
- **avg=0.499, ppl=39.79** ← **best MoE result so far**
- Beats V9 GPT 354M (0.513 avg) is ~13M fewer total params but significantly better than B-series
- Status: **already trained, eval complete**

**h1_topk_egpt_moe_r128** — Same + 128 register tokens in EGPT block
- avg=0.484, ppl=39.56 — slightly lower avg but better PPL than h1-topk-moe
- Registers + MoE: further investigation needed
- Status: **already trained, eval complete**
### New H-series: full-size BoltzmannMoE equivalents (2026-05-29)

**h1_boltz_moe_fullsize** — BoltzmannMoE in EGPT block (4 full experts × int=2048, non-iso-param)
- Direct equivalent of h1_topk: identical architecture, only routing differs
- FIXES: 1/sqrt(expert_I) routing energy scale (prevents loss spikes), full-size experts
- **avg=0.501, ppl=36.48, gsm8k=2.05%** ← beats h1_topk on avg accuracy AND PPL
- Status: **done, eval complete**

**h1_gptmoe_boltz_egpt** — Switch MoE in GPT prefix + BoltzmannMoE in EGPT (full-size)
- Switch MoE (top-2, 4 experts) on all 6 GPT prefix layers; BoltzmannMoE on EGPT block
- avg=0.486, ppl=35.52 — lowest PPL of all MoE variants, BoolQ notably low (0.476)
- Status: **done, eval complete**

**h1_boltz_topk2** — Sparse Boltzmann routing (top-2 of 4 experts) in EGPT block
- Same architecture as h1_boltz_fullsize, but with `top_k=2` parameter
- Energy-based selection (no learned router); truncated softmax (zero non-top-k, no renormalization)
- avg=0.4856, ppl=36.37, gsm8k=1.97% — **matches soft Boltzmann on PPL** (36.37 vs 36.5)
- Active params (idealized sparse impl): ~50M vs 68M for soft → 25% theoretical compute saving
- Note: current impl computes all K experts then masks; saving requires Switch-style dispatch
- Status: **done, eval complete**

**Key finding**: With full-size experts and 1/sqrt(expert_I) routing normalization,
**Boltzmann energy routing (0.501) ≥ TopK sparse routing (0.499)** at the same scale.
The energy landscape correctly identifies expert alignment without needing a learned router.
Sparse top-2 Boltzmann (h1_boltz_topk2) matches soft Boltzmann on PPL (36.37 vs 36.5)
and beats the learned-router topk on PPL (39.8) — energy-based selection generalises
to sparse regimes without any auxiliary router.



---

## Baseline comparison

GSM8K columns use lm-evaluation-harness filters: `strict-match` (the ground-truth
`####` separator only) and `flexible-extract` (also accepts "the answer is …" patterns).
EGPT-style models often emit answers in non-`####` formats; `flex` is the fairer
metric. The `flex_avg` column is the mean of `gsm8k flex` and `gsm8k_cot flex` and
is the gsm8k summary used in the scatter plot.

| Model | Params | Avg acc | WikiPPL | g_strict | g_flex | cot_flex | flex_avg |
|-------|-------:|--------:|--------:|---------:|-------:|---------:|---------:|
| V9 GPT d=1024 | 354M | **0.513** | **29.84** | 2.43% | 2.88% | 2.50% | **2.69%** |
| V0 GPT d=768 | 162M | 0.479 | 38.31 | 1.74% | 2.20% | 1.97% | 2.08% |
| V1-400M EGPT d=1024 | 354M | 0.494 | 38.61 | 0.68% | 1.67% | 2.20% | 1.93% |
| V1 EGPT d=768 | 143M | 0.481 | 47.66 | 0.45% | 1.74% | 2.20% | 1.97% |
| V58 EGPT rec 1×24 | 113M | 0.459 | 65.74 | 0.15% | 1.74% | 2.12% | 1.93% |
| B1 BoltzMoE (no reg) — old code | 407M | 0.474 | 51.90 | 0.23% | 1.36% | 1.90% | 1.63% |
| B4 BoltzMoE rep0.1 — old code | 407M | 0.466 | 51.87 | 0.53% | 1.67% | 2.12% | 1.90% |
| **B1 rerun** (1/√I routing scale fix) | 407M | **0.483** | **37.99** | 0.68% | 1.90% | 2.05% | 1.97% |
| **B4 rerun** (rep λ=0.1, fix) | 407M | **0.494** | **38.04** | 0.83% | 1.97% | 2.43% | **2.20%** |
| B5 rerun (rep+drop+WD, fix) | 407M | 0.480 | 43.05 | 0.76% | 1.82% | 1.82% | 1.82% |
| d1 BoltzMoE deep pure-EGPT | 143M | 0.477 | 50.04 | 0.45% | 1.90% | 1.97% | 1.93% |
| C1 TopK EnergyMoE | 165M | 0.474 | 47.34 | 1.14% | 2.05% | 1.90% | 1.97% |
| h1_boltz iso-param | 145M | 0.464 | 46.13 | 0.38% | **2.35%** | **2.50%** | 2.43% |
| h1_topk_egpt_moe | 145M | 0.499 | 39.79 | 0.76% | 2.20% | 1.82% | 2.01% |
| h1_topk_egpt_moe_r128 | 145M | 0.484 | 39.56 | 0.83% | 2.27% | 2.20% | 2.24% |
| **h1_boltz_moe_fullsize** | 145M | **0.501** | 36.48 | 1.06% | 2.05% | 2.20% | 2.12% |
| h1_gptmoe_boltz_egpt | 145M | 0.486 | 35.52 | 1.06% | 1.82% | 1.90% | 1.86% |
| h1_boltz_topk2 (sparse train) | 145M | 0.486 | **36.37** | 0.83% | 1.97% | 1.74% | 1.86% |
| h1_boltz_full @ top2 eval | 145M | 0.489 | 42.84 | 0.15% | **2.35%** | **2.50%** | 2.43% |
| **h1_egpt (no MoE; ISO compute to h1_boltz_full)** | 145M | 0.489 | 39.55 | 1.06% | 2.05% | 2.35% | **2.20%** |
| **580M @ step 14k (7.34B tok)** | **679M** | 0.514 | 30.39 | 0.83% | 1.90% | **2.88%** | **2.39%** |
| **580M @ step 18k (9.43B tok)** | **679M** | 0.524 | 28.97 | 1.14% | 1.97% | 2.27% | 2.12% |
| **580M @ step 30k (15.73B tok)** | **679M** | 0.537 | 26.84 | 1.67% | 1.67% | 2.12% | 1.90% |
| **580M @ step 76k (39.8B tok)** | **679M** | 0.559 | 22.41 | 1.21% | 2.20% | 2.58% | 2.39% |
| **580M @ step 102k (53.5B tok)** | **679M** | **0.580** | **20.23** | 1.90% | 2.12% | 2.50% | 2.31% |
| **scale_h3_boltz @ 104k (54.5B tok)** | **620M** | 0.556 | 22.67 | 1.59% | 1.82% | 2.96% | 2.39% |
| **scale_h3_boltz @ 120k (62.9B tok)** | **620M** | **0.569** | **21.89** | 1.67% | 1.74% | 1.74% | 1.74% |
| h1_boltz_fullsize_tanhexact (A/B fail) | 145M | 0.479 | 39.90 | 0.53% | 1.97% | 2.12% | 2.05% |
| h1_boltz_fullsize_erfexact (A/B fail) | 145M | 0.488 | 39.45 | 0.45% | 1.59% | 1.74% | 1.67% |
| **h1_boltz_fullsize_tanhexact + rep=0** | 145M | **0.495** | 39.42 | 0.38% | 1.59% | 2.27% | 1.93% |
| h2_6gpt_2egpt6x_boltz (2 EGPT blocks) | 155M | 0.487 | 37.71 | 0.15% | 1.90% | 1.59% | 1.74% |

### gelu_grad_method A/B test (2026-06-01..02) — phi' magnitude is the issue, not phi shape

Hypothesized that the legacy `phi' = sigmoid(c·x)·0.5` (half-magnitude approx)
was a bug worth fixing. Trained 4 variants of h1_boltz_fullsize at 30k steps,
7.86B tokens. Headline: the legacy half-magnitude phi' is empirically better
than the "correct" full-magnitude derivative when paired with strong repulsion.

| variant | gelu_grad | rep λ | avg | PPL | flex-avg |
|---|---|---:|---:|---:|---:|
| sigmoid (control, legacy) | sigmoid·0.5 | 0.1 | **50.10** | **36.48** | 2.12 |
| tanh_exact + rep=0 | tanh + analytic | **0.0** | **49.52** | 39.42 | 1.93 |
| erf_exact | F.gelu + analytic | 0.1 | 48.77 | 39.45 | 1.67 |
| tanh_exact | tanh + analytic | 0.1 | 47.90 | 39.90 | 2.05 |

Findings:
1. **erf_exact ≈ tanh_exact** (48.77 vs 47.90) — F.gelu and tanh-approx GELU
   give functionally equivalent results (test confirmed they differ <0.03% rel
   at random init). So the issue is the **doubled phi' magnitude**, not the
   phi shape change.
2. **tanh_exact + rep=0 recovers most of the gap** (49.52 vs 50.10 baseline,
   only −0.58pp) — confirming the **term2-cancellation hypothesis**: with strong
   repulsion forcing experts to anti-correlate AND full-magnitude phi', term2
   contributions from different experts partially cancel when summed. Without
   repulsion, less cancellation pressure.
3. The "legacy half-magnitude phi' + strong repulsion" combination falls into
   a sweet spot of effective gradient + low cancellation that the strict
   "correct" gradient overshoots.

Default unchanged (`sigmoid`). For new BoltzMoE runs, the recommended
combination is either (a) keep legacy sigmoid + strong repulsion (current
default), or (b) use erf_exact/tanh_exact with rep=0 or weak rep (≤0.01).

### h2 (2 EGPT blocks × 6 iters) vs h1 (1 EGPT × 6) — diminishing returns

h2_6gpt_2egpt6x_boltz_d768 (155M, 18 effective layers via two distinct EGPT
blocks each iterated 6× — vs h1's one block × 6 = 12 effective) trained 30k
steps. Result: avg **48.71** / PPL **37.71** / flex-avg **1.74**, **worse** than
h1_boltz_fullsize (50.10 / 36.48 / 2.12). Adding a second unique EGPT block
doesn't help at this scale; the simpler h1 architecture is preferred.

**Key lesson**: The h1_topk_egpt_moe works because:
1. The GPT prefix processes input into rich representations first
2. The MoE has full-capacity experts (not split iso-param)
3. The architecture remains balanced (FFN:Attn comparable to baselines)
4. Top-k routing with load-balance loss prevents collapse

The B-series failed primarily due to the iso-param design creating tiny (1024-dim)
experts with a 21:1 FFN-to-attention imbalance — not because Boltzmann routing
is fundamentally worse than top-k.

---

## Expert specialization (B1/B5 analyzed, 200 samples/category)

Mean-centered PCA of routing vectors reveals semantic clustering:
- **COPA** (commonsense causal reasoning): completely isolated, Mahalanobis d>7 from all others
- **GSM8k** (math): distinct cluster, d≈3–4 from MMLU
- **MMLU-Humanities/Social**: tight cluster (d≈1.2)
- **BoolQ**: moderately separated from MMLU (d≈3)

Expert dominance (B1): Expert #13 handles STEM/Medical/BoolQ/COPA/GSM8k (66–93%);
Expert #3 handles Humanities/Social/Logic (60–92%) → factual vs. reasoning split.

Cached routing arrays: `experiments/boltzmann-moe/results/routing_cache/routing_b{1-5}.pkl`

---

## What to try next

1. **C3/C4 results**: attention-MoE and paired-unit results will reveal whether
   adding MoE capacity on the attention side is more effective than the FFN side.

2. **Entropy regularization**: direct penalty on routing entropy
   `-λ E_h[H(p(·|h))]` — would prevent hard routing collapse at the source.

3. **Balanced B-series rerun**: redo B1 with `d=768, intermediate_size=2048` (same
   as V1 per expert) and `n_experts=4` — iso-param with V1, FFN:Attn preserved.

4. **Scale h1_topk**: lift the best h1_topk architecture to d=1024 / 24 layers
   for a direct comparison with V9 GPT at 354M params.

---

## 2026-09-15: ICLR draft migrated avg10 → Avg11 (pushed to Overleaf)

**Decision taken:** the ICLR draft is the active paper and standardises on the
colleague-consistent **Avg11** (`experiments/eval_scripts/compute_avg11.py`). The NeurIPS
draft and the talk are archive and stay on avg10 — do not touch them for metric work.

**No GPU re-eval was required.** All 22 `iclr_*` runs already had race + lambada_openai,
so every cited arm yields a COMPLETE Avg11 by re-aggregation from stored per-task JSON.

**Mapping was pinned by reproducing each `avg10` from source, not by run name** — this
mattered: `iclr_gptmoe/*` configs declare `model_type: energy` but, with softmax attention
and no recurrence, they *are* the energy-free plain-stack baseline. Name-based mapping
would have mislabelled `tab:pure`.

**Headline consequence — the `§sec:pure` claim reversed to parity.** Under avg10 Boltzmann
led the energy-free transformer by 0.22pp at matched params; under Avg11 the energy-free
iso-param stack scores **44.43** against **44.38** for Hopfield K=32 — 0.05pp the other
way. Per user direction the paper now claims **parity** (the slight Boltzmann advantage is
expected to come from the larger 32B runs, still training). Boltzmann still leads the
3×-budget stack, 44.38 vs 44.20. Backbone/routing decomposition moved 0.68/0.46 →
**0.40/0.45**, so the "most of the benefit is the encoder" claim was withdrawn.

**Two documentation bugs found and fixed while migrating:**
1. The paper's prose described `avg10` as scoring `sciq` with plain `acc`; every stored
   number in fact used `acc_norm` (plain `acc` would raise each arm ≈0.7pp). Published
   numbers were self-consistent; the prose was not.
2. The appendix asserted the energy-free GPT-MoE baselines "are queued / have no numbers
   yet" while `sec/experiments.tex` already reported them — an internal contradiction.
   Now updated with the three measured values (44.43 / 44.20 / 43.05).

MMLU and GSM8K-CoT are now **separate columns everywhere**, never in the mean: at 134M
both are at chance (MMLU 24.2–26.6% vs 25% random; GSM8K-CoT 1.6–2.4%), so averaging them
in compresses the spread between arms.

Full number-by-number record: **`AVG11_ICLR_MIGRATION.md`**.

---

## 2026-09-15: the seven previously-UNEVALUATED sharded ICLR arms now have Avg11

Driver: **`experiments/eval_scripts/eval_sharded_iclr_avg11_20260915.sh`**
(`{verify|submit|resubmit|status|report}`). These arms had `global_step*/model/*.distcp`
shards but **no `unsharded*` dir and no `harness_results_*.json`**, so unlike the 55-arm
race+lambada top-up they needed the full chain in one 1-GPU job: `lm_engine.unshard` →
`eval_harness.py` over the full `$EVAL_TASKS` (15 tasks) → `compute_avg11.py`. Because the
eval scores race + lambada directly, every output json is a **COMPLETE 11/11 Avg11** file
with no merge step. All 7 jobs succeeded; none failed.

| arm | step / target | Avg11 | MMLU | GSM8K-CoT | WikiPPL |
|---|---|---|---|---|---|
| `iclr_scale/scale32B_gptswitch` **FINISHED** | 61035/61035 (32.00B tok) | **50.25** | 25.69 | 2.43 | **25.00** |
| `iclr_scale/scale32B_gptswitch` (intermediate) | 58000/61035 | 49.70 | 25.55 | 2.27 | 25.25 |
| `iclr_scale/scale32B_boltz_hop` *(PARTIAL, still training)* | 6000/61035 (9.8%) | 44.64 | 24.11 | 2.05 | 42.30 |
| `iclr_decide/w1w2_K32_top2` *(PARTIAL)* | 10000/30000 (33%) | 41.63 | 25.23 | 2.20 | 56.00 |
| `iclr_slope/slope90k_1blk` *(PARTIAL)* | 40000/90000 (44%) | 41.07 | 24.66 | 2.65 | 61.28 |
| `iclr_balance/pure_hop_isoP_bal` *(PARTIAL)* | 16000/30000 (53%) | 39.79 | 23.37 | 1.74 | 80.27 |
| `iclr_big/iclr_big_hop_sandwich` *(PARTIAL)* | 4000/15000 (27%) | 39.61 | 23.56 | 1.52 | 78.38 |

**First 400M/32B headline number: `scale32B_gptswitch` = Avg11 50.25 at 25.00 word-PPL.**
Its Boltzmann partner is only at step ~7000/61035, so the pair is NOT yet comparable —
32B-scale Boltzmann-vs-Switch remains open.

> ### ⚠ DO NOT QUOTE THE FIVE PARTIAL ROWS AS RESULTS
> All five yield a task-COMPLETE 11/11 Avg11, which is exactly what makes them easy to
> misread: task-completeness is not training-completeness. They are undertrained
> snapshots, not comparable to the finished 30k-step grid.
>
> **In particular `w1w2_K32_top2` (41.63) does NOT resolve Hopfield-vs-W1W2.** It sits at
> a third of the token budget of Hopfield K=32's 44.38, and its PPL 56.00 vs 40.59 reads
> as undertrained rather than as a worse expert form. The expert-form decision stands
> unresolved; the arm is still `#PAUSED-20260914` in `watchdog_jobs.conf`.

**Method note — how the two live `iclr_scale` arms were read without touching the trainers.**
Both were mid-run (LSF 1647293/1647503) with `save_interval: 1000, max_to_keep: 2`;
gptswitch landed a checkpoint every ~23 min, so its "latest" shard is pruned by the trainer
within ~45 min — shorter than queue + 4.5 GB unshard. So at 02:12 UTC the completed
iteration named by each `latest_checkpointed_iteration.json` was **copied** (pure read) to
`results/iclr_scale_eval_staging/<arm>/global_step<N>/{model,metadata.json,training_config.yml}`,
which is all `load_checkpoint_and_unshard()` reads (`async_checkpointing: false` ⇒ `model/`
only, never `optimizer/`). Eval jobs pointed at the staging copy, so no eval job ever opened
a live training dir. After 1647503 reached its 61035 target and went DONE, that arm's real
run dir became safe to write and holds the final `unsharded_step61035/`; it is safe from
pruning because `watchdog_loop.sh:249` marks an arm DONE at `cur_step >= num_training_steps`
and never resubmits, and pruning only removes `global_step*` dirs.

**SKIPPED:** `iclr_balance/pure_hop_isoP_bal_DIVERGED_unclamped_bias_20260914` — diverged
(unclamped load-balance bias), not a valid result.


---

## 2026-09-15 (part 2): routing sign inversion, chemical-potential balancing, Sinkhorn

Full detail in `ROUTING_SIGN_BUG_20260915.md`; orientation in `HANDOFF.md` §11.

**THE BUG.** The composable Boltzmann router selects the **worst**-matching experts.
Measured on `iclr_flops/iclr_hop_K32_top2`: overlap of the selected experts 1.1358
against 1.8457 averaged over all experts, and *exactly equal* to the lowest-overlap
pair. `_HopfieldExpert` stores `E = mean(gelu(Wx)^2)` — which GROWS with overlap — and
`e_sign="neg"` then picks the smallest. The legacy `BoltzmannMoE_Energy_MLP` does it
correctly (`E = +overlap`, `softmax(+E/tau)`). Affects every `EnergyFF_BoltzmannMoE`
run: all 22 `iclr_*` arms and the live 400M. Not the Switch/learned-gate arms, not
gptswitch, not the legacy `h1_*`/`b*`/`c*` series.

**WHY IT SURVIVED.** Across 300 steps the sign makes **no resolvable difference to
lm_loss** (+0.006..+0.013 against a 0.038 noise floor) — consistent with the earlier
finding that deleting the FF branch entirely moved perplexity by +0.0003. Nothing in
the loss curve ever complained.

**ANTI-ROUTING WAS AN ACCIDENTAL LOAD BALANCER.** Correcting the sign revives the FF
branch 20-100x and collapses routing (effK 8.5 -> 1.9 of 32). Anti-routing is
self-limiting (use the worst expert, it improves, you stop using it); correct routing
is self-reinforcing (the winner gains overlap and wins more). So the headline property
"Boltzmann routing does not collapse and needs no load-balancing loss" may hold only
because of the bug.

**THE PRINCIPLED REPLACEMENT.** `p_k ∝ exp(E_k/tau)` is the entropy-regularised argmax;
adding the batch-marginal constraint `sum_tokens p_k ~ N/K` yields
`p_k ∝ exp((E_k - mu_k)/tau)` with `mu_k` a **chemical potential** — the dual variable
conjugate to occupancy. No gradient pathway, no learned gate, so the "no auxiliary
loss" property survives; and it separates the two roles the bug conflated (`E_k` =
which expert fits, `mu_k` = how crowded it is).

**SINKHORN BEATS THE CLAMPED CONTROL RULE.** `balance_rate` pinned at the +-1.0
`_BIAS_MAX` clamp in every arm. Solving the dual exactly
(`mu <- mu + log(load(mu)*K)`, 3 iterations, no clamp, no gain) at 134M, matched step
340, from the best 134M Boltzmann config:

| arm | lm_loss | effK/32 | max_share | E_mean | mu | expert cos |
|---|---:|---:|---:|---:|---:|---:|
| M1 shipped (INVERTED) | 5.0858 | 5.04 | 0.3533 | 0.5507 | -- | 0.1226 |
| M2 corrected + clamped bias | 5.1053 | 12.10 | 0.1774 | 0.1809 | 1.000 | 0.0598 |
| **M3 corrected + SINKHORN** | **5.0392** | **26.57** | **0.0718** | 0.2157 | 3.440 | **0.0582** |

Sinkhorn is **5.3x better balanced than the shipped arm** (uniform max_share would be
0.031; the shipped arm has one expert taking **35% of tokens**), has the best loss and
the best expert diversity, and reaches `mu = 3.44` — **3.4x past the clamp**, which
confirms the clamp was the binding constraint. Trends: the shipped arm's balance
**degrades** (effK 19.9 -> 5.7) while both balanced corrected arms **improve**
(M2 8.3 -> 13.3, M3 9.4 -> 26.6).

**FINAL 134M NUMBERS, step 1700** (the three arms ran to 1700 before M2/M3 were stopped
to free GPUs for the 400M; these are the numbers the ICLR appendix `app:sinkhorn`
quotes, so they need a repo source):

| arm | lm_loss | effK/32 | max_share | E_mean | mu_absmax |
|---|---:|---:|---:|---:|---:|
| M1 shipped (INVERTED) | 3.7215 | 7.80 | 0.3066 | 0.3975 | -- |
| M2 corrected + clamped bias | 3.7393 | 16.31 | 0.0805 | 0.2716 | 1.000 |
| **M3 corrected + SINKHORN** | **3.7234** | **30.80** | **0.0573** | 0.2286 | 1.97 |

Note the loss ordering CHANGED between step 340 and step 1700: at 340 Sinkhorn had the
lowest loss, at 1700 the shipped arm is ahead by 0.0019. Both gaps are far inside the
0.038 replicate noise floor, so **the honest claim is no measurable quality cost, not a
gain** -- which is what the appendix says. This is the same non-monotone trap that made
me retract the step-30 tau reading; a 340-step ordering does not survive.

`mu_absmax` FALLING 3.44 -> 1.97 is the load-balancing job getting easier as experts
specialise: less tilt is needed to keep the marginals uniform. It also means the clamp
would have stopped binding eventually -- but only after the damage at 340.

**The early effK dip is a transient, not a failure.** Both Sinkhorn arms fall hard
before recovering, because the experts are still near-identical at init so the dual has
nothing to separate:

| step | 10 | 30 | 40 | 70 | 200 | 600 | 1000 | 1600 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 134M M3 effK | 31.58 | 18.08 | 8.62 | 12.01 | 22.57 | 29.77 | 30.56 | 30.77 |
| 400M arm effK | 31.95 | 27.53 | 24.87 | 13.06 | -- | -- | -- | -- |

The 400M is tracing the same shape with a shallower dip (13.06 min vs 8.62) and had
already turned up by step 80. **Do not read a Sinkhorn arm's balance before ~step 600.**

### 2026-09-15: `max_to_keep` + a scratch restart CORRUPTS the checkpoint pointer

A compound failure that bricks an arm and makes every resubmit die in seconds. Hit
`iclr_hop_K16_dense_sink` and `slope90k_hyb_sink` during the corrected-sign rerun.

1. LSF preempts by `SSUSP -> PEND -> RUN` on the **same job id**, re-running the original
   bsub command with the original config. With no `load_args`, dolomite restarts at step 0
   even though checkpoints are on disk. `save_interval` does not protect against this --
   the checkpoints existed the whole time and were simply never loaded. The watchdog's
   auto-resume never fires because the watchdog never resubmitted.
2. The from-scratch run writes a LOW checkpoint (e.g. `global_step1000`) and points
   `latest_checkpointed_iteration.json` at it.
3. `max_to_keep: 2` prunes by **iteration number, keeping the highest**, so it deletes that
   fresh low checkpoint and retains the two high ones from the earlier run (18000, 19000).
4. The pointer now names a DELETED directory. Once `load_args` is present, every start dies
   instantly with `FileNotFoundError: .../global_step1000/training_config.yml`, and the
   watchdog burns a resubmit every 5 min (#6/400 before it was caught).

Note step 4 only becomes visible once `load_args` exists; without it the arm silently
restarts from 0 forever, which is worse and looks like slowness.

**Diagnosis:** compare `latest_checkpointed_iteration.json` against the `global_step*` dirs
present. **Repair:** rewrite the pointer to the highest surviving checkpoint (verify it has
`training_config.yml` and 7 entries, ~1.6G at 134M, matching a known-good checkpoint).

**Detecting a scratch restart:** `current_step < max_step` is NOT sufficient -- it cannot
distinguish a scratch restart from a resume at an earlier checkpoint. Use the first step
logged AFTER the last `wandb: Syncing run`: `~10` means scratch, `~ckpt+10` means resumed.

Prevention: every `configs/iclr_sink/*.yml` now carries `load_args.load_path`.

### 2026-09-15: FIRST corrected-sign Avg11 results (3 of 14 arms)

Metric via the canonical `compute_avg11.py`; the comparison script's `--selftest`
reproduces five published `tab:frontier` values to <0.02pp, and the published GSM8K reads
2.43, matching the paper exactly. So the deltas below are trustworthy.

| arm | published | corrected | delta | MMLU | GSM8K |
|---|---:|---:|---:|---:|---:|
| iclr_hop_K16_top2 | 43.91 | 43.50 | **-0.41** | 25.21 | 2.05 |
| iclr_hop_K16_top2_renorm | 43.74 | **44.54** | **+0.81** | 23.86 | 2.43 |
| iclr_hop_K16_top2_nofix* | 43.53 | 43.40 | -0.13 | 23.92 | 2.12 |

\*NOT like-for-like: tau normalised to 1.0 removed one of its four reverted knobs.

**1. The sign correction is QUALITY-NEUTRAL.** Mean delta **+0.09pp**. Consistent with the
sign making no resolvable `lm_loss` difference over 300 steps (+0.006..+0.013 vs a 0.038
noise floor) -- which is exactly why the bug survived the whole grid unnoticed. The paper's
PARITY claims therefore survive the correction; what the bug corrupted is the
routing-HEALTH narrative, not the quality numbers.

**2. It REVERSES a stated claim.** `sec:experiments` says "Making ours renormalise *costs*
0.17pp (43.91 -> 43.74)" and concludes masking-without-renormalising is right. Corrected:
renorm **44.54** vs top2 **43.50**, i.e. renorm LEADS by 1.04pp -- opposite sign.
Plausible mechanism: under anti-routing the weights were near-meaningless so renormalising
them changed little; with correct routing the weight MAGNITUDES carry signal, so
normalising them can matter. Hypothesis on n=1, not a claim.

**3. The caveat governing both readings.** The three deltas span **1.22pp**
(-0.41 .. +0.81), wider than the ~0.5pp threshold. So NO individual per-arm delta is
interpretable at this scale; only the aggregate (no systematic shift) is supportable.
Do not rewrite `tab:frontier` rows off single deltas -- wait for the full set, and treat
the renorm row as the one claim the correction might genuinely overturn.

### 2026-09-15 (update): ALL SIX core `tab:frontier` Boltzmann rows measured

| arm | k/K | published | corrected | delta | MMLU | GSM8K |
|---|---:|---:|---:|---:|---:|---:|
| iclr_hop_K32_top2 | 0.062 | 44.38 | **44.58** | +0.19 | 25.29 | 1.82 |
| iclr_hop_K16_top2_renorm | 0.125 | 43.74 | 44.54 | **+0.81** | 23.86 | 2.43 |
| iclr_hop_K16_dense | 1.000 | 44.78 | 44.40 | -0.38 | 26.28 | 2.20 |
| iclr_hop_K32_top1 | 0.031 | 44.12 | 44.19 | +0.08 | 24.47 | 2.20 |
| iclr_hop_K16_top2 | 0.125 | 43.91 | 43.50 | -0.41 | 25.21 | 2.05 |
| iclr_hop_K16_top2_nofix* | 0.125 | 43.53 | 43.40 | -0.13 | 23.92 | 2.12 |
| iclr_pure_hop_K16_top2_1blk | 0.125 | 40.21 | 38.76 | **-1.45** | 24.58 | 1.36 |

\*not like-for-like (tau normalisation removed one of its four reverted knobs).

**Mean delta over the six HYBRID arms: +0.03pp.** Not merely within noise -- essentially
exactly zero. The fully-recurrent `1blk` arm (-1.45) is the sole exception, so the split is
by ARCHITECTURE, not a spread.

**A mechanism I proposed and then REFUTED.** I hypothesised the recurrent arm suffered
because correct routing makes all iterations converge on the same experts, losing
cross-iteration diversity. The routing metrics contradict the premise: 1blk sits at
effK **15.93/16**, max_share 0.0769 -- marginally BETTER balanced than the hybrid
(15.84, 0.0733). There is no diversity collapse. The -1.45pp is UNEXPLAINED; candidates
are genuine architectural sensitivity or noisier evals on a weaker model (its baseline is
40.21 vs 43-44, i.e. nearer chance on several tasks). `pure_hop_T12_sink` (the other
fully-recurrent arm) tests whether recurrent arms pattern together at all.

**Consequences for the paper, in OPPOSITE directions:**
* STRENGTHENS the sparsity claim. Sparse K32-top2 at k/K=0.062 now BEATS dense routing
  (44.58 vs 44.40); published had dense ahead (44.78 vs 44.38). "More, narrower experts is
  simultaneously better and cheaper" goes from a within-noise ordering to the sparse arm
  winning outright at 1/16 the arithmetic.
* WEAKENS the Switch comparison. Switch stays 44.83 (learned router, correctly not rerun).
  Published best Boltzmann was dense at 44.78, a 0.05pp gap -- what "the same number to two
  decimals" rested on. Corrected best is K32-top2 at 44.58, so the gap widens to 0.25pp.
  Still small, and now achieved at HALF Switch's routing density, but the "to two decimals"
  phrasing must go.
* The renorm reversal PERSISTS (+0.81, now 2nd of the Boltzmann rows), contradicting
  `sec:experiments`' "renormalising costs 0.17pp".

**ENERGY STABILITY — the feared runaway does not happen.** The energy is evaluated on
`ln_x = self.ln(x)` (RMSNorm), not the raw residual, so it cannot grow through the
residual; only `||W||` remains, opposed by `weight_decay 0.1`. Measured
`energy_abs_mean`: the **shipped inverted arm is the worst offender** (0.107 -> 0.590,
grew 5x, now declining), while both corrected arms carry **2.5-3x less energy** and
M3 has plateaued (0.213-0.222 since step 170). `energy_abs_max` comparable (M3 14.0 vs
M1 14.2). **No activation change warranted**; if it ever trends up, weight-normalising
the energy is the one-line fix and it also retires the `routing_norm: zscore` patch.

**ALSO CORRECTED THIS SESSION** (see HANDOFF §11.7-11.9): the "~5x slower than
gptswitch" figure is substantially host placement (6.76 -> 2.49 s/step on the same
code), which also voids the launch-overhead reading; `fused_experts` is exact and 1.61x
but validated SINGLE-NODE only (it wedged at 2 nodes); repulsion is 17-21% of the step
rather than 2-4% (a units error) and is load-bearing; and the learnable rank-8 proxy
router reaches 0.942 top-2 agreement while the spectral `||Wx||^2` proxy fails with
ReLU just as it does with GELU.

### 2026-09-15 (RETRACTION + the real result): SINKHORN breaks PURE-ENERGY stacks

I proposed TWO mechanisms for the pure-energy degradation. Both are now REFUTED by
measurement. The refutations are the useful part, so they are recorded rather than quietly
replaced.

**Refuted #1 -- "cross-iteration diversity collapse".** Predicted worse balance in the
recurrent arm. Its routing metrics are marginally BETTER than the hybrid's (effK 15.93/16 vs
15.84; max_share 0.0769 vs 0.0733). No collapse.

**Refuted #2 -- "train/eval mu mismatch".** mu was solved under no_grad and applied only when
`self.training`, so eval ran with mu = 0. That IS real and is now fixed
(`sinkhorn_persist_mu`), but it is NOT the cause. Post-hoc calibration on held-out training
data (|mu|max 0.626) then re-evaluation moved `pure_1blk` only
  Avg11 38.76 -> 38.95   WikiPPL 177.09 -> 175.16
about 2% of a 107-point perplexity gap.

**The actual result**, from the control pair built for exactly this comparison (two pure-energy
isoP arms identical but for the balancer) with a hybrid reference:

| variant | Avg11 | WikiPPL |
|---|---:|---:|
| PURE isoP published (inverted sign) | 40.14 | 61.45 |
| PURE isoP corrected + SINKHORN | 36.16 | **316.21** |
| PURE isoP corrected + CLAMPED bias | **41.49** | 61.65 |
| HYBRID K16 corrected + SINKHORN | 43.50 | 40.71 |

**The corrected SIGN is fine on pure-energy stacks.** With clamped balancing it matches the
published perplexity (61.65 vs 61.45) and BEATS published Avg11 by 1.35pp. **SINKHORN is what
breaks them** (PPL 316 vs 61), and since applying mu at eval does not help, the damage is to the
TRAINED WEIGHTS, not to inference.

**Hypothesis, explicitly untested.** Sinkhorn re-solves mu on every forward CALL. A pure stack
calls the shared block 8x per forward, so routing is re-tilted differently at each iteration AND
each batch -- batch-dependent noise injected into what is meant to be a recurrent fixed-point
iteration. The clamped bias is a slowly-updated persistent buffer: one tilt, everywhere. Hybrids
are protected because only 1 block of 7 is a MoE and six GPT layers stabilise the residual.
Discriminating test: a pure arm with Sinkhorn but mu FROZEN after warmup.
FREE CONFIRMATION PENDING: `pure_hop_T12_sink` is also a pure stack (num_layers 1) with sinkhorn,
at 23000/30000. It should show the same blowup; if it does not, this hypothesis is wrong too.

**CONSEQUENCES, including one that corrects advice already given:**
1. The paper's pure-energy rows should use the CLAMPED variant -- a BETTER result than published,
   not a worse one.
2. **Sinkhorn is for HYBRIDS.** Do not recommend it for pure-energy / recurrence-heavy stacks
   until the mechanism is understood. Earlier guidance for colleagues' reruns called sinkhorn
   sound in general; that was too broad.
3. `sinkhorn_persist_mu` stays -- train/eval consistency is right on its own merits -- but must
   not be described as fixing the pure-energy problem.

### 2026-09-15 (RESOLVED): it IS the train/eval mu transition, measured directly

The retraction two entries above was WRONG, and the reason was the test, not the hypothesis.
Two measurements settle it.

**1. word_perplexity was inflating the effect size.** `word_perplexity = exp(bits_per_byte *
3.7066)` reproduces every arm to 0.00% (the constant is ln2 * bytes_per_word ~ 5.35 for
wikitext). So a 1.55x regression in bits/byte reads as a 7.8x one in word_ppl:

| arm | iters | word_ppl | bits/byte |
|---|---:|---:|---:|
| hybrid K16 + sinkhorn (1 MoE of 7) | 6 | 40.71 | 1.0000 |
| PURE isoP + clamped | 8 | 61.65 | 1.1119 |
| PURE big + sinkhorn | 4 | 103.78 | 1.2524 |
| PURE 1blk + sinkhorn | 8 | 177.09 | 1.3966 |
| PURE isoP + sinkhorn | 8 | 316.21 | 1.5530 |

**Report bits/byte, not word_perplexity**, for cross-model comparison at this scale. Describing
316 vs 62 as the effect overstated it; the real regression is 1.112 -> 1.553 bits/byte.

**2. The discontinuity is the train->eval MODE switch, on identical data.** CE on the SAME
held-out web batches (`eval_mode_vs_data_20260915.py`), so no dataset or metric confound:

| arm | train CE | eval CE | delta |
|---|---:|---:|---:|
| PURE isoP + sinkhorn | 3.4088 | **4.9959** | **+1.587** |
| PURE isoP + clamped | 3.3360 | 3.3365 | +0.0005 |
| HYBRID K16 + sinkhorn | 2.9454 | 2.9486 | +0.0032 |

Loader reported no missing/unexpected keys on any arm, so nothing was half-initialised. The
pure+sinkhorn arm loses **1.59 nats purely from switching mode**, while the arm whose balancing
is a persistent buffer loses 0.0005 and the hybrid 0.003. That is the mu tilt disappearing at
eval, and it scales with how much of the network depends on it: all 8 iterations for a pure
stack, 1 block of 7 for a hybrid.

CONCLUSION. Sinkhorn does not damage training -- pure+sinkhorn trains to CE 3.41, slightly
BETTER than pure+clamped at 3.34 on the hybrid's scale... in fact essentially the same. It
damages INFERENCE, by evaluating a router that was trained tilted with no tilt at all. The fix
is the per-iteration mu buffer; the target is to recover the 1.587 nats.

### 2026-09-15 (FIXED, measured): per-iteration mu removes the entire 1.587-nat discontinuity

Same held-out web batches, train vs eval mode, so no dataset or metric confound:

| checkpoint | train CE | eval CE | discontinuity |
|---|---:|---:|---:|
| before (mu dropped at eval) | 3.4088 | 4.9959 | **+1.587** |
| **after PER-ITERATION mu** | 3.4088 | **3.3506** | **-0.058** |
| pure + clamped (control) | 3.3360 | 3.3365 | +0.0005 |

Eval token PPL 147.81 -> 28.52. Sinkhorn + per-iteration mu now agrees with the clamped
control to 0.014 nats, which is what should happen if mu was the entire story -- and it was.
Eval CE landing slightly BELOW train CE is expected: eval has no train-time stochasticity, and
mu accumulated over 64 batches is lower-variance than any single batch's solve.

The single-averaged-mu attempt could not be re-measured here because its buffer is shape (K,)
while the code now expects (n_iter, K); that directory is stale. Its failure is already
documented and explained by the 1.56-1.68 mean spread across iterations.

**GUIDANCE CORRECTED, again.** Earlier I wrote "sinkhorn is for HYBRIDS only; do not recommend
it for pure-energy stacks". That is now wrong. Sinkhorn is fine everywhere PROVIDED the dual
reaches inference:
    sinkhorn_iters: 3
    sinkhorn_persist_mu: true
    sinkhorn_mu_iters: <this block's entry in layer_iterations>   # 8 pure, 6 hybrid
For an already-trained checkpoint, `calibrate_sinkhorn_mu_20260915.py` recovers it in minutes
without retraining. Hybrids only lose 0.003 nats without it, so it is optional there, but it
costs nothing.

**LR is a side issue.** The 6000-step sweep gives 5x (1e-2) = 3.7869 against 1x (2e-3) = 3.8285,
i.e. 0.042 nats, right at the noise floor, with 10x diverging (7.89) and lower LRs clearly worse.
So LR tuning is worth ~0.04 nats where the mu fix is worth 1.587 -- roughly 38x more. The pure
model's log-log slope (-0.097 vs -0.082 hybrid) says it wants more TOKENS, not a different LR.

### 2026-09-16: dose-response CONFIRMED out of sample, and deeper recurrence WINS once mu reaches eval

`pure_hop_T12_sink` (layer_iterations [12]) played no part in forming the mu hypothesis, so it
is an out-of-sample test. Predicted before measuring: its train->eval discontinuity should
EXCEED the 8-iteration arms' 1.587 nats.

| arm | iterations | MoE blocks | train CE | eval CE | discontinuity |
|---|---:|---:|---:|---:|---:|
| hybrid K16 | 6 | 1 of 7 | 2.9454 | 2.9486 | +0.003 |
| pure isoP | 8 | 1 of 1 | 3.4088 | 4.9959 | +1.587 |
| **pure T12** | **12** | 1 of 1 | 3.3403 | 5.1673 | **+1.827** |

Confirmed, and the per-iteration SPREAD scales the same way: mean 1.897 / max 3.623 across 12
iterations against 1.56-1.68 across 8. More iterations, more divergent duals, bigger penalty
when they are dropped. (A single averaged buffer would have been |mu|max 0.795 -- which is why
the averaged fix failed.)

After per-iteration calibration T12 goes 3.3402 -> **3.2706**, a -0.070 discontinuity, the same
signature as every other fixed arm.

**THE INVERSION WORTH KNOWING.** Ranking pure arms by EVAL CE after the fix:
    T12     (12 iters, sinkhorn + mu)   3.2706   <- best pure-energy model measured
    clamped ( 8 iters)                  3.3365
    isoP    ( 8 iters, sinkhorn + mu)   3.3506
Deeper recurrence HELPS. Before the fix T12 looked like the worst of the pure arms (its eval CE
5.1673 was the highest of any). Anyone reading the pre-fix numbers would have concluded that
deep recurrence does not pay off in this architecture; the opposite is true.

### LR sweep result (6000 steps, 200 warmup, comparable to each other only)

| arm | LR | loss@6000 |
|---|---|---:|
| clamped_5x | 1e-2 | **3.7869** |
| sink_5x | 1e-2 | 3.8014 |
| clamped_1x | 2e-3 (current) | 3.8285 |
| clamped_0p2x | 4e-4 | 4.0834 |
| clamped_10x | 2e-2 | 7.8885 (diverged) |

5x is worth **0.042 nats** over the shipped 2e-3, right at the 0.038 noise floor; 10x diverges;
lower is clearly worse. So LR is a ~0.04-nat lever where the mu fix is a 1.6-1.8-nat one, i.e.
~40x smaller. The pure model's steeper log-log slope (-0.097 vs -0.082) means it wants more
TOKENS, not a different step size.

### 2026-09-16 (HEADLINE): the mu bug INVERTED the depth scaling law of pure-energy stacks

Complete dose-response, all four pure arms plus the hybrid reference, bits/byte on wikitext
(tokenizer-independent; word_perplexity exponentiates this by ~3.7 and should not be used for
cross-model comparison):

| arm | iters | bpb, mu DROPPED | bpb, mu RESTORED | Avg11 restored |
|---|---:|---:|---:|---:|
| hybrid K16 top2 | 6 (1 MoE of 7) | 1.0000 | -- | 43.50 |
| big_hop_pure | 4 | 1.2524 | 1.2103 | 39.43 |
| pure 1blk | 8 | 1.3966 | 1.1525 | 40.43 |
| pure isoP | 8 | 1.5530 | 1.1201 | 40.42 |
| **pure T12** | **12** | **1.6034** | **1.0996** | **40.96** |

**Both columns are monotone, in OPPOSITE directions.**
  mu dropped   : 4 -> 12 iterations makes it WORSE  (1.2524 -> 1.6034)
  mu restored  : 4 -> 12 iterations makes it BETTER (1.2103 -> 1.0996)

So T12 is the best pure-energy model on every axis (bpb 1.0996, Avg11 40.96), closing on the
hybrid (1.0000, 43.50) -- and under the bug it was the WORST of the set.

WHY THIS MATTERS BEYOND THE BUG. Anyone reading the pre-fix numbers would have concluded that
depth hurts in a pure-energy stack, and would have concluded it MOST confidently from the deepest
arm, which is exactly the one that shows the opposite. Depth was never the problem: evaluating a
mu-tilted router with no tilt is, and the penalty compounds per iteration, so depth wore the
blame. Any architecture claim about recurrence depth measured before 2026-09-16 is suspect for
this reason alone.

CONSEQUENCES:
* `app:frontier`'s pure row (40.43, the 8-iteration arm) is correct but is NO LONGER the best pure
  result -- T12 reaches 40.96. Whether T12 becomes a reported row is an open editorial decision.
* A 400M design should push iterations UP, not down -- the opposite of what the pre-fix data
  implied.

## 2026-09-16 (part 2) — depth-vs-width at 400M, the annealing bonus, and the WSD redesign

Full detail in `HANDOFF.md` §12.14–12.17. Four things landed, one of which **corrects the closing
line of the entry immediately above.**

### ⚠ CORRECTION to "a 400M design should push iterations UP, not down"

That conclusion came from the 134M mu-restored depth scan (§12.2), where bits/byte was monotone in
depth (4 iters 1.2103 → 12 iters 1.0996). A direct **iso-FLOP ablation at 400M** (§12.14) found the
**opposite ordering**: three arms at 1.17 G MAC/token, d=1024, `it4` (I_e 17920, 401M params) beat
`it8` (8960, 254M) beat `it12` (5952, 204M) at **every logged step**, and `it4` was additionally
**22% faster per step** because iterations are sequential in a launch-bound block. Iterations share
weights, so holding FLOPs fixed while adding depth forces width — and parameters — down.

Both can hold: parameters help early, compute-efficiency pays late, and §12.2's arms ran 15k+ steps
against these ~1000. **But the 400M design guidance is now the reverse of what the line above says,
at least at ~1000 steps.** The it4-minus-it8 gap narrows monotonically (0.258 → 0.132 over steps
100–500) and would cross somewhere past a few thousand steps, so **"width wins" is established at
~1000 steps and NOT at 90000.** Nothing measured settles which regime a 90k run lands in. Iso-FLOP
is also not iso-time; anyone repeating this must report both axes.

### The annealing bonus is a one-time, saturating payment (§12.17) — this drove the schedule choice

`sw2k_sparse` (held at peak 1e-3) vs `sw2k_sparse_c10x` (same peak, decay completed in 2000 steps),
identical otherwise. 100-step means: the gap opens 0.005 (step 400) → 0.116 (800) → ~0.11 flat
through 1600. **The whole ~0.11 nats is realised while LR falls 1e-3 → 7.04e-4, a 1.4× reduction;
the further 3× to 2.37e-4 adds nothing.**

So the fast-drop arm does not have a better trajectory — it has the same trajectory plus a constant
offset. Decaying early forfeits progress in exchange for a bonus available at any moment. Corroborated
from the other side by §12.4 (60k steps pinned at a 2e-4 floor bought 0.015–0.019 nats) and §12.5
(LR floor inert across a 50× range).

**Also: "maybe the 2e-3 peak was too high" is NOT supported.** `sw2k_sparse_5e4` is behind the 1e-3
arm at every step with the gap not closing (step 800: 4.6525 vs 4.5515; step 1600: 4.1231 vs 4.0492).
And the "struggles early, then drops fast and steady" shape that prompted the doubt is the *expected*
signature of a high peak: measured loss = valley progress + a temperature penalty growing with lr.
**Do not lower a peak on early-loss appearance.**

Consequence: `configs/wsd90k/wsd90k_pure_it4.yml` and `wsd90k_sandwich.yml` — true WSD, warmup 1000,
**held at peak 2e-3 for 80000 steps**, cosine to 2e-4 over the last 9000. No code change;
`CosineScheduler` already does WSD once `num_constant_steps > 0`. Both at 262144 tok/step (8 GPUs),
23.59B tokens, verified by resolving the real scheduler. The deadline-relevant property: the trunk
sits at peak, so a decay can be branched off ANY checkpoint — a run stopped at 60% is still
harvestable, which is not true under cosine.

### Sandwich: mu recalibrated, and a FLOP-share / robustness dissociation (§12.15)

`iclr_big_hop_sandwich_sink` had the §12.1 bug (`persist_mu` false, `mu_iters` 1 against a 4× block).
Recalibrated: **Avg11 43.24 → 43.36**, bpb 1.0274 → 1.0247. **Use 43.36.** Small — and the smallness
is the finding: the sandwich has **pure-like FLOP concentration** (mixture = 97.6% of per-token
FLOPs) but **hybrid-like robustness** to routing corruption (0.003 nats, vs 1.6–1.8 for pure 8–12
iteration stacks). Two GPT layers worth 1.6% of FLOPs provide enough bypass that corrupting the
energy router barely matters. If that extends to an imperfect *proxy* router, the sandwich is the
best sparsity target of the three.

### A retracted result, and the mechanism that produced it

A monitor reported the sandwich proxy-selection ablation at **+0.00pp** and recommended launching the
400M sparse arm. **Retracted** (§12.16a): both Avg11 lines cited the *dense* directory and
`ablate/C_proxysel/` contained no results file. The tell was bits/byte identical to 4 decimals — if
routing changed for even a few tokens that moves. **Treat an exactly-zero delta as evidence of a
plumbing fault, not of robustness.**

Real cause (§12.16b, correcting 12.16a's guess): a **CUDA OOM** — 15.50 GiB requested, 10.39 free.
`proxy_route: true` with `sparse_forward: false` runs the dense all-K path *plus* the proxy heads,
which does not fit at `batch_size 4` for 400M `I_e=15872`. It was masked because
`bench_proxysel_one.sh` wraps the harness in `|| echo "... FAILED"`, turning a crash into a line of
stdout, after which `compute_avg11.py` globbed for the newest `harness_results*.json` in the tree and
found the dense one. **A stage that produces a number must fail loudly or verify its own output path.**

Genuine byproduct: the harness is **deterministic to 4 decimal places** across independent runs of
the same checkpoint, so a real 0.1pp Avg11 delta is signal, not variance.

---

## 2026-09-17 — ICLR appendix: new ABLATION STUDY section (`app:ablations`)

Written into `~/Code/overleaf/boltzmann-moe-ICLR-2026/sec/appendix.tex`, committed
locally as `0718b7c` (**not pushed** — Overleaf push needs explicit confirmation). Sits between
`app:threeway` and `app:setup`; a one-paragraph pointer was added to the `app:findings` overview.
Six labelled groups: `app:abl-sign`, `app:abl-balance`, `app:abl-mu`, `app:abl-depthwidth`,
`app:abl-lr`, `app:abl-proxycost`. All accuracies `Avg11`. Compiles clean at ICLR text width
(5.5 in): no errors, no overfull boxes.

### The one genuinely new result the section reports: the balancer head-to-head at full budget

`iclr_pure_hop_isoP_sink` vs `pure_hop_isoP_bal_corr` — identical except the balancing mechanism
(corrected sign, tau 1.0, d=768, one block x **8**, K=16 hopfield I_e=4480, top-2, 30000 steps).
Token-matched empirically from the logs: 22.006 and 21.991 Gtok/day at 1.029 s/step, i.e.
262144 tok/step each = 7.86B tokens, 4 GPUs each.

| balancing | effK/16 | max_share | min_share | final lm_loss | Avg11 | WikiPPL | bpb |
|---|---:|---:|---:|---:|---:|---:|---:|
| clamped proportional control (`balance_rate: 0.003`) | 14.12 | 0.1046 | 0.0227 | **3.4665** | **41.49** | **61.65** | **1.1119** |
| Sinkhorn dual (`sinkhorn_iters: 3`) | **15.93** | **0.0739** | **0.0541** | 3.4857 | 40.42 | 63.55 | 1.1201 |

**The exact dual is the better balancer and the worse model here** — behind by 1.07pp Avg11,
1.90 WikiPPL, 0.0082 bpb. The lm_loss gap (0.019) is inside the 0.038 noise floor and is not
claimed. This is the **opposite ordering** to the 134M K=32 / 2000-step comparison in
`app:sinkhorn` (§11.5), which the section states as an unresolved tension rather than resolving.
Two caveats recorded: the winning arm finished `load_bias_absmax = 1.0000`, i.e. **pinned at the
clamp** that §11.5 gives as the reason to prefer the dual; and with `sinkhorn_iters: 0` there is
no `mu`, so the §12.1 train/eval discontinuity **cannot arise** — 41.49 is read off plain
`unsharded/` and never needed a `_mucal`, unlike every Sinkhorn pure arm.

`bal_corr` is also the best pure arm on Avg11 (41.49 > T12's 40.96) at **half the depth**, while
T12 keeps the better perplexity (58.90 vs 61.65). Both metrics reported.

### Numbers re-derived this session (compute_avg11.py against stored harness output)

The mu-recovery progression, which quantifies why one shared buffer is not enough:

| arm | as trained | single shared mu | per-iteration mu |
|---|---|---|---|
| pure 8 it. I_e=2048 | 38.76 / 1.3966 | 38.95 / 1.3937 | **40.43 / 1.1525** |
| pure 8 it. I_e=4480 | 36.16 / 1.5530 | 37.21 / 1.4320 | **40.42 / 1.1201** |
| pure 12 it. I_e=4480 | 36.30 / 1.6034 | — | **40.96 / 1.0996** |

Shared buffer recovers 28% of the bpb gap on one arm and 1.2% on the other. Sandwich proxy-sel
verified independently: 42.96 / 47.49 wppl / 1.04156 bpb, i.e. **-0.40pp** and **+0.0169 bpb**
against the dense 43.36 / 44.62 / 1.024705.

### Four contradictions found in the record, flagged as `\CC` rather than resolved

1. **`app:routing`'s sign convention IS the bug.** It says "$s_k=+E_k$ for W1W2, $s_k=-E_k$ for
   Hopfield (lower is better)". With the energies as written in `app:expert-forms` the correct
   rule is the *opposite* in both cases. Needs fixing in the method section.
2. **`app:degenerate` attributes the dead FF branch to the gradient scale `c`**; §11.2 attributes
   it to the sign. Not separated — the two measurements differ in K *and* in `c`. A
   {sign} x {c} 2x2 on one shape would settle it and has not been run.
3. **`app:collapse` and the collapse audit disagree on which arm has effK 2.52** — the table says
   2.52 is the "no fixes" hybrid and 5.18 the single block; the audit in
   `iclr_pure_hop_isoP_bal`'s config header says 1.38 for pure isoP and 2.52 for pure 1blk.
   One is mislabelled.
4. **§12.14's cross-scale claim is inadmissible.** "134M T12 beat the 400M 4-iteration arm" uses
   `iclr_big_hop_pure_sink`, the arm §12.3 strikes for running at half tokens (1.97B vs 3.93B).
   7.86B vs 1.97B is not a depth comparison.

Also: the "20-100x FF-branch revival" multiplier is not reproducible from any single matched-step
pair (§11.4's own table gives 0.582 -> 12.44 = 21x; the step-20-30 A/B gives 0.0294 -> 3.6562 =
124x; §11.4's parenthetical "121x" implies a baseline of 0.103 that appears nowhere). The section
quotes the matched pair and says the ratio is step- and tau-dependent instead of giving one number.

And **PROGRESS.md's own 2026-09-16 entry above is stale**: it still carries §12.16b's diagnosis
that `compute_avg11.py` globbed the newest results file in the tree. §12.16c retracts that —
`resolve_results_path` globs only under the directory it is handed and exits 1. The real cause was
the OOM plus a monitor filter that grepped `Avg11 =` globally across a stdout in which the dense
arm had been evaluated twice.

### 2026-09-17: `sec/appendix.tex` reconciled to the corrected grid (paper repo, committed local only)

`sec/experiments.tex` had been migrated to the corrected routing sign + chemical-potential
balancing; `sec/appendix.tex` had not, so the two disagreed on nine numbers and on four
derived readings. All appendix figures now come from `compute_avg11.py` over the stored
`results/iclr_sink/*/unsharded*` harness JSONs. Commits `b41811d` (appendix) and `6472acb`
(intro + `sec:cost`) in `~/Code/overleaf/boltzmann-moe-ICLR-2026`. **Not pushed** — user reviews.

`app:frontier` table, all four columns per energy row:

| row | published | corrected |
|---|---|---|
| top-2 of 32 | 44.38 / 40.59 / 25.17 / 1.82 | **44.58 / 40.65 / 25.29 / 1.82** |
| dense soft | 44.78 / 40.73 / 24.64 / 2.20 | **44.40 / 40.79 / 26.28 / 2.20** |
| top-1 of 32 | 44.12 / 40.77 / 24.33 / 2.20 | **44.19 / 40.83 / 24.47 / 2.20** |
| top-2 of 16 | 43.91 / 40.49 / 26.55 / 2.43 | **43.50 / 40.71 / 25.21 / 2.05** |
| no fixes | 43.53 / 42.32 / 25.09 / 1.59 | **43.40 / 41.79 / 23.92 / 2.12** |

**FOUR readings changed direction**, all restated rather than patched:
1. `app:findings` item 5: "44.43 against 44.38 ... which is parity" -> energy **nominally ahead
   by 0.15pp** (44.58 vs 44.43). Kept as parity (0.15pp is inside single-seed resolution), not
   as a win. Same wording `sec:pure` already uses.
2. Hopfield vs W1W2 at matched K,k **FLIPS**: was Hopfield +0.08pp (43.91 vs 43.83); is now
   Hopfield **-0.33pp** (43.50 vs 43.83) and also behind on PPL (40.71 vs 40.00). The case for
   Hopfield is now purely structural (one matrix -> 2x the experts -> raise K).
3. The router x sparsity 2x2 **no longer changes sign with the sparsity**: learned - energy was
   +0.05 dense / -0.08 top-2, is now **+0.43 / +0.33** — learned ahead in both cells. That
   reading is explicitly withdrawn in the text.
4. "the energy formulation only reaches comparable quality by routing densely, which forfeits
   the sparsity that motivates it" is **false** now that the best energy arm is the sparsest at
   k=2. Retracted.

Recomputed deltas: Switch-over-energy at k/K=0.125 0.92 -> **1.33pp**; cost of sparsity 0.87 ->
**0.90pp**; K16->K32 gain 0.47 -> **1.08pp**; top-1-of-32 vs dense 0.66 -> **0.21pp**; energy over
the unbalanced gate 0.79 -> **0.38pp** at matched sparsity (**1.46pp** at its best); price of
arithmetic parity 0.45 -> **0.25pp** (this one was wrong in `sec:cost` too, and that file
contradicted itself — line 88 already said 0.25pp).

`app:abl-sign`'s "against the as-published values in Table~\ref{tab:frontier}" was dangling once
`tab:frontier` held the corrected values. Replaced with the pre-correction values inline
(44.38, 43.74, 44.78, 44.12, 43.91) plus a note that they are no longer tabulated anywhere.

**Two stale things deliberately LEFT and flagged** (no traceable corrected source):
* `sec/appendix.tex:40` and `:524` still carry `effective expert counts 12.3-12.8 of 16, max
  share 0.17-0.18` — pre-sign-correction training-log numbers. `sec:balance` has the corrected
  15.2-16.0 of 16 / 0.063-0.130, but those come from no JSON and no ground-truth table, and
  `app:collapse`'s own table (effK 7.65-18.03 offline) matches neither. Needs one owner.
* `sec/experiments.tex:278` (`tab:threeway`, "routing collapse: none (eff. 12.3-12.8 of 16)")
  contradicts `sec:balance:173` in the same file. Same root cause.

Also noted: `iclr_decide/w1w2_K32_top2` — the arm the paper twice says "is training"/"in
progress" — has an interim eval at **step 10000 of 30000** reading Avg11 41.63 / 56.00 wppl.
That is one third of the budget, so it is NOT the answer to the Hopfield-vs-W1W2 question and
must not be quoted; the paper's "in progress" wording is still correct.

## 2026-09-17 — colleague datamix port (`configs/cmix/`), a dynamo bug, and the projection null result

**Datamix reconciled with the colleagues.** Fetched their
`s8e4_stdmoe_fh2_boltz_topk2.yml` (`Bharat-Runwal/dolomite-engine`, branch
`bsaha/boltzmoe-iclr27`) via the existing `bharat` git remote and vendored it verbatim as
`configs/cmix/REFERENCE_s8e4_stdmoe_fh2_boltz_topk2.yml` — that file is now THE datamix
reference (documented in `CLAUDE.md` and `HANDOFF.md`).

**Their mix is 70% web / 30% math; ours was 100% web / 0% math** (they add
`megamath-web-pro_0` 0.15 and `finemath-3plus-rewritten_0` 0.15, and use `split 99,0.5,0.5`
against our `99.5,0.5,0`). This confounds EVERY cross-group comparison, not only MMLU and
GSM8K. All four paths verified present; the math sets are 52 GB and 79 GB (~13B and ~21B
tokens) so 0.15 x 32B = 4.8B each stays under one epoch. New arms use a separate
`data_cache_path` (`.cache/megatron_cmix`) because the Megatron blend index is keyed on the
mix; do NOT write into the colleagues' `cache-bsaha`.

`cmix_134M_recur.yml`, `cmix_400M_pure_it4_sparse.yml`, `cmix_400M_sandwich_sparse.yml` are
their parents with ONLY the datamix and run names changed — verified by diff, 0 lines
outside those — so each is a clean datamix A/B against a run we already have.
`cmix_1B_stacked_sparse.yml` is their architecture (12 DISTINCT layers, NO recurrence)
ported to our code.

**Three defects in the upstream file, corrected in the port:**
1. `proj_mode: unconstrained` — that spelling does not exist in our tree (0 refs); it was a
   silent no-op that happened to land on the same default. Ours says `energy_proj_type`.
2. Its header states its whole purpose is `energy_scale_mode: sqrt_inv_d` replacing a
   learnable temperature, but the body never sets it and does set `temperature: 1.0`, and
   the field does not exist in our tree either. **Raise with them**: a "fixed sqrt(1/d)
   beats learnable T" conclusion may rest on a config that never enabled it.
3. Hopfield stores ONE matrix per expert where legacy w1w2 stores two, so at their
   `intermediate_size: 76288` the port is **~793M, not ~1105M**. Restoring 1.1B is 2x `I_e`
   or 2x experts; expert count was left at their 8 for the first port.

**NEW BUG FOUND AND FIXED — `torch._dynamo` recompile_limit silently disables
torch_compile on non-recurrent stacks.** `_run_block` guards on the layer index `i`, so L
distinct blocks produce L specializations. The default `recompile_limit` is **8**: past it
dynamo falls back to EAGER for that function, silently, so `torch_compile` stops doing
anything and any wall-clock number becomes meaningless. The 12-layer port hit it at
`i == 7`. Our recurrent arms were never exposed (`num_layers` 3 and 7 — note **7 is one
under the ceiling**). Fixed at the compile site in `lm_engine/distributed.py`, scaling the
limit with the distinct-block count (`8 -> 56` for 12 blocks, confirmed in the log).
Numerics untouched — it only raises a compiler specialization ceiling.

**Projection A/B (`projab_*`, 134M, 2500 steps) is a NULL RESULT.** Median `lm_loss` over
steps >= 2000: `unconstrained` **3.6081**, `dual_unconstrained` **3.6096**, `psd_anti`
**3.6177**. Bootstrap 95% CIs are +-0.010 and overlap heavily against a 0.0096 spread, so
**no difference is resolvable**; `psd_anti` is nominally last. Step times were identical
(0.4212-0.4223 s), so the PSD-antisymmetric constraint is computationally free and buys
nothing here. This supports describing Pi as unconstrained in the paper. NOTE it does NOT
reproduce "dual consistently a tiny bit better" — plain `unconstrained` is nominally ahead,
inside noise.

**`p2n_long_outrep` (1733007) was PREEMPTED, not a code failure** — stdout says `preempt`
and the signal pattern is LSF teardown (SIGTERM + SIGABRT across ranks), not a wedge and
not an NCCL error. Ran ~25 min. Needs a re-arm to confirm it crosses the sparse switch.

**GPUs freed:** `t32B_sandwich_sparse` (1730874) and `t32B_pure_it4_sparse` (1720770) killed
at ~step 6.5k/61035 to make room for the port work; checkpoints intact for resume.

## 2026-09-17 overnight — 1B scale-up: both priorities answered, 32B run launched

**PRIORITY 1 — the scaled Boltzmann MoE works at 1B.** `cmix1B_12layers_K64k2_gptDense`
(12 distinct layers, NO recurrence, 8 DENSE-GPT + 4 energy, K=64 top-2, I_e 2846, psd_anti,
1.000B params) ran 500/500 steps to `DONE`, loss 10.4384 -> 3.8738 monotone. Routing at K=64,
windowed medians over steps >= 230:
  `effective_n_experts` **57.64 / 64**   (the shipped K=32 arm was 5.0/32 -- no collapse)
  `max_expert_load` 0.0479              (uniform is 1/64 = 0.0156, so 3x uniform)
  `expert_cos_abs_mean` 0.2114          (0.85 is the collapse bar)
  `grad_norm` 0.5857 dense / 0.2301 sparse -- both well under the 1.0 clip.
Sinkhorn balances K=64 with no auxiliary loss and no learned gate.

**PRIORITY 2a — sparse TRAINING: 2.631x at 1B.** dense 4.5478 s/step (n=18) -> sparse
1.7285 s/step (n=28) at 8192 tokens/call. Compare 1.175x at K=8: the lever is p/K, which falls
0.50 -> 0.0625. Realised 2.631x against a ~6x Amdahl ceiling (44%).

**PRIORITY 2b — sparse INFERENCE works. NEW measurement on a real trained checkpoint**
(`scripts/test_sparse_inference_20260917.py`, unsharded step-500 ckpt, one fixed batch, three
paths, identical weights):
  (1) DENSE all-K                  loss 6.228001
  (2) SPARSE p=K=64 (exact select) loss 6.228086   -> **+8.5e-5 nats**: dispatch EXACT on real weights
  (3) SPARSE p=2 (inference path)  loss 6.236036   -> **+0.008035 nats = 1.0081x ppl**
So the rank-16 proxy costs **0.8% perplexity** at inference. **`proxy_topk_agree` of 0.32
OVERSTATES the harm** -- disagreement about WHICH top-2 win is not disagreement about the
OUTPUT, because the substituted experts are near-equivalent in energy. Caveats: eval text was
README/source (out of distribution for a web+math model), so ppl 507 is not a quality figure,
only the gap is; and the checkpoint was TRAINED through the sparse path, so this answers
"would switching to dense help?" not "what does sparsifying a dense-trained model cost?".

**32B RUN LAUNCHED (rung 3 of the fallback ladder).** `cmix1B_12L_gptDense_32B`, job 1740630,
16 GPUs (2 nodes x 8) on grp_ebm, 61035 steps x 524288 tok/step = 32.0B, sparse_start_step 2000.
Projection from arm A's measured sparse step time: ~2.5 h dense warmup + ~29 h sparse
= **~32 h ≈ 1.3 days**. Without sparsity it would be ~77 h, so sparsity buys ~2 days.

**THE SINGLE-BLOCK HYBRID FAILED, AND NOT FOR THE REASON PREDICTED.**
`cmix1B_hybrid_K64k2_feas` (6 dense-GPT + ONE energy block recurred x6, I_total 773760)
EXIT 1 after 1921 s with **0 steps and 0 OOM**; peak host memory 39.4 GB of a 200 GB limit.
Signature: **`InductorError` x8 plus a timeout** -- torch.compile could not compile a single
773,760-wide recurrent energy block. The 5.90 GiB dense-warmup analysis was correct arithmetic
aimed at the WRONG failure mode. Consequence: the memory argument against single-block
recurrence at 1B stands on paper but was never the binding constraint; **compile time is.**
NEXT STEP (needs approval): the 2-energy-block variant halves I_total/block to 386880, which
shrinks the compile graph as well as the memory -- plausibly fixing the ACTUAL cause, but that
is now a hypothesis about compile scaling, not the memory claim.

**Config/process bugs found and fixed tonight**
1. `torch._dynamo.config.recompile_limit` defaults to 8; `_run_block` guards on the layer index,
   so a 12-DISTINCT-layer model exceeds it at layer 7 and dynamo silently falls back to EAGER --
   torch_compile becomes a no-op and any wall-clock number is meaningless. Patched in
   `lm_engine/distributed.py` to scale with block count (8 -> 56 here). Recurrent arms were never
   exposed, but the 134M hybrid at num_layers 7 sits ONE under the ceiling.
2. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` now set INSIDE the job script in
   `submit_train.sh`. An LSF requeue re-runs the ORIGINAL command, so submit-time env is silently
   lost -- that is exactly how one arm lost the setting and re-OOMed.
3. **A checkpoint saved with `sparse_start_step > 0` reloads with `_sparse_active = False`**
   (the flag is `sparse_forward and sparse_start_step <= 0`). So evaluating a sparse-trained
   checkpoint SILENTLY runs the dense path. Any sparse-inference measurement must set the flag
   explicitly or use `sparse_start_step: 0` in the eval config.
4. `register_model_classes()` is NOT invoked on importing `lm_engine.hf_models`. Without it,
   `from_pretrained` on our checkpoints dies with `KeyError: 'energy'` / "Transformers does not
   recognize this architecture", which reads like a corrupt checkpoint.
5. `sinkhorn_mu_iters` must equal the energy block's own `layer_iterations` entry (rule 9). The
   hybrid inherited 4 from its `[1,4,1]` sandwich parent against an entry of 6.
6. **yaml.safe_load passing is NOT validation.** A hand-edit folded stray YAML aliases into
   `distributed_args.stage` (`'0 - *gffn - *gffn...'`); yaml accepted it as a string and the job
   died in 31 s with no traceback. Configs are now built programmatically (load -> modify ->
   safe_dump) and validated with `TrainingArgs(**yaml.safe_load(...))` over the WHOLE document.
7. HF causal LMs shift labels INTERNALLY. Passing pre-shifted labels double-shifts and drives
   loss to ~ln(vocab)=11.52; the first sparse-inference run had to be discarded.

**Also tonight:** the NVLS/Fabric-Manager fault that killed one arm is a CLUSTER fault, not our
code (`Failed to bind NVLink SHARP (NVLS) Multicast memory: CUDA error 401`, host p3-r28-n3). It
names its own mitigation, `NCCL_NVLS_ENABLE=0`, which would likely remove most of the ~50%
multi-node startup flakiness at the cost of NVLS collective acceleration. NOT applied -- it
changes every run and needs a decision.

## 2026-09-17 (later) — MULTI-NODE: two distinct bugs separated; our code is CORRECT, the fabric is not

**BUG 1 (OURS) — inductor `spmd_check` compile deadlock. FIXED.**
`lm_engine/distributed.py:57` sets `torch._inductor.config.reorder_for_compute_comm_overlap = True`.
That is OUR line: every related torch default is off. It makes `_needs_spmd_graph_preservation()`
true (`post_grad.py:1061` = `enable_overlap_scheduling OR reorder_for_compute_comm_overlap`), which
runs inductor's `spmd_check` pass. That pass calls **`dist.all_gather_object` DURING COMPILE** --
twice: once for graph hashes, again for diagnostics on the mismatch path. Our two hosts compiled
structurally DIFFERENT graphs for the same frame (`spmd_check` reported ranks 0-7 with **2**
call_function nodes and ranks 8-15 with **27**, split exactly on the host boundary), so the ranks
made different numbers of blocking collective calls and DEADLOCKED.
Symptoms, both of which I initially misdiagnosed:
  - with gloo's default 30-min timeout: `InductorError: Timed out waiting 1800000ms for send` at
    1884 s / 1921 s, 0 steps, 0 OOM, 39 GB of a 200 GB host limit. I first read this as "cannot
    compile a 773,760-wide block" -- WRONG.
  - with `timeout_minutes: 180`: a 108-minute silent HANG, cpu_used flat 746 -> 748 s across 16
    ranks. Raising the timeout did NOT fix anything, it only let the deadlock persist longer.
FIX: in `distributed.py`, when `n_nodes > 1`, set `reorder_for_compute_comm_overlap = False` and
`aten_distributed_optimizations.spmd_check = False`. Gated on topology so single-node keeps the
comm/compute overlap optimisation. This is the CORRECT fix rather than silencing the detector:
since our graphs genuinely differ across hosts, the reordering it guards would itself be unsafe.
VERIFIED: `DIFFS = 0`, no `InductorError`, compile completes and reaches the first collective --
which no prior 2-node attempt ever did.
STILL UNEXPLAINED: why two hosts produce 2- vs 27-node graphs for the same frame. Worth chasing;
it may indicate real non-determinism in the sparse path.

**BUG 2 (THE CLUSTER'S) — InfiniBand `IBV_WC_RETRY_EXC_ERR`. NOT OUR CODE.**
```
NET/IB: Got completion from peer 100.126.x.y<port> with status=IBV_WC_RETRY_EXC_ERR(12)
        opcode=IBV_WC_RECV_RDMA_WITH_IMM(129) vendor_err=129  hca mlx5_N
-> ncclRemoteError: remote process exited or there was a network error
```
An RDMA write exhausted its retries without an ACK from the peer HCA, on the FIRST inter-node
allreduce (`SeqNum=1`, `last completed work: -1` -- the documented startup-fault signature).
**This is the `ncclRemoteError` that has been open as a question for days. It is the fabric.**
NOT one bad port: **6+ HCAs** (mlx5_1/3/4/6/7/9) across **7+ peers** (100.126.12.28/.58/.60/.85/.99,
100.126.20.105/.109) and **4 different host pairs**.
Mitigations tried and their outcomes:
  - `NCCL_NVLS_ENABLE=0` -- NO EFFECT, and it was the WRONG LAYER: NVLS is intra-node NVLink SHARP;
    this fault is inter-node IB. (A genuine NVLS binding fault WAS also observed separately, on
    p3-r28-n3, so the flag is kept as cheap insurance for that.)
  - `NCCL_IB_TIMEOUT=22` + `NCCL_IB_RETRY_CNT=13` -- NO EFFECT. So it is not simple congestion.
  - `NCCL_DEBUG=WARN` -- kept: it is what surfaced the HCA and peer identifiers at all.
TIMELINE EVIDENCE FOR AN ADMIN REPORT: job 1719544 ran 120 steps on this exact path ~5 h earlier;
3/3 attempts failed afterwards. The fabric DEGRADED during the session, so it may clear on its own.

**THE DECIDING CONTROL — `NCCL_IB_DISABLE=1` (TCP transport): PASSED.**
`cmix1B_2node_tcptest`, 2 nodes (p2-r27-n4 + p3-r28-n1), 40/40 steps, STAT=DONE, **0 IB errors,
0 ncclRemoteError**, loss 9.5779 -> 7.6935 monotone, and it CROSSED the dense->sparse switch at
step 20 (3.5428 s -> ~1.30 s).
=> **Our code is multi-node-correct. The sole blocker is InfiniBand.**
Indicative throughput: 262144 tok/step / 1.30 s ~ 202k tok/s vs ~303k tok/s for the 8-GPU
single-node IB run, i.e. roughly 2/3 of IB -- FAR better than TCP usually costs. **DO NOT QUOTE
THIS**: the sparse-phase median rests on TWO points (steps 30, 40), and 2x4 carries less
inter-node traffic than the 2x8 shape we actually want. A few-hundred-step 2x8 measurement with a
proper windowed median is the outstanding question.

**FSDP tensor layout (asked: is the huge collective our fused expert tensor?) -- NO.**
The experts are ALREADY separate parameters: `I_e x d = 2846 x 1024 = 2,914,304` each (5.6 MiB
bf16), 64 of them per block. `_fused_W()` builds the `(K*I_e, hidden)` view at FORWARD time; there
is no stored fused parameter, so `fused_experts` costs nothing in layout.
The 191,267,969-element collective is FSDP's FlatParameter for ONE wrapped block, and the
arithmetic matches to the element: 186,515,456 (64 experts) + 4,752,513 (attn+norms+proxy).
`distributed.py:368` wraps with `transformer_auto_wrap_policy(transformer_layer_cls=block_classes)`
-- whole blocks. So the lever is FSDP WRAPPING GRANULARITY, not expert layout.
Note: finer wrapping would be LESS bandwidth-efficient, not more -- FSDP flattens precisely to
avoid many small latency-bound messages. It buys lower peak memory and better overlap. And size is
probably not the trigger anyway: 365 MiB is a routine allreduce, and the NVLS fault failed binding
2 MiB. Untested; needs sign-off since it changes every run.
