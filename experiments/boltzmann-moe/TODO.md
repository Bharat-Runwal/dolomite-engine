# Boltzmann MoE — TODO

## 2026-09-12: inference-FLOPs work (see HANDOFF.md §7 for all evidence)

**Done**
- [x] Exact per-token FLOP model for `math_fet_boltz_hopfield_rep_*`
      (`scripts/analyze_moe_router_flops_20260912.py`). 603.59M MACs/token;
      MoE is 25.0%, lm_head 25.5%, GPT prefix 37.5%.
- [x] Measured real routing + branch magnitudes + paired-NLL ablation on 3 FET
      checkpoints (`scripts/measure_moe_routing_20260912.py`, job 1559723).
- [x] Answered the standing open question *"why does truncating to top-2 cost 6 PPL?"*
      — for the Hopfield-MEAN MoE it costs **nothing**, because routing is uniform
      (`eff_n = 7.999/8`) and the FF branch is inert. The 6 PPL belongs to the
      **w1w2** line, where routing carries real information (~3.9 PPL).
- [x] `train_utils.py:73` — `FFEnergyBase` added to the metrics isinstance gate.
      No `EnergyFF_*` run had ever logged routing-collapse metrics.
- [x] `head_dim` decoupled from `hidden_size/num_heads` in `EnergyAttention_QK`,
      enabling over-complete heads (`num_heads*head_dim > d`). Default unchanged;
      forward+backward verified, partial RoPE already supported (`rope.py:118`).
- [x] Single-block all-MoE architecture family + FLOP/param calculator
      (`scripts/design_single_block_moe_20260912.py`), 7 configs emitted to
      `configs/single_block_moe/`.
- [x] Grouped (sort-by-expert) sparse inference path + throughput benchmark
      (`scripts/bench_moe_throughput_20260912.py`) — closes the TODO below about
      needing a real dispatch to measure anything.

**Next**
- [ ] **Fit the rank-r proxy router** on the cached `(x, E_k)` pairs in
      `results/router_analysis/router_fit_cache__*.pt` (frozen backbone). For
      Hopfield the ceiling is 96.7% top-1 at r=16; for w1w2 the projection-based
      ceiling estimate plateaus near 0.5 and is **non-monotone in r**, which is not
      a credible ceiling curve — extend r up to d (=768) to validate the estimator,
      and prefer fitting a head directly on `(x, E_k)` over projecting x.
- [x] **Repulsion loss was mis-specified** — FIXED (`repulsion_form`, default
      `squared`). `L_rep = λ·E[cos]` is minimised at cos = −1 and rewards
      anti-alignment: both FET-rep and h1 sit within ~3% of the geometric floor
      `−1/(K−1)`. **But it is NOT the cause of the dead branch** — h1 runs λ=0.1
      (10× FET's) and is 86% load-bearing. Use `abs` to preserve the existing λ
      sweeps; `squared` is ~3× weaker at the same λ. Re-sweep λ either way.
- [ ] **The real root cause: the Hopfield output prefactor `4/I_e`.**
      `mean ‖g_k‖` = 0.019 (Hopfield) vs 155.66 (legacy w1w2) — 8,200×, of which
      256× is the bare constant (`energy_ff.py:644` `4/I_e` = 0.0039 vs
      `mlp.py:693` no prefactor at all). `scale_ff` = 1.52 cannot bridge that.
      Change `4/I_e → 1/√I_e` on the Hopfield output and re-run. Fixing repulsion
      alone will NOT revive this line.
- [x] Scale-free routing — DONE (`routing_norm ∈ {none, zscore, sqrt_width}` on
      `BoltzmannMoEFFEnergy._logits`). `zscore` normalises to unit std across
      experts, so pair it with `temperature ≈ 0.35` to reach `eff_n ≈ 2`; plain
      z-score alone only reaches `eff_n` 5.6–6.4 of 8.
- [ ] Re-check the FF-branch strength of the remaining `math_fet_hopfield_mean_r*`
      register variants — the whole Hopfield-MEAN line measures 1.6–10.9% FF share
      vs 85.8% for w1w2, so published deltas in that family may be comparing
      near-inert branches.
- [ ] DROPPED: "route once at iteration 0, reuse for 1..T". Iteration-to-iteration
      argmax agreement is 0.27–0.30 on the w1w2 line (TV 0.24–0.27) — routing
      genuinely changes with depth. The 0.87–0.90 agreement on the Hopfield model
      was an artefact of its degenerate uniform routing.
- [ ] Train one `configs/single_block_moe/` point once the throughput numbers justify
      it. Note the recurrence tax: k/K must be ≲ 1/T to stay at parity with a
      conventional dense transformer at iso-total-params.

## 2026-09-14: training-step speedup for scale32B_boltz_hop (6.75s vs gptswitch 1.34s)

**Done**
- [x] Training-step microbench (fwd+BWD, production shape) —
      `scripts/bench_moe_train_step_20260914.py` →
      `results/router_analysis/bench_moe_train_step_20260914.json`. Findings:
      repulsion = +65% of the MoE block (single largest cost); two_stage is SLOWER
      (0.59× at N=4096, grouped-loop backward); proxy needs a fused kernel (no FLOP
      saving at 4096 tok/call); 1-in-10 output repulsion → 1.55× block speedup;
      weight-space repulsion 3.5× cheaper AND sparse-compatible. See PROGRESS.md.
- [x] (a) Phase-B profiler config `configs/iclr_scale/scale32B_boltz_hop_PROFILE.yml`
      (throwaway; engine TorchProfiler; 4 preemptable GPUs, job 1658324) — trace →
      `results/profiler/boltz_hop_trace`. PARSED (`scripts/parse_profiler_trace_20260915.py`).
      RESULT: one step = 1712.99 ms GPU kernel time; **elementwise 61.0% / GEMM 25.3%
      / attention 4.3% / comms 0.4%**. Both elementwise AND GEMM are dominated by the
      dense-32-expert MoE (I_e=1280 projections fire 1536×/step = 32×6×8). **The ~14%
      MoE estimate was REFUTED — the 5× gap is `recurrence × dense-32-expert`.** See
      PROGRESS.md "Phase B RESULT (2026-09-15)".
- [x] (b) Intermittent-repulsion patch DRAFTED (not applied; needs sign-off) —
      `results/router_analysis/intermittent_repulsion_20260914.patch`. Adds
      `repulsion_interval` (default 1 = current) + `repulsion_scale_comp` to
      `BoltzmannMoEFFEnergy`, opt-in per config. NOTE: profiler shows repulsion is
      only ~2–4% of the whole step (not 39%: that was one isolated block-call), so
      this is a safe near-free win but NOT the headline lever.

**Next**
- [ ] **Headline levers to close the 5× gap (from the trace, biggest first):**
      (1) fused top-k sparse **back-projection** (skip 15/16 of the [·,1280]@[1280,1536]
      ≈244 ms); (2) **rank-r proxy router** (Hopfield ceiling 96.7% top-1 @ r=16 —
      TODO line 28) to skip the all-32 fwd projection ≈266 ms; (3) **fuse the
      per-expert energy loop** (grouped-GEMM + fused gelu²-mean) to cut both
      elementwise and the launch-overhead gap (79.5 k kernels/step, GPU-busy 1.71 s ≪
      6.75 s wall); (4) route-once-reuse across the 6 recurrence iters ONLY if
      verified stable on the trained config (re-check TODO line 54's 0.87–0.90).
- [ ] (b) patch: ship it opportunistically (safe, ~2–3% step) once sign-off is given;
      `repulsion_interval: 10, repulsion_scale_comp: true`. Not gating on it.
- [x] **Overleaf Avg11 — RESOLVED 2026-09-15.** User's call: the target paper is the
      **ICLR draft** (`/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`), NOT the
      NeurIPS one, and it migrates fully to Avg11. This turned out to need **no
      re-eval at all**: every `iclr_*` run already carries race+lambada, so all 22
      yield a COMPLETE Avg11. Migration applied and PUSHED to Overleaf (commit
      `Migrate all reported numbers from avg10 to canonical Avg11`). Number-by-number
      record + the claims that changed: **`AVG11_ICLR_MIGRATION.md`**.
      The NeurIPS draft was deliberately left alone — it is archive, still on avg10.
- [x] **32B gptswitch arm FINISHED (2026-09-15)** — job 1647503 DONE at step
      61,035 = the full 32B tokens, final train-loss 2.8277, 1.31 s/step. It had
      **no eval at all**; agent tasked with unshard + full harness. This is the
      completed half of the 400M headline pair and the paper's first real
      400M/32B number. **The Boltzmann half is only at step ~7370 of 61,035
      (12%)** at 6.79 s/step → ~4.2 more days as-is, ~2.6 days with
      `fused_experts` (see ACCEL_FINDINGS on the `boltz-accel` branch).
- [ ] **DO NOT paper-cite the 4 newly-evaluated arms — all are PARTIAL.**
      w1w2_K32_top2 10k/30k (33%), slope90k_1blk 40k/90k, pure_hop_isoP_bal
      16k/30k, iclr_big_hop_sandwich 4k/15k. Their Avg11 is complete (11/11 tasks)
      but not comparable to the finished 30k grid. In particular w1w2_K32_top2
      (41.63, PPL 56.00) does NOT yet resolve Hopfield-vs-W1W2 — it is at a third
      of Hopfield K=32's budget and its PPL says undertrained, not worse. The
      paper's "that arm is training" wording stays.
- [ ] **Fold in the remaining ICLR runs that had NO eval** (agent submitted the
      jobs 2026-09-15): `iclr_decide/w1w2_K32_top2` (**the arm `tab:threeway` says
      "resolves" Hopfield-vs-W1W2 — highest value**), `iclr_slope/slope90k_1blk`,
      `iclr_big/iclr_big_hop_sandwich` (the paper's one remaining acknowledged
      baseline gap), `iclr_balance/pure_hop_isoP_bal`, and the two live
      `iclr_scale/scale32B_*` arms. Skipped: `pure_hop_isoP_bal_DIVERGED_*`.

## After Mon 2026-06-01 talk

### A/B test: tanh_exact φ' (DONE — surprising negative result, follow-up needed)

**Result (2026-06-01)**: `tanh_exact` lost decisively to `sigmoid` at h1_boltz_fullsize scale, 30k steps, 7.86B tokens:

| | sigmoid (control) | tanh_exact (treatment) | Δ |
|---|---:|---:|---:|
| Avg | 50.10 | 47.90 | **−2.20pp** |
| WikiPPL | 36.48 | 39.90 | **+3.42** |
| flex-avg | 2.12 | 2.05 | −0.07 |

The fix was supposed to make φ' a faithful derivative of φ. Instead the model trained a worse representation. Two confounded changes:
  (a) φ shape switched from `F.gelu` (exact erf-based) to tanh-approx GELU
  (b) φ' magnitude doubled (legacy was uniformly half the true gelu')

**Default unchanged: keep `gelu_grad_method: sigmoid`.**

**Next A/B (cheap, isolates hypothesis (b))**: keep `phi = F.gelu` and compute `phi' = 0.5·(1 + erf(x/√2)) + x·exp(−x²/2)/√(2π)` (analytic exact derivative of `F.gelu`). One extra erf+exp vs current; still sub-millisecond per layer at our scale. Add a third `gelu_grad_method` value: `"erf_exact"`. Run h1_boltz_fullsize_erfexact and compare to both sigmoid and tanh_exact.

Hypothesis to test: if `erf_exact` beats `tanh_exact` and matches `sigmoid`, the issue is the φ-shape change (F.gelu shape matters). If `erf_exact` ≈ `tanh_exact` (both worse than sigmoid), the legacy half-magnitude φ' was acting as beneficial gradient damping and we shouldn't "fix" it.

Old plan below preserved for reference:

---

### A/B test: tanh_exact φ' (HIGH PRIORITY, cheap)

The legacy code uses `φ' ≈ sigmoid(c·x) · 0.5` paired with `φ = F.gelu(x)` —
but this `φ'` is uniformly **half** the true `gelu'(x)`, and is not actually
the derivative of `F.gelu`. Term2 in `∂E/∂h` is therefore systematically
under-magnitude; weight scales partially absorb this during training, but
it's still a non-faithful gradient.

**Fix landed (gated, default = old)**: `BoltzmannMoE_Energy_MLP` now takes
`gelu_grad_method ∈ {"sigmoid", "tanh_exact"}` (default `"sigmoid"` for
backward compat). Setting `tanh_exact` switches both φ and φ' to the
self-consistent tanh-approx-GELU pair:
  φ(x) = 0.5 · x · (1 + tanh(c·x))
  φ'(x) = 0.5 · (1 + tanh(c·x)) + 0.5 · c · x · (1 − tanh²(c·x))
with c = √(2/π). One extra elementwise mul vs current.

**A/B plan** (run two h1_boltz_fullsize variants from scratch, 30k steps,
4 GPUs preemptable — ~6 h each):
  1. h1_boltz_fullsize_sigmoid (control; current default)
  2. h1_boltz_fullsize_tanhexact (treatment; `gelu_grad_method: tanh_exact`)
Otherwise identical configs. Compare avg / WikiPPL / GSM8k flex-avg.

Hypothesis: if phi' magnitude was being absorbed by W2 scale, we expect
small net effect on quality (≤ ±0.3pp avg). If the *shape* matters
(GELU' has a peak around x≈0.6 that sigmoid·0.5 misses), expect a
meaningful improvement on representation quality. Either way, **the
correct gradient should be the new default once we have data**.

If tanh_exact ≥ sigmoid: switch default to `tanh_exact` for new training,
keep `sigmoid` available for legacy checkpoint reproducibility.

Test (already run): `experiments/boltzmann-moe/tests/test_gelu_grad_method.py`
verifies (1) backward-compat (sigmoid path bit-identical to legacy),
(2) tanh_exact φ' matches autograd, (3) the two paths produce different
outputs on the same weights. Run via bsub when iterating on the model.

---


### MoE hyperparam search (queued for later)
Goals: find the best Boltzmann MoE config at h1 scale (145M, d=768, K=4, K=8, K=16)
**without** consuming so many GPUs at once that we get deprioritized in the queue.

Submit at most 3–4 jobs at a time; let them complete before the next batch.

Axes to sweep (in priority order):
- **Sparsity**: `top_k` ∈ {2, 3, 4} of K=4; {2, 4, 8} of K=8; {2, 4, 8, 16} of K=16
  — most promising follow-up: top-3 of 4 may close most of the 1.5pp acc gap
  while preserving sparse-train benefits ([[project_sparse_boltz_perf]]).
- **Temperature τ**: {0.5, 1.0, 2.0, 4.0}. Higher τ softens routing distribution
  → less drastic truncation → smaller flex-acc gap for sparse top-k.
- **Repulsion λ**: {0.0, 0.01, 0.1, 0.3, 1.0}. B4/B5 already showed 0.1 is good
  load balance; 0.3+ untested with new routing scale.
- **Number of experts K**: {4, 8, 16, 32} at h1 scale. Bigger K + smaller per-expert
  gives sparse a clearer FLOPs win (44% saving at K=16, top_k=2 vs 25% at K=4).
- **Repulsion target**: cosine sim (current) vs L2 distance vs orthogonality
  loss on term1.
- **Anneal top_k**: train soft → gradually reduce top_k over training. Should
  recover both the soft-train gradient signal and the sparse-eval efficiency.

### Sparse Boltzmann MoE production kernel
- Implement scattermoe-style fused dispatch for top-k Boltzmann gradient terms.
  Per-expert PyTorch loop is correct but loses on backward at every K (see
  [[project_sparse_boltz_perf]]). Need this to make sparse routing a real
  training-FLOPs win, not just an inference-FLOPs win.
- Or: try `torch._grouped_mm` (CUDA 12+) as an easier first step.

### Better integration of routing diagnostics into wandb
- Per-layer `effective_n_experts`, `n_dominant_experts`, `max_expert_load` are
  already logged. Add a heatmap visualization or per-step routing-entropy
  histogram for easier spotting of collapse runs.

### Larger scale Boltzmann MoE
- 580M continues; results @ 14k & 18k are encouraging. Plan:
  - 1.5B Boltzmann MoE (K=8, d=2048) at 65B+ tokens once 580M finishes
  - Try repulsion λ scaling with K: λ ∝ 1/K?

### Open questions
- Why does the soft-trained model lose 6 PPL when truncated to top-2 at eval?
  Hint: sparse-trained model doesn't pay this cost ([[project_sparse_boltz_perf]]).
  Hypothesis: the soft model's bottom-k experts contribute small-but-essential
  refinements during training; truncating at eval drops these. Test by
  measuring per-expert L2 norm of `term2_e * p_e` in trained vs sparse-trained.
- Is 580M's GSM8k_cot flex jitter (2.88% @ 14k → 2.27% @ 18k) within stderr
  (±0.4pp) or a real regression? Run cot eval at step 22k, 26k to see trend.
