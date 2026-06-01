# Boltzmann MoE — TODO

## After Mon 2026-06-01 talk

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
