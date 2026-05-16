# BoltzmannMoE Experiments — Progress & Results

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

---

## Baseline comparison

| Model | Params | FFN:Attn | Avg acc | WikiPPL |
|-------|--------|----------|---------|---------|
| V9 GPT d=1024 | 354M | 2.0× | **0.513** | **29.8** |
| V1-400M EGPT d=1024 | 354M | 3.0× | 0.494 | 38.6 |
| **h1_topk_egpt_moe** | ~200M | balanced | **0.499** | 39.8 |
| V1 EGPT d=768 | 143M | 2.7× | 0.481 | 47.7 |
| h1_topk_egpt_moe_r128 | ~200M | balanced | 0.484 | 39.6 |
| V58 EGPT recurrent | 113M | 2.7× | 0.459 | 65.7 |
| B1 (BoltzMoE best) | 407M | **21.3×** | 0.474 | 51.9 |
| B4 (best load balance) | 407M | 21.3× | 0.466 | 51.9 |

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
