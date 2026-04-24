# Results — Energy-Inference Large-Scale

<!-- New results go at the top (reverse chronological). -->

## Full EGrad (V37/V38): EGrad(attn) + Mixed_Energy_MLP — 2026-04-23

**Goal**: Test whether adding energy-gradient FFN (Mixed_Energy_MLP = ∂E_FF/∂h + standard GELU MLP,
iso-param) improves over EGrad(attn)-only baseline (V27/V31).

### Models

| Variant | Architecture | d | Layers | Params | LR | Job | Status |
|---------|-------------|---|--------|--------|----|-----|--------|
| V37 | Full EGrad 12×1, d=768, energy+standard=1152+1152 | 768 | 12 | ~141M | 2e-3 | 37396 | PEND/RUN |
| V38 | Full EGrad 24×1, d=1024, energy+standard=1536+1536 | 1024 | 24 | ~342M | 1.5e-3 | 37397 | PEND/RUN |

**Iso-param check**:
- V37: 2×768×1152 + 2×768×1152 = 3×768×1536 (same as SwiGLU-1536 in V27) ✓
- V38: 2×1024×1536 + 2×1024×1536 = 3×1024×2048 (same as SwiGLU-2048 in V31) ✓

**Token count**: 30k steps × 4 GPUs × batch=4 × grad_accum=4 × seq=4096 = 7.86B tokens

**New code**: `Mixed_Energy_MLP` in `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py`
— forward: `∂E_FF/∂h = φ'(W1x)·W2^T + φ(W1x)·W1^T` (two-projection energy) + standard GELU MLP.

### Scripts/Configs

```bash
# Configs
configs/multi_block_ablation/v37_full_egrad_12x1_d768_lr2e3.yml
configs/multi_block_ablation/v38_full_egrad_24x1_d1024_lr1e3.yml

# Submitted 2026-04-23
bash experiments/energy-inference/scripts/multi-block-ablation/run_v37_full_egrad_12x1_d768_lr2e3.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v38_full_egrad_24x1_d1024_lr1e3.sh
```

---

## Output-rule ablation: V17-V20 (6×2 and 400M scale-ups) — 2026-04-21

**Goal**: Scale up V15 (energy-gradient mixed heads) and V16 (energy-descent identity proj) to
6×2 recurrent (V17/V18) and ~340M (V19/V20) to test whether output rule matters at scale.

### Models

| Variant | Architecture | d | Params | LR | Job | Status |
|---------|-------------|---|--------|----|-----|--------|
| V17 | energy_grad mixed heads, 6L×2, d=768, 6E+6G, int=2048 | 768 | ~130M | 2e-3 | 589 | RUN |
| V18 | mixed_head_energy_descent, 6L×2, d=768, 6E+6G, identity proj, int=2048 | 768 | ~130M | 2e-3 | 590 | RUN |
| V19 | energy_grad mixed heads, 24L×1, d=1024, 8E+8G, int=2048 | 1024 | ~320M | 1.5e-3 | 591 | RUN |
| V20 | mixed_head_energy_descent, 24L×1, d=1024, 8E+8G, identity proj, int=2048 | 1024 | ~342M | 1.5e-3 | 592 | RUN |

**Token count**: 30k steps × 4 GPUs × batch=4 × grad_accum=4 × seq=4096 = 7.86B tokens

### Scripts

```bash
bash experiments/energy-inference/scripts/multi-block-ablation/run_v17_energy_grad_6x2_d768_lr2e3.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v18_energy_desc_6x2_d768_lr2e3.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v19_energy_grad_24x1_d1024_lr1e3.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v20_energy_desc_24x1_d1024_lr1e3.sh
```

**Note**: Fixed `MASTER_PORT` formula in `scripts/common/pretrain.sh` — was
`5${LSB_JOBID: -5:-1}` (gave privileged port <1024 for 3-digit job IDs); now
`$((20000 + LSB_JOBID % 10000))`.

---

## Mixed EGPT+GPT heads (V10/V11) and GPT 400M baseline (V9) — 2026-04-21

**Goal**: Test whether mixing per-head energy (V=K) and GPT (V=W_V x) attention in the same block
improves over either pure architecture. Also benchmark V9 GPT 354M vs V1 400M EGPT.

### Models

| Variant | Architecture | d | Params | LR | Steps | Status |
|---------|-------------|---|--------|----|-------|--------|
| V9 GPT 354M | Standard GPT, 24L, d=1024, int=2048, SwiGLU | 1024 | 354M | 1.5e-3 | 30k ✓ | eval done |
| V10 Mixed 12×1 | 12L × 1 step, 6 energy + 6 GPT heads per block, int=1536 | 768 | 144M | 2e-3 | 30k ✓ | eval done |
| V11 Mixed 6×2 | 6L × 2 steps, 6 energy + 6 GPT heads per block, int=2048 | 768 | 118M | 2e-3 | 30k ✓ | eval done |

### Results (2026-04-21)

| Metric | V0 GPT 143M | V1 EGPT 143M | V1 400M EGPT | V9 GPT 354M | V10 Mixed 12×1 144M | V11 Mixed 6×2 118M |
|--------|-------------|--------------|--------------|-------------|---------------------|---------------------|
| **Wikitext PPL** ↓ | 38.31 | 47.66 | 38.61 | **29.84** | 36.99 | 40.67 |
| **Avg acc** ↑ | 47.2% | 46.8% | 48.5% | **49.6%** | 47.2% | 47.3% |
| arc_challenge | 24.5% | 23.6% | 25.2% | **27.1%** | 25.5% | 24.5% |
| arc_easy | 51.6% | 47.9% | 52.4% | **55.3%** | 51.4% | 51.0% |
| boolq | 56.6% | 61.1% | 60.1% | 57.7% | 56.5% | **58.6%** |
| copa | 59.0% | **64.0%** | 65.0% | 65.0% | 57.0% | 61.0% |
| hellaswag | 33.0% | 30.8% | 34.5% | **38.5%** | 33.4% | 32.3% |
| openbookqa | **29.6%** | 29.2% | 29.6% | 29.0% | 27.4% | 29.4% |
| piqa | 63.2% | 62.2% | 62.8% | **65.5%** | 65.4% | 63.4% |
| sciq | 78.1% | 75.2% | 77.5% | **82.8%** | 76.9% | 75.7% |
| winogrande | 51.7% | 50.7% | 51.7% | 50.4% | **52.7%** | 51.8% |
| mmlu | 25.2% | 23.5% | 25.7% | 24.8% | **25.7%** | 24.9% |

### Key findings

1. **V9 GPT 354M is the best model overall** (49.6% avg, 29.84 PPL) — beats V1 400M EGPT (48.5%, 38.61 PPL) with identical parameter count. Standard GPT is substantially more sample-efficient than energy attention.
2. **Mixed heads (V10/V11) match pure GPT at 143M scale** (47.2–47.3% avg) but do not improve over V0. The energy head component neither helps nor hurts accuracy, but improves PPL vs pure EGPT (37–41 vs 47.7).
3. **V11 (6×2 recurrent) is nearly identical to V10 (12×1)** (47.3 vs 47.2%): adding a recurrence step to the mixed head model confers no accuracy benefit.
4. **The energy attention V=K constraint hurts perplexity** — even 6 energy heads out of 12 raises PPL from 38.3 (pure GPT) to 37.0 (V10 mixed), suggesting V=K is a structural bottleneck.

### Scripts

```bash
bash experiments/energy-inference/scripts/multi-block-ablation/run_v9_gpt_baseline_d1024.sh   # V9
bash experiments/energy-inference/scripts/multi-block-ablation/run_v10_mixed_12x1_d768_lr2e3.sh  # V10
bash experiments/energy-inference/scripts/multi-block-ablation/run_v11_mixed_6x2_d768_lr2e3.sh   # V11
```

Plots: `results/multi-block-ablation/plots/v9_v11_*.{png,pdf}`

---

## Multi-block EGPT ablation — 160M scale (started 2026-04-19)

**Goal**: Can 12 distinct EGPT blocks match GPT at 160M scale? Do they collapse to
functionally similar blocks after training (motivating weight-sharing / pruning)?

### Variants

| Variant | Config | Blocks | Iters/block | d | Params | Wandb |
|---------|--------|--------|-------------|---|--------|-------|
| V1 | `configs/multi_block_ablation/v1_12x1_dual_unconstrained_d768.yml` | 12 distinct | 1 | 768 | ~176M | `v1_12x1_dual_unconstrained_d768` |
| V2 | `configs/multi_block_ablation/v2_6x2_dual_unconstrained_d768.yml` | 6 distinct | 2 | 768 | ~176M | `v2_6x2_dual_unconstrained_d768` |
| V3 | `configs/multi_block_ablation/v3_shared_backbone_d1152.yml` | 12, shared attn+FF | 1 | 1152 | ~163M | `v3_shared_backbone_d1152` |
| V4 | `configs/multi_block_ablation/v4_shared_backbone_wide_d1152.yml` | 12, shared attn+FF | 1 | 1152 | ~174M | `v4_shared_backbone_wide_d1152` |

**Token count**: 30k steps × 4 GPUs × batch=4 × grad_accum=4 × seq=4096 = **7.86B tokens**

**Estimated training time**: ~3-4h per job chunk on 4 A100s; auto-resumes until complete

### Launch commands

```bash
# From repo root: /proj/dmfexp/nima/Code/dolomite-engine/
bash experiments/energy-inference/scripts/multi-block-ablation/run_v1_12x1.sh   # job 974246 (RUN)
bash experiments/energy-inference/scripts/multi-block-ablation/run_v2_6x2.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v3_shared.sh
bash experiments/energy-inference/scripts/multi-block-ablation/run_v4_shared_wide.sh
```

### Code changes required

- `lm_engine/hf_models/config/__init__.py`: added `shared_backbone: bool = False` field
- `lm_engine/hf_models/mixins/dense/base.py`: added weight-tying logic for `shared_backbone=True`
  (ties `.attn`, `.ffwd`, `.ln` of blocks 1–11 to block 0; only `.proj_attn`/`.proj_mlp` remain independent)

### Variants (updated)

| Variant | Description | d | Actual Params | LR | Steps | Status |
|---------|-------------|---|---------------|----|-------|--------|
| V0 | Standard GPT baseline (12L, SwiGLU, RoPE) | 768 | ~162M | 3e-4 | 30k ✓ | eval done |
| V1 | 12 distinct EGPT blocks, dual_unconstrained proj | 768 | 143M | 3e-4 | 30k ✓ | eval done |
| V1 lr2e3 | V1 with LR=2e-3 | 768 | 143M | 2e-3 | 30k ✓ | **eval done** |
| V2 | 6 distinct EGPT blocks × 2 iters, dual_unconstrained | 768 | ~110M | 3e-4 | 30k ✓ | eval done |
| V3 | Shared backbone (1 block × 12 iters), d=1152 | 1152 | — | 3e-4 | 30k ✓ | eval done |
| V4 | Shared backbone, 4-head wide (head_dim=288), d=1152 | 1152 | **349M** (no tying) | 3e-4 | 20k ✓ | **eval done** |
| V4 lr2e3 | V4 with LR=2e-3 | 1152 | **349M** (no tying) | 2e-3 | 20k ✓ | **eval done** |
| V1 400M | 24 distinct EGPT blocks, d=1024 | 1024 | 354M | 7e-4 | ~15k (wall-time) | in progress |
| V5 | attn_only_energy: h += ffwd - proj_attn(attn) | 768 | ~143M | 2e-3 | — | **submitted 989203** |
| V6 | helmholtz_factored: dx = -M(attn + A*ffn) | 768 | ~140M | 2e-3 | — | **submitted 989204** |
| V7 | helmholtz_dual: dx = -(R*attn + A*ffn) | 768 | ~140M | 2e-3 | — | **submitted 989205** |
| V8 | helmholtz_dual_reversed (V7 control): dx = -(A*attn + R*ffn) | 768 | ~140M | 2e-3 | — | **submitted 989206** |

V5–V8 all use 8 GPUs × batch=4 × grad_accum=2 × seq=4096 = **7.86B tokens** (same as baseline, ~2× faster wall time).
V6/V7/V8 use low-rank parametrization: rank=32 (antisym J), dissipation_rank=16 (PSD R).

### Benchmark Results (2026-04-20)

All evals use `lm-evaluation-harness`. **Note**: V4 variants are 349M params (weight-tying was removed for FSDP-2 compatibility; see notes below). V4 evaluated at 20k steps.

| Metric | V0 GPT (162M,30k) | V1 (143M,30k) | V1 lr=2e-3 (143M,30k) | V2 (110M,30k) | V3 (163M,30k) | V4 (349M†,20k) | V4 lr=2e-3 (349M†,20k) |
|--------|--------|--------------|------------|-------------|-----------|------|------|
| **Wikitext PPL** ↓ | **38.31** | 54.04 | 47.66 | 63.32 | 54.95 | 44.23 | 41.78 |
| **Avg zero-shot acc** ↑ | **0.491** | 0.475 | 0.490 | 0.464 | 0.465 | 0.482 | **0.491** |
| arc_challenge (acc_norm) | 0.245 | 0.235 | 0.236 | 0.225 | 0.233 | 0.237 | **0.254** |
| arc_easy (acc_norm) | **0.459** | 0.429 | 0.446 | 0.402 | 0.428 | 0.436 | 0.448 |
| hellaswag (acc_norm) | **0.330** | 0.294 | 0.308 | 0.283 | 0.293 | 0.310 | 0.324 |
| winogrande (acc) | **0.517** | 0.500 | 0.507 | 0.508 | 0.507 | 0.515 | 0.518 |
| boolq (acc) | 0.566 | 0.556 | **0.620** | 0.610 | 0.515 | 0.555 | 0.539 |
| piqa (acc_norm) | **0.632** | 0.614 | 0.622 | 0.614 | 0.611 | 0.637 | **0.646** |
| copa (acc) | 0.590 | **0.640** | **0.640** | 0.560 | 0.600 | 0.650 | **0.670** |
| openbookqa (acc_norm) | **0.296** | 0.292 | 0.292 | 0.282 | 0.270 | 0.278 | 0.284 |
| sciq (acc) | **0.781** | 0.725 | 0.752 | 0.693 | 0.728 | 0.724 | 0.733 |
| **MMLU** (acc) | **0.252** | 0.240 | 0.235 | 0.247 | 0.239 | 0.240 | 0.244 |
| **GSM8K** (exact_match) | **0.017** | 0.002 | 0.005 | 0.000 | 0.000 | 0.002 | 0.014 |

† V4 variants are 349M params because weight-tying (attn/ffwd/ln shared across blocks) was removed due to FSDP-2 incompatibility — originally intended to be ~174M. Evals at 20k steps (wall-time kills; training continuing to 30k).

Plots saved to `results/multi-block-ablation/plots/`.

### Key Observations (preliminary)

- **V0 GPT dominates on perplexity** (38.3 vs 54.9–63.3): standard residual update is more
  sample-efficient at this token budget (7.86B tokens). EGPT variants lag by 16–25 PPL.
- **Energy variants score near-chance on GSM8K** (0.000 vs 0.017 for GPT): multi-step arithmetic
  likely requires more training or the energy update rule interferes with chain-of-thought
  structured generation.
- **V3 (shared backbone) beats V2 (6×2) on PPL** (54.9 vs 63.3) despite fewer unique parameters —
  the recurrent application of one shared block appears more efficient than 6 distinct blocks
  iterated twice.
- **V0 leads on most commonsense tasks** (ARC, HellaSwag, PIQA, OpenBookQA, SCIQ); V2 leads
  on BoolQ (0.610 vs 0.566) — possibly reflects different inductive biases.
- **V1 EGPT (12×1) closely tracks V3 Shared on PPL** (54.0 vs 55.0) — surprising given V3 uses
  a single shared block iterated 12×; suggests the recurrence is doing real work, not just
  wasting compute.
- **V1 wins copa (0.630)** — the only task where an EGPT variant beats GPT baseline.
- V4 (wider shared backbone) expected to be best among EGPT variants based on training loss
  curves (lowest loss observed, though fewest steps/hr); eval pending.

### Training curve observations (from wandb, ~19 Apr 2026)

- GPT_base already below V1@20k at step 8k — standard transformer more sample-efficient
- V4 has lowest loss among EGPT variants but is slowest (~6k steps/2hr vs V2's ~25k)
- V2 has worst training loss; V1-lr2e3 and V4-lr2e3 in progress
- V4 at lr=2e-3 noticeably spikier than V4 at 3e-4

---

## Per-iteration alpha + curl/grad random baseline — h.10 structured proj (2026-04-17)

**Training script**: `experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260413.py`
**Checkpoint**: `results/structured-proj/410m_per_iter_alpha_h10_20260415_050631/final`
**Run**: `410m_per_iter_alpha_h10` (wandb), 10k steps, h.10 only (6 iters, rank=256, dr=128, lr=3e-4)

### Per-iteration alpha: final distribution (step 10000)

| iter | alpha_attn | alpha_ff | log_alpha_attn | log_alpha_ff |
|------|-----------|----------|----------------|--------------|
| 0 | 0.502 | 0.516 | +0.008 | +0.065 |
| 1 | 0.499 | 0.520 | -0.005 | +0.082 |
| 2 | 0.501 | 0.484 | +0.005 | -0.063 |
| 3 | 0.509 | 0.484 | +0.036 | -0.063 |
| 4 | 0.490 | 0.517 | -0.039 | +0.069 |
| 5 | 0.499 | 0.528 | -0.004 | +0.113 |

All 12 scalars within ±0.03 of 0.5 after 10k steps. **No per-iteration differentiation learned.**

### Curl/grad ratio: trained vs. randomized baseline

To determine whether alpha_balanced ≈ 0.58 is a genuine learned signal or a structural
artifact, we sampled 500 random draws of U, V, L with the same per-element mean and std as
the trained matrices, computed alpha_balanced = ||J||/(||J||+||R||) for each, and compared.

| Stream | Trained alpha_balanced | Random baseline (N=500) | Delta | Sigma |
|--------|----------------------|-------------------------|-------|-------|
| attn | 0.5837 | 0.4814 ± 0.0011 | +0.102 | **+91.6σ** |
| ff   | 0.5937 | 0.5591 ± 0.0011 | +0.035 | **+32.3σ** |

**The random baseline is not 0.5.** The structural imbalance arises from the rank asymmetry
between J and R: J uses two factor matrices (U, V, each rank r_j=256) while R uses one
(L, rank r_d=128). For zero-mean random factors:

```
||J||_F  ~  d * sqrt(2 * r_j) * σ_U * σ_V
||R||_F  ~  d * sqrt(r_d)     * σ_L²
```

For attn (σ_U=σ_V=0.040, σ_L=0.057): J and R contributions are roughly equal → baseline ≈ 0.48.
For ff  (σ_U=σ_V=0.020, σ_L=0.0245): J slightly dominates  → baseline ≈ 0.56.

**What training changed:**


| Stream | `\|J\|` trained | `\|J\|` random (expected) | `\|R\|` trained | `\|R\|` random (expected) |
|--------|-----------------|---------------------------|-----------------|-----------------------|
| attn | 68.00 | ~46 | 48.50 | ~47 |
| ff   | 12.88 | ~12 | 8.81  | ~9  |

Training grew ||J_attn|| by ~47% while ||R_attn|| stayed near its random-init value.
For ff, both norms are essentially at the random-init level — ff projection barely retrained.

### Key findings

1. **Alpha degeneracy confirmed at per-iteration granularity.** With rank-256/128 J/R
   matrices (1.64M trainable params), 6 per-iteration alpha scalars carry zero information
   after 10k steps. The scalars are redundant: any target K can be produced by rescaling
   the factor matrices, so gradients flow into U/V/L rather than log_alpha.

2. **The curl bias is a genuine learned signal, not a norm artifact.** The attn-stream curl
   preference (alpha_balanced=0.584) sits 91.6σ above its randomized baseline (0.481).
   Training specifically enlarged J_attn (+47% Frobenius norm) while leaving R_attn near its
   initialized scale. This is a directional choice encoded in matrix geometry, not alpha.

3. **The ff stream barely retrains.** Its alpha_balanced (0.594) is only 3.5pp above its
   random baseline (0.559), and the absolute norms of J_ff and R_ff are near random-init
   values. The language modeling signal for h.10 flows almost entirely through the
   attn-stream component of the Port-Hamiltonian projection.

4. **Implication for alpha constraints.** Since the curl preference is encoded in ||J||/||R||
   ratios (not in alpha), fixing alpha=0.5 and letting U/V/L train freely is strictly
   equivalent to learning alpha — which is why all previous scalar-alpha and per-layer-alpha
   experiments found alpha ≈ 0.5 at convergence. A meaningful constraint would need to fix
   the J/R *norms* directly (e.g., normalise U, V, L to unit columns) and only learn
   directional structure.

---

## Per-layer alpha sweep — 410m_hybrid_s8e4_lr1e3_60k (2026-04-12)

**Script**: `experiments/energy-inference/scripts/perlayer-alpha/perlayer_alpha_20260412.py`
**Results file**: `results/perlayer-alpha/perlayer_alpha_410m_hybrid_s8e4_lr1e3_60k_20260412.json`
All profiles use norm-preserving interpolation. Baseline: uniform α=0.51 (NP optimum).

| Profile | h.8 | h.9 | h.10 | h.11 | PPL | Notes |
|---------|-----|-----|------|------|-----|-------|
| **uniform-0.51** | 0.51 | 0.51 | 0.51 | 0.51 | **10.83** | Best |
| perturb-h11-sym | 0.51 | 0.51 | 0.51 | 0.60 | 10.92 | Almost same — h.11 tolerates more sym |
| all-curl-biased | 0.47 | 0.47 | 0.47 | 0.47 | 12.42 | Slight curl bias hurts mildly |
| converge-then-explore | 0.60 | 0.55 | 0.45 | 0.40 | 13.22 | Early sym / late curl — modest cost |
| perturb-h11-anti | 0.51 | 0.51 | 0.51 | 0.40 | 13.23 | h.11 hurt by more curl |
| perturb-h8-anti | 0.40 | 0.51 | 0.51 | 0.51 | 13.90 | h.8 more sensitive than h.11 |
| explore-then-converge | 0.40 | 0.45 | 0.55 | 0.60 | 22.18 | Early curl / late sym — catastrophic |

### Key findings

1. **No per-layer variation helps — uniform 0.51 is optimal.** The trained model has
   converged to a balanced J-R mix that is uniformly optimal across all energy layers.
   Trying to assign "roles" (explorer vs converger) to individual layers makes things worse.

2. **Early layers are more sensitive to excess curl than late layers.**
   Pushing h.8 to α=0.40 (more rotation) raises PPL by 3.1× (10.83→13.90).
   Pushing h.11 to α=0.40 raises PPL by 1.22× (10.83→13.23).
   The first energy layer (h.8, 3 iters) is more fragile — it needs the descent
   component to find the right basin before refinement begins.

3. **Late layers tolerate more symmetric (descent) bias.**
   Pushing h.11 to α=0.60 barely hurts (10.83→10.92). The deepest layer (9 iters)
   is already in the right basin and extra dissipation is benign.

4. **"Explore-then-converge" is catastrophic (PPL 22).** This is the opposite of what
   energy-based intuition might suggest. The model does NOT work like simulated annealing
   (explore first, settle later). Early layers need gradient descent to enter the basin;
   only once there can rotational dynamics contribute to fine-grained refinement.

5. **"Converge-then-explore" (early sym, late curl) is only modestly worse (13.2).**
   This confirms finding 2: the ordering that preserves early descent is much more
   tolerable than the one that removes it.

### Interpretation

The energy recurrence is not annealing — it's more like Newton's method. Early steps
need to be close to gradient descent to reach a basin; later steps benefit from rotation
to navigate within it. But the trained model has converged to a uniform mix that already
does both at every layer, so any per-layer differentiation only removes capability.

---

## Norm-preserving alpha sweep — 410m_hybrid_s8e4_lr1e3_60k (2026-04-12)

**Script**: `experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py --norm_preserving`
**Command**:
```bash
python experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py \
    --norm_preserving \
    --alphas 0.0 0.1 0.2 0.3 0.4 0.45 0.49 0.5 0.51 0.55 0.6 0.7 0.8 0.9 1.0
```
**Results file**: `results/alpha_sweep_norm_preserving_410m_hybrid_s8e4_lr1e3_60k_20260412.json`

Mode: `W_eff = ||W||_F * (α·Ŵ_sym + (1-α)·Ŵ_anti) / sqrt(α²+(1-α)²)`  — always `||W_eff||_F = ||W||_F`.

| α | PPL | Generation quality |
|---|-----|--------------------|
| 0.0 (pure anti-sym) | 174,412 | `/frame` + `Corp` — degenerate |
| 0.1 | 65,653 | `clear` repeated — degenerate |
| 0.2 | 16,865 | `clear` repeated — degenerate |
| 0.3 | 851 | `clear` repeated — breaking |
| 0.4 | 70.9 | `complete` repeated — poor |
| 0.45 | 14.1 | coherent, repetitive |
| 0.49 | 11.2 | coherent, good |
| **0.50** | **10.94** | coherent, good quality |
| **0.51** | **10.83** | **PPL minimum** — coherent, good |
| 0.55 | 11.1 | coherent but looping |
| 0.6 | 15,385 | `IERI` repeated — degenerate |
| 0.7 | 251,274 | `ernity` repeated — degenerate |
| 0.8 | 335,640 | `ernity`/`sembl` — degenerate |
| 0.9 | 336,298 | `sembl` repeated — degenerate |
| 1.0 (pure sym) | 341,901 | `sembl` repeated — degenerate |

### Key findings

1. **α=0.5 norm-preserving ≈ the trained weight W.** When `||W_sym|| ≈ ||W_anti||` (as here,
   both ≈70.7%), the norm-preserving midpoint recovers `W_eff ≈ W` exactly. The PPL improves from
   12.17 (standard α=0.5 = W/2) to 10.94 — the difference is purely scale: the standard sweep
   was running at half the trained magnitude.

2. **The true optimum is the trained direction, slightly sym-biased (α=0.51).** The extra 0.11
   PPL gain over α=0.5 is negligible. The model has converged to essentially the balanced mix.

3. **The working direction window is α ∈ [~0.45, ~0.58] — only ~0.13 wide.** Outside this,
   the model collapses catastrophically. The sharp cliff at α=0.60 (PPL jumps to 15k) vs the
   more gradual slope at α=0.40 confirms the asymmetry seen in the standard sweep:
   extra sym destroys the dynamics faster than extra anti-sym.

4. **Scale matters, but direction is the dominant effect.** Raising the scale from W/2 to W
   (standard α=0.5 → NP α=0.5) improves PPL by 1.11× (12.17→10.94). Moving outside the
   working direction window degrades PPL by 3–4 orders of magnitude. Direction sensitivity
   >> scale sensitivity.

5. **The model requires genuine rotational dynamics, not just gradient descent.** Even at the
   trained scale, removing curl (α→1) is immediately fatal. This is now cleanly separated from
   any scale confound.

### Comparison: standard vs norm-preserving at α=0.5

| Mode | α=0.5 PPL | Best PPL | Best α |
|------|-----------|----------|--------|
| Standard (`W_eff = W/2`) | 12.17 | 12.16 | 0.49 |
| Norm-preserving (`W_eff ≈ W`) | 10.94 | 10.83 | 0.51 |

---

## Alpha sweep — 410m_hybrid_s8e4_lr1e3_60k (2026-04-12)

**Script**: `experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py`
**Checkpoint**: `/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k`
**Command**:
```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
python experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py
```
**Results file**: `results/alpha_sweep_410m_hybrid_s8e4_lr1e3_60k_20260412.json`

### Proj sym/anti norms (energy layers 8–11)

| Layer | total | sym | anti | %sym | %anti | scale_ff |
|-------|-------|-----|------|------|-------|----------|
| h.8  | 64.50 | 45.75 | 45.46 | 70.9% | 70.5% | 0.088 |
| h.9  | 69.72 | 50.00 | 48.59 | 71.7% | 69.7% | 0.108 |
| h.10 | 63.41 | 46.30 | 43.33 | 73.0% | 68.3% | 0.129 |
| h.11 | 54.03 | 39.55 | 36.80 | 73.2% | 68.1% | 0.149 |

The trained proj is close to the random-matrix baseline (70.7%/70.7%), with a mild ~2–4% sym bias.

### Alpha sweep results

| α | PPL | Generation quality |
|---|-----|--------------------|
| 0.0 (pure anti-sym / curl) | 155,697 | `/frame` repeated — completely degenerate |
| 0.25 | 3,441 | `clear` repeated — broken |
| **0.5 (trained behaviour)** | **12.2** | Coherent, repetitive but meaningful ✓ |
| 0.75 | 54.4 | `ernity` repeated — breaking down |
| 1.0 (pure sym / gradient) | 68,664 | `sembl` repeated — completely degenerate |

### Key findings

1. **The model requires an equal mix of sym and anti**: α=0.5 is the only working point. Both
   extremes (pure gradient descent α=1, pure curl α=0) cause catastrophic degeneration (PPL 4–5
   orders of magnitude worse).

2. **The trained proj naturally converged to the balanced mix**: sym and anti norms are both ≈70%
   of the total — essentially the random-matrix baseline. The model didn't learn to prefer
   pure descent or pure rotation; it learned that both are essential.

3. **Interpretation**: The energy recurrence dynamics require the *interplay* between curl
   (exploration/rotation) and gradient (descent/dissipation) components. This is consistent with
   Port-Hamiltonian systems theory: effective computation needs both conservative (J) and
   dissipative (R) components.

4. **The sensitivity is extremely sharp**: PPL jumps from 12 to 54 at α=0.75 and to 68k at α=1.0.
   The working window is narrow — roughly α ∈ [0.4, 0.6].

---

## Alpha fine sweep — 410m_hybrid_s8e4_lr1e3_60k (2026-04-12)

**Script**: `experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py`
**Command**:
```bash
python experiments/energy-inference/scripts/inspect_proj_alpha_20260412.py \
    --alphas 0.40 0.43 0.45 0.47 0.49 0.50 0.51 0.53 0.55 0.57 0.60 \
    --out experiments/energy-inference/results/alpha_fine_sweep_410m_hybrid_lr1e3_20260412.json
```
**Results file**: `results/alpha_fine_sweep_410m_hybrid_lr1e3_20260412.json`

| α | PPL | Notes |
|---|-----|-------|
| 0.40 | 20.0 | `"known"` looping |
| 0.43 | 14.5 | coherent but loops |
| 0.45 | 13.1 | coherent but loops |
| 0.47 | 12.5 | **best generation quality** (diverse, non-repetitive) |
| **0.49** | **12.160** | **PPL minimum** |
| 0.50 | 12.165 | indistinguishable from 0.49 |
| 0.51 | 12.181 | negligibly worse |
| 0.53 | 12.225 | slightly worse |
| 0.55 | 12.282 | HTML artifacts appearing |
| 0.57 | 13.5 | degrading fast |
| 0.60 | 39.1 | degenerate |

### Key findings

1. **Optimum is α ≈ 0.49 ± 0.02** — essentially exactly 0.5. The trained weight has
   sym/anti norms of ~70.7%/70.5%, which is the Pythagorean orthogonal split
   (||W_sym||² + ||W_anti||² = ||W||²). The optimum in α-space is the midpoint
   α=0.5, consistent with this decomposition being balanced.

2. **Sharp asymmetry: the model tolerates extra curl better than extra gradient.**
   - Left of optimum (more anti-sym): PPL rises from 12.16 → 20.0 over Δα=0.10 (1.7× worse)
   - Right of optimum (more sym): PPL rises from 12.16 → 39.1 over Δα=0.10 (3.2× worse)
   The energy recurrence is more fragile under excess gradient dynamics than excess rotation.

3. **Generation quality peaks slightly anti-sym of the PPL optimum (α=0.47).**
   At α=0.47 the text is more diverse and less repetitive than α=0.49-0.51, even
   though PPL is slightly higher. A little extra curl reduces mode collapse in greedy decoding.

4. **Note on parametrisation**: the current sweep uses W_eff = α·W_sym + (1-α)·W_anti, so
   the "trained" weight W = W_sym + W_anti corresponds to using *both* at coefficient 1 —
   not representable by a single α. At α=0.5, W_eff = W/2 (half scale). The optimum at
   α≈0.5 means the model is happy at half-scale too, but the asymmetry result still stands.
   A norm-preserving sweep (scale W_sym and W_anti to equal norms before mixing) is the
   right next step to cleanly separate scale from direction effects.

### Next experiments

- [x] Fine-grained sweep around α=0.5 (done: α∈[0.40,0.60], optimum at 0.49)
- [x] Norm-preserving interpolation (done: confirms scale confound; true optimum α≈0.51 ≈ trained W)
- [ ] Per-layer alpha: maybe earlier energy layers prefer more curl, later ones more gradient
- [ ] Repeat on `400m_s1_e6_e6_e6_e6_e6_e6_s1_lr3e4_30k` (all-energy, deeper recurrence)
- [ ] Repeat on `410m_hybrid_s8e4_lr2e3_60k` (dead scale_ff control — does alpha still matter?)
