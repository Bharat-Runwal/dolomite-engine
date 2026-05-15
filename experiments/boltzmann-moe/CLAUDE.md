# Boltzmann MoE — Experiment Guide

This directory documents the **BoltzmannMoE Energy FFN** experiments (series B1–B5).
The goal was to replace the standard Energy\_MLP feedforward in deep EGPT with a
Boltzmann-weighted mixture of experts and study whether the routing collapses.

---

## What is BoltzmannMoE?

Each of the 12 EGPT blocks gets a new FFN type where the energy is the
log-partition function over K parallel experts:

```
E_moe(h) = log( Σᵢ exp(Eᵢ(h)) )
∂E_moe/∂h = Σᵢ pᵢ(h) · ∂Eᵢ/∂h      pᵢ = softmax_i(E(h) / τ)
```

Each expert is an Energy\_MLP: `Eᵢ(h) = φ(W1ᵢh)ᵀ(W2ᵢh)`.
The routing energy is `Eᵢ = h · term1ᵢ` where `term1ᵢ = φ(W1ᵢh) @ W2ᵢᵀ`
(contracting only the first gradient term; the full gradient would add a
spurious Hessian contribution).

**Iso-parameter**: K experts each of size `intermediate_size // K` → same total
params and FLOPs as one Energy\_MLP with the same `intermediate_size`.

---

## Key source files

| File | What it does |
|------|-------------|
| `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py` | `BoltzmannMoE_Energy_MLP` class |
| `lm_engine/hf_models/config/mlp.py` | `_BoltzmannMoEEnergyMLPArgs` config dataclass |
| `lm_engine/hf_models/modeling_utils/mlp_blocks/__init__.py` | `get_mlp_block` dispatch |
| `lm_engine/hf_models/config/__init__.py` | Registry entry (`_MLP_CONFIG_CLASSES`) |
| `lm_engine/train_utils.py` | `get_metrics()` logging for routing collapse metrics |
| `lm_engine/arguments.py` | `SaveArgs.max_to_keep: int | None` (pydantic fix) |

---

## Config fields for `BoltzmannMoE_Energy_MLP`

```yaml
mlp_blocks:
  - mlp_type: BoltzmannMoE_Energy_MLP
    intermediate_size: 16384   # total = n_experts × per_expert_I
    n_experts: 16              # number of experts
    temperature: 1.0           # Boltzmann softmax temperature
    repulsion_coef: 0.0        # 0 = off; 0.01–0.1 = stochastic contrastive repulsion
    n_repulsion_pairs: 4       # random expert pairs sampled per step for repulsion
    dropout: 0.0               # dropout on intermediate activations
    add_bias: false
```

**IMPORTANT**: always set `max_to_keep: 2` in `save_args` — the field is
`int | None` and pydantic will reject `null` from saved checkpoint configs
without this fix.

```yaml
save_args:
  save_path: ...
  save_interval: 5000
  max_to_keep: 2
```

---

## Experiment configs

All located in `configs/boltzmann_moe/`.

### B-series (deep EGPT, iso-param, d=768):

| Config | repulsion_coef | dropout | weight_decay | Purpose |
|--------|---------------|---------|-------------|---------|
| `b1_boltz_moe_16x1024_d768_lr2e3.yml` | 0 | 0 | 0.1 | Baseline — does routing collapse? |
| `b2_boltz_moe_repulsion_16x1024_d768_lr2e3.yml` | 0.01 | 0 | 0.1 | Weak repulsion |
| `b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3.yml` | 0.01 | 0.1 | 0.3 | Dropout + high WD |
| `b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3.yml` | 0.1 | 0 | 0.1 | Strong repulsion (best load balance) |
| `b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3.yml` | 0.1 | 0.1 | 0.3 | Combined |

All use: `d=768`, 12 blocks, 16 experts × 1024 = **~422M total params**. Note: all B variants underperform V1 EGPT d=768 (143M) due to 21:1 FFN:Attn imbalance.

### C-series (design fixes, d=768):

| Config | Description |
|--------|-------------|
| `c1_topk_energy_moe_4x2048_top2_d768.yml` | 4 full-size experts, top-2 sparse, load-balance loss |
| `c2_surrogate_boltz_16x1024_d768.yml` | Boltzmann routing + learned linear surrogate router |
| `c3_attn_moe_2x_d768.yml` | MoE on **attention** (2 energy-attn experts), normal FFN |
| `c4_paired_unit_moe_2x_d768.yml` | 2 paired (attn+FFN) units, balanced FFN:Attn |

### H-series (hybrid GPT+EGPT-MoE, **best results**):

6 standard GPT layers + 1 recurrent EGPT block (×6). MoE only in the EGPT block.

| Config | Description | Avg acc | WikiPPL |
|--------|-------------|---------|---------|
| `h1_topk_egpt_moe_d768.yml` | **BEST**: 4 full experts×2048, top-2 | **0.499** | 39.8 |
| `h1_topk_egpt_moe_r128_d768.yml` | Same + 128 register tokens | 0.484 | 39.6 |
| `h1_boltz_egpt_moe_d768.yml` | Boltzmann routing (iso-param, fails) | 0.464 | 46.1 |

---

## Running experiments

### Submit a new training run

```bash
REPO=/proj/dmfexp/nima/Code/dolomite-engine
mkdir -p $HOME/bsub_logs

bsub \
    -q preemptable -G grp_preemptable \
    -J egpt_b1_boltz_moe \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 -M 64G -W 06:00 \
    -o "$HOME/bsub_logs/egpt_b1_boltz_moe_%J.stdout" \
    -e "$HOME/bsub_logs/egpt_b1_boltz_moe_%J.stderr" \
    <<'EOF'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
bash /proj/dmfexp/nima/Code/dolomite-engine/scripts/common/pretrain.sh \
    /proj/dmfexp/nima/Code/dolomite-engine/configs/boltzmann_moe/b1_boltz_moe_16x1024_d768_lr2e3.yml
EOF
```

Or use the convenience script:

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
bash experiments/energy-inference/scripts/multi-block-ablation/run_b1_b3_boltz_moe.sh
```

### Resume from checkpoint

The training script auto-detects `latest_checkpointed_iteration.json` and
resumes. For manual resume, append `load_args` to the config:

```bash
cat >> /tmp/resume_b1.yml <<'YAML'

load_args:
  load_path: /proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/b1_boltz_moe_16x1024_d768_lr2e3
YAML
# then submit with /tmp/resume_b1.yml as the config
```

### Wall-time note

At ~0.91 s/step, 30k steps takes ~7.6 hours. Use `-W 08:00` for a single
uninterrupted run. If using `-W 04:00`, the job will checkpoint at 5k-step
intervals and need resubmission.

---

## Scaling up

To scale to more experts or larger hidden size, adjust `intermediate_size` and
`n_experts` in the config. The table below shows total params for d=768, 12 blocks:

| n_experts | per_expert_I | total_I | ~Total params |
|-----------|-------------|---------|--------------|
| 4 | 1024 | 4096 | ~165M |
| 8 | 1024 | 8192 | ~243M |
| 16 | 1024 | 16384 | ~422M (B-series) |
| 16 | 2048 | 32768 | ~723M |
| 32 | 1024 | 32768 | ~723M |

**Warning**: current experiments show the MoE is severely FFN-heavy (FFN:Attn
≈ 21:1 at 422M). The V1 d=768 EGPT baseline (143M, FFN:Attn ≈ 2.7:1) scores
higher (avg 0.481 vs 0.474). Before scaling up the MoE, consider using a
balanced architecture where attention and FFN have comparable parameter counts
(e.g., increase `d` while keeping `intermediate_size` moderate, or use fewer
larger experts).

---

## Routing collapse metrics (WandB)

Logged every 10 steps under `model/energy_mlp/<block>.ffwd/`:

| Metric | Range | Meaning |
|--------|-------|---------|
| `effective_n_experts` | 1.0 → K | Per-token entropy exponentiated. **1.0 = hard routing** |
| `n_dominant_experts` | 1 → K | How many experts win argmax across the batch |
| `max_expert_load` | 0 → 1 | Fraction of tokens routed to the busiest expert (uniform = 1/K = 0.0625) |
| `mean_token_entropy_norm` | 0 → 1 | Normalized per-token routing entropy |

**Key finding**: `effective_n_experts ≈ 1` by step 500 for all variants (hard routing
emerges fast), but `n_dominant_experts = 14–16` — different tokens go to different
experts. This is **not** collapse to a single expert; it is learned specialization.
B4 (repulsion 0.1) achieves best load balance (`max_expert_load ≈ 0.37`).

---

## Expert specialization analysis

Routing vectors (per-sample, per-layer, per-expert) are cached in:

```
experiments/boltzmann-moe/results/routing_cache/routing_b{1-5}.pkl
```

Load format:
```python
import pickle
data = pickle.load(open("routing_b1.pkl", "rb"))
# data["categories"]["COPA"]["routing"]  → (N, 12, 16) float32
# data["categories"]["COPA"]["texts"]    → list of N input strings
```

Re-run deep analysis (PCA, Mahalanobis separation, routing profiles) without
re-running inference:

```bash
bsub -q normal -G grp_ebm -n 1 -M 8G -W 00:30 \
     -gpu "num=1" \
     -o $HOME/bsub_logs/pca_analysis_%J.stdout \
     -e $HOME/bsub_logs/pca_analysis_%J.stderr \
    <<'EOF'
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
cd /proj/dmfexp/nima/Code/dolomite-engine
python experiments/energy-inference/scripts/multi-block-ablation/analyze_boltz_expert_deep_20260429.py \
    --model b1 --reuse --device cpu
EOF
```

To collect new routing data for a new model (needs GPU):

```bash
# Add to MODELS dict in analyze_boltz_expert_deep_20260429.py:
#   "b6": RESULTS / "b6_.../unsharded"
# Then submit with --model b6 (no --reuse flag)
```

---

## Results summary (30k steps, 7.86B tokens)

| Model | Params | Avg acc | WikiPPL | Notes |
|-------|--------|---------|---------|-------|
| V9 GPT d=1024 | 354M | **0.513** | **29.8** | Best baseline |
| V1-400M EGPT d=1024 | 354M | 0.494 | 38.6 | |
| V1 EGPT d=768 | 143M | 0.481 | 47.7 | Beats all MoE at 1/3 params |
| B1 BoltzMoE (no reg.) | 407M | 0.474 | 51.9 | Best MoE variant |
| B4 BoltzMoE (rep 0.1) | 407M | 0.466 | 51.9 | Best load balance |
| V58 EGPT recurrent | 113M | 0.459 | 65.7 | |
| B2 (rep 0.01) | 407M | 0.462 | 52.5 | |
| B5 (rep+drop+WD) | 407M | 0.471 | 58.7 | |
| B3 (drop+WD) | 407M | 0.450 | 58.0 | Worst |

**Conclusion**: The MoE does not improve over the plain EGPT baseline at this scale.
The architectural imbalance (302M in FFN, only 14M in attention) is the primary cause.

---

## Key paper and reports

- **Local report**: `experiments/boltzmann-moe/paper/report.pdf` (10 pages)
- **Overleaf** (NeurIPS 2026 joint paper): `~/Code/energy/energy-GPT-neurips2026/`
  — nima staging area: `nima/sec/appendices/boltz_moe.tex`
- **Analysis scripts**: `experiments/energy-inference/scripts/multi-block-ablation/`
  - `analyze_boltz_moe_routing_20260428.py` — routing collapse curves from training logs
  - `analyze_boltz_expert_specialization_20260429.py` — basic PCA/heatmaps (60 samples)
  - `analyze_boltz_expert_deep_20260429.py` — deep PCA with KDE, Mahalanobis separation (200 samples)
