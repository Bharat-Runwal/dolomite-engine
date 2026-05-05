# Progress — Session 2026-04-28

## BoltzmannMoE Energy FFN implementation

### Model: `BoltzmannMoE_Energy_MLP`

New `mlp_type` implementing Boltzmann-weighted MoE for the energy FFN:

```
E_moe(h) = log( Σᵢ exp(Eᵢ(h)) )   where Eᵢ = −φ(W1ᵢh)ᵀ(W2ᵢh)
∂E_moe/∂h = Σᵢ pᵢ(h) · ∂Eᵢ/∂h   where pᵢ = softmax_i(E(h) / τ)
```

The forward pass computes both energies (for routing) and gradients (for the
update) from the same intermediate activations — no redundant work.
Fully vectorised with `torch.einsum` over the expert dimension.

**Anti-collapse mechanisms** (all independently configurable):
- `dropout`: noisy intermediate activations perturb Boltzmann routing during training
- `repulsion_coef` / `n_repulsion_pairs`: stochastic contrastive repulsion — at each
  step, K random expert pairs are sampled and their output cosine similarity penalised
  via `add_aux_loss`.  Cheap: K dot-products of size hidden_size per step.
- `weight_decay=0.3` in the optimizer: prevents dominant expert from growing large

**Key diagnostic**: `routing_entropy_norm` logged per forward pass (1.0 = uniform, 0.0 = collapsed).

### Files changed

- `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py` — `BoltzmannMoE_Energy_MLP` class
- `lm_engine/hf_models/config/mlp.py` — `_BoltzmannMoEEnergyMLPArgs`
- `lm_engine/hf_models/modeling_utils/mlp_blocks/__init__.py` — dispatch + export
- `lm_engine/hf_models/config/__init__.py` — registry entry

### Experiment configs

For ~422M params (d=768, 12 blocks, 16 experts × 1024 = 302M FFN):

| Config | Repulsion | Dropout | WD | Question |
|---|---|---|---|---|
| B1 (`b1_boltz_moe_16x1024_d768_lr2e3`) | — | 0 | 0.1 | Does Boltzmann routing collapse at 400M? |
| B2 (`b2_boltz_moe_repulsion_16x1024_d768_lr2e3`) | 0.01 | 0 | 0.1 | Does functional repulsion prevent collapse? |
| B3 (`b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3`) | 0.01 | 0.1 | 0.3 | Does dropout + high WD force specialisation? |

All log to wandb project `energy-inference-large`.

---

# Progress — Session 2026-04-27

Summary of changes made this session. Pairs with `TODO.md`.

## 1. Cluster scripts: layer cosine-similarity analyses

New scripts under `scripts/multi-block-ablation/`:

- `_layer_cosim_lib_20260426.py` — shared lib. `capture()` registers forward
  hooks on each block to extract `h_in`, `h_out`, `attn`, `ffwd`. Helpers:
  `cosine_matrix`, `consecutive_sims`, `off_diag_mean`. Plot drivers:
  `plot_heatmaps`, `plot_consecutive`, `plot_summary_bars`, `run`.
- `analyze_cosreg_layer_cosim_20260426.py` — V0/V1/V39/V52/V53/V48 (12×1
  d=768). Outputs `layer_sim_cosine_cosreg_full.{pdf,png}`,
  `consecutive_sim_cosreg_full.{pdf,png}`,
  `layer_sim_summary_cosreg_full.{pdf,png}`.
- `analyze_layer_cosim_400m_20260426.py` — V9/V1_400m/V19/V20/V31/V32/V40
  (24×1 d=1024). Same three output families with `_400m_full` suffix.
- `submit_cosreg_layer_cosim_20260426.sh`,
  `submit_400m_layer_cosim_20260426.sh`,
  `submit_all_layer_cosim_20260426.sh` — bsub wrappers (`-q normal -G grp_ebm
  -gpu "num=1" -W 02:00`); mother script fans out both. Activates the
  uv-managed venv at `/proj/dmfexp/nima/Code/nanoGPT-og/.venv`.

Status: not yet submitted on the cluster.

## 2. Metric-consistency refactor (single source of truth)

Diagnosis: cosreg table reported V0=47.5/V1=47.6 while every other table had
V0=46.7/V1=46.5. Root cause: cosreg path used a 7-task `acc` average; the
rest used the canonical 10-task `acc_norm`-where-applicable scheme.
Additional drift: `arc_easy: acc → acc_norm`, `sciq: acc_norm → acc`, plus
brittle `or`-chains over metric keys.

Fix:
- `make_tables.py` is now the single source of truth. Removed
  `AVG_TASKS_7`/`COL_AVG7`; introduced `AVG_TASKS = AVG_TASKS_10` alias.
  `get_avg()` always uses the 10-task set regardless of `n_tasks` arg.
  `cosreg_benchmarks` table spec switched to `COL_AVG10` and dropped
  per-cell `metric="acc,none"` overrides. Registered V52/V53 in `MODELS`.
- 14 plot/analysis scripts refactored to import canonical constants:
  ```python
  from make_tables import AVG_TASKS_10, TASK_DISPLAY
  ```
  Files touched: `plot_results.py`, `plot_all_variants_20260421.py`,
  `plot_scatter_v2_20260421.py`, `plot_v27_egrad_edesc_20260423.py`,
  `plot_new_variants_20260421.py`, `plot_cosreg_performance_20260425.py`,
  `plot_v9_v11_mixed_20260421.py`, `plot_v5_v8_ablation_20260421.py`,
  `plot_mixed_heatmap_20260421.py`, `plot_mixed_iso_param_20260421.py`,
  `plot_barcharts_20260421.py`, `plot_v1_vs_gpt.py` (+ dated variants),
  `analyze_v52v53_cosim_20260425e.py`. Replaced metric-or-chains with
  explicit `v.get(TASK_DISPLAY[k][1])` lookups.
- All tables and figures regenerated. V0=46.7%, V1=46.5% now consistent
  across egpt_gap / all_small / v5_v8 / mixed_results / cosreg_benchmarks.

## 3. Paper reorganisation (hierarchical layout)

`paper/paper_v2.tex` (1585-line monolith) split into:
- `paper/paper_v2.tex` — preamble + abstract + TOC + `\input{sec/<name>}`
  directives + `\appendix` + `\input{sec/appendices/<name>}`.
- `paper/sec/` — 8 main-body files: `intro`, `architecture`, `egpt_gap`,
  `probes`, `mixed_heads`, `output_rule`, `scaling`, `discussion`.
- `paper/sec/appendices/` — 7 files: `layer_merger`, `v5_v8`, `full_400m`,
  `full_benchmarks`, `all_scatter`, `cka_matrices`, `gsm8k_scatter`.

PDF output verified byte-identical (1,569,517 bytes) to pre-reorg.
Touched prose:
- `sec/architecture.tex` — Fig 20 caption rewritten to mention V52 winning
  ARC-C/PIQA, V53 leading ARC-E among EGPT variants, ~0.8–1.1 pt cost; flag
  acc_norm consistency with headline tables.
- `sec/appendices/layer_merger.tex` — cosreg numbers updated to
  V39=46.3 (−0.2 vs V1=46.5), V48=44.9 (−1.6), V52=45.7 (−0.8),
  V53=45.4 (−1.1).

## 4. Overleaf integration

Cloned the **full paper Overleaf project** from
`https://git.overleaf.com/69eb9b62c6f271a5b29323bb`.

**Local clone paths** (same repo, different machines):
- **Beast** (other server): `~/__work/LLM/energy/energy-GPT-neurips2026/`
- **BLUEVELA cluster** (this cluster, user `ndehmamy`): `~/Code/energy/energy-GPT-neurips2026/`

This is the shared NeurIPS 2026 paper repo — top level holds the joint
document (`main.tex`, `energy.tex`, `neurips_2026/`, top-level `figs/`).

Added a **`nima/` subfolder** as a self-contained staging area for the
deep-EGPT analysis slice, mirroring
`dolomite-engine/experiments/energy-inference/paper/`. The `nima/`
subfolder is independent of the top-level paper: its own `paper_v2.tex`,
`main.tex`, `sec/`, `sec/appendices/`, `tables/`, and `figs/` — so it
compiles standalone on Overleaf and can later be patched into the joint
document by the co-authors.

Self-containment fixes for `nima/` (commit `f72ca89`):
- 75 referenced figures bundled into `nima/figs/` (~2.2 MB).
- Every `\includegraphics{...}` rewritten to bare basename — zero relative
  paths leak outside the Overleaf project.
- `\graphicspath{{figs/}}` declared in both `paper_v2.tex` (line 6) and
  `main.tex` (line 7).
- Both PDFs compile clean locally: `paper_v2.pdf` (41pp, 1.58 MB),
  `main.pdf` (34pp, 1.56 MB).

Pushed `253c8c9..f72ca89 master -> master` to Overleaf.

## Outstanding (from TODO.md, untouched this session)

- Submit `submit_all_layer_cosim_20260426.sh` on the cluster; once outputs
  land, add an `app:cosreg_align` appendix.
- Substantive paper restructuring per the story arc in TODO.md
  (deep-EGPT framing, alignment promotion, Helmholtz demotion, mixed-head
  merger section). This session was mechanical: split + metric fix +
  Overleaf bundle only — no narrative changes.
- V40 / V50 400M cosreg evals; V54 / V55 parallel-GPT baseline.
