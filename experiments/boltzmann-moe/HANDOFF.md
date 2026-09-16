# Boltzmann-MoE — HANDOFF

**Entry point for a fresh session.** Read this first, then jump into the specific
doc you need. This file carries (a) orientation, (b) the things that exist
nowhere else, and (c) the 2026-09-12 session findings, which materially change
the project's conclusions.

Last updated: 2026-09-15 (ICLR draft migrated to Avg11).

> ## 📄 ACTIVE PAPER — the ICLR draft is the ONLY paper we are writing right now
>
> **Path: `/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`**
> (Overleaf remote `https://git@git.overleaf.com/6a9ace75a92fce262f38ec18`, branch `main`.)
> Files: `main.tex` + `sec/{intro,theory,experiments,appendix}.tex`.
> Table locations: `tab:frontier`, `tab:pure`, `tab:cost`, `tab:threeway` in
> `sec/experiments.tex`; `app:frontier`, `app:eval` (metric definition), `app:scale400`,
> `app:threeway`, `tab:attribution` in `sec/appendix.tex`.
>
> **ALL of its numbers are `Avg11`** as of 2026-09-15 (migrated from `avg10` that day; see
> `AVG11_ICLR_MIGRATION.md`). MMLU and GSM8K-CoT are reported as SEPARATE columns, never in
> the mean.
>
> **⚠ Do NOT read the older drafts unless a task specifically requires them — they burn
> context and they use SUPERSEDED metric conventions.** The NeurIPS draft
> (`~/Code/energy/energy-GPT-neurips2026/`) and the talk
> (`~/Code/overleaf/energy-GPT-reformulation-2026/`) are **reference/archive only**; their
> tables are still on `avg10`/`avg9`. If you find yourself opening `paper_v2.tex` or
> `boltz_moe.tex` to answer an ICLR question, stop — you are in the wrong paper.

---

## 0. Doc map — read in this order

| Doc | What it is | When to read |
|---|---|---|
| **HANDOFF.md** (this) | Orientation + 2026-09-12 findings | always, first |
| **`ROUTING_SIGN_BUG_20260915.md`** | 🔴 The composable router is SIGN-INVERTED (selects worst-matching experts). Affects the whole ICLR grid + live 400M. Includes the chemical-potential reframing of the balance property | **before any routing claim** |
| **`HANDOFF.md` §11** | 🔴 The 2026-09-15 session: sign inversion, chemical-potential balancing, Sinkhorn, energy stability | **always, with §0** |
| **`PLACEMENT_ARTIFACT_20260915.md`** | 🔴 The "5x slower" figure is mostly host placement (6.76 → 2.49 s/step, same code). Invalidates the launch-overhead reading | **before any s/step claim** |
| `ACCEL_FINDINGS_20260915.md` | fused-GEMM 1.61x (single-node ONLY — wedges at 2 nodes), repulsion sweep, proxy router 0.942, ReLU negative result | before optimisation work |
| **`/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`** | **★ THE ACTIVE PAPER.** `main.tex` + `sec/*.tex`; all numbers `Avg11` | whenever writing or quoting results |
| `AVG11_ICLR_MIGRATION.md` | What changed in the 2026-09-15 avg10→Avg11 migration, number by number | before touching any paper number |
| `CLAUDE.md` | Operational guide: config fields, how to submit runs, routing metrics, results summary | before touching configs or launching anything. **Its "Key source files" table is stale — see §10** |
| `PROGRESS.md` | Full experiment log B/C/H series + the master baseline table (lines 146-178) + the `gelu_grad_method` A/B | when you need a number or the history of a design choice |
| `BOLTZ_MOE_BEST.md` | The two ~600-680M champion runs, layer-by-layer architecture, the 65B-token iso-token leaderboard vs Switch-MoE baselines | when scaling up or writing results. **Its "Source code" line refs are stale — see §10** |
| `TODO.md` | Queued work: sparsity sweep axes, τ sweep, repulsion sweep, scattermoe kernel, open questions | when picking the next experiment |
| `SCALE_UP_PLAN.md` | 3B and 7B param-count search, LR/token/GPU recommendations, FFN:Attn sanity checks | when scaling past 680M |
| `recovered_boltz_sessions/` | Recovered transcript of the purged pre-June session (179 msgs) — the Boltzmann-Hopfield design discussion lives here | archaeology only |

Do not re-derive results tables. `PROGRESS.md:146-178` is the master table;
`BOLTZ_MOE_BEST.md:83-105` is the large-scale comparison set.

---

## 1. What the project is

Replace the feedforward block of an Energy-GPT (EGPT) layer with a
**Boltzmann-weighted mixture of energy experts**. The energy is the
log-partition function over `K` parallel expert energies; the FFN output is its
gradient w.r.t. the hidden state:

```
E_MoE(h) = τ log Σ_k exp(E_k(h)/τ)
∇_h E_MoE = Σ_k p_k(h) ∇_h E_k(h),    p_k = softmax_k(±E_k(h)/τ)
```

The selling point: **routing needs no learned router.** The energy landscape
itself ranks the experts. `PROGRESS.md:127-132` records that at h1 scale this
matches or beats a learned-router top-k (0.501 vs 0.499 avg, 36.48 vs 39.79
PPL), and that energy-based selection survives hard top-2 truncation when
trained sparse.

Two expert families:

| Expert kind | Energy | Bounded below | Params per expert |
|---|---|---|---|
| `w1w2` | `E_k = -φ(W1_k h)·(W2_k h)` | no | `2·d·I_e` |
| `hopfield` | `E_k = (1/I_e)‖gelu(W_k h)‖²` | yes (≥0) | `d·I_e` |

---

## 2. Where things live

All paths relative to `/proj/dmfexp/nima/Code/dolomite-engine` unless absolute.

| Path | Contents |
|---|---|
| `configs/boltzmann_moe/` | 31 configs — the B/C/D/H series and the scale-ups (`h1_boltz_moe_580m_8x4096_d1536.yml`, `scale_h3_8gpt_4egpt_boltz_d1280.yml`, the 3B/7B picks) |
| `configs/multi_block_ablation/` | The FET / `math_fet_*` line, including the Hopfield and Boltz-Hopfield runs |
| `experiments/boltzmann-moe/results/` | 37 run dirs (B/C/D/H series + scale-ups). ~41 GB. gitignored |
| `experiments/boltzmann-moe/results/router_analysis/` | **new 2026-09-12** — router/FLOP analysis outputs |
| `experiments/boltzmann-moe/results/routing_cache/` | `routing_b{1-5}.pkl` cached routing vectors for the specialization PCA |
| `experiments/boltzmann-moe/scripts/` | run launchers, surrogate-router training, `watchdog/` auto-resubmit, **the two new 2026-09-12 analysis scripts** |
| `experiments/boltzmann-moe/paper/` | `report.tex`/`report.pdf` (10pp local report), `make_moe_scatter.py`, `figs/` |
| `experiments/energy-inference/results/multi-block-ablation/` | The `math_fet_*` checkpoints (Hopfield, Boltz-Hopfield, register variants). **This is where the 2026-09-12 target checkpoint lives, NOT under boltzmann-moe/results/** |
| `experiments/energy-inference/scripts/multi-block-ablation/` | `analyze_boltz_moe_routing_*.py`, `analyze_boltz_expert_{specialization,deep}_*.py`, eval/bench shell scripts |
| `~/Code/GPT-experiments/projects/EGPT-action/` | The FET/action project: `RESULTS.md`, `PLANS.md`, `SPECS.md`, `TODO.md`, on-policy verifier data under `results/onpolicy/` |

**Naming trap:** the H-series and B-series checkpoints are under
`experiments/boltzmann-moe/results/`, but the `math_fet_*` Boltzmann-Hopfield
checkpoints are under `experiments/energy-inference/results/multi-block-ablation/`.
The new `measure_moe_routing_20260912.py` resolves `--run` against the latter.

---

## 3. The two model-code lineages — know which one you are looking at

This is the single most confusing thing in the codebase. There are **two
independent implementations** of Boltzmann-MoE, and the checkpoints are split
across them.

### 3a. Legacy (pre-2026-06-28) — `mlp.py`

| Item | Location |
|---|---|
| Class | `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py:540` `BoltzmannMoE_Energy_MLP` |
| Config | `lm_engine/hf_models/config/mlp.py:136` `_BoltzmannMoEEnergyMLPArgs` |
| Dispatch | `lm_engine/hf_models/modeling_utils/mlp_blocks/__init__.py:113` |
| `mlp_type` | `BoltzmannMoE_Energy_MLP` |
| Experts | W1/W2 only (no Hopfield option) |
| Routing scale | `mlp.py:591` `_routing_scale = expert_I ** -0.5`, applied at `mlp.py:672` |
| top-k truncation | `mlp.py:682` `p = p * mask` — **no renormalization** |

Also in `mlp.py`: `BoltzmannMoE_Hopfield_Energy_MLP` (`mlp.py:244`) — a
standalone Hopfield-MoE that **lacks repulsion, τ and `n_repulsion_pairs`**.
Superseded by 3b; do not use for new runs.

Sibling classes worth knowing: `TopK_Energy_MoE_MLP` (`mlp.py:894`, learned
router + load-balance loss) and `SurrogateBoltzmannMoE_Energy_MLP`
(`mlp.py:994`) — the latter is prior art for a cheap inference router: it
trains a linear `d→K` head by KL against the Boltzmann weights and swaps it in
at eval (`use_surrogate=True`). Its own docstring notes the expert gradients are
still all computed, so it saves only the router.

**Checkpoints on this lineage:** all `b*`, `c*`, `d1_*`, `h1_*`, `h2_*`,
`scale_*` runs. e.g. `h1_boltz_moe_fullsize_d768` has
`mlp_type: BoltzmannMoE_Energy_MLP`, `n_experts: 4`, `intermediate_size: 8192`
(I_e=2048), `repulsion_coef: 0.1`, `temperature: 1.0`.

### 3b. Composable (2026-06-28 refactor) — `energy_ff.py`

`lm_engine/hf_models/modeling_utils/mlp_blocks/energy_ff.py`, 770 lines.

| Symbol | Line | Role |
|---|---|---|
| `FFEnergyBase` | 114 | abstract base; owns the `_capture_energy` / `_last_energy_per_token` FSDP-2 leaf-cache contract and `get_metrics()` |
| `W1W2FFEnergy` | 163 | `E = -(1/√I)·gelu(W1h)·(W2h)`; the `1/√I` moved **into the energy** (vs legacy, which applied `1/√I_e` at the routing layer only) |
| `HopfieldFFEnergy` | 259 | `E = (1/I)‖gelu(Wh)‖²`; single shared `W`, half the params of W1W2 |
| `BoltzmannMoEFFEnergy` | 338 | the MoE wrapper: routing, `e_sign`, repulsion, `top_k` |
| `_W1W2Expert` / `_HopfieldExpert` | 502 / 558 | view-backed experts, zero-copy row-slices of a fused weight |
| `_FusedW1W2Holder` / `_FusedHopfieldHolder` | 594 / 646 | own the actual `Parameter`s |
| `FusedMoEContainer` | 685 | what gets registered as `block.ffwd`; mirrors the inner MoE's `_last_energy_per_token` and `_cached_metrics` |
| `build_boltzmann_moe` | 718 | factory; picks `e_sign` per expert kind |
| Config | `config/mlp.py:245` `_EnergyFFBoltzmannMoEArgs` | `top_k` IS exposed (`config/mlp.py:264`) |
| Dispatch | `mlp_blocks/__init__.py:179` | passes `top_k` through (`__init__.py:191`) |
| top-k truncation | `energy_ff.py:432` `p = p * mask` | **no renormalization**, same as legacy |

`mlp_type` values: `EnergyFF_W1W2`, `EnergyFF_Hopfield`, `EnergyFF_BoltzmannMoE`
(+ `expert_kind: w1w2 | hopfield`).

`e_sign` convention (`energy_ff.py:426`): `"neg"` for Hopfield (route to the
**lowest** energy), `"pos"` for W1W2 (matches the validated legacy class).

**Checkpoints on this lineage:** the `math_fet_boltz_hopfield*` runs.

### 3c. The router-cost asymmetry — the key fact for any FLOPs work

```
Hopfield: E_k = (1/I_e)‖gelu(W_k h)‖²   → needs only W_k h  (FIRST matmul)
          ∇_h E_k = (4/I_e) W_kᵀ(gelu(W_k h)⊙gelu'(W_k h))  → SECOND matmul
```
The first matmul is **shared** between the router and the gradient. So for
Hopfield experts **routing is already free** — there is no separate router cost
to remove. The router costs exactly as much as all K expert-output matmuls
combined, but you have to pay it anyway.

```
W1W2:  E_k = h·term1_k,  term1_k = φ(W1_k h) @ W2_kᵀ
```
Here `term1` is itself a second matmul, so the router genuinely is an extra
matmul on top of `W1_k h`. **The "cheap proxy router" idea therefore pays off
roughly 2× more on the W1W2 lineage than on the Hopfield lineage.**

Consequence: an inference-FLOPs story built on "the energy router is expensive"
must be told on the **W1W2** line (`h1_boltz_moe_fullsize_d768`, the 680M), not
on the Hopfield line.

### 3d. How the FF plugs into the energy block

`lm_engine/hf_models/models/energy/layer.py`, class `EnergyBlock` (line 548),
main `forward` at line 792. For `energy_proj_type: psd_anti`:

```
ln_x    = ln(h)
grad_E  = attn_out + scale_ff · ffwd_out        # layer.py:885
h      := h - proj(grad_E)                       # Π = SSᵀ + (A - Aᵀ)
```
`ffwd_out = self.ffwd(ln_x)` at `layer.py:857`. `scale_ff` is a learned scalar
(init 4.0 by default, `layer.py:604`).

**The block is iterated.** `layer_iterations` (a per-layer list, consumed in
`lm_engine/hf_models/mixins/dense/base.py:257`) controls how many times each
block runs. For the 2026-09-12 target, block 8 runs **6×** — so the MoE cost is
multiplied by 6, and `p` is recomputed on every iteration.

---

## 4. Environment & ops

### Python env — avoid the flash-attn compile
`pip install -e .` on dolomite-engine triggers a 20-40 min flash-attn CUDA
build. For inference/analysis, borrow the nanoGPT venv and set `PYTHONPATH`:

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
```
Verified present: torch 2.12.0+cu130, transformers 4.46.3, accelerate 1.13.0,
safetensors 0.8.0, wandb 0.19.6. The energy model falls back to
`F.scaled_dot_product_attention` when flash-attn is absent.

### Cluster — IBM LSF
Check `hostname` first: `loginN.*` → login node, never run compute; `pN-rM-nK.*`
→ compute node, run directly. **A compute node does not imply an allocated
GPU** — check `nvidia-smi`; if it reports "No devices were found", you still
need `bsub` for GPU work.

```bash
mkdir -p ~/bsub_logs      # always log here, never $HOME directly
bsub -q preemptable -G grp_preemptable -J <name> \
     -gpu "num=1/task:mode=exclusive_process" -n 1 -M 48G -W 02:00 \
     -o "$HOME/bsub_logs/<name>_%J.stdout" \
     -e "$HOME/bsub_logs/<name>_%J.stderr" < script.sh
```
`-G grp_preemptable` is required with `-q preemptable`. Verify 30-60 s after
submission that `STAT=RUN`, not `EXIT`.

Multi-node: `scripts/common/pretrain.sh` reads `LSB_MCPU_HOSTS` /
`LSB_JOBID` / `CUDA_VISIBLE_DEVICES` to derive master addr/port, NNODES and
GPUS_PER_NODE automatically.

### Gotchas that will bite you

1. **`max_to_keep: 2`** — always set it in `save_args`. The field is
   `int | None` and pydantic rejects `null` when reloading a saved checkpoint
   config. (`lm_engine/arguments.py`, `SaveArgs`.)
2. **Auto-resume needs an explicit `load_args.load_path`.** Without it training
   silently restarts from scratch even though a checkpoint exists. See the
   project memory note `feedback_dolomite_resume.md`.
3. **`export TMPDIR=<writable path>` in every launcher.** See §7.1.
4. **Watchdog**: `experiments/boltzmann-moe/scripts/watchdog/` auto-resubmits on
   preemption. `watchdog_jobs.conf` sets `max_resubmits`; reset the counter in
   `watchdog_state.txt` if a run hits the cap.

---

## 5. Results — the numbers that drive decisions

Full tables: `PROGRESS.md:146-178` and `BOLTZ_MOE_BEST.md:83-105`. Carry these
four rows in your head:

| Row | Avg | WikiPPL | Why it matters |
|---|---:|---:|---|
| `h1_boltz_moe_fullsize` (soft, K=4×2048) | 0.501 | 36.48 | the W1W2 reference; Boltzmann ≥ learned top-k |
| `h1_topk_egpt_moe` (learned router, top-2) | 0.499 | 39.79 | the learned-router control |
| `h1_boltz_topk2` — **trained** sparse top-2 | 0.486 | **36.37** | sparse training costs ~nothing on PPL |
| `h1_boltz_full @ top2 eval` — **post-hoc** truncation | 0.489 | **42.84** | post-hoc truncation costs **+6.4 PPL** |

**The single most important asymmetry in the project:** training with top-k is
free; truncating a soft-trained model to top-k at eval costs +6.4 PPL. Part of
that is almost certainly the missing renormalization — both implementations do
`p = p * mask` so the retained mass sums to `< 1`, shrinking the output
magnitude (`energy_ff.py:432`, `mlp.py:682`). **A top-k-with-renormalization A/B
on an existing checkpoint is a cheap, unrun experiment.** `TODO.md:107-111`
lists this as an open question with a proposed diagnostic.

Large scale (`BOLTZ_MOE_BEST.md`): at 65B iso-tokens the 680M Boltz
(679M/679M active, energy-attn + recurrent EGPT) reaches avg 58.47 / PPL 19.33,
but the matched-structure Switch ablation reaches 57.82 / 19.81 and a larger
pure-GPT Boltzmann model (962M/585M) reaches 58.02 / 19.73. The honest
conclusion recorded there: architecture advantages are **real but modest**
(~0.5-0.7pp avg, ~0.5 PPL), and 1.4× more params can substitute for them.

---

## 6. Overleaf targets

| Repo | Remote | Boltz-MoE content |
|---|---|---|
| **`/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`** | `https://git@git.overleaf.com/6a9ace75a92fce262f38ec18` | **★ ACTIVE PAPER.** `main.tex` + `sec/{intro,theory,experiments,appendix}.tex`. Tables: `tab:frontier`/`tab:pure`/`tab:cost`/`tab:threeway` (experiments), `app:frontier`/`app:eval`/`app:scale400`/`app:threeway`/`tab:attribution` (appendix). **All `Avg11` since 2026-09-15.** |
| `~/Code/energy/energy-GPT-neurips2026/` | `https://git@git.overleaf.com/69eb9b62c6f271a5b29323bb` | **ARCHIVE — still `avg10`, do not read unless asked.** `nima/sec/appendices/boltz_moe.tex` (37 KB), included from `nima/paper_v2.tex:108` |
| `~/Code/overleaf/energy-GPT-reformulation-2026/` | `https://git@git.overleaf.com/69ebe3ed5c91a9639cc3576b` | **ARCHIVE — `avg10`/`avg9`.** `moe_bs.tex` at top level + `slides/`, talk frames with the MoE results table and scatter plot |

Local report: `experiments/boltzmann-moe/paper/report.pdf`. Scatter figures are
generated by `paper/make_moe_scatter.py` into `paper/figs/`.

---

## 7. SESSION FINDINGS — 2026-09-12

**Target checkpoint throughout this section:**
```
math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu
config: configs/multi_block_ablation/<same name>.yml
ckpt:   experiments/energy-inference/results/multi-block-ablation/<same name>/unsharded/
```
Architecture: `d=1536`, 9 layers = 8 GPT (softmax attn 24 heads, swiglu
I=4096) + 1 energy block (energy_attention 24 heads, `EnergyFF_BoltzmannMoE`
`expert_kind=hopfield`, K=8, I_total=8192 → I_e=1024, τ=1.0,
`repulsion_coef=0.01`, `gelu_grad_method=sigmoid`), block 8 iterated **6×**,
`iter_dropout_range_per_block[8]=3`, `energy_proj_type=psd_anti`,
`energy_antisym_rank=32`, `energy_apply_rayleigh=true`, tied embeddings,
vocab 100352. 400.4M params (800.8 MB bf16 — file size confirms exactly).
Trained to step 32000 ≈ 33.5B tokens.

Reported evals for this checkpoint (from the JSONs in `unsharded/`):
wikitext word-PPL **26.03**, sciq 0.853, piqa 0.690, arc_easy 0.599,
hellaswag acc_norm 0.417, mmlu 0.2785, BBH mean over 28 tasks **0.2771**,
gsm8k flexible-extract 2.27%.

### 7.0 Provenance & confidence

| Finding | Basis | Confidence |
|---|---|---|
| FLOP breakdown (§7.2) | exact arithmetic over checkpoint tensor shapes | **final** |
| Expert SVD spectra (§7.3) | exact, from the weights | **final** |
| Routing uniformity (§7.4) | real BBH activations, but **CPU smoke test, N=4-6 prompts** | **provisional**, corroborated by an independent magnitude argument |
| Dead FF branch (§7.5) | real activations + paired-NLL ablation, **N=4-6 prompts / 355-545 tokens** | **provisional but very large effect** |
| Cheap-router agreement (§7.6) | 240 cached real activations, post-bugfix | **provisional** |
| Iteration stability (§7.7) | 359 tokens/iter | **provisional** |

**A full-scale confirmation run is in flight: bsub job `1559723`** (1 GPU, 810
prompts × 1024 tokens × 3 checkpoints, with `--ablate --n_ablate 250`). At the
time of writing it was `RUN` on `p3-r25-n2`, ~635 s in, still on the first
checkpoint. **If you are reading this later, check for the results first:**

```
experiments/boltzmann-moe/results/router_analysis/routing_measure__<run>.json
```
for the three runs `math_fet_boltz_hopfield_rep_*`,
`math_fet_boltz_hopfield_*` (no-repulsion sibling), and
`math_fet_hopfield_mean_*`. Also `~/bsub_logs/moe_router_measure_1559723.stdout`.
Note that `router_analysis/routing_measure_20260912.json` (old filename, no
`__<run>` suffix) is a **stale CPU smoke-test artifact** from before the
`--run` flag existed — safe to delete.

### 7.1 Folder cleanup (~190 stray temp dirs) — root cause found

`experiments/boltzmann-moe/` had grown to 201 entries, ~190 of which were dead
temp directories. **None were Claude Code artifacts.**

| Removed | Count | What it was |
|---|---:|---|
| `pymp-*/` | 113 | `multiprocessing` scratch; each held a dead `listener-*` unix socket |
| `tmp*wandb-media/`, `tmp*wandb-artifacts/` | 64 | wandb upload staging, all **empty** |
| `torchelastic_*/` | 5 | `torchrun` elastic-agent scratch (`attempt_0/<rank>`) |
| `tmp<8char>/` | 6 | `torch.distributed.rpc` scratch (`_remote_module_non_scriptable.py`) |
| `torchinductor_ndehmamy/` | 1 | empty `torch.compile` cache |

**Root cause:** Python's `tempfile` falls back to the **cwd** when `TMPDIR` is
unset and `/tmp` is unwritable — which is the case inside these LSF jobs. The
runs were launched with `cwd = experiments/boltzmann-moe` (confirmed:
`wandb/run-*/files/wandb-metadata.json` has
`root = .../experiments/boltzmann-moe`). `TMPDIR` was verifiably unset.

**Fix:** `export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "$TMPDIR"` in
every launcher. Patterns were also added to
`experiments/boltzmann-moe/.gitignore` so they never show up in `git status`
again.

### 7.2 Exact inference-FLOP breakdown and the top-k ceiling

Script: `scripts/analyze_moe_router_flops_20260912.py` (CPU). Counts weight
matmul MACs exactly from the checkpoint's tensor shapes; attention score/AV
terms are sequence-length dependent and excluded.

```
per-token matmul MACs (dense soft routing)
  gpt_layers_x8            226.49 M   37.5%
  egpt_attn_x6              42.47 M    7.0%
  egpt_proj_x6              29.49 M    4.9%
  egpt_moe 1st matmul x6    75.50 M   12.5%   <- W h: gives BOTH E_k and the pre-activations
  egpt_moe 2nd matmul x6    75.50 M   12.5%   <- gated_k @ W_k, per expert
  lm_head                  154.14 M   25.5%
  TOTAL                    603.59 M   = 1.207 GFLOP/token
```

Savings ceiling. "Exact energy router" must still compute `W h` for all experts
(so it only saves on the second matmul); a "free router" lets you skip the
first matmul for unselected experts too:

| variant | MoE | EGPT block | whole model |
|---|---:|---:|---:|
| dense soft (current) | 1.000 | 1.000 | 1.000 |
| top-4, exact energy router | 0.750 | 0.831 | 0.937 |
| top-2, exact energy router | 0.625 | 0.746 | **0.906** |
| top-1, exact energy router | 0.562 | 0.704 | 0.891 |
| top-4, free router | 0.500 | 0.661 | 0.875 |
| top-2, free router | 0.250 | 0.492 | **0.812** |
| top-1, free router | 0.125 | 0.407 | **0.781** |

**Read this carefully: up to −59% on the EGPT block but only −22% end-to-end**,
because the LM head (25.5%) and the 8 GPT layers (37.5%) dominate. top-k alone
buys 9%; the cheap router is what unlocks the rest. §9 discusses why −22% is
arguably the wrong denominator for a method claim.

### 7.3 The expert weights are near rank-2

SVD of each `W_k` (`[1024, 1536]`), from `analyze_moe_router_flops_20260912.py`.
`eff_rank` = participation ratio of the squared spectrum,
`(Σσ²)² / Σσ⁴`. Columns are the cumulative fraction of `‖W‖_F²` captured:

```
  k   ||W||_F  eff_rank   r=1    r=2    r=8    r=16   r=256
  0     7.316       2.0   0.674  0.908  0.932  0.938  0.972
  1     7.542       2.9   0.537  0.759  0.887  0.927  0.974
  2     7.159       2.6   0.596  0.776  0.887  0.929  0.971
  3     7.280       2.2   0.606  0.903  0.929  0.935  0.971
  4     7.387       2.2   0.629  0.886  0.928  0.937  0.972
  5     7.064       1.9   0.714  0.867  0.921  0.932  0.970
  6     7.100       2.2   0.632  0.855  0.920  0.933  0.970
  7     7.306       2.2   0.607  0.895  0.929  0.935  0.970
```

- Effective rank **1.9-2.9 out of 1024**. Rank-16 captures ~93%.
- `‖W‖_F` total decayed from **70.9 at init** (`0.02·√(8192·1536)`) to **20.5**.
  For scale, the GPT layers' `c_fc` have `‖·‖_F ≈ 183-190`.
- Cross-expert redundancy, `cos(W_iᵀW_i, W_jᵀW_j)`: mean **0.184**, min 0.021,
  max 0.620 — the experts are geometrically distinct, so routing has real
  content available to it in principle.

Since `W_k` is effectively rank-2, `E_k(x) = F_k(V_kᵀ x)` with `V_k ∈ ℝ^{d×r}`,
`r ≈ 2-8`. That is the whole basis for the cheap router in §7.6. It also
suggests the experts themselves are massively over-parameterized — a separate,
unexplored compression opportunity.

### 7.4 Routing is uniform

On real BBH activations (`measure_moe_routing_20260912.py`, section A):

```
  iter    tokens   eff_n    top1    top2    top4     E_mean    E_range    |out|
     0       359   7.999  0.1264  0.2527  0.5045    0.01360    0.03413    0.005
     ...
     5       359   7.999  0.1266  0.2530  0.5051    0.01473    0.04013    0.005
  (uniform reference: eff_n=8, top1=0.1250, top2=0.2500, top4=0.5000)
```

`effective_n_experts = 7.999 / 8`. Top-1 mass 0.1266 vs the uniform 0.1250.

**Cause:** `E_k ≈ 0.003-0.015` against `τ = 1.0`. The `(1/I_e)` MEAN
normalization in the Hopfield energy, compounded by weight decay shrinking
`‖W‖_F` 3.5×, puts the energies **2-3 orders of magnitude below the
temperature**, so `softmax(-E_k/τ)` is flat by construction.

An independent order-of-magnitude check reproduces this from the weights alone
(σ₁ ≈ 6, ‖ln_x‖ ≈ 19, energy spread over 1024 units with a `1/I_e` mean →
E ≈ 0.003), so the conclusion does not depend on the small prompt sample. The
one caveat that *could* have overturned it — real activations aligning with an
expert's top singular direction, which would raise `E_k` by up to ~40² — is
exactly what the real-activation measurement rules out.

Sharpening sweep on cached real activations (section C, frozen weights):

| scheme | eff_n | top1 | top2 |
|---|---:|---:|---:|
| τ=1.0 (as trained) | 7.999 | 0.1268 | 0.2533 |
| τ=0.1 | 7.905 | 0.1423 | 0.2822 |
| τ=0.03 | 7.293 | 0.1781 | 0.3463 |
| τ=0.01 | 5.754 | 0.2537 | 0.4673 |
| τ=0.003 | 3.848 | 0.4417 | 0.6931 |
| **τ=0.001** | **1.994** | 0.7325 | 0.9168 |
| z-score per token | 6.528 | 0.2157 | 0.4093 |

So τ ≈ 1e-3 is what "sharp" costs here. Note that **per-token z-scoring alone
is not enough** (eff_n 6.5) — normalizing K=8 logits to unit std still leaves a
fairly flat softmax. A z-score **plus a learned gain** (≈3×) is the natural
scale-free fix, and is the direct analogue of the `1/√expert_I` routing-scale
fix that rescued the B-series (`PROGRESS.md:110`, `PROGRESS.md:155-156`).

Per-task routing profiles (section F) showed **no specialization**: max
across-task swing in any expert's mean weight was 0.0021, i.e. every one of the
sampled BBH tasks routes essentially uniformly. (Provisional — 4 tasks in the
smoke test; the full run covers 27.)

### 7.5 **The energy-FF branch is essentially dead** ← headline

`grad_E = attn_out + scale_ff · ffwd_out` (`layer.py:885`). Measured branch
magnitudes (section A2):

| quantity | Boltz-MoE-Hopfield (target) | non-MoE `hopfield_mean` sibling |
|---|---:|---:|
| mean `‖h‖` (residual stream into block) | 433.49 | 431.81 |
| mean `‖attn_out‖` | 24.34 | 22.34 |
| mean `‖ffwd_out‖` | **0.0054** | **0.5469** |
| `scale_ff` (learned) | 1.5156 | 0.5156 |
| mean `‖scale_ff · ffwd_out‖` | **0.0082** | **0.2820** |
| **FF share of `grad_E` magnitude** | **0.03%** | **1.25%** |
| FF perturbation relative to `‖h‖` | 0.0018% | 0.0653% |

Paired-NLL ablation on **identical tokens** (section G — same tokens, same
deterministic model, so the *difference* has no sampling error; only
representativeness of the prompt set is at issue). **Full scale, 87,997 tokens**
(job 1559723; the earlier 545-token smoke-test figures are superseded):

| variant | FET-rep (Hopfield K=8) | `hopfield_mean` (no MoE) | h1_fullsize (w1w2 K=4) | h1_topk2 (sparse-trained) |
|---|---|---|---|---|
| baseline ppl | 5.9375 | 5.9645 | 7.6881 | 7.5838 |
| **branch DELETED** (`ffwd_out=0`) | **+0.0003** | +0.0033 | **+10.76** | **+10.36** |
| routing forced UNIFORM | +0.0009 | n/a | +1.94 | +1.43 |
| top-4 (truncated, no renorm) | −0.0009 | n/a | +0.0000 | +0.1201 |
| top-2 | −0.0010 | n/a | +0.68 | **+0.0000** |
| top-1 | +0.0002 | n/a | +2.40 | **+0.65** |

**Deleting the entire MoE branch changes FET-rep's perplexity by +0.0003 —
indistinguishable from zero.** The w1w2 line is the opposite: +10.8 ppl to delete,
+1.9 ppl to flatten routing. And note the **sparse-training asymmetry**: the
sparse-trained model pays *exactly zero* for top-2 and only +0.65 for top-1,
where the soft-trained one pays +0.68 / +2.40. Training sparse makes top-k free;
truncating post-hoc does not.

Four consequences:

1. **For this checkpoint the largest inference-FLOP win is deleting the MoE
   outright: −151.0M of 603.6M MACs/token = −25.0% total FLOPs, −12.6M params,
   zero measurable quality cost.** That is larger than the −22% best case from
   top-k + a free router, and it needs no finetuning at all.
2. **It explains the uniform routing.** With the branch output irrelevant to
   the loss there is no gradient pressure to differentiate experts; weight
   decay was free to shrink `W` 3.5× and drive it to rank ~2.

   **ROOT CAUSE, measured (section A3, job 1560005 — read this before blaming
   the repulsion loss).** An earlier reading of this session attributed the dead
   branch to the repulsion term. That is WRONG, and the expert-output geometry
   says so:

   | | FET-rep (K=8) | h1_fullsize (K=4) | h1_topk2 (K=4, sparse) |
   |---|---:|---:|---:|
   | `repulsion_coef` | 0.01 | **0.1 (10×)** | 0.1 |
   | mean pairwise `cos(gᵢ,gⱼ)` | −0.1394 | −0.3207 | −0.2308 |
   | geometric floor `−1/(K−1)` | −0.1429 | −0.3333 | −0.3333 |
   | fraction of the way to the floor | 97.6% | 96.2% | 69.2% |
   | cancellation `‖Σpg‖/Σp‖g‖` | 0.241 | **0.190** | **0.663** |
   | mean `‖g_k‖` | **0.019** | **155.66** | 72.18 |
   | `E` mean / cross-expert spread | 0.0126 / 0.0117 | 0.0965 / 0.4216 | 1.356 / 1.417 |
   | `effective_n_experts` | 7.999 / 8 | 3.746 / 4 | 2.413 / 4 |
   | FF share of `grad_E` | 0.04% | 85.84% | 80.87% |

   h1 did **not** avoid the anti-alignment (it is equally pinned to the geometric
   floor, at 10× the repulsion coefficient) and did **not** avoid the cancellation
   (it loses *more*: 19.0% surviving vs 24.1%). Cancellation costs both ~5×, so it
   is not the discriminator. The accounting closes as
   `‖ffwd_out‖ ≈ mean‖g_k‖ × cancellation`: 0.019×0.241 = 0.0046 (measured 0.0050)
   and 155.66×0.190 = 29.6 (measured 25.0).

   The discriminator is `mean ‖g_k‖`, 0.019 vs 155.66 — **8,200×** — and 256× of
   that is a bare structural constant in the source:

   | class | output prefactor | value at `I_e=1024` |
   |---|---|---|
   | `_HopfieldExpert` (`energy_ff.py:644`) | `4/I_e` | **0.0039** |
   | `_W1W2Expert` (`energy_ff.py:604`) | `I_e^-0.5` | 0.031 |
   | legacy `BoltzmannMoE_Energy_MLP` (`mlp.py:693`) | **none** | **1.0** |

   `scale_ff` (1.5156) cannot bridge 4 orders of magnitude; it would need ~8000.
   So the real mechanism is a **feedback loop whose sign depends on whether the
   branch is load-bearing at initialization**: the `4/I_e` prefactor makes the
   Hopfield branch weak from the start → its output barely moves the loss → weak
   gradient signal → repulsion's anti-alignment plus cancellation weakens it
   further → weight decay wins → collapse. On h1 the branch is 86% of `grad_E`, so
   the loss actively needs it and the same repulsion at 10× strength only reshapes
   geometry without shrinking magnitude. Repulsion is an **accelerant, not the
   cause** — fixing it alone will not revive the Hopfield-MoE line.

   Two independent defects, both from `1/I_e`: (a) the `4/I_e` **output** prefactor
   kills the branch magnitude; (b) the `1/I_e` **energy** scale leaves `E ≈ 0.013`
   against `τ=1`, giving uniform routing and making top-k a pure loss. Fix (a) with
   `4/I_e → 1/√I_e` on the output; fix (b) with `routing_norm` (see §7.10).
3. **The whole Hopfield-MEAN energy-FF line has a weak-to-dead FF branch**, and
   the MoE variant is ~36× worse than the plain one. The `(1/d_int)` MEAN form
   was introduced to stop the descent step growing with `intermediate_size`
   (NaNs in EGPT-RL run 1714840 — see the `HopfieldFFEnergy` docstring,
   `energy_ff.py:259-270`); at these widths it over-suppresses by ~3 orders of
   magnitude.
4. **The reported MoE-vs-Hopfield-MEAN delta was comparing two near-dead
   branches.** The config header of
   `configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml` cites
   0.5073 (Boltz-MoE) vs 0.5135 (Hopfield-MEAN) avg10_norm and attributes the
   deficit to narrow experts + MEAN scale-invariance. The real mechanism is
   that neither FF branch is load-bearing, and the MoE's is 36× weaker.
   (Metric note, 2026-09-14: those 0.5073 / 0.5135 are legacy `avg10_norm`; both
   `math_fet_*` checkpoints predate the `pyarrow>=20` fix and are **INCOMPLETE (9/11)**
   under the canonical `compute_avg11.py`, so they cannot be restated to Avg11 without
   re-evaluating. The −0.62pp delta is avg10-vs-avg10 and stays valid as a relative
   comparison.)

Note that under exactly uniform `p`, the MoE is **algebraically identical** to
a plain dense Hopfield FF of width `I_total`:
`Σ_k (1/K)(4/I_e)(gated_k @ W_k) = (4/I_total)(gated @ W_fused)`. The ablation
row "routing forced UNIFORM" (Δ −0.0040) confirms this empirically. So even if
you keep the branch, it can be collapsed to one fused GEMM instead of 8 sliced
ones — same FLOPs, better GEMM utilization, no routing.

### 7.6 Cheap-router feasibility on real activations

Section D, scored by **row-wise top-k selection agreement** with the exact
energy router (what actually matters — not energy MSE):

| variant | r | top1 | top2 | top4 | MACs/token |
|---|---:|---:|---:|---:|---:|
| spectral `‖W_k x‖²` | 2 | 0.0000 | 0.0000 | 0.2281 | 24.6 K |
| **exact-in-rank-r subspace** | 2 | 0.7792 | 0.6812 | 0.7937 | 24.6 K |
| spectral `‖W_k x‖²` | 8 | 0.0000 | 0.0021 | 0.2542 | 98.3 K |
| **exact-in-rank-r subspace** | 8 | **0.9375** | **0.8958** | 0.9323 | **98.3 K** |
| exact router (reference) | — | 1.0 | 1.0 | 1.0 | **12.58 M** |

Two results:

- **The naive `½‖W_k x‖²` proxy FAILS** (~0% top-1 agreement). The reasoning
  "GELU ≈ ReLU, and `E‖relu(z)‖² = ½‖z‖²`, so route on `‖W_k x‖²`" does not
  survive contact with these weights: `gelu(z)²` is not proportional to `z²`
  pointwise, and with `e_sign="neg"` (route to the *lowest* energy) the
  induced ordering is essentially uncorrelated. Confirmed on synthetic input
  too, where even a **full-rank** `‖Wx‖²` proxy topped out at 0.68 top-1 — so
  the failure is the nonlinearity, not the truncation.
- **Restricting the exact energy to the rank-r subspace works.** At r=8:
  0.94 top-1 / 0.90 top-2 for 98.3 K MACs vs the exact router's 12.58 M —
  a **~128× cheaper router**.

**Recommended implementation:** project `a = V_kᵀ x` (r ≈ 2-8 coefficients,
cost `d·r` per expert) then evaluate a **tiny fitted nonlinear head** on `a`.
Because `r` is only 2-8, a low-degree polynomial or a 16-unit MLP suffices and
is essentially free. Fit it with a **frozen backbone** by regression / KL
against cached `(x, E_k)` pairs — exactly what
`router_fit_cache__<run>.pt` contains. `V_k` initializes for free from the SVD
of the trained `W_k`; no training run required.

### 7.7 Iteration stability — route once, reuse 6×

Section B, comparing `p` at iteration *t* against iteration 0 of the same
forward pass:

| iter | argmax agree | total variation |
|---:|---:|---:|
| 1 | 0.9109 | 0.0008 |
| 2 | 0.8997 | 0.0009 |
| 3 | 0.8969 | 0.0010 |
| 4 | 0.8942 | 0.0011 |
| 5 | 0.8858 | 0.0011 |

Routing is nearly identical across the 6 iterations, so it could be computed
**once** and reused — a 6× router cut with no proxy and no retraining.

**Caveat that must not be lost:** this is measured in the degenerate
uniform-routing regime, where *everything* is trivially "stable". **Re-measure
once routing is actually sharp** before claiming this.

### 7.8 Bug fixed: no `EnergyFF_*` run ever logged its routing metrics

`lm_engine/train_utils.py` gated the energy-MLP metric collection on
`isinstance(module, (Energy_MLP, Compositional_Energy_MLP, Mixed_Energy_MLP,
BoltzmannMoE_Energy_MLP))` — which **omits the entire `FFEnergyBase` family**.
Result: no run using `EnergyFF_W1W2` / `EnergyFF_Hopfield` /
`EnergyFF_BoltzmannMoE` ever logged `effective_n_experts`,
`n_dominant_experts`, `max_expert_load` or `mean_token_entropy_norm`.

Verified on the target run: `wandb-summary.json` has 17 keys, and the only
`model/energy_mlp/*` entries are three `attn` norms (`W_Q_norm`,
`c_attn_norm`, `output_norm`). No routing metrics at all.

**Fix applied** (`lm_engine/train_utils.py:73-87`): added `FFEnergyBase` to the
isinstance tuple, with a `name.rsplit(".", 1)[-1] != "moe"` guard because
`FusedMoEContainer` mirrors its inner `BoltzmannMoEFFEnergy`'s metrics and
would otherwise emit every series twice. Verified: exactly one series per FF
module (`ffwd` logs, `ffwd.moe` is skipped).

**Consequence for readers of old docs:** the "experts collapsing to uniform
routing" claim in the header comment of
`configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml:12` was
**inferred, never measured**. As of §7.4 it happens to be correct — but for a
different reason than the one stated there (energies ≪ τ, not narrow experts).

### 7.9 Methodology warning: `torch.isin` silently fakes top-k agreement

`torch.isin(elements, test_elements)` **flattens** `test_elements`. So

```python
a = E.topk(k, -1).indices          # [N, k]
torch.isin(b[:, j], a)             # WRONG: compares against ALL N rows at once
```

compares each row's candidate against the union over the whole batch. With
`N=240` and `k=4` the flattened set contains nearly every expert index, so this
reports **~1.0000 agreement for anything**, including a proxy that is actually
0% correct. This produced fake `top2_overlap = 1.0000` and `top4 = 1.0000`
numbers before it was caught.

Correct row-wise form — `_row_overlap` in
`scripts/measure_moe_routing_20260912.py:56`:

```python
a = ref.topk(k, -1, largest=largest).indices
b = cand.topk(k, -1, largest=largest).indices
mask = torch.zeros_like(ref, dtype=torch.bool).scatter_(-1, a, True)
return mask.gather(-1, b).float().mean().item()
```

Any top-k overlap / routing-agreement metric anywhere in this project should be
audited against this. Note the numbers in §7.6 are post-fix; the pre-fix run
reported `exact-in-rank-8` as 0.9792/0.9729/1.0000 instead of the true
0.9375/0.8958/0.9323.

---

### 7.10 Fixes landed 2026-09-12 (code changes, not just findings)

| fix | where | default | note |
|---|---|---|---|
| `FFEnergyBase` added to the metrics isinstance gate | `train_utils.py:73` | — | no `EnergyFF_*` run had ever logged routing metrics; deduped with `name.rsplit(".",1)[-1] != "moe"` because `FusedMoEContainer` mirrors its inner wrapper |
| `repulsion_form ∈ {squared, abs, hinge, signed}` | `energy_ff.py::_repulsion_penalty`, legacy `mlp.py::_add_repulsion_loss`, config `_EnergyFFBoltzmannMoEArgs` | **`squared`** | legacy `signed` is minimised at `cos=-1` and rewards anti-alignment. **λ recalibration:** at \|cos\|≈0.3, `squared` is ~3× weaker than `signed`/`abs` (0.09 vs 0.30) because cos² vanishes quadratically near orthogonality — **`abs` is the drop-in that preserves the existing λ sweeps** (0.1 for B4/B5, 0.01 for FET); `squared` wants λ scaled up. Re-sweep. |
| `routing_norm ∈ {none, zscore, sqrt_width}` | `BoltzmannMoEFFEnergy._logits` | `none` | decouples the routing logit scale from the gradient scale, which in this class are the SAME object. On untrained Hopfield weights (K=8): `none` → `eff_n` 7.999, `sqrt_width` → 6.900, `zscore` → 5.580. `zscore` normalises to unit std, so pair it with `temperature ≈ 0.35` to reach `eff_n ≈ 2`. |
| `head_dim` decoupled from `hidden_size/num_heads` | `energy_attention.py` (`qk_dim = num_heads*head_dim`; three `c_attn.weight` slices; `add_wv_wo` assert), config `_EnergyAttentionArgs`, dispatcher | `None` = coupled | enables over-complete heads (`num_heads*head_dim > d`). Verified fwd+bwd at `D_qk = 4d`; partial RoPE already supported (`rope.py:118`). Both existing checkpoint families load with identical shapes. |

### 7.11 Measured prefill throughput (jobs 1559949/1559955/1559998/1560169)

`scripts/bench_moe_throughput_20260912.py`, H100 80GB, 32,768-token prefill,
`7b_K128` (d=2560, K=128, top_k=8):

| path | ms/call | µs/token | speedup | TFLOP/s | MAC ratio | chunk |
|---|---:|---:|---:|---:|---:|---:|
| `dense-soft` | 2421.83 | 73.91 | 1.00× | 360.9 | 1.0000 | 512 |
| `topk-masked` **(what ships)** | 2425.18 | 74.01 | **1.00×** | 360.4 | 1.0000 | 512 |
| `topk-grouped-exact` | 2027.77 | 61.88 | 1.19× | 229.0 | 0.5312 | 1024 |
| `topk-grouped-proxy` (Python loop) | 127.51 | 3.89 | **18.99×** | 431.2 | 0.0629 | 32768 |
| `topk-gmm-proxy` (`torch._grouped_mm`) | 140.45 | 4.29 | 17.24× | 391.4 | 0.0629 | 32768 |

(Job 1560169, with the weight transpose hoisted. Job 1559998 measured the gmm path
at 12.88× because it transposed the 13.3 GB expert tensor *inside* the timed
region — a bug in the first version of this benchmark, not a property of
`_grouped_mm`. Disregard any 12.88× / 6.78× figure.)

Three things to carry forward:

1. **`top_k` buys exactly 0% wall-clock today** — `p = p * mask` (`energy_ff.py:432`,
   and the legacy class) computes every expert then zeroes the rejected ones.
2. **The fix is batch size, not a kernel.** The sparse path must see large token
   batches: at a 512-token chunk each of K=128 experts gets 32 rows and the loop
   manages only 2.63×; at the full 32,768-token batch each gets 2048 rows and the
   plain per-expert **Python loop reaches 19.02× at 431.7 TFLOP/s** — at/above the
   15.9× arithmetic ceiling (it exceeds it because `dense-soft` is itself forced to
   chunk). Nothing is left for a fused kernel to recover.
3. **`torch._grouped_mm` matches but does not beat the loop** once its weight
   transpose is hoisted out of the timed region. Loop vs gmm, consistently ~2–10%
   on the loop's side:

   | shape | tokens | loop | gmm |
   |---|---:|---:|---:|
   | `3b_K32` | 8192 | 9.90× | 9.63× |
   | `3b_K32` | 32768 | 10.35× | 9.36× |
   | `7b_K64` | 32768 | 10.31× | 8.53× |
   | `7b_K128` | 8192 | 16.81× | 16.57× |
   | `7b_K128` | 32768 | 18.99× | 17.24× |

   The residual deficit is the `[N·k, d]` / `[N·k, I_e]` gather/scatter the loop
   avoids by slicing. Expect `_grouped_mm` to win in the **opposite** regime — small
   batches, decode, or very large K, where per-expert M is small and launch overhead
   dominates (at a 512-token chunk the loop managed only 2.63×). Rule of thumb:
   large batch → per-expert loop; small batch or large K → `_grouped_mm`.

Also: **`dense-soft` cannot run un-chunked at all** at production width — `W h` is
`[32768, 2.6M]` = 171 GB in bf16. The sparse path never forms it. That is an
activation-memory argument for sparse routing independent of FLOPs.

## 8. New scripts (2026-09-12)

Both live in `experiments/boltzmann-moe/scripts/`.

### `analyze_moe_router_flops_20260912.py` — CPU only, seconds

Exact per-token FLOP model derived from the checkpoint's own tensor shapes
(not a hand-written parameter count), plus per-expert SVD spectra and the
cross-expert Gram cosine.

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
OMP_NUM_THREADS=4 python experiments/boltzmann-moe/scripts/analyze_moe_router_flops_20260912.py
#   --ranks 1 2 4 8 16 32 64 128 256     which cumulative-energy ranks to report
#   --skip-spectra                        FLOPs only
```
The checkpoint is currently hardcoded (`CKPT` at the top of the file) — edit it
or parameterize if you point it elsewhere.

### `measure_moe_routing_20260912.py` — needs a GPU

The Phase-0 instrumentation. Runs real BBH few-shot prompts through the model,
hooks the MoE (or the plain FF), and reports everything in §7.4-§7.7.

```bash
python experiments/boltzmann-moe/scripts/measure_moe_routing_20260912.py \
    --run math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu \
    --n_prompts 810 --max_len 1024 --cache_per_iter 6000 \
    --ranks 1 2 4 8 16 32 64 --ablate --n_ablate 250
```

| flag | meaning |
|---|---|
| `--run` | run dir name under `experiments/energy-inference/results/multi-block-ablation/` |
| `--n_prompts` | prompts to sweep for routing stats (spread evenly over the 27 BBH task files) |
| `--max_len` | truncation length |
| `--cache_per_iter` | `(x, E_k)` pairs cached **per iteration** for the frozen-backbone router fit |
| `--ranks` | ranks to test in the cheap-router agreement table |
| `--ablate` | run section G (paired-NLL branch ablation) |
| `--n_ablate` | prompts for the ablation; note it does 6 passes over them |
| `--device` | defaults to `cuda`; `cpu` works for smoke tests |

Sections: **A** routing sharpness per iteration · **A2** branch magnitudes ·
**B** iteration stability · **C** τ / z-score sharpening sweep · **D** rank-r
cheap-router agreement · **F** per-task routing profiles · **G** `--ablate`
paired-NLL ablation (zero the branch, force uniform routing, top-4/2/1).

Outputs into `experiments/boltzmann-moe/results/router_analysis/`:
- `routing_measure__<run>.json` — all sections. Top-level keys: `run`,
  `n_prompts`, `per_iter`, `stability`, `branches`, `sharpening`, `proxy`,
  `tasks`, `ablation`.
- `router_fit_cache__<run>.pt` — `{x, E, iter, K, tau, e_sign}`; **this is the
  frozen-backbone router-fit dataset.** `x` is fp16, `E` fp32.

Notes:
- **Fully offline.** Real activations come from the BBH few-shot prompt JSONLs
  already stored next to each checkpoint
  (`unsharded/bbh_samples_*/samples_bbh_fewshot_*.jsonl`, 27 files × 250
  samples). No dataset download, no `datasets` dependency.
- **Handles the non-MoE Hopfield baseline** — pass
  `--run math_fet_hopfield_mean_8gpt_1egpt6x_d1536_int8k_lra32_itd3_lr1p5e3_33b_16gpu`
  and it runs sections A2 and G only, skipping the routing-specific ones.
- **It does NOT yet support the legacy `BoltzmannMoE_Energy_MLP` class**, i.e.
  none of the `h1_*` / `b*` / `scale_*` checkpoints. Adding support needs an
  `E_k` branch in `Recorder._post` using the legacy formula
  `E = einsum("...h,...eh->...e", x, term1) * self._routing_scale`
  (`mlp.py:672`) instead of `expert.energy_per_token(x)`, plus reading
  `W1`/`W2` rather than a single `W`. **This is the first thing to build if you
  want to evaluate the cheap router on the W1W2 lineage** — which per §3c is
  where it pays off most.

---

## 9. What to do next

### 9a. Decisions taken this session

| Question | Decision |
|---|---|
| Target inference regime | **Prefill / batched** (compute-bound), *not* single-stream decode |
| Finetune budget | **Router-only / frozen backbone** — fit the proxy on cached activations, no training run |
| If routing is uniform | **Fix sharpness first, then top-k** |
| Scope | MoE only, **plus** the free architecture-specific win (route once, reuse across iterations) |

**Why not decode:** at batch 1 the model reads 800 MB of weights per token
against 1.2 GFLOP of math — roughly 200× bandwidth-bound. Top-k on the MoE
saves reading at most ~22 MB of 800 MB (2.8%), and the energy block's `W` is
read once and reused 6× so the iterations are nearly free on bandwidth. **FLOP
reduction in the MoE buys essentially nothing at decode.** If decode ever
becomes the target, attack the 6× iteration count or quantization instead.

**Tension to be aware of:** "router-only / frozen backbone" and "fix sharpness
first" are in conflict *if* routing is genuinely uniform. Sharpening changes the
function the model computes — going from `p ≡ 1/K` to top-1 replaces a mean of
experts with a single expert, which a frozen backbone cannot absorb. In that
branch a finetune is unavoidable; **ask before launching one.** What *is*
router-only-compatible even under uniform routing: collapsing the 8 sliced
GEMMs into 1 fused GEMM (§7.5), and deleting the branch entirely.

### 9b. Concrete queue

1. **Read job 1559723's results** and confirm §7.4/§7.5 at N=810. If confirmed,
   the honest recommendation for this checkpoint is *delete the MoE branch*
   (−25.0% FLOPs, zero cost) and report the dead-branch finding.
2. **Fix the Hopfield energy scale** so the FF branch is load-bearing at all.
   The `(1/d_int)` MEAN normalization is over-suppressing by ~3 orders of
   magnitude at these widths. Candidates: `1/√d_int` instead of `1/d_int`; a
   learnable per-layer energy gain; or raising the `scale_ff` init. This is
   upstream of everything else — a dead branch makes every routing question
   moot.
3. **Make routing scale-free**: z-scored logits **× a learned gain**, or
   `τ ≈ std_k(E_k)`, or a learnable `τ`. Direct analogue of the `1/√expert_I`
   fix that rescued the B-series. §7.4 has the sweep.
4. **Port `measure_moe_routing_20260912.py` to the legacy class** and run it on
   `h1_boltz_moe_fullsize_d768` (avg 0.501, W1W2, K=4×2048, `repulsion_coef=0.1`,
   local unsharded weights available). This is the checkpoint where (a) the FF
   branch is plausibly alive, (b) routing demonstrably specialized
   (`PROGRESS.md:40-44`, `n_dominant_experts` 14-16, `max_expert_load` 0.37),
   and (c) the router genuinely is a second matmul so the cheap proxy saves ~2×
   more.
5. **top-k WITH renormalization** A/B on an existing checkpoint. Both
   implementations do `p = p * mask` with no renorm; part of the +6.4 PPL
   post-hoc truncation cost (§5) is probably just the magnitude collapse. This
   is a one-line change and a cheap eval.
6. **Fit the rank-r + tiny-nonlinear-head router** against
   `router_fit_cache__<run>.pt`, frozen backbone, and report top-k agreement +
   end-task deltas.
7. **Re-measure iteration stability** once routing is sharp (§7.7).

### 9c. Open direction — the denominator problem

The −22% whole-model ceiling (§7.2) is **the wrong denominator for a method
claim.** It is dominated by the GPT prefix (37.5%) and the LM head (25.5%),
neither of which the routing scheme touches. Two better framings:

**(a) Block-level, iso-params / iso-active-params.** Compare a Boltz-MoE block
against a standard learned-router top-k MoE block directly, at matched total
and matched active parameters. The EGPT-block column in §7.2 (down to 0.407×)
is the number that belongs in a paper. Note the counterargument that must be
pre-empted: in a standard setting you would not spend 8 layers on a GPT
prefix — all layers would be distinct MoE layers (possibly with some mid-stack
recurrence for latent reasoning). `BOLTZ_MOE_BEST.md:83-105` already has the
matched-structure Switch ablations and shows the honest margin is only
~0.5-0.7pp avg / ~0.5 PPL.

**(b) All-Boltz-MoE, single recurrent layer.** To be faithful to the
energy-based picture, put **all** parameters in one block and recurse it — no
GPT encoding layers. Sparse MoE should scale better here than a dense EGPT,
which would need to be very wide (and therefore very FLOP-heavy) to match
params. This is the comparison that would actually establish the method. It
needs FLOPs/throughput benchmarking *before* committing to training.

**Also open — decoupling `head_dim` from `d / num_heads`.** Wide attention
(e.g. `d=2048`, `d_head=128`, `num_heads=64`, so `num_heads > d/d_head`) is
**not currently supported**: `EnergyAttention_QK` hardcodes
`self.head_dim = divide_if_divisible(hidden_size, num_heads)`
(`lm_engine/hf_models/modeling_utils/sequence_mixer_blocks/energy_attention.py:75-79`),
which forces `head_dim = d / num_heads`. Supporting `num_heads · head_dim > d`
requires `c_attn` to project to `2 · num_heads · head_dim` (it currently
projects to `2 · hidden_size`, `energy_attention.py:91-96`) and a matching
change to the output projection, which reuses the Q weights. Worth scoping as a
prerequisite for wide-attention experiments.

**Reference point for "when do people use MoE":** the existing comparison set
in `BOLTZ_MOE_BEST.md` runs Switch/Boltzmann MoE ablations down at 585-962M
total params, so this project already operates well below typical production MoE
scale. Treat any "nobody uses MoE this small" objection as a framing issue for
the paper, not a blocker for the ablations.

---

## 10. Known stale references in the existing docs

Fix these when you next touch the docs; they will send you to the wrong code.

| Doc | Claim | Reality |
|---|---|---|
| `BOLTZ_MOE_BEST.md:223` | `BoltzmannMoE_Energy_MLP` at `mlp.py:282` | it is at **`mlp.py:540`**. Line 244 is now `BoltzmannMoE_Hopfield_Energy_MLP` |
| `BOLTZ_MOE_BEST.md:225` | dispatch at `mlp_blocks/__init__.py:78` | `BoltzmannMoE_Energy_MLP` dispatch is at **`__init__.py:113`**; line 78 is now `Mixed_Energy_MLP` |
| `CLAUDE.md:37-47` "Key source files" | lists only `mlp.py` / `_BoltzmannMoEEnergyMLPArgs` | **stale w.r.t. the 2026-06-28 composable refactor.** Missing `energy_ff.py` entirely (the `FFEnergyBase` family, `build_boltzmann_moe`), and `config/mlp.py:245 _EnergyFFBoltzmannMoEArgs`. A reader following this table will not find the code that the `math_fet_boltz_hopfield*` checkpoints actually run |
| `CLAUDE.md:200-215` routing metrics | says the collapse metrics are "Logged every 10 steps" | true for the **legacy** classes only. For any `EnergyFF_*` run they were never logged until the §7.8 fix |
| `configs/multi_block_ablation/math_fet_boltz_hopfield_rep_*.yml:12` | "diagnosed as experts collapsing to uniform routing because (a) per-expert width 1024 is 4× narrower…" | routing *is* uniform (§7.4), but the mechanism is `E_k ≪ τ`, not expert width. And the claim was inferred — the metrics that would have shown it were not being logged |
| `SCALE_UP_PLAN.md:31-32` | notes the `h1_boltz_moe_580m` YAML header says "~580M" but the real count is 679M | correct as written — keep the warning, the config filename is still misleading |

---

## 11. SESSION FINDINGS — 2026-09-15 (routing sign, balance, Sinkhorn, energy stability)

**This session found a bug that changes how three paper claims should be read, and a
principled replacement for the property the bug was accidentally providing.** Detail
docs: `ROUTING_SIGN_BUG_20260915.md` (sign + chemical potential + Sinkhorn + energy),
`PLACEMENT_ARTIFACT_20260915.md`, `ACCEL_FINDINGS_20260915.md`,
`AVG11_ICLR_MIGRATION.md`.

### 11.1 The composable Boltzmann router is SIGN-INVERTED  ← headline

Measured on a trained checkpoint (`iclr_flops/iclr_hop_K32_top2`), reproducing the
deployed routing exactly (zscore, `e_sign="neg"`, tau=0.35, top-2):

    overlap ||gelu(W_k x)||^2/I_e of the SELECTED experts : 1.1358
    overlap averaged over ALL experts                     : 1.8457
    overlap of the LOWEST-overlap 2 experts               : 1.1358   <- identical

**The router selects the worst-matching experts, every token** (0.62x the average).
`_HopfieldExpert` stores `E = mean(gelu(Wx)^2)`, which GROWS with overlap, and
`e_sign="neg"` then picks the smallest. The legacy `BoltzmannMoE_Energy_MLP` this was
meant to reproduce does it correctly (`E = +overlap`, `softmax(+E/tau)`); the
composable refactor added a minus to the energy AND kept the legacy softmax sign.

**Scope:** every `EnergyFF_BoltzmannMoE` run — all 22 `iclr_*` arms and the live 400M.
NOT the learned-gate/Switch arms, NOT gptswitch, NOT the legacy `h1_*`/`b*`/`c*`.

Fix is opt-in: `e_sign_override: "pos"`. Verified in isolation —
overlap(chosen)/overlap(avg) is 0.68 with the default and 1.42 with the override.

### 11.2 Anti-routing was an accidental LOAD BALANCER

Correcting the sign revives the FF branch 20-100x (`ffwd/output_norm` 0.03 -> 12) and
**collapses routing** (effK 8.5 -> 1.9 of 32). The feedback sign flips:

  * anti-routing is self-LIMITING — use the worst-matching expert, it improves,
    you stop using it. Negative feedback, so load spreads on its own.
  * correct routing is self-REINFORCING — the winner gains overlap and wins more.
    The standard MoE collapse that load-balancing losses exist to prevent.

**So two paper claims are affected, and a third was measured under the bug:**
 1. "the energy-FF branch is essentially dead" (HANDOFF 7.5) — **CAUSED by the bug**.
 2. "Boltzmann routing does not collapse and needs no load-balancing loss" —
    **depends on the bug**. Also independently shaky: the deployed arm's own balance
    DEGRADES over training (see 11.5).
 3. "energy routing is at parity with a learned gate" — measured under the inverted
    sign, so the correctly-signed router is untested.

**Do not restate the routing-health results until settled.**

### 11.3 The principled replacement: load balancing is a CHEMICAL POTENTIAL

`p_k ∝ exp(E_k/tau)` is exactly the solution of `max_p sum_k p_k E_k + tau H(p)` —
the softmax IS the entropy-regularised argmax. Imposing balance as a CONSTRAINT on
the batch-marginal occupancy (`sum_tokens p_k ~ N/K`) rather than as a penalty gives

    p_k ∝ exp( (E_k - mu_k) / tau )

with `mu_k` the Lagrange multiplier — a **chemical potential**, the quantity conjugate
to occupancy. A dual variable, not a loss term.

Why this frame is worth having: it reuses the paper's own vocabulary (Boltzmann
weights, partition function, free energy), it has **no gradient pathway and no learned
gate** so the "no auxiliary loss" property survives, and it separates the two roles the
bug had collapsed together — **`E_k` decides which expert FITS, `mu_k` decides how
CROWDED it is.** Anti-routing is a fixed, crude stand-in for `mu_k`: "prefer the expert
you match least" is a static proxy for "prefer the under-occupied expert". Correlated,
hence the balance, but it pays by inverting the SELECTION.

### 11.4 tau x balance sweep at the corrected sign (400M shape, 300 steps)

    arm  tau  mu_k | effK/32  ffwd_norm      lm_loss
    S1  0.35   --  |   4.35     0.582        5.9383   deployed (anti-routing)
    S2  0.35  off  |   2.32    12.44 (121x)  5.9443
    T2  1.0   off  |   4.47    12.13 (129x)  5.9288
    T3  3.0   off  |   7.75     2.36 ( 21x)  5.9464
    T4  0.35   ON  |   3.69    13.94 (160x)  5.9674
    T5  1.0    ON  |   7.45     9.75 (122x)  5.9177  <- best overall

**tau and mu_k are COMPLEMENTARY, not redundant.** mu_k helps at each tau
(2.32->3.69 at 0.35; 4.47->7.45 at 1.0) and tau helps at each mu
(2.32->4.47->7.75 off; 3.69->7.45 on). **tau ALONE is self-defeating**: T3 buys
balance but leaves the branch at 2.36, 5-6x below every mu_k arm, because masked
top-k weights shrink as routing softens.

⚠ **A step-30 reading of this sweep said "tau is not needed" and was WRONG** — step-200
data refuted it. S2 had already shown non-monotone early dynamics (1.03 -> 1.47 ->
2.32). **Do not draw conclusions from this system before ~200 steps.**

### 11.5 SINKHORN — the exact dual beats the clamped control rule decisively  ← ship this

`balance_rate` reaches balance by proportional control on a per-expert bias, and it sat
**PINNED at the +-1.0 `_BIAS_MAX` clamp in every arm** — delivering its result while
saturated. Raising the clamp is the obvious and wrong move (the bound exists because an
unclamped `sign()`-based version hit |bias| = 1482 and destabilised an arm, loss
4.05 -> 5.18 with 77 upward jumps).

`sinkhorn_iters` removes the multiplier: solve the dual exactly by log-domain
iteration `mu <- mu + log(load(mu) * K)`. No clamp, no gain to tune, exact not lagged.
Solved under `no_grad`, so `mu` is constant w.r.t. differentiation — correct for a
Lagrange multiplier, and it keeps the no-gradient-pathway property.

**134M results (built from `iclr_hop_K32_top2`, the BEST 134M Boltzmann arm at
Avg11 44.38; 2000-step budget, matched step 340, uniform max-share would be 0.031):**

    arm                          lm_loss  effK/32  max_share  E_mean  E_max     mu    cos
    M1 shipped (INVERTED)         5.0858     5.04     0.3533  0.5507  14.19     --  0.1226
    M2 corrected + clamped bias   5.1053    12.10     0.1774  0.1809   4.63  1.000  0.0598
    M3 corrected + SINKHORN       5.0392    26.57     0.0718  0.2157  14.00  3.440  0.0582

  * **Sinkhorn is 5.3x better balanced than the shipped arm** (effK 26.6 vs 5.0) and
    2.2x better than the clamped bias. `max_share` 0.072 against the 0.031 ideal,
    where the SHIPPED arm has one expert taking **35% of all tokens**.
  * It gets there because **mu reaches 3.44 — 3.4x past the clamp** M2 is pinned at.
    The clamp WAS the binding constraint, exactly as predicted.
  * M3 also has the **best loss** (5.0392 vs M1's 5.0858) and the **best expert
    diversity** (cos 0.058 vs 0.123). It is not buying balance by homogenising experts.
  * Loss gap 0.047 sits near a 0.038 noise floor (measured at a different shape), so
    treat it as suggestive, not established.

**effK TRENDS matter as much as the levels:**

    M1 shipped   13.8 -> 19.9 -> 12.2 -> 5.2 -> 4.7 -> 4.9 -> 5.4 -> 5.7   PEAKED then COLLAPSED
    M2 clamped    8.3 ->  7.8 ->  8.9 -> 9.4 -> 10.3 -> 11.1 -> 12.1 -> 13.3  steadily improving
    M3 sinkhorn  31.6 ->  9.4 -> 14.3 -> 16.6 -> 20.9 -> 24.6 -> 25.8 -> 26.6  steadily improving

**The shipped arm's balance DEGRADES over training while both balanced corrected arms
IMPROVE.** That is independent evidence against "Boltzmann routing does not collapse".

Sinkhorn engineering checks: converges in **3 iterations** (mu 0.165 -> 0.179 -> 0.180,
then flat); **1 graph, 0 graph breaks** under `torch.compile`, no recompilation;
survives activation checkpointing across {0,3,10} iters x {no,with} repulsion; and
**verified a 0.000e+00 no-op by default** against a clean HEAD worktree with identical
state_dict keys — twice, because the live 400M arm reads this code.

Two documented approximations: **TRAIN-ONLY** (load is a batch property, so applying it
at inference would make routing depend on batch composition — the standard
Sinkhorn-router choice, and it does introduce a train/inference mismatch), and the load
is the **LOCAL per-rank batch** (a global constraint needs a collective, deliberately
avoided given the multi-node hang in 11.7).

### 11.6 ENERGY STABILITY — the feared runaway does not happen, and the SHIPPED arm is the worst

Concern: the block ASCENDS the energy (`out = +grad E`) and the corrected router now
selects the HIGHEST-energy experts — positive feedback that could explode late.

**Structural answer:** the energy is evaluated on `ln_x = self.ln(x)` (RMSNorm), NOT the
raw residual (`models/energy/layer.py:857,1026`). So `E` cannot run away through the
residual growing; the only path left is `||W||`, in which `E` is quadratic. A SOFT
bound — RMSNorm has a learnable gain and only `weight_decay: 0.1` opposes `||W||`.

**Empirical answer (`energy_abs_mean`, added this session):**

    M1 shipped INVERTED  0.107 0.157 0.364 0.547 0.590 0.533 0.504 0.476 0.448  grew 5x, peaked, declining
    M2 corrected+clamped 0.106 0.135 0.250 0.200 0.180 0.162 0.172 0.164 0.186  peaked early, stable
    M3 corrected+sinkhorn 0.108 0.150 0.149 0.158 0.213 0.222 0.217 0.216       PLATEAUED

**The corrected arms carry 2.5-3x SMALLER energy than what is currently shipped.** The
inverted sign is the one that grew 5x. Mechanistically sensible: anti-routing selects
the LOWEST-energy experts and then ascends them, systematically pushing the bottom of
the distribution up. `energy_abs_max` is comparable (M3 14.0 vs M1 14.2), so Sinkhorn is
not introducing worse outliers than production already has.

**Conclusion: no activation change is warranted.** If `E` ever does trend up, the ranked
fix is (1) **weight-normalise the energy**, `E_k = mean(gelu(W_k x / ||W_k||)^2)` — one
line, keeps GELU and the landscape, makes `E` scale-invariant in `W`, and **retires the
`routing_norm: zscore` patch**, which exists for the same arbitrary-scale problem seen
from the too-SMALL side; (2) **normalised descent step / trust region**, leaving the
energy untouched and bounding only the step (`pref` is already just a fixed step size);
(3) **bounded phi (sigmoid/tanh) LAST** — changing phi has a hard negative result here
(`tanh_exact` lost 2.2pp avg / +3.4 PPL) and saturation FLATTENS the energy across
experts, recreating the uniform-routing failure from the opposite direction. Bounding by
saturation costs routing signal; bounding by normalisation does not.

### 11.7 Two invalidated performance conclusions

**(a) The "~5x slower than gptswitch" figure is substantially HOST PLACEMENT.** Same
unfused code, same config, resumed from the same checkpoint on a different host pair:
**6.72-6.81 -> 2.45-2.53 s/step**, 2.7x from placement alone. Identical GPU
model/driver/`gpu_factor`, no MIG; sibling contention ruled out (median 6.739 before
gptswitch finished vs 6.784 after, n=611/89). Likely dataloader starvation from shared
CPU slots: per-rank GPU-busy is 1.71 s, so 25% utilisation at 6.76 s wall vs 68% at
2.49 s. Real ratio ~1.9x, and even that is not placement-controlled.
**Rule: no multi-node s/step claim is admissible unless placement-controlled.**

**(b) The "launch-overhead bound / 79.5k kernels" reading is largely void** — it rested
on 1.71 s GPU-busy against 6.75 s wall (25% util). At 68% the step is much closer to
compute-bound.

**(c) `fused_experts` is validated SINGLE-NODE ONLY.** EXACT (float64 1.227e-15) and
1.61x at 4 GPU / 1 node, but it **WEDGED at 16 GPU / 2 nodes** — 17+ min with ZERO
inductor cache writes against 98 s to first step unfused. Reverted on the live arm.
Process lesson: exactness tests, bf16 bit-identity, checkpointing tests and an A'
replicate ALL passed — none of them can see compilation or collectives. **Exactness
does not transfer across parallelism shapes.** Bisect recorded in ACCEL_FINDINGS.

### 11.8 Repulsion: cost corrected, and it IS load-bearing

  * **Repulsion is 17-21% of the optimizer step, NOT 2-4%.** The old 2-4% came from a
    UNITS error: it divided 48 block-calls by 1536 per-expert projections, but the
    bench's 7.59 ms marginal was measured PER BLOCK CALL and already includes all 32
    experts. Correct: 7.59 x 48 = ~364 ms of 1713 ms = 21%, matched by a direct A/B
    (1.463 vs 1.210 s/step = 17%).
  * **It is load-bearing**: with NO repulsion, expert output alignment sits at 0.43 and
    RISES to 0.51 — it never decays. Every repulsion arm decays instead. So the decay
    is CAUSED by the force, not by experts settling into niches.
  * **But the benefit is steeply front-loaded**: 0 -> 0.5 pair-applications/call buys
    4.7x better alignment for 2.5% of the step; 0.5 -> 4.0 buys a further 4.4x for 15%.
  * **Intermittent firing weakens the regulariser** (alignment plateaus 4-6x higher), so
    prefer **`repulsion_space: "weight"`** — 2.20 vs 7.59 ms/call, N-INDEPENDENT
    (2.21/2.20/2.33 at N=2048/4096/8192 vs 4.06/7.59/14.91), sparse-kernel compatible,
    and full strength every step. Coefficient does NOT transfer: weight cosines are
    ~5-25x smaller than output cosines, so re-sweep (gradient-matched estimate ~0.7,
    aux-matched ~4.0; bracket {0.5, 2, 8}).

### 11.9 Cheap router: the learnable proxy works, the spectral one does not

  * **Learnable rank-8 proxy, distilled online: 0.942 top-2 agreement** with the exact
    router (chance 0.0625), matching the offline fitted-head study's 0.94 at r=8.
  * **The spectral `||Wx||^2` proxy FAILS, and ReLU does not rescue it.** Top-2
    agreement 0.214 (GELU) vs 0.209 (ReLU) — ReLU marginally WORSE; rank-r spectral is
    at or BELOW chance. Reason: the positive-part FRACTION varies per expert per token
    and carries the discriminative signal, which `sum_i z_i^2` discards. So 7.6's
    diagnosis of "the nonlinearity" named the wrong culprit.
  * **Note a conflation in older notes:** 7.6's WORKING r=8 result kept the nonlinearity
    INSIDE the rank-r subspace (`mean(gelu(W^(r)x)^2)`), which is a different and
    costlier construction than the spectral `||W^(r)x||^2`. Only the first works.
  * A **two-moment** proxy is the cheap winner in testing: 0.895 top-2 at `d(1+r)`
    (~57x cheaper than exact), beating the L1 form that costs 512x more. `mu` alone —
    one dot product, cost `d` — already gets 0.638.
  * **True sparsity is NOT implemented.** `top_k` is a post-hoc MASK: all 32 experts'
    forward AND back projections are computed then multiplied by a `p` that is zero for
    30 of them. Back-projection is the free half (~13-15% of the step, no router
    needed); the forward half needs the proxy and cuts the 61% elementwise bucket ~16x.

### 11.10 What to do next  ⚠ SUPERSEDED by §12.7 — and several §11 numbers predate the mu-at-eval fix of §12.1

 1. **Let the 134M arms finish (2000 steps)** and confirm M3 > M1 on loss and that
    `energy_abs_mean` stays plateaued.
 2. **Then a 400M Sinkhorn arm** — but validate MULTI-NODE on the throwaway
    `configs/iclr_scale/scale32B_boltz_hop_PROFILE.yml` FIRST. Sinkhorn's CPU dynamo
    check is clean (1 graph, 0 breaks) but that is not inductor+FSDP, and this is
    exactly the step that was skipped before `fused_experts` wedged the live arm.
 3. **Paper**: hold all three routing-health claims (11.2). The chemical-potential
    derivation is a STRONGER replacement for claim 2, not a retraction — balance as the
    dual of a capacity constraint, aux-loss-free and derivable rather than heuristic.
    None of the efficiency results are affected, nor anything about Switch/gptswitch.
 4. **Unresolved**: no quality evidence at this scale. Every 300-step arm sat inside the
    lm_loss noise floor, and 7.5 found deleting the FF branch entirely moved perplexity
    by +0.0003. "Corrected routing is better" needs a long run with a downstream eval.


---

## 12. SESSION STANDING — 2026-09-16 (READ THIS FIRST)

Section 11 is still correct on the routing sign and Sinkhorn, but **11.10's "what to do next" is
superseded** and several §11 numbers were measured before a large inference bug was found. Start
here.

### 12.1 The one thing that changed everything: mu never reached inference

Sinkhorn's dual `mu` was solved per forward CALL, applied only under `self.training`, and never
stored. So every Sinkhorn model was **trained with mu-tilted routing and EVALUATED with mu = 0**.

Measured on identical held-out web batches, train vs eval mode:

| arm | train CE | eval CE | discontinuity |
|---|---:|---:|---:|
| PURE isoP + sinkhorn | 3.4088 | 4.9959 | **+1.587** |
| PURE T12 + sinkhorn (12 iters) | 3.3403 | 5.1673 | **+1.827** |
| PURE isoP + clamped bias | 3.3360 | 3.3365 | +0.0005 |
| HYBRID K16 + sinkhorn | 2.9454 | 2.9486 | +0.0032 |

The cost scales with how much of the net depends on mu: all iterations of a pure stack, 1 block of
7 in a hybrid. The clamped `load_balance_bias` shows ~zero because it is a persistent buffer with
no `self.training` gate, so it always reached eval.

**FIXED** by `sinkhorn_persist_mu` + `sinkhorn_mu_iters` (a PER-ITERATION buffer, cycled by call
index — one shared buffer cannot work, the duals differ across iterations by 1.56-1.90 mean spread
and individual experts flip sign, e.g. -1.27 at iter 0 to +1.39 at iter 1). After the fix the
discontinuity is **-0.058**, i.e. gone. `calibrate_sinkhorn_mu_20260915.py` recovers mu for an
already-trained checkpoint in minutes without retraining.

**REQUIRED on every new Sinkhorn arm:** `sinkhorn_persist_mu: true` and `sinkhorn_mu_iters` = that
block's `layer_iterations` entry. Optional for hybrids (0.003 nats) but free.

### 12.2 The bug INVERTED the depth scaling law — the most consequential finding

bits/byte on wikitext (report THIS, not `word_perplexity`, which is `exp(bpb * 3.7)` and turns a
1.55x regression into a 7.8x one):

| arm | iters | mu DROPPED | mu RESTORED |
|---|---:|---:|---:|
| hybrid K16 (1 MoE of 7) | 6 | 1.0000 | — |
| pure big | 4 | 1.2524 | 1.2103 |
| pure 1blk | 8 | 1.3966 | 1.1525 |
| pure isoP | 8 | 1.5530 | 1.1201 |
| **pure T12** | **12** | **1.6034** | **1.0996** |

Both columns monotone, **opposite directions**. Under the bug deeper is worse; fixed, deeper is
BETTER, and T12 becomes the best pure model (Avg11 40.96, bpb 1.0996). Anyone reading the pre-fix
numbers would have concluded depth hurts, most confidently from the arm that proves the opposite.
**Any recurrence-depth claim measured before 2026-09-16 is suspect.**

### 12.3 Corrected-sign grid: 14 arms, all token-matched but one

Mean Avg11 delta over the six HYBRID arms is **+0.03pp** — the sign correction is quality-neutral,
so the paper's parity claims survive. Best rows: K32-top2 **44.58**, renorm 44.54, K16-dense 44.40,
K32-top1 44.19, K16-top2 43.50. Pure arms (mu restored): T12 40.96, bal_corr 41.49, 1blk 40.43,
isoP 40.42.

**`iclr_big_hop_pure_sink` is NOT comparable** — registered at 4 GPUs while its published
counterpart ran at 8, so it saw 1.97B tokens against 3.93B. Its -0.59pp is undertraining, not the
sign. Strike it from any analysis. 13 of 14 match to 0.01%.

### 12.4 Two BROKEN SCHEDULES, one of them in the published configs

`slope90k_*` set `num_training_steps: 90000` but warmup 2000 + decay 28000 = **30000**, so 60000
steps ran pinned at the 2e-4 floor. Loss falls 0.12-0.15 per 10k while decaying, then
-0.002..0.000 at the floor: the last 45-51k steps bought 0.015-0.019 nats. **The bug is in the
published `iclr_slope` configs**, so the paper's "+0.41pp for 3x the tokens" is a lower bound, and
the conclusion drawn from it has been DELETED from the paper.

Hence §12.6's replacement arms. This is check 1 of the new CLAUDE.md pre-flight.

### 12.5 LR: peak matters, floor does not, and 1e-2 is on a stability boundary

Pure isoP, 5000-6000 step grids at 262144 tok/step:

* **FLOOR is inert.** 2e-4 vs 2e-5 at peak 2e-3: -0.021. 1e-3 vs 2e-5 at peak 1e-2 (a 50x range):
  +0.008. Both inside the 0.038 noise floor, and the two contrasts disagree in sign.
* **PEAK is worth ~0.06 nats.** 2e-3 -> 1e-2 gives +0.058 at floor 2e-4 and +0.069 at floor 2e-5.
* **But 1e-2 diverges ~1 in 3.** Three arms at identical peak 1e-2 and the same default seed 42:
  two healthy (grad_norm 3.1), one blew up (grad_norm **7045**, loss 6.44 -> 8.34, ending 6.39, no
  recovery under cosine decay). `gradient_clipping: 1` was ACTIVE and did not prevent it. A reseed
  at seed 7 trained cleanly to 3.8521, the best of the grid.
* 2e-2 diverges outright (7.89). Lower LRs are simply worse.

**Recommendation: keep 2e-3 for long runs.** 0.06 nats is not worth a ~1-in-3 divergence, and the
mu fix was worth 1.6-1.8 nats, ~30x more. The pure model's steeper log-log slope (-0.097 vs -0.082
hybrid) means it wants more TOKENS, not a bigger step.

### 12.6 RUNNING RIGHT NOW

| arm | queue | GPUs | progress | note |
|---|---|---|---|---|
| `scale32B_boltz_sinkhorn` | normal | **16** | 34.5k/61035 (57%) | 400M HYBRID, 32B tokens, schedule OK, all knobs right except `persist_mu` (0.003 nats, recalibratable). **Do not restart.** ~1.5 days left |
| `t90k_switch_lastisoP` | preemptable | 4 | fresh | 90k = 23.6B, correctly scheduled. ~7.4 h |
| `t90k_hybrid_K32top2` | preemptable | 4 | fresh | ~12.9 h |
| `t90k_pure_T12` | preemptable | 4 | fresh | ~37.8 h |
| `slope90k_{1blk,hyb}_sink` | preemptable | 4 each | ~75-82k/90k | BROKEN schedule; kept only for like-for-like vs the published 44.32 |
| `iclr_big_hop_sandwich_sink` | preemptable | 4 | 8k/15k | |

`grp_ebm` is **32/32**: bsaha3's `s8e4_f5kl_distill` (16) + our 400M arm (16). Everything else is
on `grp_preemptable` (1397/6144). `blimits`, not `bjobs`, is the authoritative quota check.

### 12.7 WHAT WE ARE FOCUSED ON NEXT

1. **TRUE SPARSITY — in progress, half landed.** `sparse_backproj` (opt-in, default off) does the
   back-projection over only the top-k experts via capacity dispatch: fixed `(K, C, I_e)` buffer +
   one `bmm` + scatter-add. **Verified EXACT** vs the dense mask (relative error 3.79e-16, overflow
   0); overflow drops pairs and is COUNTED in `_sparse_overflow`. Back-GEMM saving 6.40x at K=16
   k=2, 12.80x at K=32 k=2. Deliberately not a per-expert loop — that measured 0.59x, SLOWER.
   * **NOT yet measured: actual step-time delta.** Back-projection is 13-15% of the step, so
     expect ~12% wall-clock. Benchmark this next.
   * **The big half is still blocked**: the forward projection cannot be skipped by an exact router
     (the 1/2(1+k/K) floor) and needs the proxy router (0.942 top-2 agreement at rank 8). That is
     where the 61% elementwise bucket lives, and where the pure design pays off most — the mixture
     is 98.9-99.6% of per-token FLOPs in a pure stack against ~22% in a hybrid, so sparsity is
     end-to-end there rather than capped at 22%.
2. **The 3-family x 2-scale matrix.** Have: 134M @ 7.86B complete for hybrid / GPT-switch / pure;
   400M @ 32B for GPT-switch (done) and hybrid (running). **Missing: 400M PURE** (lower priority
   per the user) and, until §12.6's arms land, any sound 134M run beyond 7.86B.
3. **Paper.** Pushed through `48da63b`. Open `\CC` items: the un-remeasured expert-cosine bound;
   the token-scaling paragraph pending §12.6; and the 400M pure router comparison, whose "+0.19pp
   in favour of energy" measures the SIGN-INVERTED router and must not be quoted as support.
4. **Untracked and NOT in the repo**: `experiments/eval_scripts/compute_avg11.py` plus six Avg11
   helpers. `compare_sign_correction_20260915.py` IMPORTS compute_avg11, so a fresh clone cannot
   run it. Decide whether to track them.

### 12.8 Process rules added this session

`CLAUDE.md` now opens with a **10-point config pre-flight**, mandatory before any long run. Each
point comes from a bug that cost real compute: schedule coverage, tokens/step matching, hosts vs
GPUs, `load_args`, the checkpoint pointer vs `max_to_keep`, glob prefix collisions
(`foo_*` matches `foo_sink_*` and `foo_s7_*` — this produced two wrong numbers in one session),
knobs reaching the model rather than just the YAML, the per-expert-kind sign direction, the
Sinkhorn requirements, and diffing against the arm being copied.

