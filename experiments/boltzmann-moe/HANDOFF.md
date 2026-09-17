# Boltzmann-MoE — HANDOFF

**Entry point for a fresh session.** Read this first, then jump into the specific
doc you need. This file carries (a) orientation, (b) the things that exist
nowhere else, and (c) the 2026-09-12 session findings, which materially change
the project's conclusions.

Last updated: 2026-09-16 (true sparsity §12.9-12.12; eval-metric banner added).

> ## 📏 EVAL METRIC — THE REFERENCE IMPLEMENTATION IS A SCRIPT, NOT A DESCRIPTION
>
> **Every headline number in this project and in the ICLR paper is `Avg11`, and the reference is
> `experiments/eval_scripts/compute_avg11.py`.** Run that; do not re-derive the recipe by hand and
> do not read an "Avg" from an older doc without checking which convention it uses.
>
> ```bash
> python experiments/eval_scripts/compute_avg11.py <run_dir_or_unsharded_dir>
> ```
>
> * **11-task unweighted mean.** `acc_norm`: arc_challenge, arc_easy, hellaswag, openbookqa, piqa,
>   sciq. `acc`: boolq, copa, winogrande, race, lambada_openai.
> * **MMLU (acc) and GSM8K-CoT (flexible-extract) are reported SEPARATELY and are NEVER folded into
>   the mean.** WikiText likewise separate — and report it as **bits/byte**, not `word_perplexity`
>   (they differ by `exp(bpb * 3.7066)`, which turns a 1.55x regression into a 7.8x one).
> * **This is the colleagues' recipe**, so our numbers are directly comparable to the EGPT-RL / FET
>   series. Source of truth: `~/Code/GPT-experiments/projects/EGPT-RL/RESULTS.md:247-249`.
>   `compute_avg11.py` is validated against their stored Avg11 on the two shared
>   `math_egptdual` seed checkpoints to within 0.004pp.
> * The script is now TRACKED in the repo (2026-09-16). It reproduces the published values exactly
>   — independently re-evaluated this session: `pure_hop_T12_sink` **40.96** (§12.2) and
>   `iclr_hop_K32_top2_sink` **44.58** (§12.3).
> * **Two superseded conventions appear in older sections of this file — never mix them with
>   Avg11.** `avg9` (9 tasks, MMLU excluded, race/lambada absent — the pre-`pyarrow>=20` bug) and
>   `avg10` (10 tasks, MMLU INCLUDED, race/lambada EXCLUDED). Avg11 runs ~3pp BELOW avg10 because
>   race (~0.28) and lambada (~0.23) sit near chance at our scale: a scoring-convention gap, not a
>   model effect. Full detail in `CLAUDE.md`'s metric block and in the script's own docstring.

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

> **⚠ CORRECTION 2026-09-16 — READ THIS BEFORE REUSING THE 0.94/0.90 ABOVE.**
> Those numbers are for the **exact energy restricted to the rank-r subspace**, i.e.
> `mean(gelu(W_k V_r V_rᵀ x)²)`, which keeps the true nonlinearity. They are **NOT** measurements
> of the "tiny fitted nonlinear head" recommended just below, and they were subsequently quoted as
> if they were — in `energy_ff.py`'s proxy comment, in HANDOFF §12.9, and in my own reasoning.
> Measured on `pure_hop_T12_sink`, the fitted diagonal-quadratic head reaches **0.348 top-2 at
> r=8** against a chance floor of `k/K` = **0.125**, and only 0.461 at r=32. A diagonal quadratic
> in `a` cannot represent `mean(gelu(·)²)`. See §12.11. The cost line above is also incomplete:
> `K·d·r` counts only the projection and omits the `K·I_e·r` term for `B_k a_k`.

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
   * ~~**NOT yet measured: actual step-time delta.**~~ **MEASURED — see 12.9.** sparse_backproj
     alone is 1.18-1.35x compiled at the real batch size and a LOSS (0.44-0.94x) at a small one.
     The full `sparse_forward` is what pays: **4.69x** compiled fwd+bwd on the pure arm, past the
     "cannot beat 2x" floor. Both depend strongly on tokens/call.
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
4. ~~**Untracked and NOT in the repo**: `experiments/eval_scripts/compute_avg11.py` plus six Avg11
   helpers.~~ **DONE 2026-09-16** — `compute_avg11.py` and the six helpers are now tracked
   (commit `8e726fa3`), so a fresh clone can reproduce every headline number. It is THE reference
   implementation of the metric; see the banner at the top of this file.

### 12.8 Process rules added this session

`CLAUDE.md` now opens with a **10-point config pre-flight**, mandatory before any long run. Each
point comes from a bug that cost real compute: schedule coverage, tokens/step matching, hosts vs
GPUs, `load_args`, the checkpoint pointer vs `max_to_keep`, glob prefix collisions
(`foo_*` matches `foo_sink_*` and `foo_s7_*` — this produced two wrong numbers in one session),
knobs reaching the model rather than just the YAML, the per-expert-kind sign direction, the
Sinkhorn requirements, and diffing against the arm being copied.


### 12.9 TRUE SPARSITY — the other half landed, and the wall clock disagrees with the FLOPs

**What is in the code now** (`energy_ff.py`, opt-in, every default unchanged):

* `sparse_forward` — skips the forward AND back projection of the K-k experts a token was not
  routed to. Requires `proxy_rank > 0` and `fused_experts` (asserted): an exact router cannot
  skip the forward projection, because it needs all K energies to decide. That is the
  `1/2*(1+k/K)` floor, and the proxy is what breaks it.
* `proxy_route` now DOES SOMETHING. It was plumbed through config, builder and constructor and
  then **never read** — `self.proxy_route` was assigned and that was the end of it. Pre-flight
  check 7, in the file that check was written for. It now moves SELECTION to the cheap router
  while the WEIGHTS stay exact, in the dense path too, which is the controlled A/B for selection
  quality with the dispatch machinery held out.
* `sparse_backproj`, `sparse_forward` and `sparse_capacity_factor` reached the pydantic args
  class and `get_mlp_block` for the first time. Yesterday's `sparse_backproj` was reachable ONLY
  by calling `build_boltzmann_moe` directly — which is exactly what its test does, so the test
  passed while a YAML using the flag would have been REJECTED outright (`extra="forbid"`). Loud
  rather than silent, so no wrong run came of it, but the flag was unusable.
* `_logits` split into `_logits_raw` + `_mu_for` so the dual is solved, and `_mu_call` ticked,
  exactly ONCE per forward even though the sparse path needs the logit map twice.
* `_dispatch_plan` factored out and shared by both sparse paths so they cannot drift.

**Exactness.** `test_sparse_forward_20260916.py`. With an ORACLE proxy (returns the exact
energies) the sparse path reproduces the dense output to **4.3e-16 - 8.0e-16** with the selection
left FREE, in all three of `routing_norm=none/renormalize`, `zscore/renormalize` and
`zscore/masked`. So **the proxy's prediction error is the ENTIRE approximation** — there is no
second error hiding in the dispatch, the denominator completion or the zscore moments. Forcing
the dense selection also gives bit-exactness (7.98e-16, 5.05e-16), and overflow is counted, not
silent. Two smaller approximations can be removed by config rather than by code:
`renormalize_topk: true` deletes the proxy-completed denominator term entirely (and was measured
at Avg11 44.54 against 44.58 for the masked form, i.e. free), and `routing_norm: none|sqrt_width`
needs no per-token moments.

**Wall clock — H100, one GPU, ms per block call, speedup vs the dense mask.** Measured four ways
because the answer CHANGES SIGN between them. `torch_compile: true` on every arm and
forward+backward is what training pays, so **the compiled f+b column at the arm's real per-call
token count is the only admissible training number**; the rest is there to show why.

tokens/call = 4096:

| shape | eager fwd | eager f+b | compiled fwd | compiled f+b |
|---|---:|---:|---:|---:|
| pure_T12 K=16 I_e=4480 · backproj | 1.02x | 1.09x | 0.89x | 0.94x |
| pure_T12 · **sparse_forward** | 2.04x | 2.82x | 2.14x | 1.36x |
| hyb_K32 K=32 I_e=512 · backproj | 0.77x | 0.89x | 0.44x | 0.53x |
| hyb_K32 · **sparse_forward** | 0.72x | 0.80x | 0.39x | 0.41x |
| big_hyb K=16 I_e=1280 · backproj | 0.79x | 0.95x | 0.57x | 0.66x |
| big_hyb · **sparse_forward** | 0.91x | 0.99x | 0.62x | 0.49x |

tokens/call = 16384 (dense f+b compiled: 24.67 / 5.44 / 8.46 ms):

| shape | eager f+b | compiled fwd | **compiled f+b** |
|---|---:|---:|---:|
| pure_T12 · backproj | 1.17x | 1.29x | 1.35x |
| pure_T12 · **sparse_forward** | 4.96x | 4.33x | **4.69x** |
| hyb_K32 · backproj | 1.12x | 0.98x | 1.18x |
| hyb_K32 · **sparse_forward** | 2.87x | 1.50x | **2.09x** |
| big_hyb · backproj | 1.08x | 1.02x | 1.21x |
| big_hyb · **sparse_forward** | 2.54x | 1.95x | **2.37x** |

Four things follow, two of which correct claims of mine:

1. **The saving is strongly batch-size dependent and the sign flips.** At 4096 tokens/call
   `sparse_forward` LOSES on both hybrid shapes (0.41x, 0.49x compiled); at 16384 it wins
   2.1-2.4x. The dispatch overhead is O(T*k) index work independent of `I_e`, while the saving
   is O(T*(K-k)*I_e), so it needs a wide expert AND enough tokens for the batched GEMM to be
   efficient — C = cf*T*k/K is 640 at T=4096 against 2560 at T=16384. **Any sparsity claim must
   state the per-call token count**; a single number is meaningless.
2. **`sparse_backproj` is small but not nothing: 1.18-1.35x compiled at 16384.** §12.7 predicted
   ~12% from FLOP arithmetic; that was right by accident at the wrong batch size and wrong at
   4096 (0.44-0.94x, i.e. a LOSS). It is dominated by `sparse_forward` everywhere and is now
   redundant with it, so it is not worth shipping on its own — but "worthless" overstated it.
3. **4.69x on the pure arm is past the "cannot beat 2x" floor, measured.** That floor applies to
   an exact router, which must compute all K energies before it can choose. The proxy is what
   breaks it, and this is the number that says the idea works rather than merely type-checks.
   4.96x eager against a 6.36x FLOP bound is 78% of theoretical, so the residue is dispatch
   overhead, not something structural.
4. **The 400M hybrid is the worst case exactly as configured.** `micro_batch_size: 1` x 4096 =
   4096 tokens/call — the column where sparsity loses (0.49x compiled). Raising micro_batch_size
   and cutting `gradient_accumulation_steps` to match keeps tokens/step identical and moves it
   into the winning regime. Cheap, and UNTESTED. Do not enable sparsity on that arm without it.

Net: this is a **pure-model lever**, which is where §12.7 expected it — the mixture is 98.9-99.6%
of per-token FLOPs in a pure stack against ~22% in a hybrid, and the pure design also wants the
wide experts that sparsity needs. It also cuts the intermediate activation footprint by
cf*k/K, i.e. 6.4x at K=16 k=2 (0.55 GiB -> 0.09 GiB at T=4096, bf16), which is what limits `I_e`.

**Not yet measured: the quality cost.** `calibrate_proxy_router_20260916.py` fits the proxy on a
trained checkpoint with no retraining — `proxy_V` from the SVD of each trained `W_k` (sound
because HANDOFF 7.3 measured the trained expert weights near rank-2), then least squares for the
2r+1 head coefficients, then it reports genuine per-row top-k set agreement and, on a recurrent
stack, the PER-ITERATION breakdown. That last column is the risk: one proxy serves all 12
applications of a shared block, which is the structure that made a single shared `sinkhorn_mu`
buffer fail (§12.1). Written and parses; NOT yet run against a checkpoint.

### 12.10 t90k_pure_T12 restarted from step 0, twice, and the mechanism is general

Found while checking arm health: T12 was at step 2670, then at step 1030 with
`learning_rate = 1.03e-3` — warmup, i.e. a fresh run. Two `step = 10` lines in its log. ~2670
steps lost.

**Mechanism.** The watchdog resolves `load_args` at SUBMIT time. LSF preemption REQUEUES a job on
the same jid and re-runs the ORIGINAL command, so an arm launched *before its first checkpoint
existed* had no `load_args` baked in and restarted from step 0 on every preemption, indefinitely.
At 1.59 s/step over 90000 steps this arm needs ~40 h uninterrupted on `preemptable`; it would
never have finished. `switch` and `hybrid` were unaffected only because they have not been
preempted yet — they were equally exposed.

**Fixed two ways.** (a) `load_args` appended to all three `configs/tok90k/*.yml`, verified through
the real loader to point at a checkpoint dir that exists (latest 22000 / 11000 / 1000) — this
protects the jobs already submitted, whose baked-in command reads the base config. Duplicate
`load_args` from the watchdog's own append is harmless: PyYAML takes last-wins and both name the
same path. (b) `watchdog_loop.sh` now resolves `load_args` INSIDE the job script, so every
requeue re-evaluates it; patched by atomic rename so the running watchdog's bash is not disturbed.
It logs `RESUME:` lines, which is the thing to grep for after any preemption.

One oddity left alone: `t90k_pure_T12/global_step2000` is a stale checkpoint from the first
attempt, newer in step number than the pointer (1000) but from a different run. The trainer only
ever reads the pointer, so it is inert — do not "fix" it by pointing at 2000.


### 12.11 The proxy router: the quad head FAILS, and §7.6's 0.90 was never a head measurement

**The negative result.** Post-hoc fit of the shipped diagonal-quadratic proxy on
`pure_hop_T12_sink` (K=16, top-2, 98,304 tokens from the training corpus tail). Chance floor for
top-2 of 16 is `k/K` = **0.125**, verified empirically at 0.1256:

| r | top-2 agreement | energy R² | x chance |
|---:|---:|---:|---:|
| 2 | 0.2460 | 0.324 | 2.0x |
| 8 | **0.3481** | 0.572 | 2.8x |
| 16 | 0.4101 | 0.682 | 3.3x |
| 32 | 0.4614 | 0.760 | 3.7x |

Above chance, nowhere near usable: routing on this would change which experts fire for ~2/3 of
tokens. **§12.9's "0.942 top-2 agreement at rank 8" is wrong** and is corrected there.

**Cause 1: a documentation error that propagated.** §7.6 measured the EXACT energy restricted to
a rank-r subspace and got 0.94/0.90; it then RECOMMENDED a fitted head as the implementation. The
0.90 was carried forward as though it described the head — the code comment records the slide
verbatim ("What works is the exact energy restricted to a rank-r subspace: r=8 gave 0.94... This
is the trainable version: a learned per-expert projection V_k plus a diagonal-quadratic head").
§7.6 now carries a correction box.

**Cause 2: one proxy cannot serve a recurrent stack.** Grouping the 96 block calls by position in
the 12-iteration cycle (means over 8 batches):

| cycle pos | 0 | 1 | **2** | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | **11** |
|---|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|--:|
| agreement | .436 | .202 | **.080** | .251 | .229 | .278 | .344 | .317 | .417 | .482 | .551 | **.592** |

A **7.4x spread**, and at position 2 it is **below the 0.125 chance floor** — anti-correlated with
the true router. Same structure that defeated a single shared `sinkhorn_mu` buffer (§12.1): a
shared block solves a different problem at each application.

**Cause 3: the wrong objective.** The fit was closed-form least squares on energy **MSE**, while
the metric is top-k SET agreement. LS spends capacity on the magnitude of experts that will never
be selected and gets no credit for ordering the two that will.

**What landed in response** (`proxy_kind`, `proxy_out_dim`, `proxy_iters`; all default to the old
behaviour):

* `proxy_kind: "subspace"` — `E_hat_k = mean(gelu(B_k a_k)²)·s_k + b_k` with `B_k = W_k V_k`.
  `W_k P_k x = B_k a_k` exactly, so the ONLY error is rank truncation, not function class.
* **`proxy_out_dim` = m, and it is not optional.** `B_k` is `(I_e, r)`, so the naive subspace form
  materialises `(T, K, I_e)` — **the same elementwise work and the same activation footprint as the
  DENSE path**, cancelling two of sparsity's three savings and leaving only the GEMM. `E_k` is a
  MEAN over `I_e` coordinates, so an m-row subsample is an UNBIASED estimator with variance ~1/m.
  At K=16, r=8, d=768, I_e=4480: m=all is 672K MAC but 71,680 elementwise (no better than dense);
  **m=512 is 164K MAC and 8,192 elementwise** — 0.15% of dense MACs and 11% of its elementwise.
* `proxy_iters` — one head per iteration, cycled by call index. **EVAL/CALIBRATION ONLY, asserted.**
  A call counter is unsound in training under activation checkpointing, which replays the forward
  during backward; `gradient_checkpointing_method: block` is set on the 400M arm. Training with
  per-iteration heads requires the block to be told its iteration index, which it is not.
* `fit_subspace_proxy_20260916.py` — data-aware basis (SVD of `W_k Σ^{1/2}`, i.e. the best rank-r
  approximation in the DATA metric rather than of `W_k` alone), closed-form scale/bias, then
  **gradient refinement on the routing KL**. All agreement numbers on a HELD-OUT split, since the
  refinement fits thousands of parameters.

**Consequence for §12.9's speedup, stated plainly:** the exact mechanism (`sparse_backproj`) is
capped at **1.73x** by `1/2(1+cf·k/K)` and delivers 1.18-1.35x. The 4.69x requires
`sparse_forward`, which requires a working selector. Until agreement is high, **4.69x is a
capability, not a result.**

### 12.12 Why the retrofit fails: with `renormalize_topk: false`, top-k is a SCALING, not a sparsification

This is the structural fact behind §12.11's numbers, and it decides how a sparse arm must be
configured.

`_route` computes `p = softmax(all K logits)` and then ZEROES all but the top-k, leaving
`sum(p) < 1`. So the surviving weights depend on **every** expert's energy through the shared
denominator. The top-k mask does not remove the other experts from the computation; it removes
their OUTPUT while keeping their influence on the SCALE of the ones that remain.

How big is that influence? From `t90k_pure_T12`'s own training log:

```
load_mean_token_entropy   = 0.523-0.536   (normalized, so effective experts/token = K^H)
=> K^H = 16^0.536 = 4.42 of 16
=> a top-2 mask captures ~2/4.42 = 45% of the softmax mass
=> the UNCOMPUTED tail is ~55% of the denominator
```

which matches the independently-known `sum(p) ~= 0.45` for this family. **A sparse path that never
evaluates K-k experts is therefore missing ~55% of the quantity that sets its own output
magnitude**, and it must either estimate it or not need it. That is why:

* over-selection barely helps (p=8 still leaves ~8 miscalibrated terms: bpb 3.03 against a dense
  1.0996), and why the curve is DISCONTINUOUS at p=K, where the tail vanishes and the zscore
  moments become exact in the same step: 3.03 -> 1.0996;
* `renormalize_topk: true` fixes it structurally -- there is no all-K sum to estimate;
* and a proxy trained on ranking alone cannot supply it. KL is nearly invariant to the energies'
  absolute scale, so `--mse_coef` adds a calibration term. Whether that is enough to rescue the
  RETROFIT is what job 1706755 measures; it is not needed at all for an arm TRAINED with
  renormalisation.

**Consequence for the retrain (TODO gate 2):** set `renormalize_topk: true`. Retrofitting it costs
+0.483 bpb because `scale_ff` was trained against `sum(p) ~= 0.45`, but TRAINING with it is free --
§12.3 measured `iclr_hop_K16_top2_renorm` at Avg11 **44.54** against **44.58** for the masked form.
With the all-K denominator gone, the remaining approximations are the zscore moments (+0.318
retrofitted, removable with `routing_norm: sqrt_width`) and SELECTION, at **+0.0165**.

A note for anyone tempted by the masked form's rationale: leaving `sum(p) < 1` was chosen to avoid
abrupt weight redistribution at routing boundaries (see the comment in `_route`). That choice is
what makes the design un-sparsifiable, and the arms trained with renormalisation show it costs
nothing to give up.

### 12.13 Sparsity: FINAL numbers, and the negative result on sparse TRAINING

**Training speedup: 2.02x, placement-controlled.** Both arms back to back in ONE allocation, same
host, same contiguous GPUs, matched micro-batch, matched 262144 tokens/step: dense **1.538** vs
sparse **0.762** s/step. Cross-validated -- the dense figure matches the live 90k arm's 1.590, the
sparse figure matches 0.776 measured on a different host.

**Do NOT compare s/step across jobs.** The same sparse config measured 0.75-0.80 s/step on one
host and 1.71-1.84 on another, a factor of 2.3, i.e. larger than the effect. Earlier figures of
2.55x / 2.15-2.31x were cross-job and are WITHDRAWN.

**We did NOT lose GPU exclusivity — be precise about this.** `mode=exclusive_process` implies
`j_exclusive=yes`, so no other job can be assigned our devices, and none was. What is shared on a
node regardless is the NVLink/NVSwitch fabric, PCIe, host memory bandwidth and the power envelope.
The slow host had its OTHER four GPUs held by four jobs belonging to OTHER USERS (osieberl x3,
keshavr x1), filling it to 8/8.

Two explanations RULED OUT, both of which I asserted before checking:
* **Not CPU/dataloader starvation** (what §11.7's placement note diagnoses): the slow host was at
  12% CPU utilisation, the fast one at 67%. Occupied CPU is fine.
* **Not allocation topology.** Every arm here gets a non-contiguous GPU set (2,3,4,6 / 0,1,2,4 /
  5,2,3,4) and most run at full speed, so scattering is the norm, not the cause. A `glink=yes`
  suggestion based on this was withdrawn.
What remains is neighbour LOAD rather than neighbour count: s90k later ran at 0.756 s/step on a
host with four neighbours. §11.8's watchdog note independently measured 2.35 -> 6.05 s/step "when
neighbours arrive", which is the same effect and larger.

**The micro-batch trap.** Dense + `fused_experts` OOMs at mbs 4 (the (4, 4096, 71680) intermediate
is 2.19 GiB), so a naive same-allocation probe forces dense to mbs 1, which costs 1.65x by itself
and inflates the ratio to 3.37x. Match the micro-batch or the number is wrong.

**Sparse TRAINING did not reach dense quality.** Matched schedule, tokens/step, device count and
init; loss gap widened monotonically: +0.004 / +0.030 / +0.042 / +0.068 / +0.081 nats at steps
500-900. Two mechanisms alongside:
* **Expert diversity is not controlled.** Output-space repulsion needs all K expert outputs, which
  the sparse path never computes. Of the substitutes, ranked by the OUTPUT alignment they achieve
  at matched steps: full output-space **0.20-0.27** (unavailable), weight-block cosines
  **0.43-0.44** from scratch but **flat at 0.72-0.75** when applied to an already-collapsed arm,
  and a subsampled output-space estimator **0.53 -> 0.74 and rising** (worst). The subsample is
  unbiased in VALUE (within 2%) but that is the wrong property -- its variance is far higher, and
  under Adam a high-variance term is damped relative to its mean. I validated the value and
  shipped a weak regulariser.
* **The proxy collapses on hand-off.** 0.703 immediately after the switch (so the two-phase
  transfer works), 0.46 within 30 steps of routing being handed over, then only +0.033 per 400
  steps. Routing on the proxy moves the energy landscape faster than a candidate-restricted
  objective tracks it.

**What is NOT separable:** "the design fails" vs "it needs a regulariser we did not find".
Reversing an alignment collapse and preventing one are different problems, and the clean experiment
-- a fresh sparse arm with weight-space repulsion from step 0, ~22 min -- was not run.

**Recommendation for the paper:** report inference sparsity (§app:throughput's 18.99x at production
width, untouched), the training implementation as a placement-controlled 2.02x capability that is
EXACT given the selection (4e-16; `sparse_candidates = K` reproduces dense bit-for-bit on a real
checkpoint), and the training-quality failure as a negative result. All three are in
`sec/appendix.tex` §app:accel-train / §app:accel-procedure as of Overleaf `c8d8b63`.

**Bugs found in this work, all mine, three of them silent:** `proxy_route` never read;
`sparse_backproj` never reaching the pydantic config; `_proxy_step` never called in the sparse path
(so `proxy_loss_coef` was a no-op and an arm trained with a FIXED RANDOM router -- alignment 0.698
-> 0.785 and no metric logged); data-dependent index shapes (wedged distributed compile, 0 steps in
8 minutes); `torch.randint` in the compiled forward (hung a job, 14 min/0 steps); duplicate
candidates double-counted after exploration was added. Every one was caught by a run misbehaving,
not by review.

### 12.14 Depth vs width at FIXED FLOPs, 400M scale — width wins at 1k steps, and it cuts against §12.2

**The design.** Three arms, all at **1.17 G MAC/token** in the mixture, d=1024, K=16, top-2, peak lr
2e-3, 131072 tok/step, 4 GPUs each. Iterations share the recurrent block's weights, so holding
FLOPs fixed while adding depth forces width — and PARAMETERS — down:

| arm | iters | I_e | total params | s/step |
|---|---:|---:|---:|---:|
| `d400_it4` | 4 | 17920 | **401M** | **3.18** |
| `d400_it8` | 8 | 8960 | 254M | 3.73 |
| `d400_it12` | 12 | 5952 | 204M | 4.09 |

**Result — `it4` wins at every step, monotonically:**

| step | it4 | it8 | it12 |
|---:|---:|---:|---:|
| 100 | **6.7398** | 6.9977 | 7.2114 |
| 200 | **6.1626** | 6.3610 | 6.4838 |
| 300 | **5.7384** | 5.9468 | 6.0699 |
| 400 | **5.5016** | 5.6543 | 5.7537 |
| 500 | **5.3381** | 5.4704 | 5.5489 |
| ~1000 | **4.7409** @1080 | 4.8069 @1120 | 4.9547 @990 |

At fixed FLOPs, **parameters beat depth**. And `it4` is additionally **22% faster per step** on
identical arithmetic, because iterations are SEQUENTIAL in a launch-bound block (§7.11: 79.5k
kernels/step, GPU-busy 1.71 s against 6.75 s wall). So iso-FLOP is NOT iso-time, and on wall-clock
`it4` wins by more than the loss table shows. **Anyone repeating this must report both axes.**

**But the gap NARROWS**: it4-minus-it8 runs 0.258 / 0.198 / 0.208 / 0.153 / 0.132 at steps
100-500, and it4-minus-it12 goes 0.472 -> 0.211 over the same window. Extrapolated it would cross
somewhere past a few thousand steps, so "width wins" is supported **at ~1000 steps and not beyond**.
The arms were killed at ~1000 steps.

**Tension with §12.2, which is the evidence that put us on T12.** There the 134M T12
(0.66 G MAC/token, 134M params) beat the 400M 4-iteration arm (1.17 G MAC, 401M params) — deeper,
narrower AND smaller winning. Here the ordering reverses. Both can hold: parameters help early,
compute-efficiency pays late, and §12.2's arms ran 15k+ steps against these 1000. **Consequence: do
not read §12.2 as licence to make the 400M cell narrow-and-deep.** It supports T12 at 134M; at 400M
this ablation favours the existing wide-shallow shape (`it4`, and the sandwich's I_e = 15872). The
T12 advantage may be specific to 134M or to long training, and nothing here settles which.

### 12.15 Sandwich: mu recalibrated (43.24 -> 43.36), and a FLOP-share / robustness dissociation

**`iclr_big_hop_sandwich_sink` had the §12.1 bug**: `sinkhorn_iters: 3` with
`sinkhorn_persist_mu: False` and `sinkhorn_mu_iters: 1` while its block runs **4x**
(`layer_iterations [1, 4, 1]`). Trained mu-tilted, evaluated at mu = 0.

Recalibrated with `calibrate_sinkhorn_mu_20260915.py` (64 batches). Mechanically correct: it probed
**4 mu solves per forward** and wrote `sinkhorn_mu` with shape **(4, 16)** — per-iteration, which is
the thing whose absence made the first attempt at this recover nothing.

| | Avg11 | bits/byte | word ppl |
|---|---:|---:|---:|
| before (mu dropped) | 43.24 | 1.0274 | 45.07 |
| **after (mu restored)** | **43.36** | **1.0247** | 44.62 |

**Use 43.36 in any table**, from `unsharded_mucal`. The gain is small: +0.12pp, -0.0027 bpb.

**And that small gain is the interesting part.** The sandwich has **pure-like FLOP concentration**
— its mixture is **97.6%** of per-token FLOPs (2 GPT wrapper layers are 1.6%, energy attention
0.8%) — yet **hybrid-like robustness** to routing corruption: 0.003 nats here against ~0.003 for
hybrids and **1.6-1.8 nats** for pure 8-12 iteration arms (§12.1), and 0.042 bpb for a 4-iteration
PURE arm (§12.2). Two GPT layers worth 1.6% of FLOPs apparently provide enough of a bypass that
corrupting the energy router barely matters.

**Why this could matter more than the 0.12pp.** Proxy-routed sparsity cost the pure T12 **-0.68pp**
Avg11 but the hybrid only **-0.03pp**. If the sandwich sits at the hybrid end of that
routing-sensitivity spectrum — which this mu result suggests — it would be the **best sparsity
target of the three**: pure-like speedup with hybrid-like tolerance of an imperfect router. Measured
sparse speedup on the sandwich is **1.91x at mbs 1** against an analytical 1.88x, and notably it
does NOT need a large micro-batch, because its experts are I_e = 15872 (3.5x the pure T12's 4480)
and sparsity needs wide experts OR many tokens. NOT placement-controlled — separate hosts.

**Untested and cheap (~30 min):** fit the proxy on `unsharded_mucal` and run the proxy-selection
ablation, exactly as done for pure and hybrid. That would say whether the sandwich escapes the
routing sensitivity that made sparse TRAINING fail on the pure arm (§12.13).

### 12.16 PLAN: LR schedule redesign (WSD), and everything in flight as of 2026-09-16 19:20

Written before a context compaction. Self-contained.

#### The proposed shape, and whether it is standard

User's proposal: **fast drop 1e-3 -> 1e-4, plateau a while, then slow decay to 10x smaller (1e-5)
across 90k steps.**

**This is NOT standard WSD, and the difference matters.** Canonical Warmup-Stable-Decay (MiniCPM,
DeepSeek) plateaus at the **PEAK**: warmup -> constant at high LR for 80-90% of the budget -> short
sharp decay over the last 10-20%. The high-LR plateau is the point: it is where the exploration
happens, and the decay merely cashes it in. The proposal instead drops to a LOW plateau early,
which gives that up. What it resembles is **step / multi-stage decay** (BERT-era, some Llama
variants), which is a real practice but a different one.

If the motivation is "the peak looked too high", the simpler equivalent is just **a lower peak**:
warmup straight to 1e-4 then WSD from there is numerically almost the same as a fast 1e-3 -> 1e-4
drop, without needing a new phase. That is testable with the arms already running.

#### What the codebase supports (checked: `lm_engine/optimization/scheduler.py:38-52`)

Exactly **three phases**: `num_warmup_steps` -> `num_constant_steps` -> `num_decay_steps`, with
cosine or linear decay to `lr * lr_decay_factor`. `num_constant_steps` is 0 in every config we have,
so WSD has never been used here, but it needs **no code change**.

The proposal needs FOUR phases (warmup, fast decay, plateau, slow decay) and would need a new
scheduler class. Do not assume it works from config alone.

#### Two implementable options

**A. True WSD, no code change (RECOMMENDED first).** Peak from the sweep below.
```yaml
lr: <peak>                    # 1e-3 or 5e-4, per the sweep
num_warmup_steps: 1000
num_constant_steps: 74000     # ~82% of budget at PEAK -- the stable phase
num_decay_steps: 15000        # ~17%, cosine
lr_decay_factor: 0.1          # 10x drop
# 1000 + 74000 + 15000 = 90000 = num_training_steps  <- PRE-FLIGHT 1
```
Property that matters for a deadline: you can branch off the stable phase at ANY point, run a short
decay, and get a properly annealed evaluable checkpoint. No need to commit to a token budget.

**B. The proposed 4-phase shape**, if A underperforms: add a `MultiStageScheduler` to
`scheduler.py` taking a list of (steps, target_lr) segments. ~30 lines, and it must be tested
against the existing 3-phase behaviour to avoid perturbing live arms.

#### The measurement that decides the peak and the decay, in flight now

* `sw2k_sparse` (1e-3, 90k-shaped decay -> stays at peak) vs **`sw2k_sparse_c10x`** (1e-3, decay
  COMPLETED in 2000 steps). Same everything else. **The step-2000 gap is the ANNEALING BONUS** at
  0.52B tokens = exactly what WSD's decay phase cashes in. Large bonus => WSD is clearly right.
  Caveat: the decay FRACTION here is 90% against WSD's 10-20%, so the transferable quantity is
  WHERE in the decay the gain lands, not its total size.
* `sw2k_sparse_5e4` (5e-4) vs `sw2k_sparse` (1e-3): the PEAK. At step 400 5e-4 was 0.13 BEHIND
  (5.0490 vs 4.9162), which is expected for a lower peak early and says nothing yet about the
  later "stunting". `sw2k_dense_5e4` was KILLED (relaunch-looping, no progress).
* 12.5 already established: LR **floor is inert** (50x range, inside noise), **peak worth ~0.06
  nats**, and **peak 1e-2 diverges ~1 in 3** — confirmed twice more today, including
  `lrt_sparse_5x` which was AHEAD at step 600 (5.8173, grad_norm 0.24) and then went to
  grad_norm 2.1e7 / loss 13.8 by step 1000. Do not use 1e-2.
* **Decay SPEED had never been varied before `c10x`** — every `pure_lr` arm shared a 5-6k schedule.
  It is also the confound behind "the 30k arms drop faster than the 90k ones": at step 9k they sit
  at lr 1.736e-3 against 1.972e-3.

#### Everything running as of 19:20, and what each settles

| job | id | settles |
|---|---|---|
| `swproxy` | 1713960 | **THE GATE.** Sandwich proxy-selection Avg11 vs dense 43.36. Near -0.03pp (hybrid-like) => launch `sw400_sparse`; near -0.68pp (pure-like) => do not, and 12.15's mu-based prediction does not transfer. |
| `sw2k_sparse_c10x` | 1714132 | the annealing bonus (above) |
| `sw2k_sparse`, `sw2k_dense` | 1713574, 1713573 | 1e-3 pair, 90k-shaped |
| `sw2k_sparse_5e4` | 1713742 | peak 5e-4 |
| `s90k_pure_T12_sparse` | 1710081 | 134M sparse arm; at step ~17.5k, loss 3.83, alignment 0.60, proxy 0.766 — all improving |
| `scale32B_boltz_sinkhorn` | 1701525 | 400M hybrid, 32B tokens, untouched |
| `t90k_pure_T12`, `t90k_hybrid_K32top2` | 1704005, 1704004 | dense 90k references |

**`configs/sw400/sw400_sparse.yml` is committed and pre-flighted, NOT launched** — 400M sandwich
sparse, 15000 steps at 262144 tok/step = 3.93B, matching `iclr_big_hop_sandwich_sink` so the delta
against its recalibrated **43.36** is like-for-like. ~6.7 h on 8 GPUs. `renormalize_topk: true` is
MANDATORY there (the fitted proxy's energy R^2 is -48 to -135000; ranking is fine, magnitudes are
not).

#### Instrumentation warning for whoever picks this up

Four monitor-filter failures today, none of them a real problem with a run: a filter that matched
nothing for 30 min on a healthy job; one that fired on a section HEADER instead of a result; one
that read a single failed `bjobs` query as job death and reported a false alarm; and the
`d400_it4_*` glob that also matched `d400_it4_5x_*` and printed two identical rows.
**Confirm anything a monitor reports by job ID before acting on it.**

#### 12.16a CORRECTION — the sandwich proxy-selection result does NOT exist yet

A monitor reported the sandwich at **+0.00pp** (dense 43.36 / bpb 1.0247, proxy-selected 43.36 /
1.0247) and concluded "hybrid-like, launch `sw400_sparse`". **That was an artifact. The gate is NOT
passed. Do not launch on it.**

Both Avg11 lines cite the SAME directory, `unsharded_mucal/` — the DENSE checkpoint — with
timestamps 19:05 and 19:39. `ablate/C_proxysel/` contains **no `harness_results*.json` at all**. So
the second eval re-evaluated the dense model; nothing measured proxy selection.

Identical-to-4-decimals bits/byte is what gave it away: if routing changed for even a few tokens
bpb would move. **Treat an exactly-zero delta as evidence of a plumbing fault, not of robustness.**

Likely mechanism: `eval_harness.py` does NOT honour `--output_path` (found earlier today — it
writes `harness_results_<timestamp>.json` next to the model), and `C_proxysel/model.safetensors` is
a SYMLINK into `unsharded_sparse_r16m512/`, so results did not land where `compute_avg11.py` looks;
it then fell back to the newest results in the tree.

**To redo it properly:** run the harness against `ablate/C_proxysel` and then VERIFY that
`ablate/C_proxysel/harness_results*.json` exists and that `compute_avg11.py` cites that path,
before reading any delta. The proxy tensors themselves are present and correctly shaped —
`proxy_V (4, 16, 1024, 16)`, `proxy_B (4, 16, 512, 16)`, i.e. 4 per-iteration heads x 16 experts —
and `C_proxysel/config.json` correctly has `proxy_route: true, fused_experts: true, proxy_rank: 16,
proxy_iters: 4, sparse_forward: false`. So only the eval bookkeeping is at fault, not the fit.

One genuine byproduct: the harness is **deterministic to 4 decimal places** across two independent
runs of the same checkpoint (43.36 / 1.0247 both times). That is a useful noise floor — it means a
real Avg11 delta of 0.1pp is signal, not run-to-run variation.

#### 12.16b CORRECTION to 12.16a — the PROXYSEL harness OOMed. It was not the symlink.

12.16a guessed the missing result was `--output_path` bookkeeping plus a symlinked
`model.safetensors`. **That guess was wrong.** The real cause, in
`swproxy_1713960.stderr` (visible only after `tr '\r' '\n'` — the progress bars make it one
giant line that defeats `grep`):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 15.50 GiB.
GPU 0 has a total capacity of 79.18 GiB of which 10.39 GiB is free.
```

`C_proxysel` has `proxy_route: true` with `sparse_forward: false`, so it runs the **dense**
all-K path *plus* the proxy heads. At 400M with `I_e=15872` that does not fit at
`--batch_size 4`, which is what worked for the 134M models. The DENSE arm on the same
checkpoint succeeded at batch 4 in the same job, which is why the loop looked healthy and
`compute_avg11.py` silently fell back to the newest results in the tree.

Resubmitted as **1715589**: `--batch_size 2`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
1 GPU, `-W 08:00`, and the job body now `ls`-es `C_proxysel/harness_results*.json` BOTH before
(confirmed absent → clean measurement) and after, so a fallback cannot be mistaken for a result.

The fit itself was fine: per-iteration heads refined to KL 0.016–0.021, top-2 recall 0.836,
top-3 0.926, top-4 0.955, top-8 0.987.

**Generalisable lesson, and it is the second time this bit us:** `|| echo "... FAILED"` in a
shell loop turns a hard crash into a line of stdout, and the NEXT stage then reads whatever
stale artifact is lying around. `compute_avg11.py` globbing for the newest
`harness_results*.json` in the tree is what converted a crash into a plausible number.
**A stage that produces a number must fail loudly, or verify its own output path exists.**

### 12.17 MEASURED: the annealing bonus saturates after a 1.4x LR reduction

This is the answer to "the sharp-drop arm has the lowest loss, how do we bank on that?" —
and it says: **there is nothing to bank. Do not decay early.**

`sw2k_sparse` (1e-3, 90k-shaped decay, so effectively pinned at peak for all 2000 steps) vs
`sw2k_sparse_c10x` (1e-3, decay COMPLETED in 2000 steps). Identical model, data, warmup 200.
100-step means:

| step | at peak | `c10x` | gap | `c10x` lr | reduction from peak |
|---|---|---|---|---|---|
| 400 | 5.0401 | 5.0353 | +0.005 | 9.73e-4 | 1.03x |
| 600 | 4.7333 | 4.7101 | +0.023 | 8.95e-4 | 1.12x |
| 800 | 4.5515 | 4.4359 | **+0.116** | 7.75e-4 | **1.29x** |
| 900 | 4.4517 | 4.3082 | +0.143 | 7.04e-4 | 1.42x |
| 1200 | 4.1921 | 4.0927 | +0.099 | 4.72e-4 | 2.12x |
| 1600 | 4.0492 | 3.9393 | +0.110 | 2.37e-4 | 4.22x |

**The whole ~0.11 nat gain is realised by a 1.4x LR reduction. The further 3x (7.04e-4 ->
2.37e-4) adds NOTHING** — the gap is flat 900->1600, and the 0.143 at step 900 is probably
a noise excursion on a 100-step mean, so quote **0.11**, not 0.14.

So the fast-drop arm does not have a better trajectory. It has the SAME trajectory plus a
one-time, saturating offset. Consistent with 12.5 ("floor is inert", 50x range inside noise)
and with 12.4 (60k steps pinned at the 2e-4 floor bought 0.015-0.019 nats): once the offset
is collected there is no more to earn, and no step size left to earn it with.

Minor confound, does not affect the contrast: `c10x` has `lr_decay_factor: 0.1` vs `0.02` on
the at-peak arm, but the at-peak arm never leaves ~1e-3 inside 2000 steps.

#### The peak: 12.16's "maybe 2e-3 was too high" is NOT supported

`sw2k_sparse_5e4` is behind `sw2k_sparse` (1e-3) at **every** step and the gap is not closing:
step 800 4.6525 vs 4.5515 (0.101), step 1600 4.1231 vs 4.0492 (0.074). Lower peak = uniformly
worse, same direction as 12.5's "peak is worth ~0.06 nats".

And the observation that originally motivated halving 2e-3 -> 1e-3 — "struggled to drop at
first, then dropped fast and steady" — is the **expected signature of a high peak**, not
evidence against it: measured loss = valley-floor progress + a temperature penalty that grows
with lr, so a high peak looks worse early and better late. **Do not lower the peak on
early-loss appearance.** 2e-3 stands unless something at 400M actually diverges.

#### Recommended WSD, and the distinction that matters

Two DIFFERENT runs; do not conflate their schedules.

1. **`sw400_sparse` (15k, 2e-3, cosine, `num_constant_steps: 0`) — DO NOT TOUCH.** It is the
   sparsity ablation against the dense **43.36**. Changing its schedule destroys the
   like-for-like comparison. It is already pre-flighted.
2. **The 90k headline runs** — this is where WSD goes. Drop-in, NO code change
   (`CosineScheduler` already implements WSD when `num_constant_steps > 0`; verified
   `scheduler.py:90-106`):

```yaml
lr: 2e-3
lr_decay_style: cosine
num_warmup_steps: 1000
num_constant_steps: 80000     # 89% at PEAK -- the stable phase
num_decay_steps: 9000         # 10%; 12.17 says the bonus lands within a 1.4x reduction
lr_decay_factor: 0.1          # -> 2e-4; floor is inert per 12.5
# 1000 + 80000 + 9000 = 90000 = num_training_steps   <- PRE-FLIGHT 1
```
`74000/15000` (17%) is the more conservative literature default if 9000 feels tight.

#### Branch a decay off the stable trunk — this is WSD's real payoff on a deadline

`LoadArgs` supports it: the assert in `arguments.py:151-157` only forbids
`load_lr_scheduler: true` with `load_optimizer: false`, so `load_optimizer: true` +
`load_lr_scheduler: false` is legal. Copy a stable-phase checkpoint, run a short decay-only
schedule, and get a properly annealed evaluable model WHILE THE TRUNK KEEPS RUNNING AT PEAK.

```yaml
# decay-only branch: num_training_steps: 4000, num_warmup_steps: 0,
#                    num_constant_steps: 0, num_decay_steps: 4000, lr: 2e-3
load_args:
  load_path: <stable trunk>
  load_optimizer: true          # keep the Adam moments
  load_lr_scheduler: false      # fresh short decay schedule
  load_starting_iteration: false
```
Consequence: **no need to commit to a token budget up front.** Evaluable checkpoints on
demand. NOT yet run end-to-end — check the `load_dataloader_state` interaction before trusting it.

#### Why the 4-phase "drop to 1e-4, plateau, crawl to 1e-5" shape is the worst option

It forfeits valley progress (12.4: a low plateau is nearly inert) AND pre-spends the annealing
bonus (12.17: it is one-time and terminal). It also needs a new scheduler class. Dropped.
On multi-stage/cyclical cosine: restarts (SGDR) and staged decay are real practices, but for a
single fixed budget they are not what people use — staging is for continued pretraining or a
data-mixture change.

#### 12.16c CORRECTION to 12.16b (and 12.16a). The tooling was fine. The MONITOR fabricated the number.

Third pass on the same incident, and this one is checked against the log line by line rather than
inferred. **Both previous diagnoses were wrong about the mechanism.**

* 12.16a blamed `--output_path` plus the `model.safetensors` symlink. **Wrong.**
* 12.16b blamed `compute_avg11.py` globbing the newest `harness_results*.json` in the tree.
  **Also wrong** — `resolve_results_path` (compute_avg11.py:80-92) globs ONLY under the directory
  it is handed and `sys.exit(1)`s if it finds nothing. It never looks outside. It cannot fall back.

What the log actually contains (`swproxy_1713960.stdout`) — exactly TWO Avg11 lines:

```
101: -------- sandwich / DENSE
128: == Avg11 aggregate from .../unsharded_mucal/harness_results_...19-05-38.json ==
129:   Avg11 = 43.36
146: -------- sandwich / PROXYSEL          <- OOMed, no number
247: -------- sandwich / DENSE             <- the script ran a SECOND time
274: == Avg11 aggregate from .../unsharded_mucal/harness_results_...19-39-22.json ==
275:   Avg11 = 43.36
292: -------- sandwich / PROXYSEL
293: sandwich/PROXYSEL HARNESS FAILED
294: sandwich/PROXYSEL AVG11 FAILED
```

`bench_proxysel_one.sh` was invoked TWICE, and both DENSE evals succeeded with the same 43.36
(which is just the 4-decimal determinism of the harness). PROXYSEL failed both times and said so.
**Every component reported correctly.** `compute_avg11.py` printed
`no harness_results_*.json under .../C_proxysel` and exited 1, exactly as designed.

**The fabrication was in the monitor.** Its filter grepped for `Avg11 *=` across the whole stdout,
collected the two DENSE lines, and presented them as "dense 43.36, proxy-selected 43.36, +0.00pp,
launch `sw400_sparse`". The two numbers were identical because they were THE SAME ARM MEASURED
TWICE — which is also why bits/byte matched to 4 decimals, the thing that (correctly) triggered
the distrust.

**The actual lessons, replacing the two wrong ones:**
1. **A monitor filter must anchor each number to its arm label**, never grep a metric name
   globally. `grep "Avg11 ="` cannot tell you WHICH model produced the line.
2. **A script invoked N times produces N of everything.** Any filter that assumes one number per
   arm per job is wrong the moment a submitter loops.
3. This is the **sixth** monitor-filter failure of the session and by far the most costly — it
   came within one step of launching a 3.9B-token 400M run on a delta that did not exist. The
   standing rule in 12.16 ("confirm anything a monitor reports by job ID before acting on it")
   is what caught it. **Keep it.**
4. What 12.16b got RIGHT and still stands: the PROXYSEL harness genuinely **OOMed** (15.50 GiB
   requested, 10.39 free) because `proxy_route: true` with `sparse_forward: false` runs the dense
   all-K path plus the proxy heads, which does not fit at `batch_size 4` for 400M `I_e=15872`.
   That is the real reason there is no proxy number, and it is fixed by `--batch_size 2`.

**Do NOT "fix" `compute_avg11.py`'s globbing or `eval_harness.py`'s `--output_path`.** Neither is
broken. A TODO item to that effect has been removed.

One real defect does remain in `bench_proxysel_one.sh`: `|| echo "... FAILED"` keeps the pipeline's
exit status at 0, so LSF reports "Successfully completed" for a job that produced nothing. Fixed
below — it now tracks failures and exits nonzero.

#### Retry log for the gate, so nobody repeats these

| job | outcome |
|---|---|
| 1713960 | the original. DENSE fine; PROXYSEL **OOMed** at `batch_size 4`. Monitor misread it as +0.00pp. |
| 1715589 | `batch_size 2` + `expandable_segments`. **Preempted (SSUSP) at 295 s** before reaching the requests; killed deliberately to add a cache. |
| 1716267 | added `--use_cache` AND `--cache_requests true`. Died in 23 s: **`--cache_requests` applies its type conversion before the argparse `choices` check**, so the literal `true` arrives as a dict repr and is rejected. Use `--use_cache` alone. GATE-FAIL fired correctly. |
| 1716546 | `--use_cache` only. RUN, GATE-CLEAN confirmed. ~2-4 h for 82639 requests at batch 2. |

### 12.18 c10x endpoint: the bonus holds at ~0.11 out to a 4.2x reduction. WSD design confirmed.

Completion of the 12.17 measurement. `sw2k_sparse_c10x` (1714132) vs `sw2k_sparse` (1713574),
100-step means, gap = at-peak minus decaying:

| step | at-peak | c10x | gap | c10x lr | reduction from peak |
|---:|---:|---:|---:|---:|---:|
| 700 | 4.6278 | 4.5768 | +0.051 | 8.39e-4 | 1.19x |
| 900 | 4.4517 | 4.3082 | **+0.143** | 7.04e-4 | **1.42x** |
| 1100 | 4.2670 | 4.1518 | +0.115 | 5.50e-4 | 1.82x |
| 1200 | 4.1921 | 4.0923 | +0.100 | 4.72e-4 | 2.12x |
| 1400 | 4.1208 | 4.0104 | +0.110 | 3.25e-4 | 3.08x |
| 1600 | 4.0492 | 3.9393 | +0.110 | 2.37e-4 | **4.21x** |

**The gap is FLAT (+0.100..+0.111) from step 1200 to 1600 while the LR falls a further 2x.** Peak
excursion 0.143 at a 1.42x reduction; durable value **~0.11**. Beyond ~1.4x, further decay buys
NOTHING. This is the strongest single justification for the 90k WSD shape: a 10% decay window
reaches a 10x reduction, which is ~7x more than needed to collect the whole bonus.

**Caveat on the numbers past step 1000.** c10x was preempted THREE times and each time resumed
from `global_step1000`, so the log contains overlapping step ranges (resets 1540->1010,
1020->1010, 1210->1010) and the 100-step buckets above MIX segments from different restarts --
same checkpoint, but different data order after each resume. Re-reading it after the third restart
moved the values by <=0.004 and changed no conclusion, but this is NOT one clean trajectory past
step 1000. If it ever needs to be a figure, plot the FIRST segment only (it reaches step 1540).

**Two operational notes.**
1. `submit_selfresuming.sh` WORKS -- three `runtime_resume_*.yml` in the save_path and every
   restart picked up step 1000 rather than 0. This is the mechanism 12.10 was written about, and
   it is now confirmed under real preemption.
2. But c10x is **LIVELOCKED**: `save_interval` 1000 against a 2000-step run means it must survive
   1000->2000 uninterrupted to checkpoint again, and it keeps dying at 1200-1540. It will likely
   never write `global_step2000`. **Its measurement is complete, so it is pure waste of 4 GPUs.**
   General lesson: `save_interval` must be << the remaining run length on a preemptable queue, or
   an arm can burn GPUs indefinitely while making zero net progress. For a 2000-step probe,
   save_interval should have been ~250.

### 12.19 THE GATE RESOLVED: sandwich proxy selection costs -0.40pp. 12.15's prediction is REFUTED.

Job **1716546**, verified properly this time: `ablate/C_proxysel/harness_results_2026-09-16T21-46-40`
exists (33497 bytes) and `compute_avg11.py` cites that exact path. The evaluated `config.json` has
`proxy_route: true, proxy_rank: 16, proxy_iters: 4, proxy_kind: subspace, sparse_forward: false`.

**All three arms measured IDENTICALLY** (`renormalize_topk: False` in both dense and proxy legs, so
the only difference is who selects the experts):

| model | dense Avg11 | proxy-routed Avg11 | delta |
|---|---:|---:|---:|
| `iclr_hop_K32_top2_sink` (hybrid) | 44.58 | 44.55 | **-0.03pp** |
| `iclr_big_hop_sandwich_sink` | **43.36** | **42.96** | **-0.40pp** |
| `pure_hop_T12_sink` | 40.96 | 40.28 | **-0.68pp** |

Sandwich WikiText 47.49 word-PPL vs the dense 44.62, i.e. ~1.0415 vs 1.0247 bpb (+0.017).

**VERDICT: the gate ("near -0.03pp => launch") is NOT passed.** -0.40pp is 13x the hybrid's cost
and 59% of the pure arm's, i.e. squarely in the middle rather than at the hybrid end.

**12.15's dissociation does not generalise.** It observed that the sandwich has pure-like FLOP
concentration (mixture = 97.6% of per-token FLOPs) but hybrid-like robustness to dropping the
Sinkhorn dual (0.003 nats vs 1.6-1.8 for pure stacks), and predicted the sandwich would therefore
tolerate an imperfect PROXY router like a hybrid. It does not. **Robustness to a mis-tilted router
(a shift in mu, shared across experts) and robustness to mis-SELECTION (picking the wrong experts)
are different properties, and the first does not imply the second.** Two GPT layers can carry a
bypass around a mis-scaled energy landscape while still being unable to substitute for the right
expert. Do not reuse the mu-sensitivity number as a proxy-sensitivity predictor for any arm.

#### What this does and does NOT settle

**Settled:** a POST-HOC proxy retrofit onto a trained dense sandwich costs -0.40pp. So
"fit a proxy to the existing 43.36 checkpoint and ship it sparse" is not viable at the -0.03pp
standard the hybrid set.

**NOT settled: from-scratch sparse TRAINING**, which is what `wsd90k_*_sparse` (1717220/1717221)
actually does. Three reasons the retrofit number does not transfer:
1. In a retrofit the model is frozen and the proxy must chase it. In training the model
   CO-ADAPTS to the proxy, and `sparse_explore` supplies exact energies for experts the proxy
   would not propose.
2. The retrofit ran `renormalize_topk: False`, so the proxy had to estimate the all-K denominator
   from energies whose R^2 is **-48 to -135000** (ranking fine, magnitudes unusable). The training
   arms set it TRUE, which deletes that term. 12.13: renormalisation is "free when TRAINED
   (44.54 vs 44.58)" but "+0.483 bpb retrofitted".
3. Consequently **neither** retrofit variant predicts a from-scratch arm: `C_proxysel` carries the
   denominator error, and `Cp_proxysel_renorm` (built, never evaluated) would carry the +0.483 bpb
   renormalisation-retrofit penalty. **Do not spend a GPU on `Cp_proxysel_renorm` expecting a
   verdict** -- it answers neither question.

**So the real gate for the launched arms is their own first ~1000 steps**, not this eval:
* `expert_cos_abs_mean` -- 12.13's ranking says weight-space repulsion should hold **0.43-0.44**
  from scratch. If it climbs toward 0.7, diversity control has failed and 12.13 is repeating.
* `proxy_topk_agree` -- chance is k/p = 2/4 = 0.50 with exploration on. The 134M s90k arm reached
  **0.7628 by step 27840**.
* loss against the dense `sw2k_dense` trajectory at matched steps (5.88 s/step at 4 GPUs).

#### Standing correction to the eval protocol, now proven to work

The verification added after 12.16a/c did its job: two failed attempts printed GATE-FAIL and
`no harness_results_*.json` rather than a plausible wrong number, and the successful one printed a
path that could be checked. **Keep the pattern: assert the output file exists, and require the
aggregator to name the path it read, before quoting any delta.**

### 12.20 SCALING TO 700M / 1B: sizing formula, and it IS reachable before Sep 24 -- with sparse

#### Exact parameter formula, calibrated against real checkpoints (not estimated)

Read off `sandwich400`'s safetensors: moe **260.0M**, embed **102.8M**, attn 9.4M, other 28.3M,
total **400.6M**. So, with `tie_word_embeddings: true` and V = 100352:

```
params(d, I_tot) = d*I_tot  +  V*d  +  9.4M*(d/1024)^2  +  28.3M*(d/1024)^2
                   ^mixture    ^embed     ^attn            ^GPT wrappers etc
```
`d*I_tot` is EXACT for the mixture (1024 x 253952 = 260.05M vs 260.0M measured) because a hopfield
expert carries ONE weight matrix, not two. Predicts 400.5M against an actual 400.6M.
Per-token mixture MACs = `d * I_tot * iterations`, and the mixture is **97.6%** of per-token FLOPs
on the sandwich, so wall clock scales with `d*I_tot` to within a couple of percent.

#### Sizing and wall clock, K=16, 90k steps at 262144 tok/step = 23.59B tokens, sparse at ~1.85x

| target | d | I_e | I_tot | FLOPs vs 400M | 8 GPU | 16 GPU | 32 GPU |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 400M | 1024 | 15841 | 253456 | 1.00x | **1.7 d** | 0.8 d | 0.4 d |
| 700M | 1024 | 34151 | 546416 | 2.15x | 3.6 d | 1.8 d | 0.9 d |
| **700M** | **1280** | **25031** | **400496** | **1.97x** | 3.3 d | **1.6 d** | 0.8 d |
| 1B | 1280 | 39679 | 634864 | 3.12x | 5.2 d | 2.6 d | 1.3 d |
| **1B** | **1536** | **30966** | **495456** | **2.93x** | 4.9 d | **2.4 d** | 1.2 d |

**Scale `d`, not `I_tot`.** 700M at d=1280 costs 1.97x against 2.15x at d=1024; 1B at d=1536 costs
2.93x against 3.12x at d=1280. A wider model needs LESS `I_tot` for the same parameter count because
the tied embedding (V*d) absorbs more of the budget -- so the wider option is simultaneously cheaper
in wall clock AND avoids the FFN:Attn imbalance that CLAUDE.md records as having sunk the B-series
("do not scale the iso-param design"). Note the embedding is 15% of a 1B model at d=1536; V=100352
is a large vocabulary at these scales.

Chinchilla-optimal (~20 tok/param) is 8B / 14B / 20B, so 23.59B is at or past optimal for all three
-- 90k steps is a defensible budget at every size, not an under-trained one.

#### Schedule: 5.5 usable days (finish ~Sep 22 for a Sep 24 deadline)

Feasible with the big two at 16 GPUs: the 400M pair (8 GPUs each) lands ~Sep 18-19, then 700M
(~1.6 d) and 1B (~2.4 d) in parallel from ~Sep 19 land ~Sep 21 and ~Sep 22. Peak demand 48 GPUs
against `grp_preemptable` at 736/6144 -- quota is not the constraint, PLACEMENT is (a single-GPU
eval sat PEND 30 min tonight).

**DENSE IS NOT REACHABLE AT THESE SIZES.** 1B dense would be ~4.5 d even at 16 GPUs, and 700M dense
~3 d. Sparsity is what puts 700M/1B inside the deadline at all, which is the strategic answer to
"3 days is too long": at 400M sparse turns 3.1-3.3 d into ~1.7 d, and at 1B it turns ~4.5 d into
2.4 d.

**And WSD means the budget need not be chosen now.** Launch with the 90k shape (stable phase at
peak), and branch a 4k decay off the trunk at whatever step the deadline forces. Report the tokens
actually reached. Under cosine an early stop leaves the model mid-decay and badly annealed; under
WSD every checkpoint on the trunk is one short decay away from being publishable. On a hard deadline
this property is worth more than the ~0.11 nat annealing bonus itself.

#### Unverified at these sizes -- check before trusting the table

* Sparse has never been run above 400M. **But the mbs-4 OOM worry is COMPUTED AWAY** -- see 12.21:
  the capacity buffer is `(K, C+1, I_e)` with `C = ceil(cf*T*p/K)`, so
  `sparse@mbsM / dense@mbs1 = cf * p/K * M = 1.25x` INDEPENDENT of I_e. At 1B that is 4.73 GiB
  against the 3.78 GiB dense mbs 1 would use -- and dense mbs 1 is what already runs everywhere.
  The fallback to mbs 1 / ga 8 remains available (the sandwich still measured 1.83-1.91x there).
* The proxy's quality at larger K*I_e is unknown. The ranking task gets harder as experts multiply;
  `proxy_out_dim: 512` and `proxy_rank: 16` were fitted at I_e ~15-18k.
* The ~1.85x sparse factor is measured on the 400M sandwich and is NOT placement-controlled.

### 12.21 Sparse activation footprint is CLOSED FORM: sparse@mbs4 = 1.25x dense@mbs1, at every size

The capacity dispatch allocates `(K, C+1, I_e)` with `C = ceil(cf * T * p / K)`,
`cf = sparse_capacity_factor` (default **1.25**, energy_ff.py:521/1273), `T = mbs*seq`, and during
TRAINING `p = sparse_candidates + sparse_explore = 4`. So the ratio to a dense mbs-1 forward is

```
sparse@mbsM / dense@mbs1  =  cf * (p/K) * M  =  1.25 * 0.25 * 4  =  1.25x
```

**independent of I_e, d, and model size.** Validated against 12.13's measured number: that section
reports the 134M dense mbs-4 intermediate `(4, 4096, 71680)` at 2.19 GiB, and the formula gives
2.19 GiB.

| shape | I_e | dense mbs1 | dense mbs4 | **sparse mbs4** |
|---|---:|---:|---:|---:|
| pure it4 400M | 17920 | 2.19 GiB | 8.75 GiB (OOMs) | **2.73 GiB** |
| sandwich 400M | 15872 | 1.94 GiB | 7.75 GiB | **2.42 GiB** |
| 700M d=1280 | 25031 | 3.06 GiB | 12.22 GiB | **3.82 GiB** |
| 1B d=1536 | 30966 | 3.78 GiB | 15.12 GiB | **4.73 GiB** |

**Consequences.**
1. `micro_batch_size: 4` under sparsity is roughly as safe as `micro_batch_size: 1` dense, which is
   the configuration every dense arm here already runs. So mbs 4 is NOT a memory gamble at any of
   these sizes, and 12.20's "may OOM at 1B" caveat is withdrawn.
2. It also explains why mbs 4 "fits where dense could not": dense mbs 4 is 4x dense mbs 1, sparse
   mbs 4 is 1.25x. The saving is `cf * p/K` = 0.3125x, i.e. **3.2x**, not the naive `k/K` = 8x,
   because the capacity factor and the exploration candidates both cost memory.
3. Lowering `sparse_explore` to 1 would drop p to 3 and the ratio to 0.94x, and raising cf to 1.5
   would push it to 1.5x. Both are levers if a larger model ever does run tight -- but note
   `sparse_explore` is what lets the proxy train at all, so cut cf first.

### 12.22 `sparse_start_step`: the two-phase schedule now runs in ONE job (code, tested)

**Use this instead of the p1dense/phase-2 config pair.** Commits `99d27a3f` (feature) and
`7c4033db` (recompile test).

**Why it was two jobs, and why that was wrong.** The dense->sparse handoff was done as two configs
sharing a `save_path`, because that needed zero code. But every submission costs queue priority on
a busy LSF, the handoff needed a human to watch `proxy_topk_agree` and launch phase 2, and the cost
recurs at every model size (three more pairs at 700M and 1B). On 2026-09-16 the scheduler was
handing out 5-12 minute RUN windows and this became the dominant time sink.

**What made it cheap.** Two pieces of plumbing already existed:
* `forward()` dispatched on `self.sparse_forward` at RUNTIME (`energy_ff.py:1099`), not at
  construction.
* `pretrain.py:400-403` already calls `set_training_step(global_step)` on every model each step
  (added for the cosreg ramp).

So the change is four small edits: the `sparse_start_step` field
(`config/mlp.py`), explicit forwarding (`mlp_blocks/__init__.py` -- pre-flight 7), the gate plus a
`set_training_step` on the MoE (`energy_ff.py`), and propagation to submodules with the module list
cached on first call (`model_wrapper/pretraining.py`).

**Design properties that matter.**
1. Construction-time asserts still key off `sparse_forward`, so a bad sparse config fails at BUILD
   time even though the first N steps run dense.
2. `_sparse_active` is a plain Python bool, so dynamo guards on it: the flip costs exactly ONE
   recompile.
3. It is derived from `global_step`, identical on every rank, so all ranks flip on the same step
   with NO communication. Rank divergence previously wedged a distributed compile, so this is
   load-bearing, not incidental.
4. Defaults are unchanged: `sparse_start_step: 0` is sparse from step 0 (the old behaviour, correct
   when resuming an already-trained proxy), and `sparse_forward: false` is dense forever with
   `set_training_step` a no-op.

**Tested (CPU, `scripts/test_sparse_start_step_20260916.py`, 11 checks):** the knob resolves onto
`.moe` through `get_mlp_block` with a real `EnergyConfig`; the gate is dense at 499 and sparse at
500; dense steps dispatch to `_forward_fused` and sparse steps to `_forward_sparse`; and in float64
the gated dense phase is **bit-identical (0.00e+00)** to a plain dense arm while the gated sparse
phase is bit-identical to a plain sparse arm -- the gate adds no numerical change.

**Tested (CPU, `scripts/test_sparse_start_recompile_20260916.py`):** under `torch.compile` the
module takes `_forward_fused` before the flip and `_forward_sparse` after, outputs differing. This
rules out the dangerous failure -- dynamo baking `_sparse_active=False` into the graph so the flag
flips, nothing errors, and the run silently stays DENSE for 90k steps. That is the shape of all
three silent bugs in 12.13, so it was the one worth testing.

**Still unverified:** whether the recompile HANGS under FSDP + activation checkpointing at
multi-GPU. That failure is LOUD (0 steps, as in the `fused_experts` 2-node wedge and the
data-dependent-shape wedge), so the first real run detects it within a minute. No dedicated GPU
test needed.

**The one thing a single job gives up: `micro_batch_size`.** It is a dataloader parameter, not a
model attribute, so it cannot be flipped mid-run. Dense needs mbs 1 (dense mbs 4 is 8.75 GiB and
OOMs), so a single job runs mbs 1 throughout and takes 4096 tok/call instead of 16384. Acceptable
at 400M+ because both shapes have wide experts (I_e 15872-17920) and 12.15 measured the sandwich at
1.83-1.91x even at mbs 1. **Do NOT copy this to a small-I_e hybrid shape, where 4096 tok/call
loses outright (0.41-0.49x).** If peak throughput matters more than job count, keep the two-config
pair and use mbs 4 in phase 2.

Configs: `configs/wsd90k/wsd90k_{pure_it4,sandwich}_1job.yml`, `sparse_start_step: 500`.

### 12.23 VALIDATED ON HARDWARE: the in-job dense->sparse switch, at 1.996x

`t32B_sandwich_sparse` (job 1718594, 8 GPUs, single node, `sparse_start_step: 300`):

| step | s/step | phase | proxy_topk_agree | expert_cos_abs_mean |
|---:|---:|---|---:|---:|
| 100 | 6.3308 | dense | 0.4702 | 0.5244 |
| 200 | 6.2853 | dense | 0.7275 | 0.7100 |
| 290 | 6.3032 | dense | **0.7858** | 0.6855 |
| 300 | 6.0345 | switching | 0.7718 | 0.6830 |
| **310** | **3.1588** | **sparse** | 0.6054 | 0.6855 |

**6.3032 -> 3.1588 s/step = 1.996x**, and the torch.compile re-trace at the switch did NOT wedge.
That was the one part of `sparse_start_step` the CPU tests could not reach (12.22), so the feature is
now validated end-to-end -- at 8 GPUs / ONE node. **Multi-node is still unverified** (the 2-node
probe 1718609 could not be scheduled: "requirements for reserving resource (ngpus_physical) not
satisfied: 229 hosts" -- two hosts with 8 free GPUs each is far harder to place than one).

The measured 1.996x lands on 12.13's placement-controlled **2.02x** rather than on the 1.85x that
12.20 projected from the sandwich's mbs-1 figure, so the earlier estimate was conservative. At
3.16 s/step the arm finishes 61035 steps in **~54 h = 2.2 days**.

**The dense phase did exactly what it exists for:** proxy agreement 0.125 (chance) -> **0.7858** by
step 290, clearing the 0.75 threshold, and it got there at step ~200 (0.7275) which is why
`sparse_start_step` was cut from 500 to 300.

**The hand-off dip is real but mild.** Agreement fell 0.7858 -> 0.6054 across the switch. 12.13 saw
0.703 -> 0.46 in the two-job setting, so this is roughly half the damage. Note the FLOOR CHANGES at
the switch and this trips people up: dense agreement is against top-2-of-16 (chance **0.125**),
sparse agreement is measured within the p=4 candidate set (chance **0.50**). 0.6054 against 0.50 is
a much weaker margin than 0.7858 against 0.125, so do not read the dip as "still fine".

**Watch `expert_cos_abs_mean`.** It sits at 0.68, against the 0.43-0.44 that 12.13 says weight-space
repulsion reaches from scratch. Too early to call at step 310 (experts have barely differentiated),
but if it is still ~0.68 at step 3000-5000 then the weight-space choice has NOT delivered and 12.13
is repeating. That is the single most likely way these arms disappoint.

### 12.24 scale32B_boltz_sinkhorn benchmarked: Avg11 49.91 at 30.4B tokens

Job 1718621. Unsharded step **58000** (the arm was stopped at 58130 of 61035 = 95.0% to free its 16
GPUs), then mu-recalibrated, then evaluated. Verified: the number is read from
`unsharded_mucal/harness_results_2026-09-17T00-28-39.json` and `compute_avg11.py` cites that path.

| | value |
|---|---|
| **Avg11** | **49.91** |
| WikiText | 25.23 word-PPL |
| tokens | 58000 x 524288 = **30.41B** |
| shape | 400M HYBRID, d=1536, K=32, top-2, `layer_iterations [1,1,1,1,1,1,6]` |

**This is the best Avg11 in the ICLR record**, against 44.58 for the 134M hybrid and 43.36 for the
400M sandwich -- but it is a DIFFERENT TIER (3x the parameters and ~8x the tokens of the 3.9B grid),
so it is not a like-for-like win over either. Quote it with its token count, always.

**Two caveats that must travel with the number.**
1. It stopped at 95% of its schedule, so the cosine decay never reached its 2e-4 floor. By 12.4
   floor steps buy 0.015-0.019 nats per 45k steps, so the finished run would be marginally BETTER:
   49.91 is a slight underestimate, not an overstatement.
2. mu recalibration was REQUIRED (the config has `sinkhorn_iters: 3` but no `sinkhorn_persist_mu`,
   with a 6x block -- the 12.1 bug). The raw `unsharded` eval is queued in the same job and will
   give the delta at this scale; the sandwich's was +0.12pp and this block runs 6x, so it may be
   larger.

#### 12.23a The switch has a TRANSIENT. Do not judge the repulsion setting inside ~200 steps of it.

I read the first 120 steps after the sandwich's switch as "weight-space repulsion is failing" and
recommended reverting both arms to the subsampled output-space setting. **That was wrong, and the
data reversed within another 90 steps.** Recorded because the shape is reproducible and the wrong
call is tempting.

`expert_cos_abs_mean` (lower = more diverse) on the sandwich, switch at step 300:

```
310 0.6855 | 340 0.7644 | 370 0.8037 | 400 0.8147 | 410 0.8159  <- PEAK
420 0.8125 | 440 0.8078 | 460 0.8044 | 480 0.8012 | 490 0.7969 | 500 0.7764 | 510 0.7743
```
Ten consecutive falls after the peak, i.e. sustained, not noise. `proxy_topk_agree` over the same
window rose 0.4617 -> 0.4930.

**Pure reproduces the identical shape ~100 steps behind**, and its proxy never dropped below the
0.50 candidate-set floor at all: 0.6850 (dense) -> 0.5161 at 350 -> 0.5386 at 390 -> 0.5287 at 400,
with cos still climbing (0.7559 at 400) i.e. still pre-peak.

**Mechanism:** handing selection to the proxy perturbs which experts see which tokens, diversity
degrades while the routing re-equilibrates, and the regulariser then reasserts. It is a transient of
the SWITCH, not a verdict on the regulariser.

**Rule: the earliest honest read on `expert_cos_abs_mean` is ~200 steps after the switch, and the
real comparison against 12.13's 0.43-0.44 target belongs at step 3000-5000.** The project record
already warned about exactly this class of error -- "a step-30 reading of the tau sweep gave the
wrong answer and had to be retracted" -- and this is the same mistake with a different metric.

Matched-age comparison against `s90k_pure_T12_sparse` (subsampled output-space, the setting 12.13
ranked WORST) is still the right yardstick, but quote it at a fair age: at step ~500 the sandwich is
at cos 0.7764 against that arm's 0.7412, a gap of 0.035 and closing, not the 0.073 it appeared to be
at step 400. That arm went on to reach cos 0.5628 / proxy 0.7632 by step 27800, so recovery over
thousands of steps is the documented expectation for either setting.

#### 12.24a The mu delta at 400M/30.4B is +0.05pp -- and it scales with GPT BYPASS, not iteration count

Both legs of job 1718621, each with its results path verified:

| | Avg11 | bits/byte | word-PPL | path |
|---|---:|---:|---:|---|
| **`unsharded_mucal`** | **49.91** | 0.8709 | 25.23 | `harness_results_2026-09-17T00-28-39` |
| `unsharded` (mu dropped) | 49.86 | 0.8720 | 25.34 | `harness_results_2026-09-17T00-57-39` |
| delta | **+0.05pp** | +0.0011 | -0.11 | |

**12.15 and I both guessed wrong.** The prediction was that this delta might EXCEED the sandwich's
+0.12pp because the energy block runs 6x here against the sandwich's 4x, so more iterations should
compound the tilt. It is smaller. Iteration count is the wrong variable:

| arm | GPT layers (bypass) | energy-block iters | mu cost |
|---|---:|---:|---|
| `scale32B_boltz_sinkhorn` (hybrid) | 6 | 6 | **+0.05pp** |
| `iclr_big_hop_sandwich_sink` | 2 | 4 | +0.12pp |
| pure 8-12 iteration arms | **0** | 8-12 | **1.6-1.8 nats** |

Monotone in the number of GPT layers, NOT in iterations. Same ordering as 12.15's FLOP-share /
robustness dissociation: a non-energy path around the mixture makes the model tolerant to a
corrupted router, and the pure stacks have none. **So predict mu sensitivity (and, by 12.19,
proxy-selection sensitivity) from the bypass, not from depth.**

Practical consequence: for hybrids the mu recalibration is nearly optional (+0.05pp is real -- the
harness is deterministic to 4 dp -- but small). For pure arms it is worth 1.6-1.8 nats and is
mandatory. The sandwich sits in between, which is consistent with it also sitting between hybrid
(-0.03pp) and pure (-0.68pp) on proxy-selection cost.

### 12.25 NEGATIVE: the 2-node wedge is NOT fixed. `repulsion_tensor_idx` was not the cause.

Job **1719429**, 2 nodes x 4 GPUs, `fused_experts: true` + `sparse_forward: true` +
`repulsion_tensor_idx: true` + `repulsion_space: weight`. Killed at **790 s**. All three of
ACCEL_FINDINGS' documented wedge signatures reproduced:

| signature | 2026-09-15 wedge | 1719429 |
|---|---|---|
| first log line -> first step | never (17+ min, killed) | **never, 790 s, 0 step lines** |
| inductor cache writes | **0** | **0 in the last 3 min** (76 total, all stale) |
| log frozen on | one dynamo warning | **a dynamo `functools.lru_cache` warning** |

**So the hypothesis in 12.22/`d0f2fef1` was WRONG.** `repulsion_tensor_idx` is labelled in
`config/mlp.py` as "the leading suspect for the fused_experts multi-node hang" and it independently
unstuck a 2-node job that had stalled 13 min -- but with it enabled AND with `repulsion_space:
weight` (which avoids the output-space einsum on data-dependent indices altogether), the wedge
persists. Suspect (2) is eliminated.

**Scope caveat:** the 2026-09-15 wedge was `fused_experts` ALONE; this probe runs `fused_experts` +
`sparse_forward`, so it does not isolate which component hangs. Operationally that does not matter --
our configs require both (`sparse_forward` asserts `fused_experts`) -- but a diagnosis must not
assume the two wedges share a cause.

**Remaining suspects, from ACCEL_FINDINGS:**
1. the single large fused GEMM shape;
3. an FSDP-gather interaction with the `weight_fn` closure that reads `holder.W.weight` inside the
   compiled region.
Bisect in flight: job **1719489**, identical but `repulsion_coef: 0.0`, which removes the repulsion
path entirely. RUNS => repulsion is implicated even with tensor indices. WEDGES => it is (1) or (3).

#### The planning consequence, which is the expensive part

**One node / 8 GPUs is the only validated shape.** And 2x8 could not even be SCHEDULED (job 1718609
sat PEND 1.5 h: "requirements for reserving resource (ngpus_physical) not satisfied: 236 hosts"),
whereas 2x4 placed instantly -- so 16-GPU runs are blocked twice over, by the wedge and by
placement. Cost the ladder at 8 GPUs:

| arm | 8 GPUs, sparse |
|---|---|
| 400M (running now) | ~2.2 d |
| 700M (d=1280, I_e 25031) | ~3.3 d |
| 1B (d=1536, I_e 30966) | ~4.9 d |

With ~7 days to Sep 24, **1B does not fit** alongside the two 400M arms. Options, in order of
preference: (a) diagnose the wedge -- suspects (1) and (3) are one config change apart and each
probe is ~10 min on 8 GPUs; (b) make 700M the top of the ladder; (c) launch 1B anyway and use WSD's
branch-decay (12.20) to harvest whatever step it reaches by the deadline, reporting the tokens
actually trained. (c) is the only option that yields a 1B number at all, and WSD is what makes it
publishable rather than half-annealed.

### 12.26 DIAGNOSED: the 2-node wedge is REPULSION, not the fused GEMM

Bisect job **1719489**: identical to the wedging probe except `repulsion_coef: 0.0`. It **RUNS** at
2 nodes x 4 GPUs -- step 10 at 4.279 s/step, step 20 at 2.970 s/step -- and crossed the
`sparse_start_step` recompile too.

**So `fused_experts` + `sparse_forward` are FINE multi-node.** ACCEL_FINDINGS' suspect (1), "the
single large fused GEMM shape", is **exonerated**: that GEMM is present and running in the bisect.
The hang is in the repulsion path, and `repulsion_tensor_idx: true` does not prevent it (12.25).

This matters beyond the wedge: it means the 1.61x fusion and the ~2x sparsity are NOT
single-node-only capabilities. Only repulsion is.

#### The suspected mechanism, and it merges suspects (2) and (3)

`build_boltzmann_moe` (energy_ff.py:2406-2411) builds the fused spec with

```python
# `weight_fn` is a closure so FSDP re-gathers are picked up
# (same reason the expert W_slice closures exist).
"weight_fn": (lambda: holder.W.weight),
```

and the repulsion branch (energy_ff.py:1177) is

```python
if self.repulsion_space == "weight":
    self._add_repulsion_loss_weight()      # goes through weight_fn -> the SHARDED parameter
else:
    self._add_repulsion_loss(expert_grads) # activations only
```

**Weight-space repulsion reads an FSDP-sharded parameter inside the compiled region**, which needs an
all-gather; at 2 nodes that is an INTER-NODE collective issued from inside a dynamo graph. That is
the deadlock shape the project memory already warned about -- *"the inductor `spmd_check` all_gather
hang that data-dependent MoE routing triggers"*. So ACCEL_FINDINGS' suspects (2) and (3) are probably
ONE mechanism, reachable specifically when `repulsion_space: weight`.

**Decisive test in flight: job 1719508**, identical but `repulsion_space: output` +
`repulsion_subsample: 64`, which routes to `_add_repulsion_loss(expert_grads)` and never touches
`holder.W.weight`.
* RUNS => the culprit is the sharded-weight read, and multi-node is available with output-space
  repulsion.
* WEDGES => repulsion hangs in either space, so the sharded read is not it and the remaining
  candidate is the repulsion graph itself.

#### If output-space wins, this changes the 700M/1B plan

The running 32B arms use `repulsion_space: weight` (12.13 ranks it better for expert diversity from
scratch, and the sandwich is currently vindicating that -- cos 0.6490 at step 1000 against the
subsampled arm's 0.7350). **Those two arms therefore cannot go multi-node, ever, without changing
the regulariser mid-run.** But 700M/1B are not launched yet, so they could take output-space
repulsion and 16 GPUs, halving their wall clock -- at the cost of the weaker diversity control, and
of not being regulariser-matched to the 400M arms.

That is a real trade to decide deliberately, not by default:
* weight-space + 8 GPUs: better diversity, matched to the 400M arms, 1B ~4.9 d (does not fit Sep 24).
* output-space + 16 GPUs: weaker diversity, unmatched, 1B ~2.4 d (fits) -- IF 2x8 can be scheduled,
  which tonight it could not (1.5 h PEND; 2x4 placed instantly).

### 12.27 THE 2-NODE WEDGE IS FULLY LOCALISED: it is WEIGHT-SPACE repulsion, and nothing else

Three probes at 2 nodes x 4 GPUs, identical but for the repulsion setting. This closes a bug that had
been open and undiagnosed since 2026-09-15.

| probe | repulsion | result |
|---|---|---|
| 1719429 | **weight**, coef 2.0 | **WEDGES** -- 790 s, 0 step lines, 0 inductor cache writes, frozen on a dynamo warning |
| 1719489 | **off**, coef 0.0 | RUNS -- 120 steps, 1.38 s/step, 0 NCCL errors, crossed the sparse switch |
| 1719544 | **output** + subsample 64 | RUNS -- step 10/20/30 at 3.77/3.10/3.03 s/step, 0 NCCL errors |

**What is therefore NOT the problem, contrary to ACCEL_FINDINGS' suspects:**
* the single large fused GEMM shape (suspect 1) -- present and running in both healthy probes;
* `fused_experts` as such -- the doc's "validated single-node only" is now **too pessimistic**;
* `sparse_forward`, the capacity dispatch, and the `sparse_start_step` recompile -- all fine at 2 nodes;
* the Python-`random` pair indices (suspect 2) -- 1719429 had `repulsion_tensor_idx: true`.

**What it is.** `energy_ff.py:1177` branches to `_add_repulsion_loss_weight()` for
`repulsion_space: weight`, and that path reaches the expert weights through the closure built at
`energy_ff.py:2411`, `"weight_fn": (lambda: holder.W.weight)` -- whose own comment says it is a
closure "so FSDP re-gathers are picked up". So weight-space repulsion **reads an FSDP-sharded
parameter from inside the compiled region**, which requires an all-gather; across nodes that is an
inter-node collective issued from within a dynamo graph, and it deadlocks. Output-space repulsion
instead consumes `expert_grads` (activations, already local) and does not touch the closure. That
also explains the asymmetry the project memory recorded -- *"the inductor `spmd_check` all_gather
hang that data-dependent MoE routing triggers"* -- as the same class of fault.

#### One methodological trap, and it nearly produced a wrong conclusion

The FIRST output-space attempt (1719508) failed with `ncclRemoteError` on the **first** collective
(SeqNum=1 ALLREDUCE, `last completed work: -1` on every rank) on hosts `p1-r15-n4` / `p2-r22-n1`,
while the repulsion-off probe had run clean on `p1-r18-n3` / `p2-r16-n1`. Read carelessly that is
"output-space also fails". **It is a DIFFERENT failure**: the wedge is a silent hang with zero NCCL
errors and zero inductor writes; this was NCCL erroring loudly before training began -- the
node-fault signature ACCEL_FINDINGS documented for `p4-r10-n4`. Retrying on other hosts
(`p4-r25-n4` / `p3-r03-n1`) ran clean. **Always separate "hung" from "NCCL-errored": they have
different causes and only the first is ours.** `submit_train.sh` now carries a `SUSPECT_HOSTS` list
(distinct from `BAD_HOSTS`) holding those two, with the promotion bar written in -- one failure
isolates a variable, two earns a blacklist.

#### The fix, and the choice it forces

**Real fix (not yet implemented):** hoist the sharded read out of the graph -- materialise
`holder.W.weight` before entering the compiled region and pass it in as a plain tensor, or compute
the weight-space repulsion under `torch.compiler.disable()`. Either removes the in-graph all-gather.
That is a contained change but it touches the hot path of two live 61035-step arms, so it must be
validated on a 2-node probe BEFORE it goes anywhere near them.

**Until then, the operational rule:** weight-space repulsion => single node, 8 GPUs max.
Output-space repulsion => multi-node available.

The two running 400M arms (1718594, 1718598) use weight-space, so they are 8-GPU-bound for their
whole life. 700M/1B are unlaunched and may pick either:

| option | expert diversity | 1B at 90k-equivalent | matched to the 400M arms? |
|---|---|---|---|
| weight-space, 8 GPUs | better (sandwich cos 0.6490 @1000 vs the subsampled arm's 0.7350) | ~4.9 d -- **misses Sep 24** | yes |
| output-space, 16 GPUs | weaker | ~2.4 d -- fits | **no** |
| fix the closure, then weight-space at 16 GPUs | better | ~2.4 d | yes |

The third row is the only one that is both fast and matched, which is the argument for spending an
hour on the fix rather than accepting the trade.

#### 12.27a OPERATIONAL: watch step time on a multi-day run and REQUEUE on a sustained 2x regression

`t32B_pure_it4_sparse` ran at ~3.2 s/step to step ~1100, then settled at **8.7-9.2 s/step** and
stayed there. Not noise, not the model: its exec host `p4-r05-n1` had **17 job slots in use** while
the sandwich's `p5-r20-n1` had **1**, and the two arms are the same size, same GPU count, same code.

Killed and resubmitted (job 1718598 -> **1720770**). It drew `p2-r20-n4` at **2 slots** and returned
to **~3 s/step** immediately, resuming from its step-1200 checkpoint.

**Why this is the only lever.** The watchdog comments already record that every alternative fails
here: `-x` (whole node) is unobtainable -- "requirement for exclusive execution not satisfied: 663
hosts" -- and left both 32B arms PEND; reserving 16 of a host's 96 slots does not schedule
("ngpus_physical not satisfied"); and `select[ut<0.5]` filters only at DISPATCH, so a host chosen at
1 slot drifts to 9. The root cause is that `-n <nnodes>` requests ONE slot per host for EIGHT GPUs,
so the job is cgroup-limited to about one core and the dataloader starves as soon as neighbours
arrive.

**The rule.** A 3x step-time regression is INVISIBLE in the loss curve -- the curve just advances
more slowly in wall-clock -- and it silently converts a 2.2-day run into 6.4 days, which was the
difference between making and missing the Sep 24 deadline. With `save_interval: 200` a requeue costs
~10 minutes. So: log step time, compare against the arm's own early median, and requeue on a
sustained 2x+ regression. Check the host's slot count (`bhosts -l <host>`, the `njobs`/`run`
columns) to confirm it is contention before blaming the code.

**Gap found while diagnosing this:** `sparse_overflow` is NOT appearing in either arm's logged
metrics, so capacity overflow could not be ruled out as a contributor by measurement -- only by the
17-vs-1 slot difference. Worth wiring into the metrics if this recurs.

#### 12.27b RETRACTION x2: `expert_cos_abs_mean` OSCILLATES. And the proxy, not diversity, is the worry.

I have now misread this metric twice in opposite directions, both times by treating one phase of an
oscillation as a trend. Recording the actual shape so nobody repeats it.

`t32B_sandwich_sparse`, every 100 steps (switch at 300):

```
 400 cos 0.8147 <- local MAX     1300 cos 0.6207 <- local MIN     2000 cos 0.7178 <- rising again
 500 0.7764   700 0.7148   1000 0.6490   1200 0.6369   1400 0.6426   1700 0.6743   1900 0.7109
```

* **First error (12.23a):** at step 400 I read the rise 0.68 -> 0.815 as "weight-space repulsion is
  failing" and recommended reverting both arms. Retracted when it fell.
* **Second error:** at step 1000 (cos 0.6490) I said weight-space was "clearly ahead on diversity"
  against the 134M reference's 0.7350. **Also wrong.** That was a downswing, and the reference arm
  oscillates in the SAME band at comparable age -- 0.6367 / 0.6615 / 0.6934 / 0.6586 across steps
  2700-3000. Neither setting is converging toward 12.13's 0.43-0.44 target.

**Rule: `expert_cos_abs_mean` swings ~0.06-0.10 on a few-hundred-step timescale. Quote a WINDOWED
median over >=500 steps, never a single reading, and never compare two arms at single points.**

#### What the data does support, and it points at the PROXY

Both arms use p=4, so the sparse-path chance floor is k/p = 0.50 for both, making this comparable:

| | proxy_topk_agree |
|---|---|
| `t32B_*` arms (weight-space repulsion) | **0.47-0.54** -- sitting ON the chance floor |
| `s90k_pure_T12_sparse` (output-space + subsample) | **0.66-0.68** -- clearly above |

**Our proxy is no better than random at proposing candidates; the reference arm's is.** Loss is
unaffected so far (3.9329 at step 2000 = 1.05B tokens, dropping normally) -- expected, because
`_forward_sparse` computes EXACT energies for the p=4 candidates and re-ranks, and `sparse_explore: 2`
supplies half the candidates from a deterministic rotation, so a useless proxy costs candidate
QUALITY rather than selection correctness.

**Hypothesis, untested: the two choices are COUPLED.** Weight-space repulsion pushes expert WEIGHTS
apart, and the subspace proxy is built from weight subspaces (`B_k = W_k V_k`). So the regulariser
may be actively degrading the structure the proxy learns to rank -- which would mean
`repulsion_space` and `proxy_kind` cannot be chosen independently, and that 12.13's ranking of
repulsion substitutes (measured WITHOUT a trained proxy in the loop) does not transfer to a
proxy-routed arm. Cheap test: one short arm, weight-space repulsion + `proxy_kind: quad` or a
higher `proxy_rank`, and see whether agreement lifts off the floor.

**Watch `proxy_topk_agree`, not `cos`, from here.** If it is still ~0.50 at step 5000 while the
reference sits at 0.68, the proxy is contributing nothing and the sparse arms are effectively running
`sparse_explore`'s deterministic rotation as their router -- which works, but is not the method the
paper describes.

#### 12.27c MULTI-NODE IS ~50% FLAKY AT STARTUP, independently of the wedge

Four 2-node x 4-GPU attempts tonight, four distinct host pairs:

| job | hosts | outcome |
|---|---|---|
| 1719489 | p1-r18-n3, p2-r16-n1 | clean, 120 steps |
| 1719508 | p1-r15-n4, p2-r22-n1 | **`ncclRemoteError` at SeqNum=1** |
| 1719544 | p4-r25-n4, p3-r03-n1 | clean, 120 steps |
| 1720214 | p2-r15-n4, p3-r10-n3 | **`ncclRemoteError` at SeqNum=1**, 8 errors |

**Two of four died on the FIRST collective**, each on a different host pair, with
`last completed work: -1` on every rank. Initially I treated this as bad hosts and added
`p1-r15-n4` / `p2-r22-n1` to a `SUSPECT_HOSTS` list. **That inference now looks wrong**: if a handful
of hosts among ~500 were at fault, drawing them twice in four attempts would be very unlikely. This
reads as general 2-node initialisation flakiness on this cluster, so the suspect list is not the
right tool and was not extended.

**Do not confuse the two multi-node failures. They are different and only one is ours:**

| | wedge (ours) | startup fault (cluster) |
|---|---|---|
| NCCL errors | **0** | **8**, `ncclRemoteError` |
| inductor cache writes | 0 | n/a, dies first |
| symptom | silent hang, log frozen on a dynamo warning | loud failure at SeqNum=1, `last completed work: -1` |
| when | after compile, never reaches step 1 | before training begins |

**Consequence for planning, and it is the important part.** Even with the wedge fixed, a 16-GPU run
has roughly a coin-flip chance of failing at startup and would need babysitting to restart. Combined
with 12.25's finding that 2x8 could not be SCHEDULED at all (1.5 h PEND against 2x4 placing
instantly), **multi-node is not a viable path for the Sep 24 deadline regardless of the wedge.**
Cost the 700M/1B ladder at 8 GPUs / 1 node: 3.3 d and 4.9 d.

The wedge verdict is still worth having for the record and for post-deadline work -- job **1722297**
is the third attempt at it -- but it is no longer on the critical path.
