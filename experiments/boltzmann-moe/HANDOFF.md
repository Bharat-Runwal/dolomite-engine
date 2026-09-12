# Boltzmann-MoE — HANDOFF

**Entry point for a fresh session.** Read this first, then jump into the specific
doc you need. This file carries (a) orientation, (b) the things that exist
nowhere else, and (c) the 2026-09-12 session findings, which materially change
the project's conclusions.

Last updated: 2026-09-12.

---

## 0. Doc map — read in this order

| Doc | What it is | When to read |
|---|---|---|
| **HANDOFF.md** (this) | Orientation + 2026-09-12 findings | always, first |
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
| `~/Code/energy/energy-GPT-neurips2026/` | `https://git@git.overleaf.com/69eb9b62c6f271a5b29323bb` | **`nima/sec/appendices/boltz_moe.tex`** (37 KB) — the appendix where `PROGRESS.md`/`RESULTS.md` numbers get copied. Included from `nima/paper_v2.tex:108` |
| `~/Code/overleaf/energy-GPT-reformulation-2026/` | `https://git@git.overleaf.com/69ebe3ed5c91a9639cc3576b` | `moe_bs.tex` at top level + `slides/`, talk frames with the MoE results table and scatter plot |

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
