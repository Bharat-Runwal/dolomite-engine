# ICLR-26 paper runs — the 22 configs the paper actually reports

Snapshot taken 2026-09-20. These are COPIES of the converged configs; the originals remain at
`configs/cmix/` and `configs/iclr_26/ablations/` and are what the live jobs were launched from.
Copies exist so the paper's arm set is one directory, reviewable without tracing which of ~200
configs mattered.

**Every arm: 32.0B tokens on the `configs/cmix/` datamix (70% web / 30% math), and
`tokens/step = GPUS x micro_batch_size x gradient_accumulation_steps x sequence_length` with
GPUS NOT IN THE CONFIG.** 134M = 262,144 tok/step x 122,070 steps; 400M/1B = 524,288 x 61,035.
Read each file's header for its intended GPU count.

---

## RESULTS AS OF 2026-09-20

**Avg11 and MMLU are accuracy in percentage points (HIGHER better). ppl is WikiText word-level
perplexity (LOWER better). GSM8K is flexible-extract exact-match %. ACTIVE / FLOPwt are millions of
active parameters / parameter-applications per token. All rows 32.0B tokens; rows are comparable
only WITHIN a scale.**

### 134M tier — complete

| config | Avg11 | ppl | MMLU | GSM8K | ACTIVE | FLOPwt | role |
|---|---|---|---|---|---|---|---|
| `abl_E_134M_6G1x6E_baseEGPT` | **45.87** | 41.00 | 24.85 | 2.27 | 123.2M | 141.7M | energy block, NO MoE |
| `abl_D_134M_6G_dense_isototal` | 45.55 | **39.94** | 24.45 | 1.74 | 134.3M | 134.3M | GPT-only, iso-TOTAL |
| `abl_B_134M_6G1x6S` | 45.43 | 39.96 | 24.43 | **2.43** | 123.5M | 143.2M | Switch, FLOP-matched |
| `abl_F_134M_6G_dense_isoactive` | 45.01 | 41.87 | **25.77** | 2.20 | 123.3M | 123.3M | GPT-only, iso-ACTIVE |
| `cmix_134M_hyb_w1w2_sparse_surr_32B` | 45.01 | 40.70 | 24.90 | 1.82 | 123.1M | 141.0M | w1w2, sparse(surrogate) |
| `cmix_134M_gptswitch_32B` | 44.87 | 40.19 | 25.28 | 1.36 | 123.5M | 123.5M | Switch, unmatched |
| `cmix_134M_hybrid_32B_sparse` | 44.82 | 41.06 | 25.57 | 1.59 | 123.2M | 141.7M | **energy hybrid (headline)** |
| `abl_I_134M_..._projUncon` | 44.32 | 41.09 | 23.73 | 2.05 | 123.1M | 141.0M | proj A/B: unconstrained |
| `cmix_134M_hyb_w1w2_surrMLP_32B` | 44.16 | 40.35 | 25.42 | 1.59 | 123.1M | 141.0M | w1w2 dense, head=router |
| `cmix_134M_sandwich_32B_sparse` | 43.45 | 40.98 | 25.20 | 1.67 | 123.2M | 141.7M | energy block moved |
| `cmix_134M_pure_32B_sparse` | 42.00 | 66.08 | 24.43 | 1.74 | 86.1M | 185.1M | pure recurrent `1x12E` |

### 400M / 1B tier — in flight except where noted

| config | status | 8B `lm_loss` | 32B Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|---|---|
| `abl_B_400M_6G1x6S` | **COMPLETE** | **2.6571** | **48.24** | **28.27** | **26.74** | **3.26** |
| `abl_H_400M_6G6S_deep` | 25%, stopped | 2.6583 | — | — | — | — |
| `abl_H_400M_6G6G_deep_isoactive` | 27%, stopped | 2.6812 | — | — | — | — |
| `abl_H_400M_6G6E_deep` | 33%, stopped | 2.7280 | — | — | — | — |
| `cmix_400M_hybrid_sparse` | ~85% RUNNING | 2.7319 | — | — | — | — |
| `cmix_400M_baseline_switch` | complete | — | 47.44 | 28.82 | 26.55 | 2.73 |
| `cmix1B_12L_gptDense_32B` | complete | — | 47.79 | 29.46 | 26.50 | 1.74 |
| `abl_G_400M_6G1x6S1x6S` | 31% RUNNING | — | — | — | — | — |
| `abl_G_400M_6G1x6E1x6E(_4gpu)` | 12% RUNNING, cannot finish by Sep 24 (13.8 s/step) | — | — | — | — | — |
| `cmix_400M_sandwich_sparse` | 28%, retired | — | — | — | — | — |

**THE FLOP-MATCHED SWITCH BASELINE IS THE STRONGEST ARM IN THE PROJECT.** `abl_B_400M_6G1x6S` at
48.24 / 28.27 / 26.74 / 3.26 leads every column, including the 1B arm (47.79 / 29.46) at 40% of its
parameters. Two consequences:

1. **HANDOFF 14.1's correction is confirmed at 400M with full benchmarks, not just FLOP arithmetic.**
   Restoring recurrence to the Switch baseline is worth **+0.80pp Avg11 and 0.55 ppl** over the
   unmatched `6G1S` (48.24 vs 47.44). The unmatched baseline really was under-provisioned, so any
   comparison against it understates the learned gate.
2. **The 400M energy-vs-gate comparison is still INCOMPLETE.** `cmix_400M_hybrid_sparse` has not
   finished. Until it does, that comparison rests only on the 8B losses (energy 2.7319 vs Switch
   2.6571, a 0.0748-nat gap). Given that `abl_I`'s 6-sigma 8B lead REVERSED by 32B, do not state the
   400M conclusion from 8B loss alone -- though the direction has been consistent across the 8B loss
   and every 134M benchmark, so a reversal of that size would be surprising.

`lm_loss` is the windowed median over the 8B crossing (steps 14,500-16,000 at 524,288 tok/step),
n=101-151 points, median SE ~0.001. LOWER better.

---

## ⚠ `energy_proj_type`: DO **NOT** SWITCH TO `unconstrained`. THE A/B WAS RUN AND IT LOSES.

`abl_I_134M_w1w2_sparse_surr_projUncon` is `cmix_134M_hyb_w1w2_sparse_surr_32B` with ONE field
changed (`energy_proj_type: psd_anti` -> `unconstrained`; the only other diffs are `save_path` and
the wandb name). Both ran the full 32.0B. Parameters differ by -0.018% (`psd_anti` costs
`d*r + d*d` = 614,400 at d=768 with the default `energy_proj_rank: 32`; `unconstrained` costs
`d*d` = 589,824). A clean single-variable A/B.

| | 8B `lm_loss` | 32B Avg11 | ppl | MMLU | GSM8K |
|---|---|---|---|---|---|
| `psd_anti` | 2.8883 | **45.01** | **40.70** | **24.90** | 1.82 |
| `unconstrained` | **2.8821** | 44.32 | 41.09 | 23.73 | **2.05** |
| delta | **-0.0062 (uncon better, ~6 sigma)** | **-0.69pp** | **+0.39** | **-1.17pp** | +0.23 |

**THE 8B LOSS ADVANTAGE INVERTED BY 32B.** Unconstrained led on training loss at 8B by ~6 sigma and
then finished WORSE on Avg11, perplexity and MMLU. Whatever `psd_anti` costs early, it pays back.

**Methodological consequence, and it is the more important finding:** an 8B training-loss lead of
6 sigma did NOT survive to the 32B benchmark — it reversed. Any "decide at the 8B milestone" rule
must therefore be treated as a TRIAGE heuristic for killing hopeless arms, NOT as a predictor of
final ranking. Do not promote an arm to the paper on an 8B loss delta alone.

### If someone still wants unconstrained variants, these are the files to change

`energy_proj_type` only does anything when the model HAS an energy block. Determined by grep, not by
assumption — `sequence_mixer_type: energy_attention` count and `mlp_type: EnergyFF*` count:

**11 arms (12 files) carry an energy block AND are still on `psd_anti`:**

| file | energy_attn | energyFF |
|---|---|---|
| `cmix_134M_hybrid_32B_sparse.yml` | 1 | 1 |
| `cmix_134M_sandwich_32B_sparse.yml` | 1 | 1 |
| `cmix_134M_pure_32B_sparse.yml` | 1 | 1 |
| `cmix_134M_hyb_w1w2_surrMLP_32B.yml` | 1 | 1 |
| `cmix_134M_hyb_w1w2_sparse_surr_32B.yml` | 1 | 1 | ← `abl_I` already tests this one |
| `abl_E_134M_6G1x6E_baseEGPT.yml` | 1 | 1 |
| `cmix_400M_hybrid_sparse.yml` | 1 | 1 |
| `cmix_400M_sandwich_sparse.yml` | 1 | 1 |
| `cmix1B_12L_gptDense_32B.yml` | 1 | 1 |
| `abl_G_400M_6G1x6E1x6E.yml` **and** `abl_G_400M_6G1x6E1x6E_4gpu.yml` | 2 | 2 |
| `abl_H_400M_6G6E_deep.yml` | 6 | 6 |

The edit is one line: `energy_proj_type: psd_anti` → `energy_proj_type: unconstrained`, plus a
distinct `save_path` and wandb `name` (pre-flight rule 6). Valid values in this tree are
`unconstrained | pos_scalar | identity | psd_anti` (`energy/layer.py:883`); `unconstrained` is the
CommonConfig default. **Do NOT write `proj_mode: unconstrained`** — that spelling does not exist
here and is a silent no-op (it appears in a colleague's config; see the DATAMIX REFERENCE block in
`experiments/boltzmann-moe/CLAUDE.md`).

**The other 9 files are ENERGY-FREE — `energy_proj_type` is inert in them and editing it does
nothing:** `cmix_134M_gptswitch_32B`, `cmix_400M_baseline_switch`, `abl_B_134M_6G1x6S`,
`abl_B_400M_6G1x6S`, `abl_D_134M_6G_dense_isototal`, `abl_F_134M_6G_dense_isoactive`,
`abl_G_400M_6G1x6S1x6S`, `abl_H_400M_6G6S_deep`, `abl_H_400M_6G6G_deep_isoactive`. They keep the
field only because they were copied from energy parents; it is dead weight, not a setting.

(12 energy files on psd_anti + 9 energy-free + `abl_I` which is the energy arm already switched to
unconstrained = 22 configs in this directory.)

---

## BEFORE LAUNCHING ANY OF THESE, read `experiments/boltzmann-moe/CLAUDE.md`

Non-negotiables that have each already cost a multi-day run: the 10-point config pre-flight; the
datamix must be the `configs/cmix/` mix WITH the math sets; `sinkhorn_persist_mu: true` and
`sinkhorn_mu_iters` == the block's `layer_iterations` entry on every Sinkhorn arm; `e_sign_override`
is `pos` for hopfield and `neg` for w1w2; head count is tier-specific (12 heads at d=768, 16 at
d=1024 -- `head_dim` must be >= `rope_dim` 64); and an arm carrying `sinkhorn_persist_mu` must run
SINGLE-NODE or it wedges silently with LSF still reporting RUN.
