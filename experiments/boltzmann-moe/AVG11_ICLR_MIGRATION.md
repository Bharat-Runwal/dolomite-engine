# ICLR draft: `avg10` → `Avg11` migration (2026-09-15)

Record of the metric migration applied to
**`/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`**. Every number below was
recomputed from raw per-task harness JSON with
`experiments/eval_scripts/compute_avg11.py`. **Nothing was rescaled, interpolated, or
converted by formula.**

## Why

The draft previously reported `avg10` (MMLU **in** the mean; race + lambada **out**;
`sciq` scored `acc_norm` despite the prose claiming `acc`). The rest of the group
(EGPT-RL / FET) headlines **`Avg11`**. Mixing the two makes our numbers
non-comparable with theirs, so the draft moved to `Avg11`:

- `acc_norm` ×6: arc_challenge, arc_easy, hellaswag, openbookqa, piqa, **sciq**
- `acc` ×5: boolq, copa, winogrande, **race**, **lambada_openai**
- **MMLU (acc), GSM8K-CoT (flexible-extract), WikiText word-PPL: reported SEPARATELY.**

`Avg11` runs **1.6–2.8 pp below `avg10`** on the same checkpoint (race ≈0.28 and
lambada ≈0.23 are near chance here, and MMLU leaves the mean). Convention artifact,
not a model effect.

**No re-eval was needed:** all 22 `iclr_*` runs already carried race + lambada.

## Checkpoint → paper row map (pinned by reproducing each `avg10` exactly)

`tab:pure` has no WikiPPL column, so its rows were pinned by reproducing the paper's
`avg10` value from source — not by run name. This matters: the `iclr_gptmoe/*` configs
say `model_type: energy`, but with softmax attention and no recurrence they *are* the
energy-free plain stack. Name-based mapping would have been wrong.

| run dir (`.../boltzmann-moe/results/`) | avg10 | **Avg11** | MMLU | GSM8K | WikiPPL |
|---|---:|---:|---:|---:|---:|
| `iclr_moebase/iclr_switch_K16_top2` | 46.72 | **44.83** | 24.18 | 1.90 | 40.02 |
| `iclr_flops/iclr_hop_K16_dense` | 46.58 | **44.78** | 24.64 | 2.20 | 40.73 |
| `iclr_flops/iclr_hop_K32_top2` | 46.26 | **44.38** | 25.17 | 1.82 | 40.59 |
| `iclr_flops/iclr_hop_K32_top1` | 46.15 | **44.12** | 24.33 | 2.20 | 40.77 |
| `iclr_moebase/iclr_switch_K16_top2_shared` | 46.04 | **43.96** | 24.94 | 1.82 | 40.11 |
| `iclr_flops/iclr_hop_K16_top2` | 45.89 | **43.91** | 26.55 | 2.43 | 40.49 |
| `iclr_flops/iclr_w1w2_K16_top2` | 45.69 | **43.83** | 25.49 | 2.27 | 40.00 |
| `iclr_flops/iclr_learn_K16_top2` | 45.66 | **43.83** | 24.57 | 2.35 | 40.13 |
| `iclr_ctrl/iclr_hop_K16_top2_renorm` | 45.35 | **43.74** | 24.55 | 2.05 | 40.36 |
| `iclr_moebase/iclr_learn_K16_top2_noLB` | 45.06 | **43.12** | 25.07 | 1.67 | 40.08 |
| `iclr_flops/iclr_hop_K16_top2_nofix` | 45.20 | **43.53** | 25.09 | 1.59 | 42.32 |
| `iclr_flops/iclr_pure_hop_K16_top2_1blk` | 42.94 | **40.21** | 24.73 | 1.82 | 69.66 |
| `iclr_gptmoe/gptmoe_last_isoP` (plain stack, iso-param) | 46.04 | **44.43** | 25.17 | 1.90 | 40.76 |
| `iclr_gptmoe/gptmoe_last_3x` (plain stack, 3× budget) | 46.18 | **44.20** | 24.51 | 1.97 | 38.98 |
| `iclr_gptmoe/gptmoe_all_isoP` (plain stack, every layer) | 44.86 | **43.05** | 23.91 | 2.05 | 44.03 |
| `iclr_slope/slope90k_hyb` (90k-step token scaling) | 46.18 | **44.32** | 25.66 | 1.97 | 39.89 |
| `iclr_big/iclr_big_hop_pure` (400M energy) | 42.70 | **40.02** | 24.53 | 1.36 | 69.42 |
| `iclr_big/iclr_big_learn_pure_1node` (400M learned) | 42.43 | **39.83** | 24.98 | 1.67 | 61.87 |

`app:eval` correction table (h1 series, legacy lineage):
`h1_boltz_moe_fullsize` 0.4578 · `h1_topk_egpt_moe` 0.4533 ·
`h1_egpt` 0.4476 · `h1_boltz_topk2` 0.4455.

## Claims that CHANGED (not just renumbered)

1. **`§sec:pure` headline reversed → now parity.** Was "Boltzmann-MoE outperforms the
   energy-free transformer by **0.22 pp** at matched parameters". Under Avg11 the
   energy-free iso-param stack scores **44.43** vs **44.38** for Hopfield K=32 — i.e.
   0.05 pp the *other* way. Rewritten as **parity** ("level, within 0.05 pp") in
   `sec/experiments.tex`, `sec/intro.tex` and `app:findings`. Boltzmann still leads the
   3×-budget stack (44.38 vs 44.20).
2. **Backbone/routing decomposition no longer favours the backbone.** Backbone worth
   0.68 → **0.40 pp**; routing 0.46 → **0.45 pp**. The claim "most of the benefit lies
   in the recurrent encoder" was removed — the two terms are now equal within noise.
3. **Renormalisation penalty shrank 0.54 → 0.17 pp.** Direction preserved, magnitude
   claim softened; explicitly flagged as not separable from seed noise.
4. **Energy-vs-learned parity strengthened to an exact tie**: 43.83 vs 43.83.
5. **Aux-loss ablation grew 0.60 → 0.71 pp** (43.83 → 43.12).
6. **Dense→sparse costs**: energy 0.69 → **0.87 pp**, learned 1.06 → **1.00 pp**.
   Router grid `learned − energy`: +0.14/−0.23 → **+0.05/−0.08**.
7. **Baseline debt discharged.** The appendix said the energy-free GPT-MoE baselines
   "are training and have no numbers yet" / "arms are queued" while
   `sec/experiments.tex` already reported them — an internal contradiction. Updated
   with the three measured values (44.43 / 44.20 / 43.05). **Still outstanding: the
   sandwich variant** (`iclr_big/iclr_big_hop_sandwich`, unevaluated).
8. **`sciq` documentation bug fixed.** The old prose said `avg10` used plain `acc` for
   `sciq`; every stored number in fact used `acc_norm` (plain `acc` would raise each
   arm ≈0.7 pp). The published numbers were self-consistent; the prose was not. Avg11
   uses `acc_norm`, so the definition now matches the code.

Ranking in `tab:frontier` is **unchanged** by the migration (only the W1W2
energy/learned pair becomes an exact tie). `tab:pure` was re-sorted, since the
plain-stack iso-param row moves above Hopfield K=32.

## Typesetting

`tab:frontier`, `tab:pure`, `app:frontier` gained MMLU + GSM8K columns →
`\footnotesize` + reduced `\tabcolsep`. Overfull hboxes went from **2 (one at 25.7 pt)
before** to **1 after** (the remaining 4.27 pt in `tab:threeway` is pre-existing).
Local `pdflatex` cannot build this repo (`eso-pic.sty` and Courier TFMs missing —
pre-existing); verified with a stubbed `eso-pic` off-repo: 24 pages, no undefined refs.

## ⚠ THE NEWLY-EVALUATED ARMS ARE PARTIAL CHECKPOINTS -- DO NOT PUT THEM IN THE PAPER

Evaluated 2026-09-15. All four give a COMPLETE Avg11 (all 11 tasks present), but
**none of them has finished training**, so none is comparable to the 30k-step grid:

| run | Avg11 | PPL | steps done | target | % |
|---|---:|---:|---:|---:|---:|
| `iclr_decide/w1w2_K32_top2` | 41.63 | 56.00 | 10,000 | 30,000 | **33%** |
| `iclr_slope/slope90k_1blk` | 41.07 | 61.28 | 40,000 | 90,000 | 44% |
| `iclr_balance/pure_hop_isoP_bal` | 39.79 | 80.27 | 16,000 | 30,000 | 53% |
| `iclr_big/iclr_big_hop_sandwich` | 39.61 | 78.38 | 4,000 | 15,000 | 27% |

**In particular, `w1w2_K32_top2` does NOT yet resolve the Hopfield-vs-W1W2
question** that `app:threeway` / `tab:threeway` say it will. It scores 41.63
against Hopfield K=32's 44.38, but at ONE THIRD the token budget, and its PPL of
56.00 against Hopfield's 40.59 is the signature of an undertrained run, not of a
worse architecture. Reading that gap as "Hopfield wins" would be a conclusion
drawn from a 33%-trained checkpoint. **The paper's existing wording ("That arm is
training") remains correct and should stay.**

Likewise the sandwich variant at 27% does not discharge the appendix's
acknowledged baseline gap.

The eval was still worth running: every one of these now has all 11 tasks scored,
so when they finish, restating them is pure re-aggregation with no GPU cost.

## Still to fold in

- `iclr_decide/w1w2_K32_top2` — the "W1W2 at K=32 is training" arm that
  `app:threeway`/`tab:threeway` say resolves the Hopfield-vs-W1W2 question.
- `iclr_big/iclr_big_hop_sandwich`, `iclr_slope/slope90k_1blk`,
  `iclr_balance/pure_hop_isoP_bal`.
- `iclr_scale/scale32B_{boltz_hop,gptswitch}` — the 32B-token 400M pair, still training
  (LSF 1647293 / 1647503). These are the "larger runs show a slight Boltzmann
  advantage" evidence and the compute-matched column `§sec:cost` asks for.
