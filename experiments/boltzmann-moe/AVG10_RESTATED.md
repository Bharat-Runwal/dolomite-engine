# Avg restated on the `avg10` convention (2026-09-12)

> ## ⚠ SUPERSEDED 2026-09-14 — `avg10` is no longer the headline convention
>
> The canonical headline metric is now **`Avg11`** (colleague-/paper-consistent):
> compute it with `experiments/eval_scripts/compute_avg11.py`, restate all stored
> evals with `experiments/eval_scripts/restate_to_avg11_20260914.py --md`, and see
> the Avg11 ICLR grid at the top of `PROGRESS.md`. **Avg11 runs ~3pp BELOW avg10**
> (it drops MMLU and adds near-chance race + lambada), so avg10 and Avg11 must never
> share a table.
>
> **This file stays as the avg9→avg10 record; do NOT read its avg10 numbers as
> Avg11.** The runs restated below (`h1_boltz_moe_fullsize`, `h1_topk_egpt_moe`,
> `h1_egpt`, `h1_boltz_topk2`) were evaluated before the 2026-08-03 `pyarrow>=20`
> fix, so their `harness_results.json` is missing `race` + `lambada_openai` and
> `compute_avg11.py` reports them **INCOMPLETE (9/11)**. They therefore CANNOT be
> converted to Avg11 and remain on avg10/avg9 until re-evaluated. Re-run their
> harness eval before quoting any Avg11.

Published "Avg" figures in `PROGRESS.md`, `CLAUDE.md` and `BOLTZ_MOE_BEST.md` are
**`avg9`**: nine tasks, **MMLU excluded**, `acc_norm` on six (incl. `sciq`). MMLU was
dropped because of a dataset installation problem; `race`/`lambada_openai` were also
absent until the 2026-08-03 `pyarrow>=20` pin.

The convention here is **`avg10`**: ten tasks, **MMLU included**, `acc_norm` on
five (`sciq` uses `acc`). This is what `compute_aggregates.py` computes. It has since
been **superseded by `Avg11`** (see the banner above); avg10 is retained here only to
document the avg9→avg10 gap, not as a paper headline.

The avg9 formula was recovered by fitting all five published h1 numbers
simultaneously; it reproduces them to a worst-case error of 0.0004.

## Headline corrections

| claim | avg9 (as published) | avg10 (correct) |
|---|---|---|
| Boltzmann soft `h1_boltz_moe_fullsize` | 0.501 | **0.4832** |
| learned-router top-2 `h1_topk_egpt_moe` | 0.499 | **0.4817** |
| Boltzmann − learned-router margin | +0.17pp | **+0.15pp** |
| iso-compute no-MoE `h1_egpt` | 0.489 | **0.4750** |
| sparse-trained `h1_boltz_topk2` | 0.4856 | **0.4714** |

Direction is unchanged — Boltzmann routing still edges the learned router — but the
margin is **+0.15pp**, which is well inside seed noise for a ten-task average and
must not be presented as a win without error bars.

**Not restatable here:** the 585M/680M runs (`*_580m`, `gptswitchmoe-680M`,
`scale_gptmoe_*`) have no stored `unsharded/harness_results*.json` under
`experiments/boltzmann-moe/results/`, so their `BOLTZ_MOE_BEST.md` figures stay on
avg9. That comparison is still internally valid because **both** arms are avg9
(680M Boltz 58.47 vs gptswitchmoe-680M 57.82, +0.65pp); only the absolute values are
on the old scale. Re-evaluate those checkpoints if absolute numbers go in the paper.

## Full restatement

