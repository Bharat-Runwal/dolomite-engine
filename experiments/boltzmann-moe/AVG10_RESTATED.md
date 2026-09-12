# Avg restated on the current `avg10` convention (2026-09-12)

Published "Avg" figures in `PROGRESS.md`, `CLAUDE.md` and `BOLTZ_MOE_BEST.md` are
**`avg9`**: nine tasks, **MMLU excluded**, `acc_norm` on six (incl. `sciq`). MMLU was
dropped because of a dataset installation problem; `race`/`lambada_openai` were also
absent until the 2026-08-03 `pyarrow>=20` pin.

The current convention is **`avg10`**: ten tasks, **MMLU included**, `acc_norm` on
five (`sciq` uses `acc`). This is what `compute_aggregates.py` computes and what the
FET series already reports. **Use avg10 for the paper.**

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

