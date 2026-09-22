# bsaha400m — the five iso-param 400M arms (30k steps / 31.46B tokens)

Five arms, all at **30000 steps / 31.46B tokens**, evaluated 2026-09-21. They come from TWO
different code trees, and **each config only runs on the tree that trained it**:

| config | code tree / branch | runs on this branch? |
|---|---|---|
| `bs400m_s12_stdmoe.yml` | `nima/main` + 2 fixes = this branch | yes |
| `bs400m_s8e4_fh2_sparse.yml` | same | yes |
| `bs400m_s8e4_f5kl_surrogate.yml` | same | yes |
| `fh2_400m.yml` | bsaha fork, branch `bsaha/boltzmoe-iclr27` | **NO** |
| `fh2kl_400m.yml` | same bsaha fork | **NO** |

The last two are included **for reference/comparison only** — their `mlp_type`s
(`BoltzmannMoE_Energy_MLP`, `MoE_Energy_F5`) do not exist on `nima/main`, so this branch cannot
build them. This branch = Nima's `a94909d2` + 2 commits: `7dd53054` (tf-4.57/5.x
`_no_split_modules` compat) and `ed1a1f6a` (keep repulsion pair indices on the activations' device).

## Only ONE of the five is sparse

The `fh2` in `bs400m_s8e4_fh2_sparse` and in `fh2_400m` means different things — they share a label
and almost nothing else:

| arm | expert class | n_experts | top_k | forward | width |
|---|---|---|---|---|---|
| `s12_stdmoe` | std `MoE` (all-softmax baseline) | 8 | 2 | dense | 824 |
| `s8e4_fh2_sparse` | hopfield + Sinkhorn, `sparse_forward: true` | **64** | 2 | **SPARSE** | 31232 (I_e=488) |
| `s8e4_f5kl_surrogate` | learned-gate surrogate | 8 | 2 | dense | 16384 |
| `fh2_400m` | `BoltzmannMoE_Energy_MLP` (dense w1w2) | 8 | 2 | dense | 16384 |
| `fh2kl_400m` | `MoE_Energy_F5` (KL/distillation) | 8 | 2 | dense | 2048 |

`s8e4_fh2_sparse` is the **only** arm with `sparse_forward`, the only one at K=64, and the only one
using a rank-16 proxy + Sinkhorn. `fh2kl_400m` carries `learnable_temperature: true` +
`distillation_weight: 0.01` and is unrelated to the sparse arm.

## Parameters

| arm | total params | true active |
|---|---|---|
| `s12_stdmoe` | 396,223,488 | 213,967,872 |
| `s8e4_fh2_sparse` | 396,449,284 | **187,586,052** |
| `s8e4_f5kl_surrogate` | 396,481,540 | 210,883,588 |
| `fh2_400m` | 396,448,772 | 210,850,820 |
| `fh2kl_400m` | 396,481,544 | 210,883,592 |

Iso-**total**-param spread across all five: **0.065%**. See the active-param caveat below — they are
NOT iso-active.

## Geometry

Global batch is **identical across all five** (32 sequences x 4096 tokens = 1,048,576 tok/step ->
31.46B at 30k steps), but the micro-batch split differs by tree:

| arms | nodes x GPU | micro_batch | grad_accum | torch_compile |
|---|---|---|---|---|
| the three nima-code arms | 1 x 8 | 4 | 8 | true, except **false** for f5kl_surrogate |
| `fh2_400m`, `fh2kl_400m` | 1 x 8 | 2 | 16 | true / **false** respectively |

mb x ga = 32 in every case, so the training is comparable. **Do NOT compare wall-clock s/step
across arms**: mb differs (2 vs 4) and `torch_compile` is off for the two learned-gate/KL arms, so
throughput is confounded three ways.

## Sizing notes that took measurement, not arithmetic

- s12: uniform `intermediate_size: 824`.
- fh2_sparse: `intermediate_size: 31232` -> I_e = 31232/64 = 488. **Hopfield stores ONE matrix per
  expert (not W1+W2), so it needs ~2x the width of the w1w2 form** for the same param count; a first
  attempt at 16384 came out 61M short. Must stay divisible by `n_experts: 64`.
- f5kl_surrogate: 16384 on 4 blocks.

## Two caveats before comparing these numbers

**1. `active parameters in the model` in the training log is WRONG for the energy arms.**
`lm_engine/model_wrapper/base.py` only applies the top-k discount when a block defines
`get_num_active_parameters()`. `BoltzmannMoE_Energy_MLP` does not, so its expert bank is counted as
fully dense — over-reporting by ~100-124M. The `true active` column above is recomputed from expert
tensor shapes x top_k/n_experts.

**2. `fh2_sparse` is NOT iso-active.** At 187.6M it is ~11% below the other four, because K=64 top-2
activates 2/64 of its bank instead of 2/8. Both its quality deficit and its wall-clock advantage are
partly explained by doing less work per token, so it is not a controlled comparison. It also differs
from the paper's fh2 in four other ways at once (hopfield class, K 8->64, aux-loss->Sinkhorn,
dense->sparse).

## Site-specific paths to change off Blue Vela

`data_path` / `data_cache_path` / `tokenizer_name` / `save_path` / `load_path`. Tokenizer lives at
`/proj/datasets/tokenizers/granite-4.0-tiktoken`. `load_path` == `save_path` in every config so a
resubmit resumes rather than restarts (`save_interval: 500`).

## Eval convention used for the reported numbers

lm-evaluation-harness **v0.4.9.2**, transformers **4.57.1** (5.1.0 silently clobbers the tied `wte`
and all configs set `tie_word_embeddings: true`).

- zero-shot 12-task: `max_length=4096, use_cache=False`, 0-shot
- MMLU: `--num_fewshot 5`, `max_length=4096, use_cache=False`
- `gsm8k_cot`: `--num_fewshot 5` (overrides the task default of 8), `max_length=2048, use_cache=True`
- plain `gsm8k`: no flag, but the task default is **5-shot**, `max_length=2048, use_cache=True`

`Avg-11` = unweighted mean over arc_challenge, arc_easy, boolq, copa, hellaswag, lambada_openai,
openbookqa, piqa, race, sciq, winogrande. **Two variants are not interchangeable:** the published
`Average` is plain `acc` on all 11; an acc_norm-preferred variant (norm on the 6 tasks that report
it) runs 0.5-0.9pp HIGHER. Say which one you mean.

## Results (30k steps / 31.46B tokens, harness v0.4.9.2, tf 4.57.1)

`AvgN` = Avg-11 **acc_norm-preferred**. `cot*` = `gsm8k_cot` 5-shot; `pln*` = plain `gsm8k`
(also 5-shot — that is the upstream task default, despite no `--num_fewshot` flag).

| arm | code | loss | AvgN | wikiPPL | lambPPL | MMLU | cotFlex | cotStr | plnFlex | plnStr |
|---|---|---|---|---|---|---|---|---|---|---|
| `s12_stdmoe` | nima | **2.3486** | **.4963** | **26.51** | **28.03** | **.2765** | .0281 | .0205 | **.0387** | **.0341** |
| `s8e4_fh2_sparse` | nima | 2.4367 | .4844 | 29.71 | 38.09 | .2462 | .0265 | .0174 | .0220 | .0174 |
| `s8e4_f5kl_surrogate` | nima | 2.4312 | .4885 | 29.87 | 33.87 | .2455 | .0281 | **.0235** | .0281 | .0197 |
| `fh2_400m` | bsaha | 2.3705 | .4947 | 27.47 | 32.27 | .2653 | .0144 | .0114 | .0258 | .0190 |
| `fh2kl_400m` | bsaha | 2.4367 | .4775 | 29.64 | 35.01 | .2468 | .0212 | .0167 | .0167 | .0091 |

**The all-softmax baseline wins every metric that discriminates** (loss, AvgN, both perplexities,
plain gsm8k). `fh2_400m` is second and within one-seed noise of it on AvgN (-0.16pp) — parity, not
a win. The other three sit 0.8-1.9pp below on AvgN.

**MMLU and gsm8k do NOT rank these arms.** MMLU: only the baseline (.2765) is off the .25 chance
floor; two arms are *below* it. gsm8k: everything is 0.9-3.9%. Below the top arm the four energy
arms reorder freely depending on which of these columns you pick, which is the signature of
sub-noise differences. Rank on loss / AvgN / PPL only.

Caveat on `AvgN`: the **published** `Average` in the paper CSV is plain `acc` on all 11 tasks, NOT
norm-preferred (verified: 18 paper rows reproduce exactly under plain-acc, 0 under norm-preferred).
Norm-preferred runs 0.5-0.9pp higher. Quoting one against the other accounts for ~1pp of apparent
difference, so always say which you mean.

n=1 per arm — no seeds.
