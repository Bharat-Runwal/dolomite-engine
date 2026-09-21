# bsaha400m — three iso-param 400M arms on `nima/main`

Built 2026-09-18 at Nima's request ("rerun s12, s8e4_fh2, s8e4_f5kl on top of the new code").
All three trained to **30000 steps / 31.46B tokens** and evaluated 2026-09-21.

| config | arch | total params | true active |
|---|---|---|---|
| `bs400m_s12_stdmoe.yml` | all-softmax + 12 std MoE (baseline) | 396,223,488 | 213,967,872 |
| `bs400m_s8e4_fh2_sparse.yml` | 8 softmax + 4 energy, hopfield K=64 + Sinkhorn + sparse | 396,449,284 | **187,586,052** |
| `bs400m_s8e4_f5kl_surrogate.yml` | 8 softmax + 4 energy, learned-gate surrogate | 396,481,540 | 210,883,588 |

Iso-**total**-param spread: 0.065%. See the active-param caveat below — they are NOT iso-active.

## Geometry (identical across all three)

1 node x 8 GPUs, `micro_batch_size: 4`, `gradient_accumulation_steps: 8`, `sequence_length: 4096`
-> 8 x 4 x 8 x 4096 = **1,048,576 tokens/step** -> 31.46B at 30k steps.
`torch_compile: true` for s12 and fh2_sparse; **false** for f5kl_surrogate (learned gate).

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

**2. `fh2_sparse` is NOT iso-active.** At 187.6M it is ~11% below the other two, because K=64 top-2
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
