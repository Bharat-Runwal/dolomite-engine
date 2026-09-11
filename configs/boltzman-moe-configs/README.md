# The two 400M reference configs

Reconstructed from each checkpoint's own stored `training_config.yml` (not
hand-written), with `load_args` stripped and `save_path`/wandb name retargeted so
each file is directly launchable.

| file | role | source checkpoint |
|---|---|---|
| `s12_stdmoe_topk2_400m_30k_2e-3.yml` | baseline | `boltzmann_sweep/stdmoe_only_topk2_lr2e-3_400m_30k` |
| `s8e4_fh2_boltzmoe_topk2_400m_30k_2e-3.yml` | best Boltzmann MoE | `boltzmann_sweep/s8e4_stdmoe_fh2_boltz_topk2` |

## Architectures (verified by re-parsing the emitted files)

Both: `model_type: energy`, d=1024, 12 layers, `num_iterations: 1`, vocab 100352,
rope (dim 64), rmsnorm, tied embeddings, seq len 4096, lr 2e-3 cosine
(2k warmup / 28k decay), AdamW (0.9, 0.95) wd 0.1, grad clip 1.0,
30k steps, micro_batch 1, grad accum 4, bf16, FSDP-2.

**Baseline** — 12x softmax attention; all 12 MLPs `mlp_type: MoE`, swiglu,
intermediate 3200, 8 experts, top-2, `normalized_topk: true`.
389M active / 1097M total.

**BoltzMoE** — 8x softmax + 4x energy attention (16 heads,
`attention_multiplier: 0.125`); MLPs 0-7 standard `MoE` (swiglu, intermediate
1664, 8 experts, top-2); MLPs 8-11 `BoltzmannMoE_Energy_MLP` with
`n_experts: 8, top_k: 2, temperature: 1.0, intermediate_size: 76288`
(9536/expert), `repulsion_coef: 0.1, n_repulsion_pairs: 4`.
387M active / 1101M total.

## 400M results (30k steps, 31.5B tokens, single seed)

| metric | baseline | BoltzMoE | delta |
|---|---|---|---|
| Avg (11 zero-shot acc) | 50.86 | 50.84 | -0.02 |
| WikiText word-PPL | 27.52 | 27.63 | +0.11 |
| MMLU (5-shot) | 27.08 | 26.20 | -0.88 |
| GSM8K (5-shot cot, flex) | 5.08 | 5.14 | +0.06 |
| ARC-Easy | 62.33 | **65.61** | **+3.28** |
| ARC-Challenge-norm | 31.83 | **34.39** | **+2.56** |

Aggregates are a tie (inside the ~0.3pp / 0.5 PPL run-to-run bound); the real
win is ARC. BoltzMoE does this with no learned router at all -- routing comes
from the expert energies.

## Quick check that they run

```bash
# both, 1 node x 8 GPU, 50 steps each
./configs/boltzman-moe-configs/submit_smoketest.sh

# or one at a time
./configs/boltzman-moe-configs/submit_smoketest.sh boltzmoe

# override placement / size
NODES=2 QUEUE=normal GROUP=grp_ebm STEPS=20 ./configs/boltzman-moe-configs/submit_smoketest.sh
```

`STEPS` (default 50) rewrites `num_training_steps` into a temp config so a smoke
test cannot silently become a 30k-step run; it also rescales warmup/decay, pushes
`save_interval` past the end, and sets `log_interval: 1`. Pass `STEPS=0` to run the
configs exactly as-is.

**Passing** = the log reaches `step = 10` with a finite train-loss and non-zero
`billion_tokens_per_day`. That proves the model builds, data loads, FSDP-2 shards
it, and the optimizer steps.

```bash
bjobs -w | grep smoke_
grep -a 'step = ' /proj/dmfexp/energy-gpt/logs/boltzmoe-smoketest/*.err | head
```

Already verified on this branch (CPU, before submitting anything): both configs
build and run a forward pass, with parameter counts matching the paper --
baseline **1,096,934,400** and BoltzMoE **1,101,091,844**.

## Two things that will bite whoever runs these

1. **Venv.** Use `.venv-nima` (transformers 4.57.1). This repo's `.venv` has
   transformers 5.1.0, which silently clobbers the tied `wte` -- and both configs
   set `tie_word_embeddings: true`. `submit_smoketest.sh` exports
   `PRETRAIN_VENV=.venv-nima` for you; `launch-scripts/pretrain.sh` now honours
   that variable (it used to hardcode `.venv`).

2. **`top_k` truncation is unnormalized.** With `n_experts: 8, top_k: 2` the class
   softmaxes over all 8 then zeroes outside the top-2 *without* renormalizing, so
   routing weights sum to ~0.56 (token-dependent). That is how the reported run
   trained. Deliberate, but it differs from Eq. 4 of the paper -- changing it
   means the numbers above no longer apply.
