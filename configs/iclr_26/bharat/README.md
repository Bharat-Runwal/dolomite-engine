# For Bharat — a 1B pair where EVERY layer is a routed MoE

**★ MAIN SUGGESTION.** Two configs, an experiment **only as a pair**: identical datamix, budget,
tokens/step, schedule, width, depth and attention. The **only** difference is the FFN router.

| config | GPUs | total | ACTIVE | MoE layers | K | k | I_e | k·I_e / d |
|---|---|---|---|---|---|---|---|---|
| `bharat_1B_18L_allMoE_boltz.yml` | 8 | **1022.4M** | **512.8M** | **18 of 18** | 32 | 8 | 768 | **4.0×** |
| `bharat_1B_18L_allMoE_switch_baseline.yml` | 8 | **1004.4M** | **494.8M** | 18 of 18 | 32 | 8 | 256 | — |

d=1536, 18 layers, 24 heads (head_dim 64 = rope_dim 64), **softmax attention in every block**.
524,288 tok/step × 61,035 = **32.0B**. Both meta-build and agree with `audit_config` to the byte.
Switch uses `I_s = I_e/3 = 256` because swiglu stores three matrices per expert and hopfield one —
that is what makes them iso-parameter at the same K and k.

## Why this shape, and why not ours

Built to match what MoE models actually do, verified from their own `config.json`:

| model | L | d | MoE layers | K | k | I_e | k·I_e / d |
|---|---|---|---|---|---|---|---|
| OLMoE-1B-7B | 16 | 2048 | **all 16** | 64 | 8 | 1024 | **4.0×** |
| Qwen1.5-MoE-A2.7B | 24 | 2048 | **all 24** (+shared) | 60 | 4 | 1408 | 2.8× |
| our 400M hybrid | 7 | 1024 | 1 of 7 (×6 rec) | 32 | 2 | 5871 | **11.5×** |
| our existing 1B | 12 | 1024 | 4 of 12 | 64 | 2 | 2846 | 5.6× |
| `fallback/` 8S4E | 12 | 1024 | 12 of 12 | 32 | 8 | 3000 | 23.4× |

The field puts an MoE in **every** layer and keeps the effective per-token FFN width `k·I_e` near
**4×d** — the width a dense model's FFN would have. Our arms invert that: very few MoE layers, each
absurdly wide. This config follows the field.

Note this **retires an old worry of ours** that `I_e=1024` is "too narrow for sparsity to pay":
OLMoE uses exactly `I_e=1024` and works, because `k=8` and `K=64` make `k·I_e` right. Narrow experts
are fine; `k·I_e` is what matters.

## The number that motivates all of this

**Our existing 1B activates like a 400M model, which is why it performs like one.**

| | total | ACTIVE | bank | bank ACTIVE | Avg11 | ppl |
|---|---|---|---|---|---|---|
| `cmix1B_12L_gptDense_32B` (K=64, k=2) | 1002.1M | 279.3M | 746.1M | **23.3M (3.1%)** | 47.79 | 29.46 |
| `abl_B_400M_6G1x6S` (our best arm) | 400.0M | 219.7M | 192.4M | 12.0M | **48.24** | **28.27** |
| **`bharat_1B_18L_allMoE_boltz`** | 1022.4M | **512.8M** | 442.4M | **110.6M (25%)** | — | — |

The old 1B stores 746M of experts and uses 23M per token. Its ACTIVE count is only **+27%** over the
400M arm and it **loses** to that arm on every column.

## Choices you may want to overrule

- **K=32, k=8.** K=8 (the earlier suggestion) is too few experts to route among at this width;
  K=64/k=2 is the trap above. `p = k+4 = 12` of 32, so the sparsity ceiling is `K/p = 2.7×`.
- **d=1536, not 2048.** More standard would be 2048, but our vocab is **100,352** — 3× TinyLlama's —
  so the tied embedding alone costs 205.5M at d=2048 and total passes 1.5B. At d=1536 it is 154.1M.
- **COSINE, not WSD.** No controlled WSD-vs-cosine A/B exists at 32B; WSD was confirmed only on short
  probes. What matters is that the pair shares one schedule, which it does (`2000 / 0 / 59035`).
- **hopfield + rank-16 subspace PROXY, not the surrogate head.** Measured at 134M, same budget:
  proxy **0.20 s/step** vs surrogate **0.30** — the proxy is **1.5× faster**, and the quality gap
  (Avg11 45.01 vs 44.82) is inside noise.

## For 128B tokens

`num_training_steps` → **244,140** and `num_decay_steps` → **242,140**, so
`warmup + constant + decay == num_training_steps`. Not optional: published `slope90k_*` configs once
ran 60,000 of 90,000 steps pinned at the LR floor because that was violated.

## Launch

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
export WANDB__SERVICE_WAIT=300
for a in bharat_1B_18L_allMoE_boltz bharat_1B_18L_allMoE_switch_baseline; do
  bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \
       $a configs/iclr_26/bharat/$a.yml 8 preemptable 24:00 200G 4
done
```

Verify within 60 s — **step lines are in STDERR**, and `grep -m1 DeviceMesh <stderr>` must NOT say
`'cpu'` (a host with failed CUDA init makes the trainer fall back to a CPU mesh and train nothing
while LSF reports RUN). Sub-60-second multi-node deaths are transient (~50%): just resubmit.

## `fallback/` — the earlier 8S4E pair

`bharat_1B_8S4E_boltz` (998.7M / 364.0M active) and `bharat_1B_12S_switch_baseline` (998.1M /
364.7M). Kept because they are verified and matched, but **superseded**: 8S4E puts `k·I_e` at 23×d,
nothing like field practice, and reaches only 364M active against this pair's 513M at the same total.
Use them only if the main pair hits a problem.
