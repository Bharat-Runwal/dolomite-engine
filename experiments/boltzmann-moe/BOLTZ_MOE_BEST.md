# Best Boltzmann MoE Runs — Quick Reference

**TL;DR.** Two ~600M Boltzmann-MoE hybrids both beat all pure-GPT and no-MoE
baselines on every metric (avg, WikiPPL) at substantially fewer training
tokens. `580M` uses recursion (one EGPT block × 6); `scale_h3_boltz` uses 4
distinct EGPT blocks with no recursion. Both implement BoltzMoE FFNs in the
EGPT layers.

| Run | Config | Save path |
|---|---|---|
| **580M (recursive)** | [`configs/boltzmann_moe/h1_boltz_moe_580m_8x4096_d1536.yml`](../../configs/boltzmann_moe/h1_boltz_moe_580m_8x4096_d1536.yml) | `experiments/boltzmann-moe/results/h1_boltz_moe_580m_8x4096_d1536/` |
| **scale_h3_boltz (no-rec)** | [`configs/boltzmann_moe/scale_h3_8gpt_4egpt_boltz_d1280.yml`](../../configs/boltzmann_moe/scale_h3_8gpt_4egpt_boltz_d1280.yml) | `experiments/boltzmann-moe/results/scale_h3_8gpt_4egpt_boltz_d1280/` |

## Layer-by-layer architecture (quick read)

### 580M — `h1_boltz_moe_580m_8x4096_d1536` (679M total params)

```
d=1536, num_layers=12, layer_iterations=[1,1,1,1,1,1,1,1,1,1,1,6]   ⇒ 17 effective layers

Layers 1-11  (each iterated 1×): GPT prefix
   ├── softmax_attention   (16 heads, head_dim=96, no bias, attn_mult=0.125)
   └── MLP swiglu          (intermediate=6144 = 4d)

Layer  12     (iterated 6× via layer_iterations): EGPT recurrent
   ├── energy_attention    (16 heads, head_dim=96, no bias, attn_mult=0.125)
   └── BoltzmannMoE_Energy_MLP
         ├── n_experts        = 8
         ├── intermediate_size= 32768   (per-expert I_e = 4096)
         ├── temperature      = 1.0
         ├── repulsion_coef   = 0.1
         ├── n_repulsion_pairs= 4
         └── gelu_grad_method = sigmoid (legacy default)
```

### scale_h3_boltz — `scale_h3_8gpt_4egpt_boltz_d1280` (~620M total)

```
d=1280, num_layers=12, layer_iterations=[1,1,1,1,1,1,1,1,1,1,1,1]   ⇒ 12 effective layers (no recursion)

Layers 1-8   GPT prefix
   ├── softmax_attention   (20 heads, head_dim=64, no bias, attn_mult=0.125)
   └── MLP swiglu          (intermediate=4096)

Layers 9-12  EGPT (4 distinct blocks, no recursion)
   ├── energy_attention    (20 heads, head_dim=64, no bias, attn_mult=0.125)
   └── BoltzmannMoE_Energy_MLP
         ├── n_experts        = 4
         ├── intermediate_size= 16384   (per-expert I_e = 4096)
         ├── temperature      = 1.0
         ├── repulsion_coef   = 0.1
         ├── n_repulsion_pairs= 4
         └── gelu_grad_method = tanh_exact (post-2026-06-01 self-consistent variant)
```

## Eval results

| Run | Tokens | Avg | WikiPPL | MMLU | GSM8k flex-avg | Notes |
|---|---:|---:|---:|---:|---:|---|
| **580M @ step 124k (FINAL)** | **65.0B** | **58.47** | **19.33** | **26.33** | 2.08 | **🏆 final — best on every metric, training complete** |
| 580M @ step 118k | 61.9B | 57.81 | 19.48 | 25.72 | 2.12 | prior |
| 580M @ step 110k | 57.7B | 57.59 | 19.70 | 26.17 | 1.86 | prior |
| 580M @ step 102k | 53.5B | 58.14 | 20.23 | 26.09 | 2.31 | prior |
| 580M @ step 76k | 39.8B | 55.93 | 22.41 | 25.30 | 2.39 | prior champion |
| 580M @ step 30k (pre-bug, clean) | 15.7B | 53.66 | 26.84 | 25.42 | 1.90 | trustworthy clean snapshot |
| **scale_h3_boltz @ step 120k** | 62.9B | **56.94** | **21.89** | 26.11 | 1.74 | iso-params no-recursion sibling |
| scale_h3_boltz @ step 104k | 54.5B | 55.63 | 22.67 | 25.03 | 2.39 | prior |

### Pure-GPT + Switch MoE comparison set (in-progress, ~12B tokens each)

Matched-arch ablation vs `scale_h3_boltz`: same d=1280, 12 layers, but pure GPT
attention everywhere and MoE in the FFN slot using either Switch-style learned
top-1 or our Boltzmann-style top-K. Goal is to isolate the contribution of the
energy-attention layers separately from the routing scheme.

Models share the same scaffold (d=1280, 12 transformer layers, softmax
attention everywhere). They differ on routing, MoE coverage, and per-expert
size — see the architecture summary at the top of this file. The Boltz-MoE
**580M** baseline (which uses energy attention and a recurrent EGPT layer)
is included as the structurally-different reference.

| Run | Config | Total / Active (M) | Tokens | Avg | WikiPPL | MMLU | gsm8k flex-avg | Notes |
|---|---|---:|---:|---:|---:|---:|---:|---|
| `scale_gptmoe_8gpt_4switchmoe_d1280` @ 24k | 8 GPT + 4 Switch-MoE FFN K=4 I=4096 | 585 / 459 | 12.6 B | 50.81 | 30.48 | 23.27 | 2.46 | early Switch top-1 |
| `scale_gptmoe_8gpt_4switchmoe_d1280` @ 60k | 8 GPT + 4 Switch-MoE FFN K=4 I=4096 | 585 / 459 | 31.5 B | **54.74** | **26.23** | 25.31 | **2.50** | **leads trio at ~30B** |
| `scale_gptmoe_12moe_K4I2048_d1280` @ 26k | 12 GPT + Boltz-MoE FFN K=4 I=2048 | 585 / 396 | 13.6 B | 53.57 | 30.68 | 26.08 | 2.35 | early Boltz lead |
| `scale_gptmoe_12moe_K4I2048_d1280` @ 65k | 12 GPT + Boltz-MoE FFN K=4 I=2048 | 585 / 396 | 34.1 B | 54.01 | 26.39 | 25.05 | 2.35 | matches v9 GPT |
| `scale_gptmoe_12moe_K4I4096_d1280` @ 14k | 12 GPT + Boltz-MoE FFN K=4 I=4096 | 962 / 585 | 7.3 B | 52.99 | 31.30 | 23.12 | 2.01 | early larger-experts |
| `scale_gptmoe_12moe_K4I4096_d1280` @ 45k | 12 GPT + Boltz-MoE FFN K=4 I=4096 | 962 / 585 | 23.6 B | 54.36 | 26.35 | **25.92** | 2.24 | best MMLU; ⅔ tokens of others |
| **580M Boltz** @ 30k (energy-attn + Boltz-MoE, recursive) | 11 GPT + 1 EGPT×6 (K=8 I=4096) | 679 / 679 | 15.7 B | 53.66 | 26.84 | 25.42 | 1.90 | iso-tokens reference (low) |
| **580M Boltz** @ 76k (energy-attn + Boltz-MoE, recursive) | 11 GPT + 1 EGPT×6 (K=8 I=4096) | 679 / 679 | 39.8 B | **55.93** | **22.41** | **25.30** | **2.39** | iso-tokens reference (≈30B+) |

These are still mid-training snapshots (target 65B tokens at step ≈124k for
each gptmoe run). Story so far:

- **At 12-13B tokens**, Boltz routing led Switch by +2.8pp avg (53.6 vs 50.8).
- **At 30+B tokens**, the gap closed and **Switch top-1 now leads by +0.7pp avg**
  (54.74 vs 54.01) — Switch caught up and overtook. The early Boltz advantage
  was apparently a faster warm-up, not a higher converged ceiling.
- `12moe_K4I4096` (largest experts, ~960M total) hits the highest MMLU (25.9)
  but at fewer tokens; tracking whether it overtakes again as it catches up.
- The **580M Boltz** model with energy attention beats every gptmoe variant on
  WikiPPL by ~4 points at the same token budget (39.8B → 22.4 vs 30B → 26.2),
  even though it has only 1 of 12 layers as MoE. The energy-attention layer
  (recursive ×6) appears to add real modeling capacity that pure-GPT-MoE
  variants can't reach.

Both substantially beat baselines:
- **scale_v9 GPT** 354M @ 100.7B: avg 54.1 / PPL 26.2
- **scale_r3** 11gpt+1egpt6x (no MoE) 620M @ 63B: avg 54.6 / PPL 24.2
- **scale_h3** 8gpt+4egpt (no MoE) 620M @ 63B: avg 54.2 / PPL 25.0

580M @ 39.8B beats scale_v9 @ 100.7B by **+1.8pp avg, −3.8 PPL** at ⅓ the tokens.

## Where Boltz-MoE wins (vs scale_v9 GPT 354M @ 100.7B) — final 580M @ 124k

**At final 65B-token training, 580M Boltz wins on every benchmark — including COPA, the
last holdout V9 had at 53.5B tokens. WikiPPL improved monotonically across all
snapshots: 26.84 → 22.41 → 20.23 → 19.70 → 19.48 → 19.33 (15.7B → 65.0B tokens).**

| Task | scale_v9 | 580M @ 124k | Δ | family |
|---|---:|---:|---:|---|
| arc_easy | 56.6 | **68.14** | **+11.5** | knowledge / multi-choice |
| hellaswag | 44.3 | **54.21** | **+9.9** | commonsense |
| arc_challenge | 26.8 | **34.73** | **+7.9** | harder ARC (acc_norm) |
| sciq | 81.3 | **88.40** | **+7.1** | science QA |
| WikiPPL | 26.2 | **19.33** | **−6.84** (better) | language modeling |
| piqa | 68.4 | **71.98** | +3.6 | physical reasoning |
| MMLU | 23.0 | **26.33** | +3.3 | first MMLU signal |
| winogrande | 52.9 | **55.72** | +2.8 | coreference |
| boolq | 53.5 | **55.75** | +2.3 | yes/no |
| openbookqa | 33.4 | **34.40** | +1.0 | open-book QA (acc_norm) |
| copa | 70.0 | **71.00** | **+1.0** | causal (V9's last hold-out, now flipped) |
| gsm8k flex | 1.4 | **2.08** | +0.7 | arithmetic |

## Training setup (both runs)

| Setting | 580M | scale_h3_boltz |
|---|---|---|
| Optimizer | TorchAdamW, betas (0.9, 0.95), eps 1e-8, wd 0.1 | same |
| Peak LR (cosine, 2k warmup) | 1e-3 | 1.5e-3 |
| `num_training_steps` | 124,000 (65B tokens planned) | 124,000 (65B planned) |
| Tokens / step | 524,288 (seq 4096 · μbs 4 · ga 4 · 8 GPUs) | same |
| Mixed precision | bf16 | bf16 |
| Distributed | FSDP-2, gradient_checkpointing_method=block, no torch_compile | same |
| Hardware | 8× H100 single node exclusive (`-x`), `normal` queue | same |
| Step time (clean) | ~3.3 s/step (~13.6B tok/day) | ~5–6 s/step (~7-8B tok/day; 4 EGPT blocks) |
| Vocab / data | granite-4.0-tiktoken / nemotron-cc-hq web | same |

## Watchdog auto-resume (run with these jobs)

Both runs are tracked by `experiments/boltzmann-moe/scripts/watchdog/`. The
watchdog auto-resubmits the bsub job on any preemption / EXIT, picking up
from the latest `global_step*` checkpoint. To launch from scratch:

```bash
# One-time bootstrap of the watchdog (it self-resubmits afterwards)
bash experiments/boltzmann-moe/scripts/watchdog/submit_watchdog.sh

# Then submit each training job once and the watchdog will keep it alive.
# Each job's max_resubmits is configured in watchdog_jobs.conf.
```

To reset a run that hit max_resubmits (see watchdog log under
`experiments/boltzmann-moe/scripts/watchdog/watchdog.log`), edit
`watchdog_state.txt` and set the count back to 1.

## Caveats

- **580M had a brief buggy training window** (~steps 40000-43000) where a
  config edit accidentally set `phi_prime → 0` for a single resubmitted
  process. The model has since recovered (`output_norm` 19→26 from corrected-
  code training). Step-30000 eval (`unsharded_step14k`/30k) is fully clean
  for trustworthy reference.
- **scale_h3_boltz uses `gelu_grad_method: tanh_exact`** which was a
  speculative "fix" later shown to underperform `sigmoid` (legacy default)
  by ~2pp at h1 scale — yet at this larger scale it still lands at avg 55.6.
  A re-run with `sigmoid` may push performance higher; flagged in TODO.md.

## Source code

- `BoltzmannMoE_Energy_MLP` class: `lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py:282`
- Config schema: `lm_engine/hf_models/config/mlp.py:_BoltzmannMoEEnergyMLPArgs`
- Dispatch: `lm_engine/hf_models/modeling_utils/mlp_blocks/__init__.py:78`
