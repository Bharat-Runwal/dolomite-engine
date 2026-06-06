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
| **580M @ step 110k** | 57.7B | 57.59 | **19.70** | 26.17 | 1.86 | new WikiPPL low; avg flat (boolq fluctuated 60→55) |
| **580M @ step 102k** | 53.5B | **58.14** | 20.23 | 26.09 | 2.31 | best avg in 580M progression |
| 580M @ step 76k | 39.8B | 55.93 | 22.41 | 25.30 | 2.39 | prior champion |
| 580M @ step 30k (pre-bug, clean) | 15.7B | 53.66 | 26.84 | 25.42 | 1.90 | trustworthy clean snapshot |
| **scale_h3_boltz @ step 120k** | 62.9B | **56.94** | **21.89** | 26.11 | 1.74 | iso-params no-recursion sibling |
| scale_h3_boltz @ step 104k | 54.5B | 55.63 | 22.67 | 25.03 | 2.39 | prior |

### Pure-GPT + Switch MoE comparison set (in-progress, ~12B tokens each)

Matched-arch ablation vs `scale_h3_boltz`: same d=1280, 12 layers, but pure GPT
attention everywhere and MoE in the FFN slot using either Switch-style learned
top-1 or our Boltzmann-style top-K. Goal is to isolate the contribution of the
energy-attention layers separately from the routing scheme.

| Run | Config | Tokens | Avg | WikiPPL | MMLU | GSM8k flex-avg | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| `scale_gptmoe_8gpt_4switchmoe_d1280` @ 24k | 8 GPT + 4 Switch-MoE FFN | 12.6B | 50.81 | 30.48 | 23.27 | 2.46 | classical Switch top-1 |
| `scale_gptmoe_12moe_K4I2048_d1280` @ 26k | 12 GPT + Boltz-MoE FFN K=4 I=2048 | 13.6B | **53.57** | **30.68** | **26.08** | 2.35 | leads at iso-tokens |
| `scale_gptmoe_12moe_K4I4096_d1280` @ 14k | 12 GPT + Boltz-MoE FFN K=4 I=4096 | 7.3B | 52.99 | 31.30 | 23.12 | 2.01 | larger experts; half the tokens, ≈ matches I=2048 |

These are early snapshots (target 65B tokens at step ≈124k). Currently
`12moe_K4I2048` leads at iso-token, suggesting Boltzmann routing > Switch top-1
when controlling for FFN capacity. `12moe_K4I4096` shows the strongest per-token
trajectory (matches I=2048 at half the tokens). Compare against
`scale_h3_boltz` (8gpt + 4 energy-attn) — the gap will say whether
energy-attention itself adds capacity or just the routing does.

Both substantially beat baselines:
- **scale_v9 GPT** 354M @ 100.7B: avg 54.1 / PPL 26.2
- **scale_r3** 11gpt+1egpt6x (no MoE) 620M @ 63B: avg 54.6 / PPL 24.2
- **scale_h3** 8gpt+4egpt (no MoE) 620M @ 63B: avg 54.2 / PPL 25.0

580M @ 39.8B beats scale_v9 @ 100.7B by **+1.8pp avg, −3.8 PPL** at ⅓ the tokens.

## Where Boltz-MoE wins (vs scale_v9 GPT 354M @ 100.7B)

| Task | scale_v9 | 580M @ 76k | Δ | family |
|---|---:|---:|---:|---|
| arc_easy | 56.6 | **62.4** | +5.8 | knowledge / multi-choice |
| arc_challenge | 26.8 | **32.3** | +5.5 | harder ARC |
| hellaswag | 44.3 | **49.8** | +5.5 | commonsense |
| WikiPPL | 26.2 | **22.4** | −3.8 (better) | language modeling |
| piqa | 68.4 | 70.9 | +2.5 | physical reasoning |
| MMLU | 23.0 | 25.3 | +2.3 | both ≈ random; slight edge |
| boolq | **53.5** | 50.4 | −3.1 | yes/no — V9 wins |
| copa | **70.0** | 67.0 | −3.0 | causal — V9 wins |

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
