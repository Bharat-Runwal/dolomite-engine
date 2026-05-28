# Boltzmann TopK Energy MoE — head-to-head with vanilla switch-routed MoE

This directory contains two configs designed to be trained head-to-head.

| Config | Architecture | Active / Total |
|---|---|---|
| `s8e4_stdmoe_fh1_topk2.yml` | First-8: softmax + std MoE. Last-4: energy-attention + **fh1** (TopK_Energy_MoE_MLP). No looping. | ~387M / ~1101M |
| `stdmoe_only_topk2.yml`     | All 12 layers: softmax + std MoE.                                                              | ~389M / ~1097M |

Both are 12 layers, `d=1024`, 16 heads × 64 head_dim, 8 experts top-2, trained 30k steps with 1.05M tok/step.

## What is `fh1` (TopK_Energy_MoE_MLP)?

Code: `lm_engine/hf_models/modeling_utils/mlp_blocks/topk_energy_moe.py`.

The fh1 block is a sparse MoE where each expert is a **full-size Energy MLP** (intermediate_size is *per-expert*, not divided across experts) and the output is the **energy gradient** of a Boltzmann-style scalar energy, computed analytically from `W1` and `W2`.

For input `x`, with `phi = GELU(W1 x)` and `phi' = sigmoid(sqrt(2/pi) · W1 x) / 2`:

```
expert_grad = phi · W2_e^T  +  (phi' ⊙ W2 x) · W1_e^T
```

A linear router picks the **top-k** experts per token, their `expert_grad` outputs are weighted by softmax over the selected logits, and combined. The block uses only a switch-style **load-balance** auxiliary loss (no z-loss, no KL distillation, no expert repulsion).

Why this can beat vanilla MoE:
- Experts share gradient structure of a coherent energy function — encourages coordinated specialization rather than orthogonal "feature shards".
- Each expert is the full-width MLP, so an active expert has the full representational capacity of the dense baseline.
- Routing weights modulate energy contributions, not arbitrary outputs — gradient signal to the router is more meaningful.

## Empirical result at 400M active / 30k steps

`s8e4_stdmoe_fh1_topk2` (no looping) beats matched `stdmoe_only_topk2` on every headline metric:

| Metric | `stdmoe_only_topk2` | `s8e4_stdmoe_fh1_topk2` | Δ |
|---|---|---|---|
| 12-task Avg ↑ | 0.5086 | **0.5120** | +0.0034 |
| MMLU 5-shot ↑ | 0.2708 | **0.2782** | +0.0074 |
| gsm8k_cot 5-shot ↑ | 0.0508 | **0.0546** | +0.0038 |
| Wiki word PPL ↓ | 27.52 | **27.37** | −0.14 |

Across 18 logged metrics fh1 wins 12, std-MoE wins 6 (mostly Lambada-style narrative tasks).

## How to submit

```bash
# Two-config head-to-head, both on 8 nodes × 8 H100s
bash submit_boltzmann_sweep.sh
```

Equivalent direct `bsub` for either config:

```bash
LOG_DIR=/proj/dmfexp/energy-gpt/logs/boltzmann_sweep
mkdir -p "$LOG_DIR"

# fh1 hybrid
bsub -r -q preemptable -G grp_preemptable -M 2000G -hl -n 8 \
  -J "s8e4_stdmoe_fh1_topk2" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/s8e4_stdmoe_fh1_topk2.out" \
  -eo "${LOG_DIR}/s8e4_stdmoe_fh1_topk2.err" \
  blaunch bash launch-scripts/pretrain.sh \
    configs/boltzmann_moe/s8e4_stdmoe_fh1_topk2.yml

# matched stdmoe-only baseline
bsub -r -q preemptable -G grp_preemptable -M 2000G -hl -n 8 \
  -J "stdmoe_only_topk2" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/stdmoe_only_topk2.out" \
  -eo "${LOG_DIR}/stdmoe_only_topk2.err" \
  blaunch bash launch-scripts/pretrain.sh \
    configs/boltzmann_moe/stdmoe_only_topk2.yml

# Watch progress
bjobs -w | grep -E 's8e4_stdmoe_fh1_topk2|stdmoe_only_topk2'
tail -f $LOG_DIR/s8e4_stdmoe_fh1_topk2.err
```

## Scaling to 9B active

To run these two configs at 9B active / larger token budget:

1. **Resize the model** — increase `hidden_size`, `num_layers`, `num_attention_heads`, and the per-expert `intermediate_size` so total active stays at the target (e.g. d=4096, 32+ layers, top-2 of 8 experts at proportional ipe). Keep the layer split (8 softmax + 4 energy-attention) and the routing config (`n_experts: 8, top_k: 2`) identical between the two YAMLs so the comparison stays apples-to-apples.
2. **Resize training** — bump `num_training_steps` (and `num_decay_steps`) so each model sees the same token budget; tune `lr` if needed (smaller models tolerate `2e-3`, 9B usually wants `~5e-4` to `~1e-3`).
3. **Resize batch** — adjust `micro_batch_size`, `gradient_accumulation_steps`, and `NODES` in `submit_boltzmann_sweep.sh` for the new memory footprint. Keep effective batch (mb × ga × world_size) roughly constant or larger to maintain stable optimization.
4. The fh1 block is dispatched by `mlp_type: TopK_Energy_MoE_MLP`; no other code changes needed.
