# 7B Energy-MoE Configs — for review

Six 7B-class configs built from the design discussion. The target structure:

> GPT layers use **standard MoE**; the recurrent (EGPT) layers use **Boltzmann
> MoE**. At 7B, scale up — 40 layers, 128 experts. Try **20 GPT + 4 EGPT×[6,6,9,9]**
> (24 stored layers, 50 effective). Also a no-recurrence variant ("h3"). And a
> baseline that is the **same structure but only RecGPT** — to balance num params
> and FLOPs (24 actual weights / 50 effective layers vs a 40-layer 7B).

All six are validated on transformers 4.57.1: they parse, build at the param
counts below, and pass a forward+backward pass.

---

## Shared design (all 6)

| Setting | Value |
|---|---|
| `model_type` | `energy` |
| `hidden_size` | 1536 (reference 7B width) |
| attention heads | 24 (head_dim 64), no GQA |
| **GPT layers** | `softmax_attention` + **standard MoE** (128 experts, top-8, per-expert `intermediate_size=256`, `shared_intermediate_size=1024`, swiglu) |
| `energy_proj_type` | `dual_unconstrained` — applies only on energy layers |
| data | math mix: 0.35/0.35 nemotron-cc-hq + 0.15 megamath-web-pro + 0.15 finemath-3plus |
| tokenizer | granite-4.0-tiktoken, vocab 100352, tied embeddings |
| optimizer | TorchAdamW, lr 3e-4, wd 0.1, betas (0.9,0.95), cosine, 2k warmup |
| steps | 124000 (placeholder — adjust to token budget) |
| precision / dist | bf16, FSDP-2, gradient_checkpointing=block, stage 0, no torch_compile |
| wandb project | `dolomite` |

`use_interleaved_weights: false` on the standard MoE blocks — the sonicmoe kernel
is **not** enabled in this build, and the non-kernel MoE path requires it to be
false. Enable sonicmoe + flip this to true only in a kernel-ready env.

---

## The 6 configs

Two axes: **recurrent-layer type** (Boltzmann energy / standard-MoE energy / RecGPT)
× **recurrence** (recursive `[6,6,9,9]` / h3 no-recursion).

| # | Config file | Recurrent layers | Recurrence | Stored layers | Eff. depth | **Total** | **Active/token** |
|---|---|---|---|---|---|---:|---:|
| 1 | `s7b_egpt_boltz_rec_20gpt4egpt6699_d1536.yml` | energy_attn + **BoltzmannMoE** | `[1]×20+[6,6,9,9]` | 24 | 50 | **3.90B** | **1.07B** |
| 2 | `s7b_egpt_boltz_h3_32gpt8egpt_d1536.yml` | energy_attn + **BoltzmannMoE** | `[1]×40` (none) | 40 | 40 | **6.33B** | **1.80B** |
| 3 | `s7b_egpt_stdmoe_rec_20gpt4egpt6699_d1536.yml` | energy_attn + **std MoE** | `[1]×20+[6,6,9,9]` | 24 | 50 | **4.12B** | **0.73B** |
| 4 | `s7b_egpt_stdmoe_h3_32gpt8egpt_d1536.yml` | energy_attn + **std MoE** | `[1]×40` (none) | 40 | 40 | **6.77B** | **1.11B** |
| 5 | `s7b_recgpt_rec_20l4rec6699_d1536.yml` | **softmax + std MoE** (no energy) | `[1]×20+[6,6,9,9]` | 24 | 50 | **4.12B** | **0.73B** |
| 6 | `s7b_recgpt_h3_40l_d1536.yml` | **softmax + std MoE** (no energy) | `[1]×40` (none) | 40 | 40 | **6.77B** | **1.11B** |

Notes:
- **Recursion adds no parameters** (weights reused), only FLOPs → recursive configs
  store 24 layers but run 50 effective layers. h3 stores all 40.
- **Boltzmann active = total on the energy layers**: soft routing computes *all* 128
  experts per token (top_k does not reduce FLOPs for `BoltzmannMoE_Energy_MLP`), so
  configs 1–2 have higher active params than the std-MoE / RecGPT counterparts.
- **Configs 3↔5 and 4↔6 are parameter-identical** (energy attn `2d²` + dual proj `2d²`
  = softmax attn `4d²`), so RecGPT is an exact iso-param / iso-FLOP-depth control.

---

## How the configs form the ablation

Two recurrence families (recursive depth-50, and h3 depth-40), each a 3-rung ladder:

| Rung | Recursive (24L/eff50) | h3 (40L) | Isolates |
|---|---|---|---|
| **RecGPT baseline** | #5 | #6 | recursion + MoE only — *no energy* |
| **+ energy attention** (std MoE) | #3 | #4 | the energy attention / dual-proj update |
| **+ Boltzmann routing** | #1 | #2 | energy attention **and** Boltzmann MoE |

Reading a column top-to-bottom isolates each added ingredient at matched params/FLOPs.
The RecGPT rung is the "same structure, only RecGPT" baseline.

---

## How to run

Each config is launched with `launch-scripts/pretrain.sh` under `bsub`/`blaunch`.
The launcher activates `.venv-boltz` (transformers 4.57.1) — override with
`PRETRAIN_VENV=/path/to/venv` for a different env (e.g. a kernel-ready 7B env).

Single config (4 nodes × 8 GPUs = 32 GPUs):

```bash
CFG=configs/boltzmann_moe/s7b_egpt_boltz_h3_32gpt8egpt_d1536.yml
NAME=$(basename "$CFG" .yml)
LOG_DIR=/proj/dmfexp/energy-gpt/logs/boltzmann_sweep
mkdir -p "$LOG_DIR"

bsub -r -q preemptable -G grp_preemptable -M 2000G -hl -n 4 \
  -J "$NAME" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "$LOG_DIR/$NAME.out" -eo "$LOG_DIR/$NAME.err" \
  blaunch bash launch-scripts/pretrain.sh "$CFG"
```

Or submit with the helper script:

```bash
bash submit_7b_configs.sh                                   # all 6
bash submit_7b_configs.sh s7b_egpt_boltz_h3_32gpt8egpt_d1536  # a subset
```

Adjust node count to the available cluster. Token budget per step =
`num_nodes × 8 × micro_batch_size(4) × grad_accum(4) × seq_len(4096)`; tune
`num_training_steps` (and `num_decay_steps = steps − 2000`) to the target tokens.

### Throughput caveat
The energy MoE (`BoltzmannMoE_Energy_MLP`) and the energy attention path are pure
PyTorch — **no sonicmoe / scattermoe kernel support yet under FSDP-2**. Configs run
correctly but throughput will be well below a kernel-backed standard-MoE baseline,
so wall-clock timelines are hard to predict. Correctness is the goal of these
configs; performance tuning comes after kernels land.

---

## Provenance

Generated by `tools/bs/gen_7b_energy_configs.py` (YAML anchors keep the 24/40-layer
lists exact). Param counts verified by building each model on meta device with
transformers 4.57.1.
