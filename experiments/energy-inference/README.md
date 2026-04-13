# Energy-Based Inference — Large-Scale Experiments

Scaling up energy-GPT inference dynamics experiments from small (10–100M) models in
`GPT-experiments/projects/energy-inference/` to 400M+ models trained with dolomite-engine.

## Background

See `GPT-experiments/projects/energy-inference/README.md` for the full theoretical background.
The short version:

- **Energy Transformer (EGPT)** recurrence: `x_{s+1} = x_s - η · ∇E(x_s)` where `η` (proj) is a
  learnable matrix.
- `η = η_sym + η_anti` decomposes into gradient (symmetric) and curl (anti-symmetric) dynamics.
- At small scale, RL on the explore/exploit policy α improves reasoning benchmarks.

## Goals

1. **Verify scale transfer**: Do alpha/RL/split-proj gains observed at 10–100M transfer to 400M?
2. **Inference-only changes**: Load a frozen 400M checkpoint and only modify the inference
   strategy (α schedule, Langevin noise, multi-walker) — no retraining required.
3. **Fine-tuning with RL**: Graft a small policy head onto a frozen 400M backbone and RL-tune
   just the energy dynamics (as in `train_graft_curriculum_proj_20260408.py` at small scale).
4. **Split-proj at scale**: Compare single vs. split projection variants on 400M models.

## Checkpoints

Pre-trained 400M energy-GPT checkpoints (dolomite FSDP format + unsharded HF):
```
/proj/dmfexp/energy-gpt/checkpoints-bsaha/egpt_400m/
/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/
```

## Directory Structure

```
experiments/energy-inference/
├── CLAUDE.md          # Project instructions for Claude
├── README.md          # This file
├── TODO.md            # Task list
├── PLANS.md           # Experiment plans (reverse chronological)
├── RESULTS.md         # Results (reverse chronological)
├── configs/           # YAML configs for dolomite-engine
├── scripts/           # Dated experiment scripts
└── results/           # Saved metrics, pkl files, JSON summaries
```

## Quick Start

```bash
# Interactive GPU session
bsub -Is -n 1 -gpu "num=2/task:mode=exclusive_process" /bin/bash

# Load an unsharded checkpoint for inference
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/<name>"
)

# Run a pretrain/finetune job
bsub ... < scripts/energy_inference/run_alpha_sweep.sh
```
