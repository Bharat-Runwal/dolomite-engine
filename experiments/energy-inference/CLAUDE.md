# Energy-Based Inference Experiments — dolomite-engine

## Overview

Porting energy-inference experiments from `GPT-experiments/projects/energy-inference/` to larger
models (400M+) trained with dolomite-engine. Key ideas being scaled up:

- **Alpha / curl-grad mixing**: Decompose the proj matrix into symmetric (gradient) and
  anti-symmetric (curl) parts; sweep the mixing coefficient α across recurrence steps.
- **RL-steered inference**: Use REINFORCE to learn a per-step policy (noise level, α, step scale)
  that maximises correctness reward at the end of the recurrence.
- **Split-proj variants**: Compare single-proj vs split-proj energy blocks at scale.
- **Free-energy inference**: Multi-walker Langevin dynamics with annealing at test time.

## Checkpoints

Pre-trained 400M/410M energy-GPT checkpoints are in:
```
/proj/dmfexp/energy-gpt/checkpoints-bsaha/egpt_400m/
/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/
```

Key checkpoints:
| Name | Type | Steps |
|------|------|-------|
| `400m_s1_e6_e6_e6_e6_e6_e6_s1_lr3e4_30k` | Energy (s1+6×EE+s1) | 30k |
| `400m_par_recgpt_s1_s6_s6_s6_s6_s6_s6_s1_lr3e4_30k` | Par-RecGPT | 30k |
| `410m_hybrid_s8e4_lr1e3_60k` | Hybrid (8 soft + 4 energy) | 60k |
| `E9F_GaussianBoltz_32x1024_6iter_slim_nima_cc_lr3e-4_30k` | EnergyMoE 9-layer | 30k |

Unsharded (ready to load without special tools):
```
/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/
```

## Key dolomite-engine concepts

- Training entry: `python -m lm_engine.pretrain --config <yml>` (wrapped by `scripts/common/pretrain.sh`)
- Config format: YAML — see `configs/pretraining-examples/local/pretrain-1.yml`
- model_type `energy` with `energy_attention` blocks is the EGPT variant
- `num_iterations` in config = recurrence depth (L in paper)
- `GaussianBoltzmannMoE` mlp_type = energy MoE variant used in bsaha experiments
- Checkpoints use FSDP and need unsharding for inference: `tools/pt_to_safetensors.py`

## Running on IBM cluster (bsub)

See `bsub_scripts/` at repo root for interactive session and job submission helpers.

Quick interactive session:
```bash
bsub -Is -n 1 -gpu "num=2/task:mode=exclusive_process" /bin/bash
```

Batch job:
```bash
bsub -q standard -J energy_exp -gpu "num=8/task:mode=exclusive_process" \
     -n 1 -M 16G -W 04:00 \
     -o $HOME/%J.stdout -e $HOME/%J.stderr \
     < scripts/common/pretrain.sh
```

Python helper: `bsub_scripts/mybsub.py` (uses `tyro`).

## Conventions

- Configs for this project: `experiments/energy-inference/configs/`
- Scripts: `experiments/energy-inference/scripts/` (date-stamped, e.g. `train_alpha_sweep_20260411.py`)
- Results: `experiments/energy-inference/results/`
- Wandb project: `energy-inference-large`
- Never modify existing scripts — create new dated copies.
- Use unsharded checkpoints when loading for inference/grafting.
