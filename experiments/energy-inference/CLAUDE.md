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

- Configs for this project: `experiments/energy-inference/configs/`; multi-block ablation configs: `configs/multi_block_ablation/`
- Scripts: `experiments/energy-inference/scripts/` (date-stamped, e.g. `train_alpha_sweep_20260411.py`)
- Results: `experiments/energy-inference/results/`
- Wandb project: `energy-inference-large`
- Tokenizer path: `/proj/datasets/tokenizers/granite-4.0-tiktoken` (NOT `/proj/checkpoints/dmf-lh-checkpoints/...` which is stale)
- Never modify existing scripts — create new dated copies.
- Use unsharded checkpoints when loading for inference/grafting.

## Figure and paper rules

1. **Always save figures as both PDF and PNG** — every `fig.savefig()` call must save both extensions
   (e.g. `for ext in ["png", "pdf"]: fig.savefig(plots_dir / f"{stem}.{ext}", ...)`).
2. **Always update the paper and recompile after new results** — after any new eval results or plots,
   update `experiments/energy-inference/paper/main.tex` with the new numbers/figures and run
   `pdflatex -interaction=nonstopmode main.tex` from the `paper/` directory.
3. **Always include every new plot in the paper** — after producing any new analysis figure, add a
   `\includegraphics` block for it in `main.tex` and recompile. Never leave a plot that exists on disk
   but is missing from the paper.

## Mandatory for every training run

1. **Always log to wandb** — set `experiments_tracker_name: wandb` and `project: energy-inference-large` in every config.
2. **Record the full run command/script in RESULTS.md** — paste the full bsub command or script name, config path, and key hyperparams.
3. **Estimate training time before submitting** — compute: `tokens = steps × num_gpus × micro_batch × grad_accum × seq_len`. Use ~1 TFLOP/token/GPU × GPU TFLOPS × 0.4 efficiency for wall-clock estimate.
4. **Save checkpoints every 5000 steps** (or 2000 for short runs) — set `save_interval` in config.
5. **Use auto-resume scripts** (see `scripts/multi-block-ablation/run_v*.sh` as template) so preempted jobs continue automatically.

## Preemptable queue

The cluster has a `preemptable` queue for lower-priority long-running jobs. Jobs can be
killed at any time when higher-priority jobs need GPUs, but otherwise run indefinitely.

```bash
# Submit to preemptable queue (4 GPUs, 4h wall time, auto-resume on kill)
bsub \
    -q preemptable \
    -G grp_ebm \
    -J job_name \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 \
    -M 64G \
    -W 04:00 \
    -o $HOME/job_name_%J.stdout \
    -e $HOME/job_name_%J.stderr \
    < job_script.sh
```

Key differences from `-q standard`:
- Jobs can be preempted at any time → MUST checkpoint frequently + auto-resubmit
- Typically less wait time than standard for 4 GPUs when queue not crowded
- Can run for days if not preempted (observed: jobs running 6+ days)
- Use wall time `-W 04:00` (4h chunks) to limit blast radius if something goes wrong

Auto-resume pattern (in job script):
```bash
LATEST_JSON="${SAVE_PATH}/latest_checkpointed_iteration.json"
if [ -f "${LATEST_JSON}" ]; then
    # append load_args to config and resume
fi
# At end: check if complete, resubmit if not
```

Check queue status:
```bash
bjobs -q preemptable -u all | head -30   # who is running
bqueues preemptable                       # queue stats (NJOBS, NRUN, NPEND)
```

## CRITICAL: EGPT attention output IS the true energy gradient

**Do not repeat the mistake of saying EGPT uses V=K without W_Q^T.**

`EnergyAttention_QK` (sequence_mixer_type: `energy_attention`, used in V1 EGPT):
1. Sets V = K inside flash attention (efficiency: head dim d_h ≪ d, fewer FLOPs)
2. After attention, applies W_Q^T via `einsum("bhts,hcs->btc", attn_output, W_Q_permuted)`
3. Output = `sum_h W_{Q,h}^T [A_h K_h]_t` = **exact true gradient** ∇_{h_t} E_h

The V=K step is a factored computation trick, NOT an approximation.

**Contrast with V10 MixedHeadAttention:**
- Energy heads compute `A_h K_h` but output goes through **shared W_O** (not W_Q^T)
- Therefore V10 energy heads do NOT compute the true gradient
- This is the actual bottleneck fixed by EGrad (V27+)

**EGrad (EnergyGradMixedHeadAttention):**
- Brings W_Q^T output to energy heads in the **mixed-head** context
- Energy heads: `W_{Q,h}^T A_h K_h` (true gradient); GPT heads: `W_{O,gpt} A_h V_h`
- EGrad = EGPT's correct output rule + mixed-head flexibility (some standard GPT heads)
