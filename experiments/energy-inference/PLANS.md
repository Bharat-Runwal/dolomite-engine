# Plans — Energy-Inference Large-Scale

<!-- Plans go at the top (reverse chronological). Annotate with outcome after running. -->

## Plan: MAB trajectory diversity + Langevin noise training (2026-04-12)

### Motivation

Alpha sweeps showed the trained model has an extremely narrow working window (α ∈ [0.45, 0.58]).
Hypothesis: the energy landscape co-adapted to one fixed trajectory; it is not robust to
alternative traversal strategies. The structured-proj retraining with uniform alpha noise
(current experiment) is a first step, but two more principled extensions:

### Multi-armed bandit over trajectory profiles

Instead of sampling α uniformly during training, treat each trajectory profile (or α value)
as an arm and learn a distribution over arms that maximises validation loss improvement.

- **Arms**: discretised α values or per-layer profiles (e.g. the 7 profiles from the per-layer sweep)
- **Reward**: negative validation perplexity on a held-out batch after one gradient step
- **Algorithm**: UCB or Thompson sampling; update arm weights every K steps
- **Why**: steers training toward trajectories that are hardest (most informative), rather
  than wasting capacity on trajectories that are already handled well

Implementation:
- Maintain a weight vector `w[i]` over arms; sample arm each step via softmax(w)
- After each step, compute reward = -val_loss; update w[arm] += lr_bandit * reward
- Log arm distribution to wandb to visualise which trajectories drive learning

### Langevin noise at recurrence steps

Add `σ·ε` (ε ~ N(0, I)) to hidden states at each recurrence step during training:
    `h_{t+1} = h_t + (J - R) @ grad_E(h_t) + σ·ε`

- **Why**: Langevin dynamics = gradient descent + noise. Noise prevents the recurrence
  from collapsing to a single sharp attractor; it regularises the energy landscape so
  the basin is broader (wider α window at inference time).
- **σ schedule**: anneal from σ_max → 0 over training (start noisy, finish clean). Or
  fixed small σ throughout — empirically determine which works better.
- **Connection to temperature**: σ² / 2 = kT in statistical mechanics. High T = broad
  exploration; low T = sharp convergence. Training with noise teaches the landscape to
  be well-behaved across temperatures.

Implementation: during forward, after each recurrence iteration in `EnergyBlock.forward`,
add `torch.randn_like(hidden_states) * sigma` when `self.training and sigma > 0`.

### Combined approach

1. Structured proj (J-R split) so dynamics are physically interpretable
2. Alpha noise (uniform) as baseline exploration — current experiment
3. → MAB replaces uniform alpha: smarter curriculum over trajectory strategies
4. → Langevin noise on recurrence: landscape regularisation

### Scripts created
- `scripts/mab-langevin/train_mab_proj_20260413.py` — MAB + Langevin version (done)
- `scripts/mab-langevin/run_mab_proj.sh` — MAB + Langevin bsub submission
- `scripts/mab-langevin/run_mab_only.sh` — MAB-only ablation (no Langevin)

### Warm-start init bug fix (2026-04-13)
The J init was missing singular values — used heuristic `scale` instead of `sqrt(S/2)`.
The correct factoring for `J = UV^T - VU^T = -W_anti` with SVD `W_anti = U_svd Σ Vh`:
```
U = -U_svd[:,:k] * sqrt(S[:k] / 2)   # (d, k)
V =  Vh_svd[:k].T  * sqrt(S[:k] / 2)  # (d, k)
```
Proof: UV^T = -S/2 * U_svd Vh, VU^T = -S/2 * Vh^T U_svd^T = +S/2 * W_anti/S = W_anti/2
→ J = -W_anti/2 - W_anti/2 = -W_anti ✓.
Fixed in both `train_structured_proj_20260412.py` and `train_mab_proj_20260413.py`.
Job 875085 resubmitted with fix.

**Status**: Scripts complete 2026-04-13. Job 875085 (warm-start v3, fixed init) running.

---

## Plan: Initial Port — Alpha Sweep + Multi-Walker Inference on 400M (2026-04-11)

### Motivation
Small-scale experiments (10–100M, `GPT-experiments/projects/energy-inference/`) show:
- RL-steered α (curl/grad mixing) improves reasoning benchmarks over fixed GD
- Multi-walker Langevin at inference improves best-of-N accuracy
- Split-proj variants affect energy landscape quality

The question: do these gains transfer to 400M models that were already pretrained with
dolomite-engine on large-scale data (DCLM)?

### Phase 1: Inference-only (no retraining, ~1 GPU-hour)

**Goal**: Change only the inference dynamics of a frozen 400M checkpoint; measure MMLU/BBH.

| Variant | α schedule | Walkers | Notes |
|---------|-----------|---------|-------|
| baseline | α=1.0 (pure GD) | 1 | Current inference |
| all_curl | α=0.0 | 1 | Pure rotation |
| curl_to_grad | 0→1 linear | 1 | Explore then exploit |
| multi_walker_4 | α=1.0 | 4 | Best-of-4, Langevin T=0.01 |
| multi_walker_8 | α=1.0 | 8 | Best-of-8 |
| multi_walker_free | α=curl→grad | 8 | Full pipeline |

Checkpoint: `400m_s1_e6_e6_e6_e6_e6_e6_s1_lr3e4_30k` (best LR on paren at small scale)

**Success**: Any inference variant outperforms baseline on ≥1 benchmark without retraining.

### Phase 2: RL grafting (retraining, ~8 GPU-hours)

Port `train_graft_curriculum_proj_20260408.py` logic:
- Freeze backbone (all GPT layers)
- Train only: `proj` matrices + `scale_ff` scalars in energy blocks
- RL reward: correctness on reasoning task samples
- Budget: 5K steps, LR 3e-4

**Success**: Grafted RL model > frozen baseline on BBH CoT subset.

### Expected FLOPs / wall time
- Phase 1: ~1h on 1×A100 (inference only, no gradient)
- Phase 2: ~8h on 2×A100 (RL with small batch, frozen backbone)

### Script names (to be created)
- `scripts/eval_alpha_inference_20260411.py`
- `scripts/train_rl_graft_20260411.py`

**Status**: Planned 2026-04-11.
