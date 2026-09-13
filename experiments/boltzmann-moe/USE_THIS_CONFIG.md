# Sparse Boltzmann-MoE — which config to use (2026-09-13)

## Start from this one

```
configs/iclr_flops/iclr_hop_K32_top2.yml
```

**Best sparse energy arm measured**: avg10_norm **46.26**, WikiPPL 40.59, at `k/K = 0.062`.
Trained 30k steps (7.86B tokens), d=768, 7 blocks, `layer_iterations [1,1,1,1,1,1,6]`.

Two alternatives, same family:
- `iclr_hop_K32_top1.yml` — `k/K = 0.031`, avg10 **46.15**. Half the arithmetic of top-2 for
  −0.11pp. Take this if inference cost dominates.
- `iclr_hop_K16_top2.yml` — `k/K = 0.125`, avg10 **45.89**. Only if you need K=16.

Do **not** start from `iclr_hop_K16_dense.yml` (46.58). It is dense-routed, so it forfeits
the sparsity the method is for, and a standard Switch MoE beats it anyway (see below).

## The settings that actually matter

```yaml
- mlp_type: EnergyFF_BoltzmannMoE
  expert_kind: hopfield          # 1 matrix/expert; router cost is reused by the experts
  intermediate_size: 16384       # TOTAL across K, so I_e = 512 at K=32
  n_experts: 32
  top_k: 2
  hopfield_grad_scale: sqrt_consistent   # REQUIRED. "mean" is the old 4/I_e prefactor.
  routing_norm: zscore           # REQUIRED. Without it routing is ~uniform (see below).
  temperature: 0.35
  repulsion_form: abs            # REQUIRED. "signed" rewards anti-alignment (old bug).
  repulsion_coef: 0.1
  n_repulsion_pairs: 4
  gelu_grad_method: sigmoid
  activation_function: gelu
```

Three of those are fixes, not preferences, and each is worth real accuracy:

- **`routing_norm: zscore`** — the Hopfield energy is a mean over `I_e` units, so `E ~ 1e-2`
  against `tau`, and raw softmax gives *uniform* routing (measured: effective 7.999 of 8
  experts on the shipped 33B checkpoint, i.e. top-k is pure loss rather than a trade-off).
  z-scoring the logits across experts makes routing scale-free and informative.
- **`repulsion_form: abs`** — the original `signed` form is *minimised* at cosine $-1$, so it
  rewarded anti-aligned experts instead of merely diverse ones. Worth **+0.69pp / −1.83 ppl**
  (`iclr_hop_K16_top2` 45.89 vs `..._nofix` 45.20).
- **`hopfield_grad_scale: sqrt_consistent`** — output prefactor `4/sqrt(I_e)` rather than
  `4/I_e`. Revert to `mean` only if training destabilises.

## New knobs, all default-off and safe to ignore

| knob | default | what it does |
|---|---|---|
| `track_load` | `true` | logs `load_effective_n_experts`, `load_max_share` **every step, under `torch_compile`**. The older metric path was traced away, so routing health was previously invisible. Leave on; it is nearly free. |
| `balance_rate` | `0.0` | >0 enables aux-loss-free per-expert logit-bias balancing (DeepSeek-V3 style; no loss term, no gate params). **We have not shown it is needed** — routing does not collapse on real data. |
| `renormalize_topk` | `false` | makes the masked top-k weights sum to 1, as the baselines do. Ours sum to ~0.45 at K=16,k=2. Changes output scale, so do not switch it on mid-run. |

**Do not set `balance_rate` on a run resuming from an existing checkpoint** unless you start
fresh: it adds a persistent buffer, i.e. a state_dict key, and a checkpoint written without
it will fail strict load.

## Know this before you invest in the method

At **matched parameters** (134.90M vs 134.49M) and **matched sparsity** (`k/K = 0.125`), a
conventional Switch SwiGLU MoE in the same slot scores **46.72** — better than every energy
arm, including the dense-routed one (46.58), while routing 8x sparser. At 400M in the purely
recurrent configuration the learned-router gap widens (train `lm_loss` 3.4903 vs 3.5800).

What does hold up: energy routing reaches parity with a learned router **over the same
energy-gradient experts** (45.69 vs 45.66) with **no gate parameters and no load-balancing
loss**; routing does **not** degenerate (effective expert count 12.3–12.8 of 16 on real
hidden states); and quality is remarkably flat across a 32x sparsity range.

Full context: `STATUS_20260912.md` (append-only, newest at the bottom; the restart banner at
the top has the current state) and `HANDOFF.md`.
