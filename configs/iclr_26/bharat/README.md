# For Bharat — a 1B pair that actually activates like a 1B

Two configs, and **they are an experiment only as a pair**: identical datamix, budget, tokens/step,
schedule, width and depth. Run both or neither.

| config | GPUs | total | ACTIVE | FLOPwt | structure |
|---|---|---|---|---|---|
| `bharat_1B_8S4E_boltz.yml` | 8 | **998.7M** | **364.0M** | 364.0M | 8x (softmax + Switch-MoE) then 4x (energy-attn + **BoltzmannMoE**) |
| `bharat_1B_12S_switch_baseline.yml` | 8 | **998.1M** | **364.7M** | 364.7M | 12x (softmax + Switch-MoE), energy-free |

Matched to **0.06% on total and 0.19% on active**. 524,288 tok/step x 61,035 steps = **32.0B**.
Both meta-build and agree with `audit_config` to the byte.

## Why not just re-run `s8e4_stdmoe_fh2_boltz_topk2.yml`

These are **not** derived from that file. They are built from our `cmix1B` skeleton plus the energy
block of `configs/cmix/cmix_400M_hybrid_sparse.yml` — the 400M arm we have actually trained to 32B —
so every routing knob is one with measurements behind it: hopfield experts, `e_sign_override: pos`,
`sinkhorn_iters: 3`, `sparse_forward` with the rank-16 subspace proxy, `renormalize_topk: true`,
output-space repulsion with subsample 64. Three defects in the upstream file are documented in
`experiments/boltzmann-moe/CLAUDE.md` (a `proj_mode` spelling that does not exist in this tree and is
a silent no-op; a header claiming `energy_scale_mode` that the body never sets; and a 1.1B parameter
count that is really ~793M once hopfield's one-matrix-per-expert is accounted for).

## The one number that motivates all of this

**Our existing 1B activates like a 400M model, which is why it performs like one.**

| | total | ACTIVE | bank total | bank ACTIVE | Avg11 | ppl |
|---|---|---|---|---|---|---|
| `cmix1B_12L_gptDense_32B` (K=64, k=2) | 1002.1M | 279.3M | 746.1M | **23.3M (3.1%)** | 47.79 | 29.46 |
| `abl_B_400M_6G1x6S` (FLOP-matched Switch) | 400.0M | 219.7M | 192.4M | 12.0M | **48.24** | **28.27** |
| **`bharat_1B_8S4E_boltz` (K=32, k=8)** | 998.7M | **364.0M** | ~393M | **~98M (25%)** | — | — |

The old 1B stores a 746M expert bank and uses 23M of it per token. Its ACTIVE count is only **+27%**
over the 400M arm, and it **loses** to that arm on every column. K=32/k=8 raises activation to 25% of
the bank and ACTIVE to **1.66x** the 400M arms.

## Choices you may want to overrule

**`K=32, k=8`, not K=8 and not K=64/k=2.** K=8 is too few experts to route among at this width;
K=64/k=2 is the trap above. `I_e = 96000/32 = 3000` sits near the 400M hybrid's proven `I_e=5871`
and well above the 134M arm's `I_e=1024`, which we flagged as too narrow for sparsity to pay.

**There is a hard tension here.** `active_bank = (k/K) x bank`, and `bank` cannot exceed ~700M inside
1B. So high ACTIVE forces high `k/K`, which shrinks the sparsity win: at `p=12 of K=32` the ceiling on
expert-compute saving is `K/p = 2.7x`, against `8x` for the 400M hybrid's `p=4 of 32`. This config
takes the middle. **`k=16` would give 2.0x the 400M active** but forfeits most of the sparsity story —
your call if GPUs allow.

**COSINE, not WSD.** We have no controlled WSD-vs-cosine A/B at 32B; WSD was confirmed only on short
probes. Cosine is standard and is what our whole 134M tier used. What actually matters is that the
**pair shares one schedule**, which it does (`2000 / 0 / 59035`).

**hopfield + sparse(proxy), not w1w2 + surrogate.** Measured at 134M, same budget:
`hopfield+proxy 0.20 s/step` vs `w1w2+surrogate 0.30 s/step` — **the proxy is 1.5x faster**. w1w2 is
nominally better on quality (Avg11 45.01 vs 44.82, ppl 40.70 vs 41.06) but that gap is within noise,
and it costs 50% more wall clock. For a 32B run at 1B, take the speed.

## For a 128B-token run

Multiply `num_training_steps` by 4 → **244,140**, and set `num_decay_steps = 242,140` so
`warmup + constant + decay == num_training_steps`. That check is not optional: published `slope90k_*`
configs once ran 60,000 of 90,000 steps pinned at the LR floor because it was violated.

## Launch

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
export WANDB__SERVICE_WAIT=300
for a in bharat_1B_8S4E_boltz bharat_1B_12S_switch_baseline; do
  bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \
       $a configs/iclr_26/bharat/$a.yml 8 preemptable 24:00 200G 4
done
```

Then verify within 60 s — **step lines are in STDERR**, and `grep -m1 DeviceMesh <stderr>` must NOT
say `'cpu'` (a host with failed CUDA init makes the trainer fall back to a CPU mesh and train
nothing while LSF reports RUN). Sub-60-second deaths on multi-node are transient (~50%): resubmit.
