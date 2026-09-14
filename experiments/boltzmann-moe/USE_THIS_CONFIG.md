# Sparse Boltzmann-MoE — what to run, and what will bite you

Last updated 2026-09-14. Everything here is measured on a fixed backbone (d=768, 7 blocks,
`layer_iterations [1,1,1,1,1,1,6]`), mixture held at 12.58M params, 30k steps = **7.86B tokens**,
scored as `avg10_norm` (10 tasks, `acc_norm` wherever a task reports one) by a version-pinned
harness. Total params agree to within 0.3% across every arm quoted.

---

## 1. Start from this config

```
configs/iclr_flops/iclr_hop_K32_top2.yml
```

**Best sparse Boltzmann arm measured: avg10 46.26, WikiPPL 40.59, at k/K = 0.062.**

| alternative | k/K | avg10 | when to use it |
|---|---:|---:|---|
| `iclr_hop_K32_top1.yml` | 0.031 | 46.15 | inference cost dominates; −0.11pp for half the arithmetic |
| `iclr_hop_K16_top2.yml` | 0.125 | 45.89 | only if you need K=16 |
| `iclr_hop_K16_dense.yml` | 1.000 | 46.58 | **don't** — forfeits the sparsity that motivates the method |

## 2. How it actually compares

| model | params | avg10 | k/K |
|---|---:|---:|---:|
| Switch MoE on **our energy backbone** | 134.90M | 46.72 | 0.125 |
| **Boltzmann Hopfield K=32 top-2** | 134.49M | **46.26** | **0.062** |
| pure GPT + standard MoE, 3× mixture budget | 159.67M | 46.18 | 0.125 |
| **pure GPT + standard MoE, iso-param** | 134.49M | 46.04 | 0.125 |

**Boltzmann beats an energy-free GPT + standard MoE by +0.22pp at matched parameters and half
the routing density**, and beats the 3×-budget version too (which carries 19% more params).
A Switch mixture placed on *our* backbone is ahead (46.72) — the energy backbone is worth about
+0.68pp to an identical Switch mixture, so most of that gap is the backbone, not the router.

## 3. Settings that are FIXES, not preferences

```yaml
- mlp_type: EnergyFF_BoltzmannMoE
  expert_kind: hopfield          # 1 matrix/expert; the router's work is reused by the experts
  intermediate_size: 16384       # TOTAL across K, so I_e = 512 at K=32
  n_experts: 32
  top_k: 2
  hopfield_grad_scale: sqrt_consistent   # REQUIRED. "mean" is the old 4/I_e prefactor.
  routing_norm: zscore           # REQUIRED. Without it routing is EXACTLY uniform.
  temperature: 0.35
  repulsion_form: abs            # REQUIRED. "signed" rewards anti-alignment (old bug).
  repulsion_coef: 0.1
  n_repulsion_pairs: 4
  gelu_grad_method: sigmoid
  activation_function: gelu
```

Each of the three REQUIRED lines is worth real accuracy:

- **`routing_norm: zscore`** — the Hopfield energy is a mean over `I_e` units, so `E ~ 1e-2`
  against `tau`; raw softmax then gives *uniform* routing (measured: effective 7.999 of 8 experts
  on the shipped 33B checkpoint, making top-k a pure loss). z-scoring the logits across experts
  makes routing scale-free.
- **`repulsion_form: abs`** — the original `signed` form is *minimised* at cosine −1, so it
  rewarded anti-aligned experts. Worth **+0.69pp / −1.83 ppl** (45.89 vs 45.20).
- **`hopfield_grad_scale: sqrt_consistent`** — output prefactor `4/sqrt(I_e)` not `4/I_e`.
  Revert to `mean` only if training destabilises.

## 4. Knobs added recently — all default-off, and one you should leave alone

| knob | default | verdict |
|---|---|---|
| `track_load` | `true` | **leave on.** Logs `load_effective_n_experts` / `load_max_share` every step, *and under `torch_compile`*. The older metric path is traced away, which is why routing health used to be invisible. Nearly free. |
| `renormalize_topk` | `false` | **leave false — we tested it and it HURTS.** Making the masked weights sum to 1 (as Switch does) scored 45.35 against 45.89 for the default, i.e. −0.54pp. Our `sum p ≈ 0.45` is not a handicap. |
| `balance_rate` | `0.0` | **leave 0 unless experimenting.** Not needed: routing does not collapse (effective 12.3–12.8 of 16 experts on real data, max share 0.17–0.18 against 0.0625 uniform). See the warning below before enabling. |

**Routing stays balanced with no balancing loss** — and that is not just an absence. Removing the
auxiliary load-balancing loss from a *learned* router costs 0.60pp (45.66 → 45.06). Boltzmann
reaches 45.89 with no such loss to remove.

## 5. Traps that will cost you a day each

1. **Standard/learned MoE classes need `torch_compile: false`.** Data-dependent expert shapes
   cause rank-dependent recompiles. Symptoms: **5.77 s/step instead of 0.29 s/step** (19.7×
   slower), and/or gloo crashes (`Connection closed by peer`, `spmd_check` timeouts). Applies to
   `MoE` and `TopK_Energy_MoE_MLP`. `EnergyFF_BoltzmannMoE` compiles fine — static shapes.
2. **On a preemptable queue, `save_interval` must be short.** With `save_interval: 5000` and
   preemption every few minutes, an arm never checkpoints, so every resubmit restarts from step 0.
   Two of our baselines made **zero net progress across 28 submissions** and the status view
   honestly read step 0 each time, so it looked like slowness. Use ≤1000, or use a
   non-preemptable queue for anything a conclusion depends on.
3. **Multi-node needs `blaunch`, and must NOT use `-x`.** `pretrain.sh` derives `NODE_RANK` from
   `$HOSTNAME`, so it must start on every host; plain `bsub < script` starts it on the first only
   and torchrun dies at exactly 901 s. And `-x` (whole-node exclusive) makes a 2-node job PEND for
   45+ min. Use `-n <nodes> -R "span[ptile=1]" -gpu "num=<per-host>/task"` plus
   `blaunch bash pretrain.sh`.

   **On GPUs per host:** hosts have 8 H100s. Whether `num=8/task` schedules depends on how busy
   the cluster is, not on a hard limit — an earlier note here claimed 8-per-host "cannot be
   satisfied", which was true of one busy afternoon and wrong in general (checked 2026-09-14: 48
   hosts had all 8 free). Prefer `num=8/task` when it schedules, since it removes inter-node
   traffic and the InfiniBand retry exhaustion (`IBV_WC_RETRY_EXC_ERR`) that 4-per-host multi-node
   jobs hit; fall back to `num=4/task` with more nodes if 8-free hosts are scarce. Check with
   `bhosts -gpu` before assuming either.
4. **`balance_rate > 0` changes the checkpoint format.** It registers a persistent buffer, so a
   run started with it cannot strict-load a checkpoint written without it, and vice versa.
   `load_checkpoint_and_unshard` now drops the stray key, but don't toggle it mid-run.
5. **A derived config silently loses its parent's fixes.** Three separate day-long failures came
   from copying a config and editing only the field in mind — losing `torch_compile: false`, then
   `save_interval`, then `wandb_args.name`. **Diff any derived config against its parent before
   launching.**
6. **wandb: the run name is ignored on resume.** `tracking.py` uses the checkpoint's stored
   metadata when resuming, so fixing `wandb_args.name` in the config does nothing for an existing
   run — you must also edit `name` in `<save_path>/global_step*/experiments_tracker.json`.
7. **`grep nan` on these logs is a false positive** — it matches the venv path `nanoGPT-og`. Use
   `grep -E "train-(lm_)?loss = (nan|-?inf)"`.

## 6. Cost: read this before claiming a speedup

The router is **not cheap — it is shared.** In both expert forms it performs exactly half the
block's matrix multiplies, and that half is reused by whichever experts it selects. So it
evaluates *all* K experts and cannot be skipped: exact-router top-k costs `½(1 + k/K)` of dense
and **cannot beat 2×**, however small k is.

| configuration | MACs/token | vs Switch |
|---|---:|---:|
| Switch SwiGLU K=16 k=2 | 1.58M | 1.00× |
| Boltzmann K=32 k=2, **exact** router | 13.37M | **8.43×** |
| Boltzmann K=32 k=2, **proxy** router | 1.60M | **1.01×** |

Only with a cheap proxy router does the method reach FLOP parity. Measured end to end: proxy +
grouped dispatch = **19.0×** over dense soft routing; masking alone = **1.00×** (it skips no
arithmetic). **None of this is a training saving** — every arm evaluates all experts and masks.
The claim is inference-only.

**The proxy's price differs by expert form**, and this is the open question:
`W1W2` −0.03 to −0.69 ppl (free or better) vs `hopfield` +0.13 to +1.66 ppl. So the
better-quality form (Hopfield) and the freely-routable form (W1W2) are currently different ones.
`configs/iclr_decide/w1w2_K32_top2.yml` is running to settle it — **check its result before
committing to an expert form for a large run.**

## 7. If you are scaling up

- Our runs are **7.86B tokens at 134M**. That ranks variants; it does not establish scaling.
- One 400M arm at 30B tokens needs ~114k steps at 262k tok/step — about **25 h on 32 GPUs**.
- Our own 400M pair is **inconclusive** and should not be cited: only 15k steps (3.93B tokens),
  ending at ppl 62–69, where avg10 sits near the suite floor and the three metrics disagree.
- Token scaling at 134M, measured: 30k → 90k steps gave **45.89 → 46.18 (+0.29pp for 3× tokens)**.

Provenance and the full history: `STATUS_20260912.md` (append-only, newest at the bottom; the
banner at the top has current state) and `HANDOFF.md`.
