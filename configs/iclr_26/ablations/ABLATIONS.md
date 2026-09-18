# ICLR-26 ablations — router / expert ablations on the hybrid

Purpose: isolate **what the Boltzmann energy router contributes**, by holding the whole
architecture fixed and swapping only the routing mechanism and/or the expert form.

All structures below are **read from the configs**, not assumed. Notation: `G` = GPT block
(softmax attention + dense MLP), `E` = energy block (energy attention + Boltzmann MoE),
`S` = Switch MoE block (learned gate, swiglu experts). `1x6E` = **one** block applied **6
times** (recurrence) — not six distinct blocks.

---

## 0. Reference arms (the thing being ablated)

| scale | config | structure | d | energy block |
|---|---|---|---|---|
| 134M | `configs/cmix/cmix_134M_hybrid_32B_sparse.yml` | **6G1x6E** | 768 | hopfield, K=16, k=2, I_e=1,024 |
| 400M | `configs/cmix/cmix_400M_hybrid_sparse.yml` | **6G1x6E** | 1024 | hopfield, K=32, k=2, I_e=5,871 |

Both: Boltzmann router (`sinkhorn_iters: 3`), `e_sign_override: pos`, `sparse_forward: true`,
32.0B tokens, cmix 70/30 web/math datamix.

> **CONFIRMED 2026-09-18: six iterations, `1x6E`, at both scales.** A `1x4E` in an earlier note
> was a slip and is settled. The `1x4E` shape does exist, but it is the 400M *sandwich*
> (`1G1x4E1G`) — not the hybrid.

---

## 1. Ablations to run

Two ablations, each at both scales. Tick the box and record the filename when the config is
written (configs live in this folder).

### A. Boltzmann router → Switch router, **experts unchanged (hopfield)**
Isolates the *router* alone: same recurrent block, same hopfield experts, same parameter
budget; the energy-based selection is replaced by a learned gate.

**BLOCKED — not expressible in the current schema (found 2026-09-18).**
`_EnergyFFBoltzmannMoEArgs` has NO learned-gate field: routing is always energy-based
(`temperature`, `routing_norm`, `sinkhorn_*`, `balance_rate`, `proxy_*`). There is no way to put
a Switch-style learned gate over hopfield experts without a code change. Options:
  (i) add a `router: "learned"` branch to the composable class (small, but touches a frozen file);
  (ii) port the LEGACY `SurrogateBoltzmannMoE_Energy_MLP` KL-distilled router to the composable
       class — see section 4, this also unblocks w1w2 sparse routing;
  (iii) `proxy_route: true`, which routes by the rank-r distilled proxy. NOT the same experiment:
        still energy-derived, no free-learned gate.

- [ ] **134M** — structure `6G1x6E`, hopfield K=16 k=2 I_e=1,024, learned gate — config: `________`
- [ ] **400M** — structure `6G1x6E`, hopfield K=32 k=2 I_e=5,871, learned gate — config: `________`

### B. Recurrent gptswitch — **no energy at all**
Isolates the *energy block* as a whole: the recurrent block becomes a standard Switch MoE
(learned gate + swiglu experts), so recurrence and depth are held fixed while the energy
mechanism is removed entirely.

- [x] **134M** — structure `6G1x6S`, swiglu K=16 k=2, **I_s = 341** — config: **`abl_B_134M_6G1x6S.yml`**
      verified: TOTAL 125M / ACTIVE 114M / FLOP-wt 134M == reference hybrid; schedule sums to 122,070; perGPU 32,768
- [x] **400M** — structure `6G1x6S`, swiglu K=32 k=2, **I_s = 1,957** — config: **`abl_B_400M_6G1x6S.yml`**
      verified: TOTAL 375M / ACTIVE 194M / FLOP-wt 276M == reference hybrid exactly; schedule sums to 61,035; perGPU 65,536

#### Why `I_s = I_e / 3`
swiglu stores **three** matrices per expert, hopfield **one**. For fixed K and k, iso-total and
iso-active are the *same* constraint (both linear in I), so `I_s = I_e/3` matches the hopfield
block on **total, active AND FLOPs simultaneously**. 134M: 1,024/3 = 341. 400M: 5,871/3 = 1,957.

#### Untested code path — smoke test required
**Recurrence on a Switch block has never been run in this codebase.** The plumbing looks
generic (`energy/layer.py` builds its FFN via `get_mlp_block`, `layer_iterations` lives in the
shared config, and `cmix_400M_baseline_switch` already runs an `MoE` block inside
`model_type: energy`) — but the recurrence path carries energy-specific hooks
(`_capture_energy`, `_last_energy_per_token`, per-iteration alpha) that have never seen a
swiglu MoE block. **Run a ~150-step smoke test before any long run.**

---

## 2. Consistency requirements — every ablation config MUST match its reference

Verified by diffing against the reference config; account for EVERY differing line.

1. **Datamix** — byte-identical `datasets:` block to the reference (cmix 70/30 web/math), and
   `data_cache_path: /proj/dmfexp/nima/.cache/megatron_cmix`.
2. **Tokens/step** — `GPUS x mbs x ga x seq`. **GPUS is NOT in the config.** 134M = 262,144
   at 8 GPUs (mbs 4, ga 2); 400M = 524,288 at 8 GPUs (mbs 2, ga 8). Verify empirically from the
   log: `billion_tokens_per_day * 1e9 * step_time / 86400`.
3. **Total tokens = 32.0B** — 134M: 122,070 steps; 400M: 61,035 steps.
4. **Schedule covers the run** — `num_warmup + num_constant + num_decay == num_training_steps`.
5. **Unique `save_path` AND wandb `name`**; beware prefix collisions when globbing.
6. **`e_sign_override` is expert-kind dependent** — hopfield needs `"pos"`, composable w1w2
   needs `"neg"`. A blanket `"pos"` is a silent no-op on w1w2. Switch/swiglu has no e_sign.
7. **Sinkhorn arms need `sinkhorn_persist_mu: true`** and `sinkhorn_mu_iters` == this block's
   `layer_iterations` entry. **The three 134M cmix arms are MISSING `persist_mu` — do not copy
   that defect forward.** Penalty is iteration-dependent: ~0.003 nats at 6 iterations, **+1.827
   at 12**, applied at EVAL (trained tilted, evaluated untilted).
8. **`cos_probe_interval` should be absent** (rule 9). All current cmix arms set it to 100; it
   is `no_grad` and forward-only so it cannot corrupt results, but it costs throughput and
   causes a stochastic branch under `torch.compile`. New configs: leave it out.
9. **`stage: 0`** (ZeRO stage 0 = replicated, DDP-equivalent; no sharding across nodes).
10. **Knobs must RESOLVE, not just parse** — `get_mlp_block` forwards kwargs explicitly, so a
    field can land on the pydantic args object and never reach the builder. Verify by resolving.

---

## 3. Lower priority (design still open)

- **260M swiglu sandwich** as an additional baseline.
- **Dense-EGPT sandwich `1G1x4(denseEGPT)1G`** vs the 260M sandwich. Matching *active* params
  makes the dense FFN tiny (no K to amortise: iso-active forces I ~ k*I_e), matching *total*
  gives it ~8x the active params. These answer different questions; both rows may be needed,
  clearly labelled.
- **W1W2 experts under the Boltzmann router** (at least one 134M arm). The h1-series wins were
  all w1w2; every current arm is hopfield. `sparse_forward` is hopfield-only
  (`energy_ff.py:938`), so this is **dense unless the sparse path is extended** — extension in
  progress in a separate file.

---

## 4. KL-distilled surrogate router — already implemented, never finished

`SurrogateBoltzmannMoE_Energy_MLP` (`mlp.py:994`, args at `config/mlp.py:483`) already does
"replace the energy in the exponent with a cheap learned head, trained Switch-style":

- `p_surr` from a learned **linear** layer `d -> K`; `KL(p_surr || p_boltz.detach())` added to the
  aux loss with weight `surrogate_coef`; `use_surrogate: true` uses the cheap router at eval.
- It lives on the **legacy** class, whose experts are **w1w2** — so it is ALREADY expert-kind
  agnostic, which is exactly the property the low-rank proxy lacks.

**Cost per token at the 134M shape** (d=768, K=16, I_e=1024, r=16):

| router | MACs | scales with I_e? |
|---|---|---|
| linear surrogate `d->K` | **12,288** | **no** |
| MLP surrogate `d->256->K` | 200,704 | **no** |
| subspace proxy at m=I_e | 458,752 | yes, `K*I_e*r` |
| (the sparse mixture it gates) | 3,145,728 | — |

**Status: the hypothesis was never answered.** `PROGRESS.md:327` records C2 as *"running (job
254254) — preempted at step 20k, resubmitted"*; no result is recorded. So "can a cheap distilled
head replace energy routing?" is open.

**Why this matters beyond cost:** a distilled head is agnostic to the energy's functional form,
so it sidesteps the failure that blocks sparse w1w2 — the signed-cancellation result
(`|sum|/sum|term|` = 0.0254 for w1w2 vs 1.000 for hopfield) that makes an m-row subsample
useless there. One mechanism would unblock ablation A **and** w1w2 sparsity.

**Claim to restate if we adopt it:** the paper says energy routing needs no learned router. With
a distilled head the honest framing is that the *target* is parameter-free — KL to a **detached**
Boltzmann distribution, no LM-loss gradient into the gate, no load-balancing loss — which is a
real distinction from Switch, but it must be stated rather than glossed.
