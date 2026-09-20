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
      **LAUNCHED 2026-09-19, job 1775570-series, 4 GPUs x mbs4 x ga4 = 262,144 tok/step = 32.0B.**
      Re-audited with code (my earlier hand numbers were low by n*d*I): TOTAL 134.491M /
      ACTIVE 123.492M / FLOPwt **143.214M**, i.e. **+1.1% of 6G1x6E's 141.720M** -- so this is the
      FLOP-MATCHED baseline. The `6G1S` arm we actually ran is **-12.9%** FLOPs, so the published
      44.87 is against an under-provisioned comparator. CPU build+forward verified before launch
      (recurrence on a Switch block had never run in this codebase).
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

### C. Boltzmann MoE → BASE (non-MoE) EGPT energy FFN, iso-active and iso-FLOP
Isolates **the mixture itself**. Same skeleton (`6G1x6E`), same energy attention, same
`psd_anti` projection, same 6x recurrence, same datamix, same schedule, same 32.0B budget. The
energy block's FFN goes from `EnergyFF_BoltzmannMoE` (K=16 hopfield experts, top-2, Sinkhorn,
sparse proxy selection) to `EnergyFF_Hopfield` — one monolithic hopfield energy FFN, no experts,
no router, no partition function, no balancing, no sparsity.

Answers: **does routing over a mixture buy anything over a single energy FFN of the same
per-token size**, or is the hybrid's advantage just "an energy block is present"? This is the
`denseEGPT` row anticipated in section 3, built at the **iso-active** choice.

- [x] **134M** — structure `6G1x6E`, single hopfield FFN **I = 2,472** — config:
      **`abl_E_134M_6G1x6E_baseEGPT.yml`** — launched 2026-09-20, job 1796776, 4 GPUs
      (mbs 4 x ga 4 = 262,144 tok/step), preemptable
- [ ] **400M** — hold until the 400M hybrid/sandwich pair reports; sizing would be
      `I = k x I_e + router/d` at K=32 k=2 I_e=5,871 — config: `________`

**Verified against the reference with `energy_ff_paramcount.audit_config` AND a meta-device
build (they agree to the byte). Columns: parameters in millions; FLOPwt = millions of
parameter-applications per token. Both rows 134M at 32.0B tokens, 262,144 tok/step.**

| arm | TOTAL | ACTIVE | FLOPwt |
|---|---|---|---|
| `cmix_134M_hybrid_32B_sparse` (reference) | 134.253M | 123.243M | 141.720M |
| `abl_E_134M_6G1x6E_baseEGPT` | 123.241M | **123.241M** (-0.002%) | **141.708M** (-0.009%) |

TOTAL is 8.2% lower **by construction and that is the point**: the MoE stores 16 experts and
applies 2, so it carries 11.0M parameters it never spends on a token. This arm is iso-ACTIVE and
iso-FLOP, deliberately NOT iso-total. Its residual -0.11% on the FFN is a *deficit*, so any win
it posts cannot be a compute artifact.

#### Why `I = 2,472` and not 2,048
2,048 = `k x I_e` matches the active EXPERT width exactly but leaves FLOPwt **1.4% low**, because
the MoE's 327,712 router parameters (`proxy_V`/`proxy_B`, rank 16, subspace) are applied to
EVERY token. HANDOFF §14.1 is the cautionary tale: a 12.9% FLOP deficit in the Switch baseline
REVERSED a headline claim. 2,472 also divides by 8 for bf16 tensor cores; 2,475 would match to
+0.012% but is odd and pads.

#### `hopfield_grad_scale` WAS UNREACHABLE — code fix required (2026-09-20)
`_EnergyFFHopfieldArgs` did not declare the field and `get_mlp_block` did not forward it, so
`HopfieldFFEnergy` silently used `"mean"` whatever the YAML said — **consistency rule 10 in the
wild**. At I=2,472 `"mean"` gives prefactor 0.00162 against the MoE's 0.125 at I_e=1,024, a ~77x
weaker descent step: the regime `_hopfield_grad_prefactor`'s docstring records as "the branch was
inert" (`||ffwd_out||` 0.005 vs `||attn_out||` 18.53). **Without the fix this ablation would have
compared a live MoE branch against a dead single-FFN branch and "proved" the MoE wins.** Fixed in
`config/mlp.py` (field + assert) and `mlp_blocks/__init__.py` (forward it); default stays
`"mean"`, and no pre-existing config used `EnergyFF_Hopfield`, so nothing else is affected.
Resolution verified on the built model: `transformer.h.6.ffwd` I=2472 grad_scale=sqrt_consistent.

#### What necessarily differs (intrinsic to removing the mixture)
No expert repulsion (a pairwise term; one FFN has no pairs), no Sinkhorn dual / temperature /
`e_sign` (no selection to bias), no proxy router and no sparse path (nothing to select among, so
the block is dense). `_EnergyFFHopfieldArgs` sets `extra="forbid"`, so a leftover MoE key is a
hard parse error, not a silent no-op — which is why they are deleted rather than zeroed.

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
  **PARTLY DONE 2026-09-20:** the iso-active row now exists on the HYBRID skeleton as ablation C
  (`abl_E_134M_6G1x6E_baseEGPT`). The iso-TOTAL counterpart (dense FFN at I ~ K*I_e = 16,384,
  ~8x the active params, so ~+9.6% FLOPwt) is still unbuilt and is the row that would test
  "is the expert BANK worth storing"; note it cannot be iso-FLOP at the same time.
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

---

## 5. W1W2 + sparse + surrogate selection — QUEUED, blocked on two code dependencies

Target once the surrogate drives sparse selection: run W1W2 Boltzmann-MoE **sparse** at both scales.
Rationale: every h1-series win was w1w2, every current arm is hopfield, and the surrogate is the only
selector that works for w1w2 (the rank-r subspace proxy fails there — w1w2's energy sums SIGNED terms
that cancel, `|sum|/sum|term|` = 0.0254 vs 1.000 for hopfield, so an m-row subsample needs m = I_e).

**Precomputed iso-param specs** (w1w2 stores TWO matrices per expert, so halving `I_total` matches the
hopfield arm on total AND active simultaneously at the same K and k):

| scale | reference hopfield | w1w2 substitute | active (hopfield → w1w2) |
|---|---|---|---|
| 134M | K=16 k=2 `I_total`=16,384 (`I_e`=1,024) | K=16 k=2 **`I_total`=8,192** (`I_e`=512) — exact | 1.57M → 1.57M |
| 400M | K=32 k=2 `I_total`=187,872 (`I_e`=5,871) | K=32 k=2 **`I_total`=93,952** (`I_e`=2,936) — +0.017% bank | 12.02M → 12.03M |

Budgets: 134M 122,070 steps @ 262,144 tok/step; 400M 61,035 steps @ 524,288 tok/step. Both 32.0B.

**Dependency 1 — register the w1w2 sparse path.** `energy_ff_w1w2_sparse.py` exists and its dispatch is
exact (oracle proxy 5.22e-16, gradients 6.02e-16) but is NOT in `get_mlp_block`. Registering it also
needs `energy_ff.py:938`'s `assert fused_spec["kind"] == "hopfield"` relaxed to accept `"w1w2"`.

**Dependency 2 — surrogate as sparse selector.** In progress. Gate: the **nomination-recall** curve
(what fraction of the true top-k appears among the nominated p). If a realistic head cannot nominate
the true winners, do not run these arms.

**BLOCKER FOR THE 400M ARM — multi-node.** `use_surrogate` + `sinkhorn_iters > 0` REQUIRES
`sinkhorn_persist_mu: true` (`energy_ff_surrogate.py:307`), and persist_mu's data-dependent
`int(self._mu_call.item())` inside the compiled region **HUNG at 2 nodes** (job 1775117: log dead 29
min, LSF still RUN, no NCCL error — same signature as the documented fused-repulsion 2-node wedge).
The 134M arm was rescued by going single-node (4 GPUs x mbs4 x ga4 = 262,144, unchanged budget).
400M needs 524,288 tok/step, so the options are:
  (a) **1 node x 8 GPUs** — hosts have 8, but `submit_train.sh` maps 8 GPUs to 2 nodes x 4; needs a
      `-n 1 -gpu num=8/task` shape. CLEANEST if a whole host is free.
  (b) 4 GPUs x mbs2 x ga16 — single node, correct budget, roughly 2x the step time.
  (c) Fix the persist_mu multi-node hang (make `_mu_call` rank-invariant or move the `.item()` out of
      the compiled region). Best long-term; also de-risks the three live multi-node arms that carry
      persist_mu today.
