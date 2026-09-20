# ICLR-26 run priorities — what must finish, in what order, and why

**Deadline: 2026-09-24.** Written 2026-09-20. Supersedes ad-hoc ordering; if this file and a
HANDOFF "next" section disagree, HANDOFF's LAST numbered section wins and this file should be
updated to match.

> **How to read the progress column:** `steps done / steps total` from
> `<save_path>/latest_checkpointed_iteration.json`, which is the only trustworthy source — LSF
> `STAT=RUN` does NOT mean an arm is progressing, and `bjobs` alone missed five dead arms on
> 2026-09-20. **Step lines go to STDERR, not stdout** (`bsub_logs/*.stderr`); grepping stdout
> shows nothing but the NCCL banner and looks identical to a hang.

---

## THE TWO SPARSE MECHANISMS — never write "sparse" unqualified again

These are different contributions and the paper currently conflates them in `tab:status`.

| name | what it is | config signature | where it works |
|---|---|---|---|
| **sparse(proxy)** | the router's weight matrices are replaced by a rank-$r$ subspace projection with a small output dim, so selection costs $O(Kdr)$ instead of $O(KdI_e)$ | `proxy_kind: subspace`, `proxy_rank: 16`, no surrogate | hopfield ONLY |
| **sparse(surrogate)** | a small MLP head is KL-distilled to reproduce the *ranking* of the exact energies, and nominates $p \ge k$ candidates which the exact energy then re-ranks to $k$ | `surrogate_kind: mlp`, `surrogate_hidden: 256`, `surrogate_replaces_proxy: true`, `proxy_rank: 0` | both, and is the ONLY option for w1w2 |

**Why the proxy cannot serve w1w2 (§14.6):** the bilinear energy sums SIGNED terms that cancel
(`|sum|/sum|term|` = 0.0254 for w1w2 vs 1.000 for hopfield), so an $m$-row subsample carries
relative error $\sim\sqrt{I_e/m}$ and is forced to $m = I_e$ — where it costs MORE per token than
the mixture it exists to cheapen. The surrogate head's cost has no $I_e$ in it at all.

**⚠ THIS CREATES A CONFOUND IN THE CURRENT GRID.** Every hopfield arm is sparse(proxy); every
w1w2 arm is sparse(surrogate) or dense. So "w1w2 vs hopfield" and "surrogate vs proxy" are the
SAME contrast in the data we have, and neither can be attributed. See P1.3 for the missing arm
that separates them.

---

## ⚠ REVISION 2026-09-20 (later) — THE 134M TIER IS DIAGNOSTIC, NOT EVIDENTIAL. GO DEEPER AT 400M.

**Why the plan changed.** At 134M the MoE is too small to be the thing under test:

| 134M hybrid | params |
|---|---|
| total | 134.25M |
| **embedding (tied)** | **77.07M — 57% of the model** |
| non-embedding total | 57.18M |
| non-embedding ACTIVE | 46.17M |
| **energy expert bank (total)** | **12.58M = 9.4% of the model** |
| **expert params ACTIVE per token** | **1.57M** |
| + router (proxy) | 0.33M |
| **MoE share of non-embedding ACTIVE** | **~4%** |

So the routed component is ~4% of the active non-embedding parameters. The user's read is correct and
stronger than the guess: **at 134M the MoE cannot express a large effect, and the measured effect is
slightly NEGATIVE** — `abl_D` (six dense GPT layers, no MoE, no energy block) scores **45.55 / 39.94**
against the hybrid's **44.82 / 41.06** while spending 5.2% FEWER parameter-applications per token.
At 400M the picture is different: the expert bank is 192.4M of 297.0M non-embedding total (65%) and
12.02M of 116.7M active (10.3%) — an order of magnitude more of the model.

**Consequence: stop adding 134M architecture arms. Spend the GPUs at 400M and DECIDE AT THE 8B
MILESTONE** rather than paying 32B for each arm.

### NEW ARMS — SPECS READY, NOT SUBMITTED (awaiting confirmation per the CLAUDE.md rule)

All 400M arms: d=1024, 61,035 steps x 524,288 tok/step for the full 32B, but **judged at the 8B
anchor (step 15,258, ~10.6 h at 8 GPUs)** before committing further.

| # | arm | structure | sizing | purpose |
|---|---|---|---|---|
| N1 | `abl_F_134M_6G_dense_isoactive` | `6G` | **I = 2320** → total=active=**123.308M**, +0.053% vs the MoE's 123.243M active | closes the abl_D loophole: abl_D was iso-TOTAL and carried +9% active params. This is iso-ACTIVE. ~6.8 h at 4 GPUs. |
| N2 | `abl_G_400M_6G1x6E1x6E` | 6 GPT + **two** energy blocks, each applied 6x | `I_total` = **93,952** per block (K=32, `I_e`=2936) so the two blocks together hold the same 192.4M bank and 12.02M active as today's single block | doubles energy DEPTH (12 applications) at unchanged parameters |
| N3 | `abl_G_400M_6G1x6S1x6S` | same skeleton, E→Switch | `I_s` = `I_e`/3 = **979** per block | N2's FLOP-matched learned-gate baseline |
| N4 | `abl_H_400M_6G6E_deep` | 6 GPT + **6 distinct** energy blocks, NO recurrence | `I_total` = **31,296** per block (K=32, `I_e`=978) → bank 192.2M, active 12.01M | tests whether DISTINCT depth beats recurrent depth at equal params and equal applications |
| N5 | `abl_H_400M_6G6S_deep` | same, E→Switch | `I_s` = **326** per block | N4's learned-gate baseline |
| N6 | `abl_H_400M_6G6G_deep` | 12 dense GPT layers, no MoE | size to match N4's **ACTIVE** (~200M, exact figure to be audited) | the abl_D question at the scale where the MoE is 10% of active, not 4% |

**Every one needs the five-point confirmation before `bsub`:** datamix diff empty against
`configs/cmix/cmix_400M_hybrid_sparse.yml`, parent named and diffed line-by-line, `num_layers` +
full `layer_iterations` + per-block mixer/mlp types stated, `GPUS x mbs x ga x seq`, and
`audit_config` TOTAL/ACTIVE/FLOPwt against the comparison arm.

### KILL LIST — what to stop to make room

| arm | state | verdict |
|---|---|---|
| `cmix_400M_sandwich_sparse` | **DIED** at 17,400/61,035 (28.5%), ~30 h remaining | **DO NOT RELAUNCH.** Its 134M sibling is the WORST arm measured (43.45, −1.37pp vs hybrid); the true sandwich `abl_C` is not close either. 30 h of 8 GPUs for a shape that loses. Its 8 grp_ebm GPUs are already free. |
| `abl_C_134M_1G1x6E1G_isototal` | RUN, 52,000/122,070 (42.6%), ~3.9 h | **KILL.** 134M sandwich variant — the tier is diagnostic-only now and sandwiches underperform. Frees 4 GPUs. |
| `abl_E_134M_6G1x6E_baseEGPT` | RUN, 0/122,070, ~6.8 h | **KEEP** but demote. It is the mixture-isolation middle term between abl_D and the hybrid, and it is the cheapest way to finish the 134M decomposition. Reconsider if 400M needs the 4 GPUs. |
| `cmix_134M_pure_32B_sparse` | RUN, 93.4%, **~0.8 h** | let it finish |
| `cmix1B_12L_gptDense_32B` | RUN, 95.0%, **~1.7 h** | let it finish; releases 8 grp_ebm GPUs |
| `cmix_400M_hybrid_sparse` | RUN, 60.0%, ~25 h | **KEEP — P0.2** |
| `abl_B_400M_6G1x6S` | RUN, 60.6%, ~16 h | **KEEP — P0.1** |

Freed within ~2 h: 8 (dead sandwich) + 4 (abl_C) + 8 (1B) + 8 (pure) = **28 GPUs**, enough for three
8-GPU 400M arms plus a 4-GPU 134M arm.

## P0 — the headline cannot be stated without these

**P0.1 `abl_B_400M_6G1x6S` — 35,800/61,035 (58.7%), job 1796831 PEND (preemptable).**
The FLOP-matched Switch baseline at 400M. §14.1 is the whole reason this is P0: at 134M the
unmatched `6G1S` baseline was under-provisioned by **12.9% of FLOPs**, and fixing it **reversed**
the parity claim (energy 44.82 vs matched Switch **45.43**). Reporting a 400M row against the
unmatched baseline would repeat exactly the error that already cost one headline. **This arm has
the most remaining work of anything on the critical path (~25,200 steps) and is the single most
likely thing to miss the deadline.** It should hold a non-preemptable `grp_ebm` slot as soon as
one frees.

**P0.2 `cmix_400M_hybrid_sparse` — 35,800/61,035 (58.7%), RUN on grp_ebm.**
The 400M energy arm that P0.1 is compared against. ~25,200 steps left.

**P0.3 Eval of `cmix_134M_hyb_w1w2_sparse_surr_32B` — jobs 1796826 / 1796827, submitted.**
Training is COMPLETE (122,070/122,070) and had **no eval at all**; the arm sat finished and
unmeasured. This is the gate on P3.1/P3.2 (whether 400M gets rebuilt on w1w2).

---

**P0.4 ⚠ NEW 2026-09-20 — SEEDS. Two headline margins are now smaller than the unmeasured noise.**
`abl_D` (six dense GPT layers, NO MoE, NO energy block, NO recurrence) scored **Avg11 45.55 / ppl
39.94** at 32.00B — the best 134M arm on both, at **5.2% FEWER parameter-applications per token**
than the energy hybrid (44.82 / 41.06). So the two claims the paper would rest on are:

  * §14.1: the FLOP-matched learned gate leads the energy router by **0.61pp**;
  * §15.7: **no router at all** leads both, by **0.73pp** over the hybrid and 0.12pp over Switch.

**Both are single-seed, and the multi-seed arms have been PAUSED in `watchdog_jobs.conf` since
2026-09-12, so seed spread at this scale is UNQUANTIFIED.** A sub-1pp ordering among three arms
cannot be asserted without it, and this is not a point in favour of either side — it makes the
Switch-leads sentence as unsupported as the dense-leads one.

A 134M arm costs ~5.3 h at 4 GPUs (abl_D's measured wall clock), so **2 extra seeds of the hybrid
and of `abl_B_134M_6G1x6S` is ~21 GPU-hours x 4 and fits before Sep 24.** That buys an error bar on
the single comparison the paper is actually about. Prefer it over any new architecture arm.
If the seeds are not run, the paper must state the ordering as within-noise rather than as a result.

## P1 — needed for the ablation section to be honest

**P1.1 `abl_D_134M_6G_dense_isototal` — 116,000/122,070 (95%), RUN.** ~6,000 steps.
**P1.2 `abl_E_134M_6G1x6E_baseEGPT` — 0/122,070, RUN (job 1796776).** The non-MoE base-EGPT
control: iso-ACTIVE (-0.002%) and iso-FLOPwt (-0.009%), TOTAL -8.2% by construction. Answers
whether the mixture buys anything over one energy FFN of the same per-token size. Fresh run, so
it is the longest 134M arm outstanding.

**P1.3 ⚠ MISSING ARM — `hopfield + sparse(surrogate)` at 134M. NOT BUILT, NOT SCHEDULED.**
This is the arm that breaks the confound above. With it:
  * vs `cmix_134M_hybrid_32B_sparse` (hopfield + proxy) → isolates **the selector**;
  * vs `cmix_134M_hyb_w1w2_sparse_surr_32B` (w1w2 + surrogate) → isolates **the expert form**.
Without it, every w1w2-vs-hopfield sentence in the paper must carry "confounded with the
selector". Cheap: copy the hybrid config, swap the proxy keys for the surrogate keys already
validated in the w1w2 sparse config. **Build this before starting any 400M w1w2 arm** — it is
strictly more informative per GPU-hour than a second 400M run.

**P1.4 `cmix_400M_sandwich_sparse` — 16,600/61,035 (27.2%), RUN on grp_ebm.** ~44,400 steps, the
largest single remaining block of compute in the project. It is P1 not P0 because the sandwich is
a parameter-efficiency story, not the headline routing comparison.

**P1.5 `abl_C_134M_1G1x6E1G_isototal` — 44,000/122,070 (36%), RUN.**

---

## P2 — supporting, cheap, or nearly done

**P2.1 `cmix1B_12L_gptDense_32B` — 56,000/61,035 (92%), RUN.** ~2.3 h left; finishing it releases
8 GPUs, which is the natural slot for P0.1. Lowest priority per GPU-hour *remaining* but highest
per GPU-hour *released*.
**P2.2 `cmix_134M_pure_32B_sparse` — 108,000/122,070 (88.5%), job 1796726 PEND.**

---

## P3 — CONDITIONAL on the P0.3 verdict

**Trigger: does w1w2 sparse(surrogate) beat hopfield sparse(proxy) at 134M by a margin worth a
400M rebuild?** Reference to beat: **Avg11 44.82, ppl 41.06** (`cmix_134M_hybrid_32B_sparse`).

Evidence so far (partial, and NOT a clean expert-form test):
`cmix_134M_hyb_w1w2_surrMLP_32B` — w1w2 **dense**, surrogate head used as the ROUTER
(`use_surrogate: true`, `surrogate_coef: 1.0`) — scores **Avg11 44.16 / ppl 40.35 / MMLU 25.42 /
GSM8K 1.59**. That is **-0.66pp Avg11** against hopfield but **-0.71 ppl** (better). It differs
from the reference in THREE ways at once (expert form, dense-vs-sparse, and router), so it cannot
settle the expert form. Note also that as a DENSE arm it evaluates all $K=16$ experts per token
(`sparse_forward: false` → `_forward_fused`), so its true FLOPwt is ~207M against the reference's
141.7M — **~46% more compute for a lower Avg11**. `audit_config` reports 140.96M for it because it
charges `top_k`; that figure is WRONG for any dense-masked arm and must not be quoted.

- **If w1w2 sparse wins clearly:** build `cmix_400M_hyb_w1w2_sparse_surr` and
  `cmix_400M_sandwich_w1w2_sparse_surr`. Iso-param sizing is precomputed in
  `ablations/ABLATIONS.md` §5: **K=32, k=2, `I_total` = 93,952 (`I_e` = 2,936)**, +0.017% on the
  bank. **Blocker: `use_surrogate`/surrogate + `sinkhorn_iters > 0` requires
  `sinkhorn_persist_mu: true`, which HUNG at 2 nodes** (§14.4). 400M needs 524,288 tok/step, so
  use 1 node x 8 GPUs, or 4 GPUs x mbs2 x ga16.
- **If it does not:** hopfield stays the expert form; spend the GPUs on P1.3 and P0.1 instead.

**P3.3 400M base-EGPT control** (the 400M twin of P1.2). Hold until the 400M pair reports.

---

## GPU accounting as of 2026-09-20 04:30 UTC

`grp_ebm` is a **32-GPU** quota and was found **8/32** — 24 free, the first time it has not been
at 32/32. It is now 32/32 with 400M hybrid (8), 400M sandwich (8), 1B (8) plus the two 4-GPU
ablation arms. `grp_preemptable` sits ~620/6144, so preemptable schedules but can be preempted.

**Reallocation rule:** when P2.1 (1B) completes, move **P0.1** into the freed grp_ebm slot. Do NOT
`bstop` a pending GPU job to make room — it invalidates `CUDA_VISIBLE_DEVICES` and every parked
eval dies (§14.2a). Kill and resubmit instead.

---

## Process failures this list exists to prevent

1. **Five arms were found dead simultaneously** (2026-09-20): 400M hybrid, 400M sandwich, 1B,
   134M pure, and `abl_B_400M_6G1x6S`. **None is registered in `watchdog_jobs.conf`** — that file
   holds only the older `iclr_*` grid arms — so no automatic resubmission exists for the arms the
   paper depends on. `scripts/health_check_arms.py` also did not cover the four `abl_*` arms, which
   is why the 400M FLOP-matched baseline stayed dead after the first four were relaunched.
2. **A finished arm can sit unevaluated.** `cmix_134M_hyb_w1w2_sparse_surr_32B` was complete with
   no eval; `cmix_134M_hyb_w1w2_surrMLP_32B` had a finished eval nobody had read. Run
   `scripts/auto_eval_on_finish.sh` and `compute_avg11.py`, do not assume.
