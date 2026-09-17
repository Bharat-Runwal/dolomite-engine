# Boltzmann MoE — TODO

## 2026-09-16: TRUE SPARSITY — ⭐ TOP PRIORITY IF THE PROXY BENCHMARK LANDS WELL

### ⭐ THE DECISION THAT MATTERS: train the long pure T12 with sparsity ON
- [ ] **If the proxy-selection benchmark holds up on both models, retrain the full long
      `t90k_pure_T12` (90k steps / 23.6B tokens) with the new setting, and do it IMMEDIATELY.**
      Rationale: measured **4.69x** on the pure block (compiled, fwd+bwd, at its real 16384
      tokens/call), and the mixture is 98.9-99.6% of per-token FLOPs in a pure stack. If that
      converts to anything close to a 4x step-time cut, the pure arm goes from ~40 h to ~10 h and
      the missing **400M PURE** cell in the 3-family x 2-scale matrix becomes affordable inside the
      deadline. This is the single highest-leverage item open.
      * GATE 1: proxy selection must cost < ~0.05 bits/byte and < ~0.5pp Avg11 on BOTH benchmarked
        models (job 1706727 pure, 1706728 hybrid). Measured so far on pure T12: **+0.0165 bpb**.
      * GATE 2 — RESOLVED, and it changes the recipe. p-ladder (1706725) on the RETROFITTED
        checkpoint: p=K returns exactly 1.0996 (machinery proven correct), but p=2/3/4/6 give
        3.43/3.28/3.26/3.13 against a dense 1.0996. Over-selection does NOT rescue it: the K-p
        remaining denominator terms come from the proxy, whose ABSOLUTE energy scale was never
        calibrated (the KL objective trains ranking, not magnitude).
        **=> THE RETRAIN MUST SET `renormalize_topk: true`.** That deletes the all-K denominator
        entirely, so there is nothing to estimate. Retrofitting it costs +0.483 bpb because
        scale_ff was trained for sum(p) ~= 0.45, but TRAINING with it is FREE: §12.3 measured
        iclr_hop_K16_top2_renorm at Avg11 44.54 against 44.58 for the masked form, a tie.
        **Consider also `routing_norm: sqrt_width` instead of `zscore`**, which removes the
        per-token moments (worth a further +0.318 retrofitted) -- but zscore was adopted for a
        reason (the mean-normalised Hopfield energy is ~1e-2 against tau=1, giving effK 7.999/8),
        so sqrt_width on the PURE line is untested and is a second variable. Prefer:
        phase 1 = renorm only; add sqrt_width only if the moments prove to matter when trained.
        With those two gone, the ONLY approximation left is SELECTION, at +0.0165 bpb.
      * GATE 3: **measure the end-to-end step time, not the block time.** Every 4.69x here is a
        BLOCK benchmark on ONE GPU. Confirm on the real 4-GPU arm before believing it.
      * CONSTRAINT: training uses a SHARED proxy head (0.842), not the per-iteration heads (0.880)
        — a call counter is unsound under activation checkpointing.
      * CONSTRAINT: needs a DENSE warm-up phase to fit the proxy, since the sparse path never
        computes the all-K targets the distillation needs. Two-phase recipe, or use
        `proxy_route: false` for phase 1 and flip it for phase 2.
      * `micro_batch_size` MUST stay >= 4096 tokens/call or sparsity LOSES; and note fused_experts
        OOMs at mbs 4 with I_tot=71680, so use mbs 1 x ga 16 unless sparse_forward is on (it cuts
        that intermediate 6.4x, which may re-enable mbs 4).

### THE RETRAIN RECIPE — settled parts (as of 2026-09-16, pending 2 inputs)

| setting | value | evidence |
|---|---|---|
| `renormalize_topk` | **true** | removes the all-K denominator, which is ~55% of the softmax mass (§12.12). FREE when trained: §12.3 has 44.54 vs 44.58. Retrofitting costs +0.483. |
| `sparse_forward` | true | 4.69x at p=k; expect ~3.2x at p=4 (76% of the 4.3x ceiling) |
| `sparse_candidates` | **4** | recall 0.973, and 0.946 of tokens get BIT-IDENTICAL routing to dense. p=6 gives 0.988/0.976 for a 3.2x ceiling instead. |
| `proxy_kind` / `proxy_rank` / `proxy_out_dim` | subspace / 16 / 512 | 0.30% of dense MACs; the "quad" head fails (0.348 vs a 0.125 floor) |
| `proxy_iters` | **1** | per-iteration heads (+0.05) are EVAL-only: a call counter is unsound under activation checkpointing |
| `proxy_loss_coef` | 0.01 | online KL distillation, x detached so it cannot degrade the model |
| `proxy_route` | phase 1 false, phase 2 true | phase 1 lets the proxy learn from the exact router; with renorm + proxy_route, flipping sparse_forward on is then ~numerically inert |
| `repulsion_space` | weight | REQUIRED: output-space needs all K expert outputs. coef **BLOCKED** on rep_* sweep |
| `micro_batch_size` | 4 (16384 tok/call) | below ~16384 the speedup collapses (1.36x at 4096). Sparse cuts the intermediate 6.4x so the mbs-4 OOM should not recur — VERIFY before the long run. |
| `fused_experts` | true | required for the proxy to train at all (`_forward_looped` never calls `_proxy_step`) |

**Two inputs still missing:** (a) the weight-space repulsion coefficient (rep_* sweep), (b) the
full Avg11 delta for proxy selection on both models (benchpure/benchhyb). Do NOT launch without
both, and confirm the END-TO-END step time on 4 GPUs before trusting any of the block numbers.

**Calibrating the proxy's magnitudes (`--mse_coef`) IS a large part of the fix** — I claimed the
opposite an hour earlier, from agreement and the selection rung, and the target metric overturned
it. With lambda = 1.0 the full sparse path at p=2 goes **3.4346 -> 1.7920** bits/byte, i.e. the
denominator error falls from +2.32 to +0.66 nats. The cost is real but small: agreement 0.883 ->
0.850, and selection alone 1.1161 -> 1.1339.

The two fixes are COMPLEMENTARY and address different halves of the same error:
  * calibration fixes the SCALE of the proxy-supplied denominator terms;
  * over-selection reduces their NUMBER (recall 0.882 -> 0.973 at p=4, with 0.946 of tokens
    getting bit-identical routing);
  * `renormalize_topk: true` removes the term ENTIRELY, and is free when trained.
Use all three unless a measurement says otherwise. LESSON: agreement and the selection rung are
NOT proxies for the denominator error -- only a sparse rung measures it.

### In progress
- [ ] `pladder2` (1706725): p in {K,3,4,6,8} bits/byte. p=K is the correctness self-test.
- [ ] `benchpure` (1706727) / `benchhyb` (1706728): FULL Avg11 + wikitext, dense vs
      proxy-selection, on pure_hop_T12_sink and iclr_hop_K32_top2_sink.
- [ ] `ptrain_T12_proxy16` (1706723): can the proxy be distilled ONLINE? Beat the post-hoc
      shared-head 0.842.

### Open
- [ ] `renormalize_topk: true` costs **+0.483 bits/byte** on its own (scale_ff was trained for
      sum(p) ~= 0.45). Do NOT use it to fix the denominator — over-selection is the right fix.
- [ ] Per-iteration proxy heads in TRAINING would need `layer.py` to pass the block its iteration
      index. Worth +0.038 agreement. Touches every energy model, so not before the deadline.
- [ ] Decode/generation with sparsity is UNMEASURED and probably a LOSS (T = batch size, and
      small T loses). Do not claim inference speedup for autoregressive decode.
- [ ] End-to-end (not block-level) speedup on a real multi-GPU arm, placement-controlled.


## 2026-09-15: Avg11 for the seven previously-unevaluated sharded ICLR arms — DONE

Driver: `experiments/eval_scripts/eval_sharded_iclr_avg11_20260915.sh`

**Done**
- [x] All 7 arms that had `global_step*` shards but no `unsharded*` dir and no
      `harness_results_*.json` now have a COMPLETE 11/11 Avg11 + separate MMLU /
      GSM8K-CoT / WikiText word-PPL. Full table in `PROGRESS.md` (2026-09-15 entry).
- [x] **First 400M/32B headline number**: `scale32B_gptswitch` at its full
      61035 steps / 32.00B tokens = **Avg11 50.25, WikiPPL 25.00**.
- [x] Read the two live `iclr_scale` arms without touching the trainers, via a
      pure-read shard copy in `results/iclr_scale_eval_staging/` (rationale in
      the script header and `PROGRESS.md`).

**Next**
- [ ] **The 32B Boltzmann-vs-Switch comparison is still OPEN.** `scale32B_boltz_hop`
      is only at step ~7000/61035 (Avg11 44.64 / PPL 42.30 at step 6000). Re-eval at
      the 61035 target and only then compare against gptswitch's 50.25.
- [ ] Do NOT quote the five PARTIAL rows as results — task-complete 11/11 Avg11 is
      not training-complete. `w1w2_K32_top2` (41.63 @ 33% of budget) does **not**
      resolve Hopfield-vs-W1W2; that arm is still `#PAUSED-20260914`. To settle it,
      resume it to 30000 steps and compare against Hopfield K=32's 44.38.
- [ ] Optional cleanup: `results/iclr_scale_eval_staging/` holds ~3 GB of copied
      shards plus two unsharded models; the gptswitch step-58000 row there is
      superseded by `unsharded_step61035` in the real run dir.

## 2026-09-12: inference-FLOPs work (see HANDOFF.md §7 for all evidence)

**Done**
- [x] Exact per-token FLOP model for `math_fet_boltz_hopfield_rep_*`
      (`scripts/analyze_moe_router_flops_20260912.py`). 603.59M MACs/token;
      MoE is 25.0%, lm_head 25.5%, GPT prefix 37.5%.
- [x] Measured real routing + branch magnitudes + paired-NLL ablation on 3 FET
      checkpoints (`scripts/measure_moe_routing_20260912.py`, job 1559723).
- [x] Answered the standing open question *"why does truncating to top-2 cost 6 PPL?"*
      — for the Hopfield-MEAN MoE it costs **nothing**, because routing is uniform
      (`eff_n = 7.999/8`) and the FF branch is inert. The 6 PPL belongs to the
      **w1w2** line, where routing carries real information (~3.9 PPL).
- [x] `train_utils.py:73` — `FFEnergyBase` added to the metrics isinstance gate.
      No `EnergyFF_*` run had ever logged routing-collapse metrics.
- [x] `head_dim` decoupled from `hidden_size/num_heads` in `EnergyAttention_QK`,
      enabling over-complete heads (`num_heads*head_dim > d`). Default unchanged;
      forward+backward verified, partial RoPE already supported (`rope.py:118`).
- [x] Single-block all-MoE architecture family + FLOP/param calculator
      (`scripts/design_single_block_moe_20260912.py`), 7 configs emitted to
      `configs/single_block_moe/`.
- [x] Grouped (sort-by-expert) sparse inference path + throughput benchmark
      (`scripts/bench_moe_throughput_20260912.py`) — closes the TODO below about
      needing a real dispatch to measure anything.

**Next**
- [ ] **Fit the rank-r proxy router** on the cached `(x, E_k)` pairs in
      `results/router_analysis/router_fit_cache__*.pt` (frozen backbone). For
      Hopfield the ceiling is 96.7% top-1 at r=16; for w1w2 the projection-based
      ceiling estimate plateaus near 0.5 and is **non-monotone in r**, which is not
      a credible ceiling curve — extend r up to d (=768) to validate the estimator,
      and prefer fitting a head directly on `(x, E_k)` over projecting x.
- [x] **Repulsion loss was mis-specified** — FIXED (`repulsion_form`, default
      `squared`). `L_rep = λ·E[cos]` is minimised at cos = −1 and rewards
      anti-alignment: both FET-rep and h1 sit within ~3% of the geometric floor
      `−1/(K−1)`. **But it is NOT the cause of the dead branch** — h1 runs λ=0.1
      (10× FET's) and is 86% load-bearing. Use `abs` to preserve the existing λ
      sweeps; `squared` is ~3× weaker at the same λ. Re-sweep λ either way.
- [ ] **The real root cause: the Hopfield output prefactor `4/I_e`.**
      `mean ‖g_k‖` = 0.019 (Hopfield) vs 155.66 (legacy w1w2) — 8,200×, of which
      256× is the bare constant (`energy_ff.py:644` `4/I_e` = 0.0039 vs
      `mlp.py:693` no prefactor at all). `scale_ff` = 1.52 cannot bridge that.
      Change `4/I_e → 1/√I_e` on the Hopfield output and re-run. Fixing repulsion
      alone will NOT revive this line.
- [x] Scale-free routing — DONE (`routing_norm ∈ {none, zscore, sqrt_width}` on
      `BoltzmannMoEFFEnergy._logits`). `zscore` normalises to unit std across
      experts, so pair it with `temperature ≈ 0.35` to reach `eff_n ≈ 2`; plain
      z-score alone only reaches `eff_n` 5.6–6.4 of 8.
- [ ] Re-check the FF-branch strength of the remaining `math_fet_hopfield_mean_r*`
      register variants — the whole Hopfield-MEAN line measures 1.6–10.9% FF share
      vs 85.8% for w1w2, so published deltas in that family may be comparing
      near-inert branches.
- [ ] DROPPED: "route once at iteration 0, reuse for 1..T". Iteration-to-iteration
      argmax agreement is 0.27–0.30 on the w1w2 line (TV 0.24–0.27) — routing
      genuinely changes with depth. The 0.87–0.90 agreement on the Hopfield model
      was an artefact of its degenerate uniform routing.
- [ ] Train one `configs/single_block_moe/` point once the throughput numbers justify
      it. Note the recurrence tax: k/K must be ≲ 1/T to stay at parity with a
      conventional dense transformer at iso-total-params.

## 2026-09-14: training-step speedup for scale32B_boltz_hop (6.75s vs gptswitch 1.34s)

**Done**
- [x] Training-step microbench (fwd+BWD, production shape) —
      `scripts/bench_moe_train_step_20260914.py` →
      `results/router_analysis/bench_moe_train_step_20260914.json`. Findings:
      repulsion = +65% of the MoE block (single largest cost); two_stage is SLOWER
      (0.59× at N=4096, grouped-loop backward); proxy needs a fused kernel (no FLOP
      saving at 4096 tok/call); 1-in-10 output repulsion → 1.55× block speedup;
      weight-space repulsion 3.5× cheaper AND sparse-compatible. See PROGRESS.md.
- [x] (a) Phase-B profiler config `configs/iclr_scale/scale32B_boltz_hop_PROFILE.yml`
      (throwaway; engine TorchProfiler; 4 preemptable GPUs, job 1658324) — trace →
      `results/profiler/boltz_hop_trace`. PARSED (`scripts/parse_profiler_trace_20260915.py`).
      RESULT: one step = 1712.99 ms GPU kernel time; **elementwise 61.0% / GEMM 25.3%
      / attention 4.3% / comms 0.4%**. Both elementwise AND GEMM are dominated by the
      dense-32-expert MoE (I_e=1280 projections fire 1536×/step = 32×6×8). **The ~14%
      MoE estimate was REFUTED — the 5× gap is `recurrence × dense-32-expert`.** See
      PROGRESS.md "Phase B RESULT (2026-09-15)".
- [x] (b) Intermittent-repulsion patch DRAFTED (not applied; needs sign-off) —
      `results/router_analysis/intermittent_repulsion_20260914.patch`. Adds
      `repulsion_interval` (default 1 = current) + `repulsion_scale_comp` to
      `BoltzmannMoEFFEnergy`, opt-in per config.
      **CORRECTION 2026-09-15: the "~2–4% of the whole step" note below was WRONG.**
      It was a profiler bucketing artefact — only repulsion's `F.normalize` landed
      in reduce/norm, while its dominant cost (the backward through normalize+cos)
      fell into the generic elementwise bucket. Repulsion is **17–21% of the
      optimizer step**: the bench's 7.59 ms/call x 48 calls/step (6 recurrence x 8
      grad-accum) = ~364 ms of 1713 ms = 21%, matching a direct A/B (1.463 vs 1.210
      s/step = 17%). So repulsion IS a headline-sized lever. But intermittent
      firing buys that speed by weakening the regulariser (see
      ACCEL_FINDINGS_20260915.md on the `boltz-accel` branch) — prefer
      **weight-space repulsion**, 2.20 vs 7.59 ms/call and sparse-compatible, at
      full strength every step.

**Next**
- [ ] **Headline levers to close the gap (from the trace, biggest first).**
      ⚠ **The "5×" gap is substantially a HOST-PLACEMENT artifact** — same unfused
      code on a different host pair runs 2.45–2.53 s/step vs 6.72–6.81, GPU model
      identical, sibling contention ruled out. Real ratio to gptswitch ≈1.9×, and
      the "launch-overhead bound" reading (25% GPU util) becomes 68% util, i.e.
      largely void. See PLACEMENT_ARTIFACT_20260915.md before acting on these:
      (1) fused top-k sparse **back-projection** (skip 15/16 of the [·,1280]@[1280,1536]
      ≈244 ms); (2) **rank-r proxy router** (Hopfield ceiling 96.7% top-1 @ r=16 —
      TODO line 28) to skip the all-32 fwd projection ≈266 ms; (3) **fuse the
      per-expert energy loop** (grouped-GEMM + fused gelu²-mean) to cut both
      elementwise and the launch-overhead gap (79.5 k kernels/step, GPU-busy 1.71 s ≪
      6.75 s wall); (4) route-once-reuse across the 6 recurrence iters ONLY if
      verified stable on the trained config (re-check TODO line 54's 0.87–0.90).
- [ ] (b) patch: ship it opportunistically (safe, ~2–3% step) once sign-off is given;
      `repulsion_interval: 10, repulsion_scale_comp: true`. Not gating on it.
- [x] **Overleaf Avg11 — RESOLVED 2026-09-15.** User's call: the target paper is the
      **ICLR draft** (`~/Code/overleaf/boltzmann-moe-ICLR-2026/`), NOT the
      NeurIPS one, and it migrates fully to Avg11. This turned out to need **no
      re-eval at all**: every `iclr_*` run already carries race+lambada, so all 22
      yield a COMPLETE Avg11. Migration applied and PUSHED to Overleaf (commit
      `Migrate all reported numbers from avg10 to canonical Avg11`). Number-by-number
      record + the claims that changed: **`AVG11_ICLR_MIGRATION.md`**.
      The NeurIPS draft was deliberately left alone — it is archive, still on avg10.
- [x] **32B gptswitch arm FINISHED (2026-09-15)** — job 1647503 DONE at step
      61,035 = the full 32B tokens, final train-loss 2.8277, 1.31 s/step. It had
      **no eval at all**; agent tasked with unshard + full harness. This is the
      completed half of the 400M headline pair and the paper's first real
      400M/32B number. **The Boltzmann half is only at step ~7370 of 61,035
      (12%)** at 6.79 s/step → ~4.2 more days as-is, ~2.6 days with
      `fused_experts` (see ACCEL_FINDINGS on the `boltz-accel` branch).
- [ ] **DO NOT paper-cite the 4 newly-evaluated arms — all are PARTIAL.**
      w1w2_K32_top2 10k/30k (33%), slope90k_1blk 40k/90k, pure_hop_isoP_bal
      16k/30k, iclr_big_hop_sandwich 4k/15k. Their Avg11 is complete (11/11 tasks)
      but not comparable to the finished 30k grid. In particular w1w2_K32_top2
      (41.63, PPL 56.00) does NOT yet resolve Hopfield-vs-W1W2 — it is at a third
      of Hopfield K=32's budget and its PPL says undertrained, not worse. The
      paper's "that arm is training" wording stays.
- [ ] **Fold in the remaining ICLR runs that had NO eval** (agent submitted the
      jobs 2026-09-15): `iclr_decide/w1w2_K32_top2` (**the arm `tab:threeway` says
      "resolves" Hopfield-vs-W1W2 — highest value**), `iclr_slope/slope90k_1blk`,
      `iclr_big/iclr_big_hop_sandwich` (the paper's one remaining acknowledged
      baseline gap), `iclr_balance/pure_hop_isoP_bal`, and the two live
      `iclr_scale/scale32B_*` arms. Skipped: `pure_hop_isoP_bal_DIVERGED_*`.

## 2026-09-15: IF WE ADOPT SINKHORN, THE PAPER'S MODEL CLAIMS MUST BE CORRECTED

- [ ] **Correct every model/routing claim in the ICLR draft if the Sinkhorn variant
      becomes the reported model.** The adopted config differs from every trained arm in
      the paper on THREE axes: routing sign (`e_sign_override: pos`), temperature
      (1.0 vs 0.35), and balancing (`sinkhorn_iters: 3`). Specifically:
      * **`sec/theory.tex`** — add the exact derivation: `p_k ∝ exp(E_k/tau)` is the
        entropy-regularised argmax, and load balance is the DUAL of a capacity
        constraint, i.e. a chemical potential `p_k ∝ exp((E_k - mu_k)/tau)`. This is
        needed in the MAIN paper, not just the appendix, because it changes what the
        routing rule IS.
      * **`app:eval` / `app:routing` / `app:collapse`** — the routing-health numbers
        (effective experts 12.3-12.8 of 16, max share 0.17-0.18) were all measured under
        the INVERTED sign. Re-measure or relabel.
      * **`tab:threeway` / `sec:balance`** — "no gate parameters and no load-balancing
        loss" survives literally (the chemical potential has no gradient pathway and no
        learned gate) but "the energy supplies for free what a learned router must be
        regularised to maintain" does NOT: balance now comes from an explicit dual
        variable. Rewrite as balance-as-constrained-variational-result.
      * **`app:degenerate`** — the `routing_norm: zscore` patch exists because the
        Hopfield energy scale is arbitrary. If we also weight-normalise the energy the
        patch is retired; note whichever we ship.
      * Every `Avg11` number in the paper is from an INVERTED-sign checkpoint. Adopting
        the corrected variant means the whole grid would need re-running before those
        tables describe the shipped model. **Decide explicitly whether the paper reports
        the shipped (inverted) model or the corrected one — do not mix.**
      Refs: `ROUTING_SIGN_BUG_20260915.md`, `HANDOFF.md` §11.

      **DECIDED 2026-09-15: the paper reports the CORRECTED model.** The rerun is
      LAUNCHED — 14 arms, `configs/iclr_sink/*_sink.yml`, jobs 1667233-1667246, on
      preemptable with `save_interval: 1000`, watchdog-tracked, plus the 400M
      `scale32B_boltz_sinkhorn` (1667174). Every arm carries
      `e_sign_override: "pos"` + `temperature: 1.0` + `sinkhorn_iters: 3` +
      `repulsion_tensor_idx: true`, and no `cos_probe_interval`.

      **Scope is 14 arms, not 22.** Two classes need NO rerun and keep their numbers:
      the 11 learned-gate / Switch arms (`TopK_Energy_MoE_MLP`, `MoE`) have no energy
      sign, and the 6 legacy `BoltzmannMoE_Energy_MLP` arms (`w1w2_K32_top2`,
      `iclr_w1w2_K16_top2`, 4x `iclr_seeds/iclr_boltz*`) already route CORRECTLY --
      that class is the one the composable refactor mis-copied. So `tab:frontier`'s
      w1w2 rows, all Switch rows and the `iclr_seeds` replicates stand as published.

      **TRAP for whoever regenerates configs: the sign fix is direction-dependent.**
      Measured overlap(chosen)/overlap(avg): hopfield default `neg` = 0.68 (worst-match),
      override `"pos"` = 1.42 (best-match); composable w1w2 default `pos` = -69.2
      (worst-match), override `"neg"` = +70.2. A blanket `"pos"` is correct for hopfield
      and a SILENT NO-OP for composable w1w2. All 14 launched arms are hopfield.

      **Do not judge a rerun arm's balance before ~step 600** -- effK dips to 8-13
      first (see the transient table in `PROGRESS.md`).

- [ ] **Log the routing sign as a training metric.** `e_sign` is NOT in
      `train_utils.py:get_metrics()`, so the single setting that silently inverted the
      entire 22-arm grid is invisible in every wandb run. `sinkhorn_iters`,
      `repulsion_interval` and `repulsion_space_is_weight` ARE logged; the sign is the one
      that mattered and it is missing. Add `ffwd.e_sign_is_pos` (1.0/0.0) alongside them.
      **Deliberately NOT done on 2026-09-15**: `energy_ff.py` is loaded by 15 running jobs
      and the rule is not to touch running-job model code without a green light. It is a
      pure additive metric with no numerical effect, so it is safe to land at the next
      natural gap. Until then, verify a config's sign by RESOLVING it, not by reading YAML:
      `build_boltzmann_moe(**kw).moe.e_sign` -- note `e_sign` lives on `.moe`, NOT on the
      `FusedMoEContainer` that the builder returns, so a probe reading the container gets
      `None` for everything and will happily report whatever its fallback branch says.

- [ ] **The `nofix` negative control is no longer the published ablation.** Setting
      `temperature: 1.0` on every rerun arm silently removed one of the FOUR knobs that
      the published "top-2 of 16, no fixes" row reverted (`sec/appendix.tex` `app:frontier`,
      Avg11 **43.53**): nofix used tau 1.0 and its baseline used 0.35, so tau was itself one
      of the reverted knobs. The rerun control now reverts THREE knobs at matched tau
      (`hopfield_grad_scale`, `routing_norm`, `repulsion_form`).
      This is defensible and arguably cleaner -- tau stops being a confound bundled into
      "the fixes", and the arm stays genuinely degenerate because the `app:degenerate`
      failure REQUIRES tau ~ 1 (raw Hopfield energies are tiny relative to it, so
      `routing_norm: none` drives routing uniform). But the new number is **not** a
      like-for-like replacement for 43.53, and the appendix must say which comparison it is
      reporting. A true like-for-like rerun would need the baseline back at tau 0.35, which
      contradicts the tau=1.0 decision the whole batch rests on.
      Same trap to check for elsewhere: a bundled ablation whose "reverted" set happens to

- [ ] **PHYSICS SIGN AUDIT: exactly one slot is inconsistent, and fixing it is NOT cheap.**
      Verified numerically by `scripts/energy_sign_slots_20260915.py` against the convention
      `p_k = exp(-E_k/tau)/Z`, `E_FF = -tau*log Z_FF`, `E = E_AT + s*E_FF`, forward `h -= proj(grad E)`:
      * SLOT 1 (definition of E_FF): `_HopfieldExpert` stores `S = mean(gelu(Wx)^2) >= 0`, which
        GROWS with overlap, so the physics energy is `E_FF = -S`. The code calls `S` itself "E"
        (see the `out = +grad E` comment at energy_ff.py:559).
      * SLOT 2 (Boltzmann weight): `e_sign="pos"` gives `p == exp(-E)/Z` EXACTLY. **Correct.**
        `"neg"` does not. So the sign correction we shipped is the physics-correct one.
      * SLOT 3 (free energy): the exported `-tau*logsumexp(logits)` EQUALS `-tau*log Z_FF`
        exactly under `"pos"`. **Correct as originally written** -- I briefly "fixed" this to
        `+tau*LSE` and reverted it; that change was wrong.
      * SLOT 4 (forward): the expert emits `+c*grad S = -c*grad E_FF`, so
        `h <- h - proj(out)` ASCENDS `E_FF` (measured `dE = +1.49e-09` along `-out`).
        **This is the single inconsistency.**
      Attention uses the SAME convention (`attn_out = +grad LSE = -grad E_AT`), so both branches
      of `E = E_AT + s*E_FF` emit `-grad E` and the model is internally self-consistent -- it
      descends `-E = log Z_AT + s*log Z_FF`. A physics-consistent fix must flip BOTH branches,
      i.e. `layer.py` / `energy_attention.py`, affecting EVERY energy model and requiring
      retraining, because `scale_ff` and `proj` are learned and currently absorb the sign.
      **Impact today: NONE.** Slot 4 only corrupts the action/energy aux-loss, and
      `_capture_energy` defaults to False (`layer.py:832`) and is enabled by no `iclr_*` config.
      Do not enable the action loss until slot 4 is fixed -- it would penalise the quantity the
      forward pass increases.
      BOUNDS (asked for separately, and they already hold): `E_FF = -tau*log sum_k exp(S_k/tau)`
      with `S_k in [0, S_max]` gives `-tau*log K - S_max <= E_FF <= -tau*log K`. `S_max` is finite
      because the energy is evaluated on RMSNorm'd `ln_x` with finite `||W||`, so an action loss
      on `E_FF` cannot run to `-inf`.
      CAVEAT: the identity `out = grad E_FF` is exact only for `routing_norm` in
      {`none`,`sqrt_width`} plus the (h-independent) Sinkhorn `mu` and balance bias. The shipped
      arms use `zscore`, where the per-token mean/std make the logits nonlinear in all `E_k`, so
      descent is approximate there. Separately, `cos(expert_out, grad S) = 0.983` not 1.0,
      because `gelu_grad_method="sigmoid"` is a surrogate for gelu' -- so the emitted vector is
      an approximate gradient at the ~2% angular level regardless of signs.
      include a knob the rerun normalises.

- [ ] **True sparsity is NOT implemented** (never was). `top_k` is a post-hoc MASK: all
      K experts' forward AND back projections are computed, then multiplied by a `p` that
      is zero for K-k of them. Two separable pieces:
      * **back-projection** — needs no router (p is known by then), ~13-15% of the step.
        Requires capacity-based dispatch (gather -> one bmm -> scatter-add); the NAIVE
        grouped-loop form was measured at **0.59x, i.e. SLOWER**, so this is a real
        kernel task, not a flag.
      * **forward projection** — cannot be skipped by the exact router at all, since it
        needs all K energies to decide (the `1/2(1+k/K)` floor, "cannot beat 2x"). Needs
        the proxy router (measured 0.942 top-2 agreement at r=8). This is the big one:
        it cuts the **61% elementwise** bucket ~16x, not just GEMM.
      **Do NOT block the 400M Sinkhorn run on this.** Sinkhorn is +0.46%, so the run is
      already ~as fast as the current 400M arm; sparsity is a separate multi-hour change
      with real bug risk (see the activation-checkpointing failure in ACCEL_FINDINGS).

## After Mon 2026-06-01 talk

### A/B test: tanh_exact φ' (DONE — surprising negative result, follow-up needed)

**Result (2026-06-01)**: `tanh_exact` lost decisively to `sigmoid` at h1_boltz_fullsize scale, 30k steps, 7.86B tokens:

| | sigmoid (control) | tanh_exact (treatment) | Δ |
|---|---:|---:|---:|
| Avg | 50.10 | 47.90 | **−2.20pp** |
| WikiPPL | 36.48 | 39.90 | **+3.42** |
| flex-avg | 2.12 | 2.05 | −0.07 |

The fix was supposed to make φ' a faithful derivative of φ. Instead the model trained a worse representation. Two confounded changes:
  (a) φ shape switched from `F.gelu` (exact erf-based) to tanh-approx GELU
  (b) φ' magnitude doubled (legacy was uniformly half the true gelu')

**Default unchanged: keep `gelu_grad_method: sigmoid`.**

**Next A/B (cheap, isolates hypothesis (b))**: keep `phi = F.gelu` and compute `phi' = 0.5·(1 + erf(x/√2)) + x·exp(−x²/2)/√(2π)` (analytic exact derivative of `F.gelu`). One extra erf+exp vs current; still sub-millisecond per layer at our scale. Add a third `gelu_grad_method` value: `"erf_exact"`. Run h1_boltz_fullsize_erfexact and compare to both sigmoid and tanh_exact.

Hypothesis to test: if `erf_exact` beats `tanh_exact` and matches `sigmoid`, the issue is the φ-shape change (F.gelu shape matters). If `erf_exact` ≈ `tanh_exact` (both worse than sigmoid), the legacy half-magnitude φ' was acting as beneficial gradient damping and we shouldn't "fix" it.

Old plan below preserved for reference:

---

### A/B test: tanh_exact φ' (HIGH PRIORITY, cheap)

The legacy code uses `φ' ≈ sigmoid(c·x) · 0.5` paired with `φ = F.gelu(x)` —
but this `φ'` is uniformly **half** the true `gelu'(x)`, and is not actually
the derivative of `F.gelu`. Term2 in `∂E/∂h` is therefore systematically
under-magnitude; weight scales partially absorb this during training, but
it's still a non-faithful gradient.

**Fix landed (gated, default = old)**: `BoltzmannMoE_Energy_MLP` now takes
`gelu_grad_method ∈ {"sigmoid", "tanh_exact"}` (default `"sigmoid"` for
backward compat). Setting `tanh_exact` switches both φ and φ' to the
self-consistent tanh-approx-GELU pair:
  φ(x) = 0.5 · x · (1 + tanh(c·x))
  φ'(x) = 0.5 · (1 + tanh(c·x)) + 0.5 · c · x · (1 − tanh²(c·x))
with c = √(2/π). One extra elementwise mul vs current.

**A/B plan** (run two h1_boltz_fullsize variants from scratch, 30k steps,
4 GPUs preemptable — ~6 h each):
  1. h1_boltz_fullsize_sigmoid (control; current default)
  2. h1_boltz_fullsize_tanhexact (treatment; `gelu_grad_method: tanh_exact`)
Otherwise identical configs. Compare avg / WikiPPL / GSM8k flex-avg.

Hypothesis: if phi' magnitude was being absorbed by W2 scale, we expect
small net effect on quality (≤ ±0.3pp avg). If the *shape* matters
(GELU' has a peak around x≈0.6 that sigmoid·0.5 misses), expect a
meaningful improvement on representation quality. Either way, **the
correct gradient should be the new default once we have data**.

If tanh_exact ≥ sigmoid: switch default to `tanh_exact` for new training,
keep `sigmoid` available for legacy checkpoint reproducibility.

Test (already run): `experiments/boltzmann-moe/tests/test_gelu_grad_method.py`
verifies (1) backward-compat (sigmoid path bit-identical to legacy),
(2) tanh_exact φ' matches autograd, (3) the two paths produce different
outputs on the same weights. Run via bsub when iterating on the model.

---


### MoE hyperparam search (queued for later)
Goals: find the best Boltzmann MoE config at h1 scale (145M, d=768, K=4, K=8, K=16)
**without** consuming so many GPUs at once that we get deprioritized in the queue.

Submit at most 3–4 jobs at a time; let them complete before the next batch.

Axes to sweep (in priority order):
- **Sparsity**: `top_k` ∈ {2, 3, 4} of K=4; {2, 4, 8} of K=8; {2, 4, 8, 16} of K=16
  — most promising follow-up: top-3 of 4 may close most of the 1.5pp acc gap
  while preserving sparse-train benefits ([[project_sparse_boltz_perf]]).
- **Temperature τ**: {0.5, 1.0, 2.0, 4.0}. Higher τ softens routing distribution
  → less drastic truncation → smaller flex-acc gap for sparse top-k.
- **Repulsion λ**: {0.0, 0.01, 0.1, 0.3, 1.0}. B4/B5 already showed 0.1 is good
  load balance; 0.3+ untested with new routing scale.
- **Number of experts K**: {4, 8, 16, 32} at h1 scale. Bigger K + smaller per-expert
  gives sparse a clearer FLOPs win (44% saving at K=16, top_k=2 vs 25% at K=4).
- **Repulsion target**: cosine sim (current) vs L2 distance vs orthogonality
  loss on term1.
- **Anneal top_k**: train soft → gradually reduce top_k over training. Should
  recover both the soft-train gradient signal and the sparse-eval efficiency.

### Sparse Boltzmann MoE production kernel
- Implement scattermoe-style fused dispatch for top-k Boltzmann gradient terms.
  Per-expert PyTorch loop is correct but loses on backward at every K (see
  [[project_sparse_boltz_perf]]). Need this to make sparse routing a real
  training-FLOPs win, not just an inference-FLOPs win.
- Or: try `torch._grouped_mm` (CUDA 12+) as an easier first step.

### Better integration of routing diagnostics into wandb
- Per-layer `effective_n_experts`, `n_dominant_experts`, `max_expert_load` are
  already logged. Add a heatmap visualization or per-step routing-entropy
  histogram for easier spotting of collapse runs.

### Larger scale Boltzmann MoE
- 580M continues; results @ 14k & 18k are encouraging. Plan:
  - 1.5B Boltzmann MoE (K=8, d=2048) at 65B+ tokens once 580M finishes
  - Try repulsion λ scaling with K: λ ∝ 1/K?

### Open questions
- Why does the soft-trained model lose 6 PPL when truncated to top-2 at eval?
  Hint: sparse-trained model doesn't pay this cost ([[project_sparse_boltz_perf]]).
  Hypothesis: the soft model's bottom-k experts contribute small-but-essential
  refinements during training; truncating at eval drops these. Test by
  measuring per-expert L2 norm of `term2_e * p_e` in trained vs sparse-trained.
- Is 580M's GSM8k_cot flex jitter (2.88% @ 14k → 2.27% @ 18k) within stderr
  (±0.4pp) or a real regression? Run cot eval at step 22k, 26k to see trend.

- [ ] **Re-derive the pure-energy paper rows from the RECALIBRATED checkpoints.** The published
      `app:frontier` pure rows and everything I measured before 2026-09-15 evening evaluated a
      mu-tilted router with no tilt, costing 1.587 nats (eval token PPL 147.8 vs 28.5). Those
      corrected-sign pure numbers (Avg11 36.16 / 38.76 / 39.04) are therefore NOT the model's
      quality and must not go into a table. The `unsharded_mucal2/` evals supersede them.
- [ ] **Tell colleagues `sinkhorn_persist_mu: true` + `sinkhorn_mu_iters: <layer_iterations>`
      is REQUIRED** for pure/recurrence-heavy stacks. This supersedes the earlier
      "sinkhorn for hybrids only" note, which was based on the pre-fix measurements.

## 2026-09-16 (part 2) — 400M launch queue and the WSD schedule

- [ ] **THE GATE: sandwich proxy-selection Avg11 vs the dense 43.36.** Job `1716267` (PEND as of
      20:55, no free exclusive GPU; `grp_preemptable` only 736/6144, so it is GPU scarcity not
      quota). Near **−0.03pp** (hybrid-like) ⇒ launch `sw400_sparse`; near **−0.68pp** (pure-like)
      ⇒ do not, and §12.15's mu-based prediction does not transfer.
      **Before reading any delta, verify `ablate/C_proxysel/harness_results*.json` EXISTS and that
      `compute_avg11.py` cites that path** — the first attempt reported +0.00pp off the dense
      checkpoint (§12.16a/b). The job now `ls`-es the path before and after and prints
      GATE-CLEAN / GATE-FAIL.
- [ ] **Launch the two 90k WSD arms** — `configs/wsd90k/wsd90k_pure_it4.yml` and
      `wsd90k_sandwich.yml` (committed `0cc32ebe`, all 10 pre-flight checks pass).
      **8 GPUs each, non-negotiable**: at 4 they silently halve tokens/step to 131072, the error
      that made `iclr_big_hop_pure_sink`'s −0.99pp unreadable. ~3.3 and ~3.1 days respectively.
      Use `submit_selfresuming.sh` (resolves `load_args` per requeue) — do NOT add `load_args` to
      the configs.
      ⚠ 16 new GPUs on top of ~28 already committed, on a queue that currently cannot place a
      single-GPU eval. Consider starting one and watching its placement before committing both.
- [ ] **`sw400_sparse` — committed and pre-flighted, NOT launched.** Gated on the item above.
- [ ] **Validate the decay-branch recipe end-to-end** (bottom of `wsd90k_pure_it4.yml`).
      `load_optimizer: true` + `load_lr_scheduler: false` is legal per `arguments.py:151-157`, but
      the `load_dataloader_state` interaction with `load_starting_iteration: false` is unchecked.
      This is what makes a partial 90k run harvestable, so it should be proven on a throwaway
      checkpoint BEFORE it is needed under deadline pressure.
- [ ] **Read out `sw2k_sparse_c10x` (1714132) at step 2000** — the completed-decay endpoint against
      `sw2k_sparse`'s at-peak 3.9413. §12.17 predicts the gap stays ~0.11, not that it grows.
- [ ] **Decide the pure 400M iteration count for a 90k budget.** `it4` is in the config because it
      is the only shape measured at 400M, but §12.14's ordering is established at ~1000 steps and
      the gap was narrowing. The iso-FLOP hedge is a three-line change (`layer_iterations: [8]`,
      `sinkhorn_mu_iters: 8`, `intermediate_size: 143360`) — all three together or not at all.
- [x] **`bench_proxysel_one.sh` now fails loudly** — tracks failures, verifies each results file
      landed, exits nonzero so LSF stops reporting "Successfully completed" for a job that
      produced nothing. `--batch_size` is now a parameter defaulting to 2 (4 OOMs at 400M), and
      each arm gets its own `--use_cache` so a preemption resumes.
      ⚠ **The rest of that item was WRONG and is withdrawn — see HANDOFF 12.16c.**
      `compute_avg11.py` does NOT glob outside the directory it is handed; `resolve_results_path`
      (compute_avg11.py:80-92) searches only under that path and `sys.exit(1)`s otherwise. It
      behaved correctly throughout. `eval_harness.py`'s `--output_path` is not at fault either.
      **Do not "fix" either of them.** The phantom +0.00pp came from a MONITOR filter that grepped
      `Avg11 =` globally and labelled two DENSE evals (from two invocations of the script) as
      dense-vs-proxy. The fix is in how results are monitored: anchor every number to its arm
      label, and never assume one number per arm per job.

## 2026-09-16 (part 3) — after the sparse pivot

- [x] **`sparse_start_step` implemented and tested** (`99d27a3f`, `7c4033db`) — dense→sparse in ONE
      job. Use `wsd90k_*_1job.yml` for new arms; it removes a submission and a manual handoff per
      arm, which matters most at 700M/1B. See HANDOFF 12.22 for the mbs caveat.
- [x] **`s90k_pure_T12_sparse` killed AND deregistered** — worst 134M repulsion settings
      (subsampled output-space, `expert_cos_abs_mean` 0.5664).
- [x] **`t90k_pure_T12` / `t90k_hybrid_K32top2` deregistered**, `sw2k_dense` killed.
      ⚠ The two `t90k` arms are STILL RUNNING (~26 h left on pure at 34%) and hold 4 GPUs each.
      Deregistering only stops resubmission. Kill them if the GPUs are needed.
- [ ] **DECIDE: realign the arms to scale32B?** scale32B runs **61035 steps at 524288 tok/step =
      32.0B tokens**; ours are 90000 at 262144 = 23.59B. Matching STEPS at our tok/step gives only
      16.0B — half — which is the `iclr_big_hop_pure_sink` error that made a −0.99pp delta
      unreadable. To match both, set `micro_batch_size 4 / ga 4` at 8 GPUs (524288) and 61035 steps,
      WSD becoming 1000 / 54000 / 6035. Costs ~2.3 d at 8 GPUs or ~1.13 d at 16 — the 16-GPU option
      is FASTER than the current 90k/8-GPU plan and trains on 36% more tokens.
      Note 90k was never a validated budget: it came from the `slope90k_*` lineage whose schedules
      were broken (60000 steps pinned at the floor, 12.4).
- [ ] **134M sparse LR: no better value is validated.** 2e-3 stands. 1e-2 wins by 0.04–0.05 nats
      when it survives (3.7872/3.8018 vs 3.8289) but the one SPARSE arm at 1e-2 diverged to
      grad_norm 2.4e7 while dense arms at 1e-2 mostly survived — sparse looks less stable at high
      LR. The grid has NOTHING between 2e-3 and 1e-2; 4e-3 is the untested midpoint.
- [ ] **Phase-1 handoff threshold**: `proxy_topk_agree > 0.75`, reference 0.1193→0.7906 in 280 dense
      steps (job 1709497). Judge `expert_cos_abs_mean` (weight-space repulsion, the 12.13 experiment
      never run) only at step ~300-500 — it is meaningless at step 30-100.

## 2026-09-17 — from writing the ICLR ablation section (`app:ablations`, Overleaf `0718b7c`, NOT pushed)

- [x] **New appendix section written** — six groups (sign, balance, mu, depth-vs-width, LR,
      proxy-selection cost), all `Avg11`, compiles clean at 5.5 in. Local commit only.
- [ ] **PUSH DECISION**: `0718b7c` is committed in
      `~/Code/overleaf/boltzmann-moe-ICLR-2026` and needs the user's explicit go-ahead
      before `git push`.
- [ ] **FIX `app:routing`'s sign sentence** — as written it states the *inverted* convention
      ("$s_k=-E_k$ for Hopfield ... lower is better"). With `app:expert-forms`' energies the
      correct rule is $s_k=+E_k$ for Hopfield and $s_k=-E_k$ for the W1W2 form as written there.
      This is a method-section error, not a results one, and it is the most visible remaining
      artefact of the sign bug.
- [ ] **Separate the sign from the gradient scale.** `app:degenerate` blames `c` (mean vs
      1/sqrt(I_e)); §11.2 blames the routing sign. The two arms differ in K *and* `c`, so nothing
      is attributed. A {sign} x {c} 2x2 at one shape, ~300 steps, would settle it. Until then
      `app:degenerate` must not be presented as an energy-scale finding.
- [ ] **Reconcile the routing-collapse numbers.** `app:collapse` reports effK 5.18/16 for the
      single-block arm and 2.52 for "no fixes"; `audit_expert_collapse.py` (quoted in
      `configs/iclr_balance/pure_hop_isoP_bal.yml`) reports 1.38 for pure isoP and 2.52 for pure
      1blk. One label is wrong. Re-run the audit on both checkpoints and pick one probe.
- [ ] **Strike the §12.14 cross-scale claim** ("134M T12 beat the 400M 4-iteration arm"): the 400M
      arm is `iclr_big_hop_pure_sink`, which §12.3 already strikes for half tokens. Either re-run
      it token-matched at 8 GPUs or drop the comparison. The within-scale statements stand.
- [ ] **Discriminate the balancer result** (the one new number in the section): a pure arm with
      `sinkhorn_iters: 3` but **mu frozen after warm-up**. That separates "balance beyond effK ~14
      buys nothing" from "per-forward re-solution injects batch-dependent noise into a recurrent
      fixed-point iteration". ~30k steps at 4 GPUs to be comparable with the pair above.
- [ ] **`app:accel-train`'s activation-footprint figure is the inference one.** It quotes
      `c_f k/K` = 6.4x; during training the buffer is sized by `p = candidates + explore = 4`, so
      it is 3.2x (§12.21). The section now states both — check the earlier sentence still reads
      correctly once someone edits `app:accel-train`.
- [ ] **`app:findings` is stale on the sandwich**: it says the sandwich variant "has not
      finished". It has — 43.36 Avg11 (mu-recalibrated). One-sentence fix, deliberately not made
      while adding a new section.
- [ ] **No completed WSD run exists.** `app:abl-lr` carries a `\CC` saying so; `wsd90k_*_sparse`
      (1718594 / 1718598) are the arms that would remove it.

## Sign-convention unification (post-deadline)

- [ ] **Make the stored energy BE the state energy `S_i = -overlap`, and default `e_sign="neg"` for
      both expert kinds.** Full proposal, risks and sequencing in
      `experiments/boltzmann-moe/SIGN_CONVENTION_UNIFICATION.md`.
      **Not a bug** — with today's `e_sign_override`s the arms implement the Boltzmann form
      correctly. It is a naming/defaults inconsistency: the variable called `E_k` holds `-S_i` for
      hopfield and `+S_i` for w1w2, so the correct `e_sign` differs per kind and every config must
      carry an override (CLAUDE.md pre-flight 8).
      ⚠ **Requires a config-version guard**: every existing checkpoint stores
      `e_sign_override: "pos"` for hopfield, which becomes WRONG once the storage sign flips.
      Without the guard every published number re-evaluates with an inverted router.
      Gate on a bit-identity test (tolerance 0, this is a pure relabelling) modelled on
      `scripts/test_logits_refactor_equiv_20260916.py`.
      Do NOT touch the two live 61035-step arms; they must finish on current code.
