# Boltzmann MoE — Experiment Guide

> ## 📄 ACTIVE PAPER — the ICLR draft is the ONLY paper we are writing right now
>
> **Path: `/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`**
> (Overleaf remote `https://git@git.overleaf.com/6a9ace75a92fce262f38ec18`, branch `main`.)
> Files: `main.tex` + `sec/{intro,theory,experiments,appendix}.tex`.
> Table locations: `tab:frontier`, `tab:pure`, `tab:cost`, `tab:threeway` in
> `sec/experiments.tex`; `app:frontier`, `app:eval` (metric definition), `app:scale400`,
> `app:threeway`, `tab:attribution` in `sec/appendix.tex`.
>
> **ALL of its numbers are `Avg11`** as of 2026-09-15 (migrated from `avg10` that day; see
> `AVG11_ICLR_MIGRATION.md`). MMLU and GSM8K-CoT are reported as SEPARATE columns, never in
> the mean.
>
> **⚠ Do NOT read the older drafts unless a task specifically requires them — they burn
> context and they use SUPERSEDED metric conventions.** The NeurIPS draft
> (`~/Code/energy/energy-GPT-neurips2026/`) and the talk
> (`~/Code/overleaf/energy-GPT-reformulation-2026/`) are **reference/archive only**; their
> tables are still on `avg10`/`avg9`. If you find yourself opening `paper_v2.tex` or
> `boltz_moe.tex` to answer an ICLR question, stop — you are in the wrong paper.

> ## 🔴 TWO 2026-09-15 FINDINGS THAT INVALIDATE EARLIER CONCLUSIONS — READ BEFORE TRUSTING ANY ROUTING OR s/step CLAIM
>
> **1. The composable Boltzmann router is SIGN-INVERTED.**
> `ROUTING_SIGN_BUG_20260915.md`. Measured on a trained checkpoint: the router selects
> the experts with the **LOWEST** `||gelu(W_k x)||^2` — 0.62x the average expert's
> overlap, exactly the lowest-overlap set. `_HopfieldExpert` stores an energy that
> GROWS with overlap and `e_sign="neg"` then picks the smallest. The legacy
> `BoltzmannMoE_Energy_MLP` does it correctly (`E = +overlap`, `softmax(+E/tau)`).
> Affects **every `EnergyFF_BoltzmannMoE` run**: the whole 22-arm ICLR grid and the
> live 400M. NOT the learned-gate/Switch arms, not gptswitch, not the legacy
> `h1_*`/`b*`/`c*` series.
> Correcting it (`e_sign_override: "pos"`) revives the FF branch **20–100x**
> (`ffwd/output_norm` 0.03 → 12) and **collapses routing** (effK 8.5 → 1.9 of 32),
> because anti-routing was acting as an accidental load balancer. So TWO paper claims
> are affected: "the energy-FF branch is essentially dead" (caused by the bug) and
> "Boltzmann routing does not collapse and needs no load-balancing loss" (may hold
> only because of it). A third — "energy routing is at parity with a learned gate" —
> was measured under the inverted sign. **Do not restate the routing-health results.**
> Note: across 300 steps the sign made NO resolvable `lm_loss` difference
> (+0.006..+0.013 vs a 0.038 noise floor), which is why it went unnoticed.
>
> **2. The "~5x slower than gptswitch" figure is substantially a HOST-PLACEMENT artifact.**
> `PLACEMENT_ARTIFACT_20260915.md`. Same unfused code, same config, resumed from the
> same checkpoint on a different host pair: **6.72–6.81 → 2.45–2.53 s/step**, i.e.
> 2.7x from placement alone. Identical GPU model/driver/`gpu_factor`, no MIG; sibling
> contention ruled out (median 6.739 before gptswitch finished vs 6.784 after).
> Likely dataloader starvation from shared CPU slots — per-rank GPU-busy is 1.71 s, so
> 25% utilisation at 6.76 s wall vs 68% at 2.49 s. **Consequences:** the real ratio to
> gptswitch is ~1.9x (still not placement-controlled), and the "launch-overhead bound /
> 79.5k kernels" reading is largely void at 68% utilisation.
> **Rule: no multi-node s/step claim is admissible unless placement-controlled** —
> same hosts for both arms, or several placements with the spread reported.
>
>
> **3. The balance property has a PRINCIPLED replacement, and Sinkhorn beats what we ship.**
> `ROUTING_SIGN_BUG_20260915.md`, `HANDOFF.md` §11.3-11.6. Load balancing is the
> **chemical potential** `p_k ∝ exp((E_k - mu_k)/tau)` — the dual variable of the
> batch-marginal constraint, with no gradient pathway and no learned gate, so the "no
> auxiliary loss" claim survives it. `balance_rate` (proportional control) pinned at the
> +-1.0 clamp in every arm; **`sinkhorn_iters` solves the dual exactly** and at 134M
> reaches `effK` **26.6/32** against the shipped arm's **5.0** (max_share 0.072 vs
> **0.353** — the shipped config has one expert taking 35% of tokens), with the best
> loss and best expert diversity of the three, at `mu = 3.44` i.e. 3.4x past the clamp.
> Trends: shipped balance **degrades** (effK 19.9 -> 5.7), both corrected+balanced arms
> **improve**.
> **Energy stability:** the feared runaway does NOT happen — the energy is evaluated on
> RMSNorm'd `ln_x`, and the SHIPPED inverted arm carries 2.5-3x MORE energy (0.59 peak)
> than either corrected arm. No activation change warranted.
> **Do not draw conclusions from these probes before ~200 steps** — a step-30 reading of
> the tau sweep gave the wrong answer and had to be retracted.
>
> Acceleration work: `ACCEL_FINDINGS_20260915.md`. `fused_experts` is EXACT (1.227e-15)
> and 1.61x at **4 GPU / 1 node**, but it **WEDGES at 16 GPU / 2 nodes** and is
> reverted on the live arm — validated single-node only.

> ## ⚡ TRUE SPARSITY — measured 2026-09-16. Read before quoting any sparsity or FLOPs number.
>
> `HANDOFF.md` §12.9 has the full tables. The three things that get misquoted:
>
> **1. `sparse_forward` is the one that works: 4.69x** (compiled, forward+backward, H100) on the
> pure_T12 block shape at its real per-call size. That is **past the `1/2*(1+k/K)` "cannot beat 2x"
> floor**, which applies only to an exact router — the rank-r proxy is what breaks it.
> `sparse_backproj` alone is 1.18-1.35x and is redundant with it.
>
> **2. THE SPEEDUP DEPENDS ON tokens/call AND THE SIGN FLIPS.** At 4096 tokens/call
> `sparse_forward` LOSES on hybrid shapes (0.41-0.49x compiled); at 16384 it wins 2.1-2.4x.
> Dispatch overhead is O(T*k) regardless of `I_e`; the saving is O(T*(K-k)*I_e). **Never quote a
> sparsity speedup without the per-call token count** — `micro_batch_size x sequence_length`, NOT
> tokens/step. The 400M hybrid at `micro_batch_size: 1` sits in the losing column.
>
> **3. The proxy's prediction error is the ENTIRE approximation.** With an oracle proxy the sparse
> path reproduces the dense output to 4e-16 with the selection free
> (`scripts/test_sparse_forward_20260916.py`). So there is nothing to audit in the dispatch — audit
> the proxy. `renormalize_topk: true` and `routing_norm: none|sqrt_width` remove the two smaller
> approximations by config rather than by code.
>
> Training a sparse arm needs a DENSE phase first: the proxy is distilled against the exact all-K
> routing distribution, which the sparse path does not compute. Fit post-hoc on a trained
> checkpoint with `scripts/calibrate_proxy_router_20260916.py`.

> ## 🖥 ANY GPU WORK GOES THROUGH bsub — INCLUDING ONE-OFF TESTS AND BENCHMARKS
>
> **An interactive session is frequently on a CPU-ONLY compute node.** Verified 2026-09-16 on
> `p2-r05-n3`: `hostname` says compute node, and `nvidia-smi` says **"No devices were found"**.
> So "I am on a compute node, I can run directly" is only true for CPU work. A GPU script run
> in-shell there either dies or silently falls back to CPU — and a *timing* script that falls
> back to CPU returns numbers that look plausible and are meaningless.
>
> **Submit every GPU test.** Use the saved wrappers rather than hand-rolling a bsub each time:
>
> | script | use |
> |---|---|
> | `scripts/bsub/submit_gpu_test.sh <job> "<cmd>" [gpus] [wall] [mem]` | generic one-off GPU test/benchmark |
> | `scripts/bsub/bench_sparse.sh` | the exact sparsity wall-clock sweep behind §12.9 |
>
> ```bash
> bash scripts/bsub/submit_gpu_test.sh mytest \
>     "python experiments/boltzmann-moe/scripts/test_something.py --train"
> ```
>
> **`preemptable` is the right queue for tests** (`-q preemptable` REQUIRES `-G grp_preemptable`).
> Tests are short, a preempted one costs only a resubmit, and `grp_ebm` is capacity-limited —
> check it with `blimits`, NOT `bjobs`. Logs go to `$HOME/bsub_logs/`, never `$HOME`.
> Then verify 30–60 s later: `bjobs -J <job>` must show `RUN`, not `EXIT`.
>
> Only these stay safe to run in-shell: file edits, `git`, `ls`/`find`/`grep`,
> `bsub`/`bjobs`/`bkill`, LaTeX, and CPU-only Python (a float64 exactness test on tiny tensors is
> fine — a timing run is not).

> ## 🛑 CONFIG PRE-FLIGHT — RUN THIS BEFORE ANY LONG RUN. NO EXCEPTIONS.
>
> Every buggy long run in this project came from a config that *looked* right. A week of
> deadline does not buy time to re-run; it makes each wasted run unaffordable. Before launching
> anything longer than ~1 h, verify EVERY line below against the config as parsed, not as
> written, and against the arm it will be compared with.
>
> **1. Schedule covers the run.** `num_warmup_steps + num_constant_steps + num_decay_steps`
> MUST equal `num_training_steps`. The `slope90k_*` arms set 2000+28000=30000 against 90000, so
> **60000 steps ran pinned at the 2e-4 floor** and bought 0.015–0.019 nats across 45–51k steps.
> This bug is in the PUBLISHED configs, so it corrupts a paper claim, not just a rerun.
>
> **2. Tokens/step matches the arm you will compare against.**
> `tokens/step = GPUS × micro_batch_size × gradient_accumulation_steps × sequence_length`.
> GPUS is NOT in the config — it comes from the launcher (watchdog conf field 5, or the bsub
> `-gpu num=N/task` × tasks). `iclr_big_hop_pure_sink` was registered at 4 GPUs while its
> published counterpart ran at 8, so the rerun saw **half the tokens** and its −0.99pp delta was
> read as a sign effect when it was an undertraining effect.
> Verify empirically, not from the config:
> `tokens/step = billion_tokens_per_day × 1e9 × step_time / 86400` from the training log.
>
> **3. GPU count is what you think.** `nexec_host` is HOSTS, not GPUs. `bjobs -l` shows
> `num=8/task` and `gpus=0..7`; 2 hosts × 8 = 16. Do not infer 4/host.
>
> **4. `load_args` present iff resuming.** Absent ⇒ a restart silently begins at step 0 (LSF
> requeues a preempted job on the SAME jid, re-running the original command). Present but
> pointing at an empty dir ⇒ a fresh run fails. The watchdog appends it only when a checkpoint
> exists, which is correct; hand-written configs must match that.
>
> **5. `latest_checkpointed_iteration.json` names a directory that EXISTS.** With
> `max_to_keep: 2`, a from-scratch restart writes a LOW checkpoint, points at it, and pruning
> (which keeps the HIGHEST) deletes it — every subsequent start then dies instantly.
>
> **6. Names and paths are unique.** A distinct `save_path` AND wandb `name` per arm. Beware
> **prefix collisions** when globbing logs or results: `foo_*` matches `foo_sink_*` and
> `foo_s7_*`. Anchor on the numeric job id (`foo_\d+\.stderr`). This produced two wrong
> numbers in one session, both plausible-looking rather than erroring.
>
> **7. Knobs reach the model, not just the YAML.** `get_mlp_block` forwards kwargs EXPLICITLY,
> so a field can parse onto the pydantic args object and never reach the builder. Verify by
> RESOLVING: `build_boltzmann_moe(**kw).moe.e_sign` — and note `e_sign` lives on `.moe`, not on
> the `FusedMoEContainer` the builder returns, so a probe reading the container gets `None` for
> everything and its fallback branch will report whatever you told it to.
>
> **8. The sign is direction-dependent per expert kind.** hopfield needs
> `e_sign_override: "pos"`; composable w1w2 needs `"neg"`. A blanket `"pos"` is a silent no-op
> on w1w2.
>
> **9. Required for Sinkhorn arms:** `sinkhorn_persist_mu: true` and
> `sinkhorn_mu_iters` = this block's `layer_iterations` entry. Without them the router is
> trained tilted and EVALUATED untilted: **+1.587 nats at 8 iterations, +1.827 at 12**, ~0.003
> for a hybrid. Also `repulsion_tensor_idx: true`, and NO `cos_probe_interval`.
>
> **10. Diff against the arm you are copying.** `diff <(grep -vE '^\s*#|^\s*$' old.yml)
> <(grep -vE '^\s*#|^\s*$' new.yml)` and account for EVERY line. Unexplained differences are
> bugs; expected-but-absent differences are also bugs.
>
> Then, 30–60 s after launch, confirm the arm reached a first step and that its logged
> `learning_rate` and tokens/step are the intended values.

> ## ⚠ METRIC CONVENTION — READ BEFORE QUOTING ANY "Avg" IN THIS FILE
>
> **CANONICAL (2026-09-14 onward): `Avg11`**, the paper `tab:scaling` recipe, the
> SAME number the EGPT-RL / FET colleagues headline — so ours are directly
> comparable to theirs. **Compute it with `experiments/eval_scripts/compute_avg11.py`**
> (validated to reproduce the colleagues' stored Avg11 on the two shared
> `math_egptdual` seed checkpoints to +0.004pp). Recipe:
> - **11-task unweighted mean.** `acc_norm`: arc_challenge, arc_easy, hellaswag,
>   openbookqa, piqa, sciq. `acc`: boolq, copa, winogrande, **race**, **lambada_openai**.
> - **MMLU (acc) and GSM8K-CoT (flex-extract) are reported SEPARATELY**, never averaged in.
> - Source of truth: `~/Code/GPT-experiments/projects/EGPT-RL/RESULTS.md:247-249`.
>
> **Two older conventions appear in tables below — never mix them with Avg11:**
> - **`avg9`**: 9 tasks, MMLU EXCLUDED, race/lambada absent (the pre-`pyarrow>=20`
>   Arrow/parquet bug, fixed 2026-08-03). `acc_norm` on all six that report it.
> - **`avg10`** (old `compute_aggregates.py`): 10 tasks, **MMLU INCLUDED**,
>   race/lambada EXCLUDED. `compute_aggregates.py` is now **DEPRECATED for headlines**.
>
> **Avg11 runs ~3pp BELOW avg10** because race (~0.28) and lambada (~0.23) sit near
> chance at our scale — a scoring-convention gap, not a model effect. avg10 is
> 1.2–2.7pp below avg9. **Do not put avg9 / avg10 / Avg11 numbers in the same table.**
>
> Restated ICLR-grid Avg11 table: `python experiments/eval_scripts/compute_avg11.py <run_dir>`
> (all 22 iclr_* runs already have race+lambada, so all yield a COMPLETE Avg11).
> Bulk-restate EVERY stored eval to Avg11 (marks INCOMPLETE runs):
> `python experiments/eval_scripts/restate_to_avg11_20260914.py --md`.
> Legacy avg9→avg10 restatement: `restate_avg9_to_avg10_20260912.py --md` / `AVG10_RESTATED.md`.

This directory documents the **BoltzmannMoE Energy FFN** experiments (series B1–B5).
The goal was to replace the standard Energy\_MLP feedforward in deep EGPT with a
Boltzmann-weighted mixture of experts and study whether the routing collapses.

> **Recovered session dialogue** (2026-09-07): the original pre-June boltzmann-moe
> Claude Code session was auto-purged (Claude Code's `cleanupPeriodDays`, default 30).
> The surviving boltz discussion (h1 Boltzmann-MoE config, gradient-through-energy,
> FPT scaling) was extracted to `recovered_boltz_sessions/`:
> - `boltz_10073753_jun14-jul16.md` — 179 msgs, the substantive record (launched from `Code/GPT-experiments`)
> - `boltz_c3ee2c6c_apr30-aug11.md` — 4 incidental mentions (paper baselines)
> - `ADMIN_snapshot_recovery_request.md` — draft email for GPFS snapshot recovery of files purged today

---

## What is BoltzmannMoE?

Each of the 12 EGPT blocks gets a new FFN type where the energy is the
log-partition function over K parallel experts:

```
E_moe(h) = log( Σᵢ exp(Eᵢ(h)) )
∂E_moe/∂h = Σᵢ pᵢ(h) · ∂Eᵢ/∂h      pᵢ = softmax_i(E(h) / τ)
```

Each expert is an Energy\_MLP: `Eᵢ(h) = φ(W1ᵢh)ᵀ(W2ᵢh)`.
The routing energy is `Eᵢ = h · term1ᵢ` where `term1ᵢ = φ(W1ᵢh) @ W2ᵢᵀ`
(contracting only the first gradient term; the full gradient would add a
spurious Hessian contribution).

**Iso-parameter**: K experts each of size `intermediate_size // K` → same total
params and FLOPs as one Energy\_MLP with the same `intermediate_size`.

---

## Key source files

**Two lineages** — the B/C/H-series and all `h1_*` checkpoints use the *legacy*
class; every `math_fet_*` / `EnergyFF_*` checkpoint uses the *composable* refactor
(2026-06-28). See `HANDOFF.md` §3 for how to tell them apart and why it matters.

| File | What it does |
|------|-------------|
| `.../mlp_blocks/mlp.py:540` | **legacy** `BoltzmannMoE_Energy_MLP` (w1w2 experts) |
| `.../mlp_blocks/energy_ff.py` | **composable** family: `FFEnergyBase`, `W1W2FFEnergy`, `HopfieldFFEnergy`, `BoltzmannMoEFFEnergy`, `FusedMoEContainer`, `build_boltzmann_moe` |
| `lm_engine/hf_models/config/mlp.py` | `_BoltzmannMoEEnergyMLPArgs` (legacy) and `_EnergyFFBoltzmannMoEArgs` (composable) |
| `.../mlp_blocks/__init__.py:113` / `:179` | `get_mlp_block` dispatch — legacy / composable |
| `lm_engine/hf_models/config/__init__.py` | Registry entry (`_MLP_CONFIG_CLASSES`) |
| `lm_engine/train_utils.py:73` | `get_metrics()` logging for routing collapse metrics |
| `lm_engine/arguments.py` | `SaveArgs.max_to_keep: int | None` (pydantic fix) |
| `.../sequence_mixer_blocks/energy_attention.py:75` | `head_dim` (optional; decouples from `d/num_heads` for over-complete heads) |

---

## Config fields for `BoltzmannMoE_Energy_MLP`

```yaml
mlp_blocks:
  - mlp_type: BoltzmannMoE_Energy_MLP
    intermediate_size: 16384   # total = n_experts × per_expert_I
    n_experts: 16              # number of experts
    temperature: 1.0           # Boltzmann softmax temperature
    repulsion_coef: 0.0        # 0 = off; 0.01–0.1 = stochastic contrastive repulsion
    n_repulsion_pairs: 4       # random expert pairs sampled per step for repulsion
    dropout: 0.0               # dropout on intermediate activations
    add_bias: false
```

**IMPORTANT**: always set `max_to_keep: 2` in `save_args` — the field is
`int | None` and pydantic will reject `null` from saved checkpoint configs
without this fix.

```yaml
save_args:
  save_path: ...
  save_interval: 5000
  max_to_keep: 2
```

---

## Experiment configs

All located in `configs/boltzmann_moe/`.

### B-series (deep EGPT, iso-param, d=768):

| Config | repulsion_coef | dropout | weight_decay | Purpose |
|--------|---------------|---------|-------------|---------|
| `b1_boltz_moe_16x1024_d768_lr2e3.yml` | 0 | 0 | 0.1 | Baseline — does routing collapse? |
| `b2_boltz_moe_repulsion_16x1024_d768_lr2e3.yml` | 0.01 | 0 | 0.1 | Weak repulsion |
| `b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3.yml` | 0.01 | 0.1 | 0.3 | Dropout + high WD |
| `b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3.yml` | 0.1 | 0 | 0.1 | Strong repulsion (best load balance) |
| `b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3.yml` | 0.1 | 0.1 | 0.3 | Combined |

All use: `d=768`, 12 blocks, 16 experts × 1024 = **~422M total params**. Note: all B variants (original iso-param code) underperform V1 EGPT d=768 (143M) due to the 21:1 FFN:Attn imbalance — but see the revised results at the bottom: the 1/√I routing-scale fix lifts the B4 rerun to 0.494 (≈ V1-400M EGPT).

### C-series (design fixes, d=768):

| Config | Description |
|--------|-------------|
| `c1_topk_energy_moe_4x2048_top2_d768.yml` | 4 full-size experts, top-2 sparse, load-balance loss |
| `c2_surrogate_boltz_16x1024_d768.yml` | Boltzmann routing + learned linear surrogate router |
| `c3_attn_moe_2x_d768.yml` | MoE on **attention** (2 energy-attn experts), normal FFN |
| `c4_paired_unit_moe_2x_d768.yml` | 2 paired (attn+FFN) units, balanced FFN:Attn |

### H-series (hybrid GPT+EGPT-MoE, **best results**):

6 standard GPT layers + 1 recurrent EGPT block (×6). MoE only in the EGPT block.

| Config | Description | Avg acc | WikiPPL |
|--------|-------------|---------|---------|
| `h1_boltz_moe_fullsize` | **BEST**: Boltzmann routing, 4 full experts×2048 + 1/√I routing scale | **0.501** | **36.5** |
| `h1_topk_egpt_moe_d768.yml` | 4 full experts×2048, top-2 (learned router) | 0.499 | 39.8 |
| `h1_gptmoe_boltz_egpt` | Switch-MoE GPT prefix + Boltzmann EGPT (full-size) | 0.486 | 35.5 |
| `h1_boltz_topk2` | Sparse top-2 Boltzmann in EGPT (no learned router) | 0.486 | 36.4 |
| `h1_topk_egpt_moe_r128_d768.yml` | h1_topk + 128 register tokens | 0.484 | 39.6 |
| `h1_boltz_egpt_moe_d768.yml` | Boltzmann routing, **iso-param** (experts too small — fails) | 0.464 | 46.1 |

**Update (2026-06-05)**: With **full-size experts** (not iso-param split) **and
1/√(expert_I) routing-energy normalization**, Boltzmann routing (0.501) matches or
slightly beats learned-router top-k (0.499). See the revised conclusion below.

---

## Running experiments

### Submit a new training run

```bash
REPO=/proj/dmfexp/nima/Code/dolomite-engine
mkdir -p $HOME/bsub_logs

bsub \
    -q preemptable -G grp_preemptable \
    -J egpt_b1_boltz_moe \
    -gpu "num=4/task:mode=exclusive_process" \
    -n 1 -M 64G -W 06:00 \
    -o "$HOME/bsub_logs/egpt_b1_boltz_moe_%J.stdout" \
    -e "$HOME/bsub_logs/egpt_b1_boltz_moe_%J.stderr" \
    <<'EOF'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
bash /proj/dmfexp/nima/Code/dolomite-engine/scripts/common/pretrain.sh \
    /proj/dmfexp/nima/Code/dolomite-engine/configs/boltzmann_moe/b1_boltz_moe_16x1024_d768_lr2e3.yml
EOF
```

Or use the convenience script:

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
bash experiments/energy-inference/scripts/multi-block-ablation/run_b1_b3_boltz_moe.sh
```

### Resume from checkpoint

The training script auto-detects `latest_checkpointed_iteration.json` and
resumes. For manual resume, append `load_args` to the config:

```bash
cat >> /tmp/resume_b1.yml <<'YAML'

load_args:
  load_path: /proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/results/b1_boltz_moe_16x1024_d768_lr2e3
YAML
# then submit with /tmp/resume_b1.yml as the config
```

### Wall-time note

At ~0.91 s/step, 30k steps takes ~7.6 hours. Use `-W 08:00` for a single
uninterrupted run. If using `-W 04:00`, the job will checkpoint at 5k-step
intervals and need resubmission.

---

## Scaling up

To scale to more experts or larger hidden size, adjust `intermediate_size` and
`n_experts` in the config. The table below shows total params for d=768, 12 blocks:

| n_experts | per_expert_I | total_I | ~Total params |
|-----------|-------------|---------|--------------|
| 4 | 1024 | 4096 | ~165M |
| 8 | 1024 | 8192 | ~243M |
| 16 | 1024 | 16384 | ~422M (B-series) |
| 16 | 2048 | 32768 | ~723M |
| 32 | 1024 | 32768 | ~723M |

**Warning (applies to the B-series iso-param design only)**: the *deep iso-param*
B-series is severely FFN-heavy (FFN:Attn ≈ 21:1 at 422M), and there the V1 d=768 EGPT
baseline (143M, FFN:Attn ≈ 2.7:1) scored higher (0.481 vs 0.474). **This has since
been fixed — do not scale the iso-param design.** Use the H-series recipe instead:
a GPT prefix + one recurrent EGPT block whose MoE uses **full-size experts**
(int=2048 each; top-k or Boltzmann) with **1/√(expert_I) routing-energy
normalization**. That keeps FFN:Attn balanced, and once applied the Boltzmann MoE
matches/beats top-k and scales cleanly — a 679M model reaches 0.580 avg / 20.2 PPL at
53.5B tokens (see updated results below).

---

## Routing collapse metrics (WandB)

Logged every 10 steps under `model/energy_mlp/<block>.ffwd/`.

> **Caveat (fixed 2026-09-12)**: `train_utils.py:73` gated on the legacy classes
> only, so **no `EnergyFF_*` run ever logged these** — every `math_fet_*` wandb run
> has attention norms and nothing else. `FFEnergyBase` is now in the isinstance
> tuple, but runs completed before this date have no routing metrics to plot, and
> any claim about their routing collapse was inferred rather than measured.

| Metric | Range | Meaning |
|--------|-------|---------|
| `effective_n_experts` | 1.0 → K | Per-token entropy exponentiated. **1.0 = hard routing** |
| `n_dominant_experts` | 1 → K | How many experts win argmax across the batch |
| `max_expert_load` | 0 → 1 | Fraction of tokens routed to the busiest expert (uniform = 1/K = 0.0625) |
| `mean_token_entropy_norm` | 0 → 1 | Normalized per-token routing entropy |

**Key finding**: `effective_n_experts ≈ 1` by step 500 for all variants (hard routing
emerges fast), but `n_dominant_experts = 14–16` — different tokens go to different
experts. This is **not** collapse to a single expert; it is learned specialization.
B4 (repulsion 0.1) achieves best load balance (`max_expert_load ≈ 0.37`).

---

## Expert specialization analysis

Routing vectors (per-sample, per-layer, per-expert) are cached in:

```
experiments/boltzmann-moe/results/routing_cache/routing_b{1-5}.pkl
```

Load format:
```python
import pickle
data = pickle.load(open("routing_b1.pkl", "rb"))
# data["categories"]["COPA"]["routing"]  → (N, 12, 16) float32
# data["categories"]["COPA"]["texts"]    → list of N input strings
```

Re-run deep analysis (PCA, Mahalanobis separation, routing profiles) without
re-running inference:

```bash
bsub -q normal -G grp_ebm -n 1 -M 8G -W 00:30 \
     -gpu "num=1" \
     -o $HOME/bsub_logs/pca_analysis_%J.stdout \
     -e $HOME/bsub_logs/pca_analysis_%J.stderr \
    <<'EOF'
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:${PYTHONPATH:-}
cd /proj/dmfexp/nima/Code/dolomite-engine
python experiments/energy-inference/scripts/multi-block-ablation/analyze_boltz_expert_deep_20260429.py \
    --model b1 --reuse --device cpu
EOF
```

To collect new routing data for a new model (needs GPU):

```bash
# Add to MODELS dict in analyze_boltz_expert_deep_20260429.py:
#   "b6": RESULTS / "b6_.../unsharded"
# Then submit with --model b6 (no --reuse flag)
```

---

## Results summary

### Initial B-series (30k steps, 7.86B tokens) — pre-fix, iso-param

| Model | Params | Avg acc | WikiPPL | Notes |
|-------|--------|---------|---------|-------|
| V9 GPT d=1024 | 354M | **0.513** | **29.8** | Best baseline |
| V1-400M EGPT d=1024 | 354M | 0.494 | 38.6 | |
| V1 EGPT d=768 | 143M | 0.481 | 47.7 | Beat all *pre-fix* MoE at 1/3 params |
| B1 BoltzMoE (no reg.) | 407M | 0.474 | 51.9 | Best pre-fix MoE variant |
| B4 BoltzMoE (rep 0.1) | 407M | 0.466 | 51.9 | Best load balance |
| V58 EGPT recurrent | 113M | 0.459 | 65.7 | |
| B2 (rep 0.01) | 407M | 0.462 | 52.5 | |
| B5 (rep+drop+WD) | 407M | 0.471 | 58.7 | |
| B3 (drop+WD) | 407M | 0.450 | 58.0 | Worst |

### Updated results — after full-size experts + 1/√(expert_I) routing scale

| Model | Params | Avg acc | WikiPPL | Notes |
|-------|--------|---------|---------|-------|
| **580M @ 102k (53.5B tok)** | 679M | **0.580** | **20.2** | best overall in this line |
| scale_h3_boltz @ 120k (62.9B tok) | 620M | 0.569 | 21.9 | Boltzmann, scales cleanly |
| `h1_boltz_moe_fullsize` | 145M | **0.501** | 36.5 | Boltzmann ≥ top-k |
| h1_topk_egpt_moe | 145M | 0.499 | 39.8 | learned-router top-k |
| h1_egpt (no MoE, iso-compute) | 145M | 0.489 | 39.6 | MoE now beats plain EGPT |
| h1_boltz_topk2 (sparse) | 145M | 0.486 | 36.4 | no learned router; matches soft on PPL |
| **B4 rerun (1/√I fix)** | 407M | **0.494** | 38.0 | was 0.466/51.9 → now matches V1-400M |
| B1 rerun (1/√I fix) | 407M | 0.483 | 38.0 | was 0.474/51.9 |

**Conclusion (revised 2026-06-05)**: The earlier "MoE does not beat plain EGPT"
verdict was an artifact of two *fixable* B-series flaws — the iso-param design (tiny
experts, FFN:Attn ≈ 21:1) and an **unnormalized routing-energy scale** that caused
loss spikes. After (a) using **full-size experts** and (b) normalizing the routing
energy by `1/√(expert_I)`:

- Full-size **Boltzmann MoE (0.501 / 36.5)** ≥ learned-router **top-k (0.499 / 39.8)**
  at 145M, and both beat the iso-compute plain-EGPT baseline (0.489 / 39.6).
- The same fix rehabilitates the B-series: **B4-rerun 0.494 / 38.0** (was 0.466 / 51.9),
  now matching V1-400M EGPT (0.494 / 38.6).
- It scales: a 679M model reaches **0.580 avg / 20.2 PPL at 53.5B tokens**.

Net: **Boltzmann energy routing is competitive with, and slightly ahead of, top-k**
once experts are full-size and the routing scale is normalized. The energy
landscape selects experts without a learned router and generalizes to sparse top-2.
Full detail and the gelu_grad / h2 A/B studies are in `PROGRESS.md`.

---

## Key paper and reports

- **★ ACTIVE — ICLR 2026 paper** (Overleaf): `/u/ndehmamy/Code/overleaf/boltzmann-moe-ICLR-2026/`
  — main file: `main.tex`; sections in `sec/{intro,theory,experiments,appendix}.tex`
  — remote: `https://git@git.overleaf.com/6a9ace75a92fce262f38ec18` (branch `main`)
  — **all numbers are `Avg11`**; metric defined in `sec/appendix.tex` `\label{app:eval}`
  — **this is the only paper being written. Start and finish here.**
- **Local report**: `experiments/boltzmann-moe/paper/report.pdf` (10 pages)
- **Scatter plot script**: `experiments/boltzmann-moe/paper/make_moe_scatter.py`
  — generates `paper/figs/moe_scatter_total_params.pdf` and `moe_scatter_active_params.pdf`
- **NeurIPS 2026 paper — ARCHIVE, still on `avg10`, do not read unless asked**:
  `~/Code/energy/energy-GPT-neurips2026/` (main `nima/paper_v2.tex`,
  appendix `nima/sec/appendices/boltz_moe.tex`)
- **Talk slides — ARCHIVE, still on `avg10`/`avg9`**:
  `~/Code/overleaf/energy-GPT-reformulation-2026/talk_v3.tex`
- **Analysis scripts**: `experiments/energy-inference/scripts/multi-block-ablation/`
  - `analyze_boltz_moe_routing_20260428.py` — routing collapse curves from training logs
  - `analyze_boltz_expert_specialization_20260429.py` — basic PCA/heatmaps (60 samples)
  - `analyze_boltz_expert_deep_20260429.py` — deep PCA with KDE, Mahalanobis separation (200 samples)
