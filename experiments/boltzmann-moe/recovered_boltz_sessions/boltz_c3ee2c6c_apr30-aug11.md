# Recovered boltz-relevant dialogue — c3ee2c6c (EGPT-optim/RMSNorm, Apr 30 – Aug 11)

Source session: `/u/ndehmamy/.claude/projects/-proj-dmfexp-nima-Code-GPT-experiments/c3ee2c6c-855d-4fbd-bf57-24773c865979.jsonl`
Extraction: all user prompts + assistant text blocks whose text mentions 'boltz' (case-insensitive), chronological. Tool-result/file-dump blocks excluded.



===== [2026-08-01T05:42:32.778Z] ASSISTANT =====
`~/Code` is essentially empty (512 B — the real tree is under `/proj/dmfexp/nima/Code`). The consumer is **dolomite-engine = 3.1 TB, ~all in `experiments/`**: 2.8 TB in energy-inference/results + **288 GB in boltzmann-moe/results**. Let me check the boltzmann-moe dir (any active runs there?) and the primary Code root:


===== [2026-08-01T05:44:25.464Z] ASSISTANT =====
Investigation complete. Here's the full report.

## Disk cleanup report

**All usage is under `/proj/dmfexp/nima/Code` (3.4 TB); `~/Code` is empty (512 B).** The consumer is **dolomite-engine checkpoints (3.1 TB)**.

| Area | Size | Reclaimable | Notes |
|---|---|---|---|
| `energy-inference/…/multi-block-ablation` | 2,788 GB | **~2,564 GB** | 210 runs — only **4 active**, 206 old |
| `boltzmann-moe/results` | 288 GB | **~270 GB** | all inactive since early June; includes explicit `*_v0_old` dupes |
| GPT-experiments results (verifier/RL/action) | ~95 GB | TBD | npz/pkl/json artifacts — needs a finer pass, kept for now |
| wandb (dolomite + GPT-exp) | ~3.2 GB | ~3 GB | old local run logs |

**Reclaimable from checkpoints alone: ~2.5–2.8 TB.**

**Policy** (matches your "remove most checkpoints, especially if unsharded exists / too many points on one trajectory"):
- Inactive run **with** `unsharded/` → delete **all** sharded `global_step*` (the HF unsharded ckpt + results JSON are preserved). Most runs.
- Inactive run **without** `unsharded/` → keep **only the final** `global_step`, delete earlier ones (e.g. the `r*_smooth` runs each have **20** checkpoints at 60–87 GB).
- **Active 4 runs** → keep the **last 2** checkpoints (resume window), delete older intermediates (~155 GB).

**DO-NOT-TOUCH (training right now):** `math_egptdual_r0` (88k), `r128sb` (58k), `seed42` (2.6k), `seed1234` (2.4k) — plus all `.git`, code, configs, `.venv`, paper repos.

This is a large irreversible delete, so I want your sign-off on how aggressive before I run a single `rm`:


===== [2026-08-01T05:52:08.810Z] ASSISTANT =====
Cleanup complete and verified.

## Cleanup done — freed 2.49 TB

| Dir | Before | After |
|---|---|---|
| `multi-block-ablation` | 2,788 GB | **485 GB** |
| `boltzmann-moe/results` | 288 GB | **41 GB** |
| **dolomite-engine total** | 3.1 TB | **586 GB** |

**937 dead-run checkpoint dirs deleted, 2,550 GB (2.49 TB) freed.** Every deleted run kept either its unsharded HF checkpoint or its final sharded checkpoint, so no model was lost.

**All 4 active runs fully intact** — R0 kept its whole rolling window (now latest=98000, advanced *during* the cleanup — training never paused), R128 (58k), seed42 (2600), seed1234 (2500). All 6 jobs (2×1B, 2 seeds, 2 guardians) still RUN. The seeds are climbing nicely on the 4-GPU path (2500–2600, checkpoints sticking).

(Note: `/proj/dmfexp` still shows 99% — it's a 300 TB pool shared across the group, so our 2.5 TB is real headroom back but a small fraction of the whole.)

Now logging the interim 1B benchmark to RESULTS.md as promised — let me check its format:


===== [2026-08-04T07:56:55.784Z] USER =====
<task-notification>
<task-id>aafc638f75e2a21aa</task-id>
<tool-use-id>toolu_bdrk_01MMAPXJ5mh3ri7pDqr3mrEN</tool-use-id>
<output-file>/tmp/claude-700186/-proj-dmfexp-nima-Code-GPT-experiments/c3ee2c6c-855d-4fbd-bf57-24773c865979/tasks/aafc638f75e2a21aa.output</output-file>
<status>completed</status>
<summary>Agent "Diff Bharat's 1B config vs ours" finished</summary>
<note>A task-notification fires each time this agent stops with no live background children of its own. The user can send it another message and resume it, so the same task-id may notify more than once.</note>
<result>I have everything needed. Let me compile the findings.

## Summary

**Branch resolution:** The URL parses as branch **`bharat/energy`** (ref `refs/heads/bharat/energy`), path `configs/energy`. Confirmed via `git ls-remote --heads bharat`. I fetched it and 6 sibling energy branches (`bsaha/energy`, `bsaha/energy-v2`, `bsaha/energy-exp`, `bharat/energy_moe`, `bharat/energy_logs`, `hoo/energy`) using fetch + `git show`/`ls-tree` only — no checkout, working tree untouched. Node: compute node `p3-r07-n3`, internet worked.

### Key finding: the 1B EGPT-Dual config does NOT exist in the colleague's dolomite-engine repo

I inventoried **every** `model_type: energy` config across all 7 energy branches. The `configs/energy/` directory holds only small prototypes:
- `baseline_3EGPT_9iter*.yml` — d=1024, 3 layers, iters [9,9,9]
- `energy_loop.yml` / `energy_32gpu.yml` / `energy_bs.yml` — **d=2048 but only 5 layers**, num_iterations 4–8, intermediate_size 4096, GPT-2-style tokenizer (vocab 50259), lr 0.01 power schedule, dclm/nemo data
- `energy_test.yml`, `energy_debug.yml`, gmm/workshop configs — tiny

**None** is a 12-layer, d=2048, ~944M EGPT-Dual with iterations ending in 6. The string `dual_unconstrained` appears **only in code** (`lm_engine/hf_models/models/energy/layer.py`, `analyze_helmholtz.py`) as a supported `energy_proj_type` — it is never wired into any committed config. So the colleague's repo has the *capability* but no 1B config.

### Where the real EGPT-Dual configs live

Our own config header says the paper 1B was "trained in energy-gpt, a different codebase." I checked `/proj/dmfexp/energy-gpt/` (bsaha's checkpoints). There, plain EGPT-Dual (`mlp_type: Energy_MLP`, `dual_unconstrained`) exists **only at 400M** — `checkpoints-bsaha/table2-400m/egpt_dual_seed{42,43,44}/.../training_config.yml` (d=1280, 12 layers). The only d≈1536 duals (`boltzmann_sweep/*d1536`) are **Boltzmann-MoE** (`BoltzmannMoE_Energy_MLP`, int 32768), not plain dual, and trained on math-free 50/50 web data. **No d=2048 1B plain-EGPT-Dual config exists anywhere** — our config is a genuine scale-up with no exact original.

### Diff: our 1B config vs the canonical paper EGPT-Dual (`table2-400m`, d1280/400M)

Data and tokenizer are the reference recipe; the fields are grouped into scale-driven (expected) vs recipe choices (candidate gap drivers).

| Field | Paper EGPT-Dual (table2-400m) | Ours (1B) | Note |
|---|---|---|---|
| **energy_apply_rayleigh** | **true** | **false** | Recipe deviation — biggest suspect |
| **layer_iterations** | [1×8, **3,4,6,9**] (Σ=30) | [1×8, **2,3,4,6**] (Σ=23) | Ours does ~23% fewer energy-descent passes |
| **iter_dropout_range_per_block** | [0×8, 1,2,2,4] | [0×8, 1,1,2,3] | Coupled to above |
| **gradient_clipping** | 1.0 | 0.5 | Tighter clip |
| energy_full_per_token_grad | absent (default) | false | Likely same effective value |
| lr | 2e-3 | 1.5e-4 (muP: 2e-4@d1536 ×1536/2048) | Intentional muP scale-down; transfer to d2048 is an assumption |
| torch_compile | true | false | Numerically fine (disabled for FSDP/iter-dropout deadlock) |
| loss_mask | output_only | (pretraining default) | Worth confirming |
| model_type | energy | register_energy, n_registers 0 | Asserted no-op (model.py:158) |
| hidden_size / heads / head_dim / intermediate | 1280 / 20 / 64 / 5120 | 2048 / 16 / 128 / 8192 | Scale-driven (expected) |
| normalization / pos-emb / init_range | rmsnorm / rope / 0.02 | same | Identical |
| **data_path blend** | 0.35/0.35/0.15/0.15 (web-nemotron ×2, megamath-web-pro, finemath-3plus) | **identical** | Rules data OUT as gap cause |
| **tokenizer / vocab / seq_len** | granite-4.0-tiktoken / 100352 / 4096 | **identical** (same tokenizer, different path) | Rules tokenizer OUT |
| optimizer | AdamW, wd 0.1, betas .9/.95, eps 1e-8, cosine, warmup 2000, decay_factor 0.1 | same (except lr/clip) | Identical schedule shape |

### Assessment — most plausible gap drivers

Data mixture, weights, tokenizer, vocab, sequence length, and the optimizer/schedule shape are **identical** to the canonical EGPT-Dual recipe, so they are ruled out. The differences that plausibly explain our R0 sitting ~3.6pp (MMLU) / ~1.8pp (GSM8K) below paper EGPT-Dual are, in order:

1. **`energy_apply_rayleigh: false` (ours) vs `true` (paper).** The single clearest deviation from the canonical EGPT-Dual recipe; Rayleigh-normalizing the energy gradient is part of the paper's dual model. Top suspect.
2. **Shallower iteration schedule** — last-4 layers iterate [2,3,4,6] vs the paper's [3,4,6,9] (23 vs 30 effective passes), with correspondingly less iter-dropout. Fewer energy-descent steps = less test-time compute, which most directly hits reasoning benchmarks (MMLU/GSM8K).
3. **`gradient_clipping: 0.5` vs `1.0`** and the **muP LR down-scaling to 1.5e-4 at d=2048** (a transfer assumption, not a tuned value) — secondary; either could mildly under-optimize.

Two important caveats: (a) I could **not** find a 1B paper config, so the comparison is against the 400M canonical recipe — the paper's actual 1B EGPT-Dual may differ; (b) our config runs 120k steps ≈ 126B tokens, and if the paper 1B saw more tokens the gap could be partly budget, though the project's own prior finding (memory: "GSM8K gap is scale-emergent/architectural, NOT undertraining") argues against that.

**Files:** ours = `/proj/dmfexp/nima/Code/dolomite-engine/configs/multi_block_ablation/math_egptdual_r0_split_d2048_int8192_126b_16gpu.yml`; canonical reference = `/proj/dmfexp/energy-gpt/checkpoints-bsaha/table2-400m/egpt_dual_seed42/global_step30000/training_config.yml`.</result>
<usage><subagent_tokens>2</subagent_tokens><tool_uses>18</tool_uses><duration_ms>405180</duration_ms></usage>
</task-notification>
