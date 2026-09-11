# Boltzmann TopK Energy MoE — head-to-head with vanilla switch-routed MoE

Adapted from `configs/boltzman_moe_all/README.md`. Three configs designed to be
trained head-to-head, copied unmodified from that directory except for one fix
(tokenizer path, see below).

| Config | Attention | MLP blocks | Active / Total |
|---|---|---|---|
| `s8e4_stdmoe_fh2_boltz_topk2.yml` | 8 softmax + 4 energy | 8 std MoE + 4 **fh2** (`BoltzmannMoE_Energy_MLP`) | ~387M / ~1101M |
| `s12_stdmoe_only_topk2.yml`       | 12 softmax           | 12 std MoE (ipe=3200, top-2) | ~389M / ~1097M |
| `s8e4_stdmoe_only_topk2.yml`      | 8 softmax + 4 energy | 12 std MoE (ipe=3200, top-2) | ~389M / ~1097M |

All three: 12 layers, `d=1024`, 16 heads × 64 head_dim, 8 experts top-2, no
looping (`num_iterations: 1`), 30k steps × 1.05M tok/step = **31.5B tokens**,
lr 2e-3, micro_batch 1, grad accum 4.

**Why 3 configs?** `s12_stdmoe_only_topk2` is the baseline reported in the paper.
`s8e4_stdmoe_only_topk2` is the cleaner ablation — it shares the attention layout
with the fh2 hybrid, so the *only* difference between it and fh2 is the MLP block
type on the last 4 layers. Run all three so the attention change and the MLP
change can be separated.

## What is `fh2` (`BoltzmannMoE_Energy_MLP`)?

Code: `lm_engine/hf_models/modeling_utils/mlp_blocks/boltzmann_moe.py`.

Each expert is an Energy MLP whose output is the analytic gradient of a scalar
energy, computed from `W1` and `W2`. With `phi = GELU(W1 x)` and
`phi' = sigmoid(sqrt(2/pi) · W1 x)/2`:

```
expert_grad = phi · W2_e^T  +  (phi' ⊙ W2 x) · W1_e^T
```

**There is no learned router.** Routing comes from the expert energies
themselves — `E_i = <x, phi · W2_e^T> / sqrt(expert_I)` and
`p = softmax(E/tau)` — which is what makes this the Boltzmann variant. Contrast
`fh1` (`TopK_Energy_MoE_MLP`) in `configs/boltzman_moe_all/`, which uses the *same*
energy-gradient experts but selects them with a learned `nn.Linear` gate; fh1 is
**not** a Boltzmann router, despite what the `boltzman_moe_all` README says.

Two `fh2` details that matter and are easy to miss:

- **`top_k` truncation is unnormalized.** With `n_experts: 8, top_k: 2` the class
  softmaxes over all 8 experts, then zeroes everything outside the top-2 *without*
  renormalizing — so routing weights sum to ~0.56 on average, token-dependent.
  This is how the reported run trained. Deliberate (it avoids the discontinuity
  when the top-k set changes) but it differs from the paper's Eq. 4.
- **All experts are computed, then discarded.** `term1`/`term2` are evaluated for
  all 8 experts before the top-2 are gathered, so `top_k` buys regularization,
  not FLOPs.

## Empirical result at 400M active / 30k steps / 31.5B tokens

`s8e4_stdmoe_fh2_boltz_topk2` vs the matched `s12_stdmoe_only_topk2` baseline:

| Metric | `s12_stdmoe_only_topk2` | `s8e4_stdmoe_fh2_boltz_topk2` | Δ |
|---|---|---|---|
| 11-task Avg ↑ | **0.5086** | 0.5084 | −0.0002 |
| Wiki word PPL ↓ | **27.52** | 27.63 | +0.11 |
| MMLU 5-shot ↑ | **0.2708** | 0.2620 | −0.0088 |
| gsm8k_cot 5-shot ↑ | 0.0508 | **0.0514** | +0.0006 |
| ARC-Easy ↑ | 0.6233 | **0.6561** | **+0.0328** |
| ARC-Challenge-norm ↑ | 0.3183 | **0.3439** | **+0.0256** |

Aggregates are a **tie** (both inside the ~0.3pp / 0.5 PPL run-to-run bound
measured from a repeated identical config). The real win is ARC (+3.3 / +2.6 pp,
outside that bound); the real loss is MMLU (−0.9 pp). Per-task it is 7 wins / 6
losses. The claim the data supports is *matches standard MoE while removing the
learned router entirely*, not *beats standard MoE*. Single seed.

## How to submit

The repo-root launcher already covers these three (it points at
`configs/boltzman_moe_all/`, the same files):

```bash
bash submit_boltzmann_sweep.sh          # all three, 8 nodes x 8 H100 each, full 30k steps
```

To launch one directly from this directory:

```bash
LOG_DIR=/proj/dmfexp/energy-gpt/logs/boltzman_moe
mkdir -p "$LOG_DIR"
mkdir -p /proj/dmfexp/energy-gpt/checkpoints-bsaha/boltzman_moe   # save_path parent is NOT auto-created

NAME=s8e4_stdmoe_fh2_boltz_topk2         # or s12_stdmoe_only_topk2 | s8e4_stdmoe_only_topk2
export PRETRAIN_VENV=$PWD/.venv-nima     # REQUIRED, see below
bsub -r -q normal -G grp_ebm -M 2000G -hl -n 8 \
  -J "${NAME}" \
  -gpu "num=8/task:mode=exclusive_process" \
  -oo "${LOG_DIR}/${NAME}.out" \
  -eo "${LOG_DIR}/${NAME}.err" \
  blaunch bash launch-scripts/pretrain.sh "configs/boltzman_moe/${NAME}.yml"
```

**Sanity check that a launch is healthy**: the log should reach `step = 10` with a
finite train-loss and non-zero `billion_tokens_per_day`. For a cheap check without
committing to 30k steps, copy a config and set `num_training_steps` (and
`num_decay_steps`) to ~50.

All three were verified to build and run a forward pass on this branch:
**1,096,934,400** (s12) / **1,092,736,004** (s8e4) / **1,101,091,844** (fh2) params,
loss ~11.7-12.1 = ln(100352), correct for random init.

## Two things that will bite whoever runs these

1. **Venv.** Use `.venv-nima` (transformers 4.57.1). This repo's `.venv` is
   transformers 5.1.0, which silently clobbers the tied `wte` — and all three
   configs set `tie_word_embeddings: true`. Export
   `PRETRAIN_VENV=/proj/dmfexp/bishwajit/Code/dolomite-engine/.venv-nima` before
   launching; `launch-scripts/pretrain.sh` honours that variable (it used to
   hardcode `.venv`).

2. **Tokenizer path was stale.** The originals in `configs/boltzman_moe_all/` point at
   `/proj/dmfexp/energy-gpt/data/granite-4.0-tiktoken`, which no longer exists.
   The copies here point at `/proj/datasets/tokenizers/granite-4.0-tiktoken`
   (verified: loads, vocab 100352 = `vocab_size`). This is the **only** change
   made to the copied files.
