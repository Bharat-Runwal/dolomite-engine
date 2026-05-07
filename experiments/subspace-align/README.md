# Subspace Alignment Analysis

Scripts for reproducing the subspace-alignment diagnostics from the NeurIPS 2026 paper
*"Energy Attention Enables Distributed Refinement"*.

## What these scripts measure

For a model with multiple recurrent blocks (e.g. 8-GPT + 4-EGPT), we test:

1. **Cross-block cosine similarity** — do blocks share write-operator structure?
2. **Per-iteration L⊥ trajectory** — do EGPT iterations pull hidden states toward L?
3. **LM-head alignment (top-k and bottom-k)** — does the write operator concentrate in
   dominant or near-null vocabulary directions?
4. **Singular value distribution** — how rank-deficient is W_U? Is there a real L⊥?

## Setup

```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
```

Models must be **unsharded** (HuggingFace format). To unshard an FSDP checkpoint:
```bash
python -m lm_engine.unshard --config /tmp/unshard.yml
# where /tmp/unshard.yml contains:
# load_args:
#   load_path: /path/to/fsdp/checkpoint
# unsharded_path: /path/to/output
# mixed_precision_args:
#   dtype: bf16
```

**Note for Python 3.13 checkpoints loaded on Python 3.12:**
If you get `ModuleNotFoundError: No module named 'pathlib._local'`, prepend this fix:
```python
import sys, pathlib, types
m = types.ModuleType('pathlib._local')
m.Path = pathlib.Path; m.PurePath = pathlib.PurePath
sys.modules['pathlib._local'] = m
```
Or run with the wrapper: `python /proj/dmfexp/nima/Code/dolomite-engine/unshard_compat.py --config ...`

## Scripts

### `01_cross_block_cosim.py` — Cross-block functional and weight similarity

**What it computes:**
- δ-h off-diagonal cosim: `ρ^δ = cos(h_out - h_in)` averaged over token positions and passages
- W_QK cross-block cosim: `ρ^J = cos(W_Q^T @ W_K)` — gauge-invariant attention kernel
- Π@J or Π_mlp@W2^T@W1: full write operator including projection
- W_OV: `W_O @ W_V` for standard attention models

**Key results (matched pair, 8GPT+4×[3,4,6,9], d=1280):**
| Model | δ-h | ρ^J |
|-------|-----|-----|
| EGPT (value-tied) | 0.392 | 0.490 |
| RecGPT (W_V indep.) | 0.027 | -0.001 |

**Usage:**
```bash
# Single model
python 01_cross_block_cosim.py --model 410m_hybrid_s8e4 --n_batches 8

# Add your model to MODELS dict in the script, then:
python 01_cross_block_cosim.py --model my_egpt_800m --n_batches 8
```

**To add a new model**, edit `MODELS` dict:
```python
MODELS = {
    ...
    "my_egpt_800m": BASE / "my_model/unsharded",   # path to HF unsharded checkpoint
    "my_recgpt_800m": BSAHA / "my_recgpt/unsharded",
}
```

**Output:** `plots/egpt_block_cosim_stats_{model}.json` + PNG/PDF heatmaps

---

### `02_lperp_trajectory.py` — Per-iteration L⊥ trajectory (Exp B/D2)

**What it computes:**
For each recurrent block at each iteration step, measures:
```
ε(h) = ||(I - P_{L_k}) h|| / ||h||
```
where L_k = top-k right-singular vectors of W_U. A decreasing ε means the
hidden state moves toward the prediction subspace; increasing ε means scratch
accumulation.

**Key results (k=256, 54% energy for d=1280):**
- EGPT (8GPT+4EGPT[3,4,6,9]): backbone GPT layers increase ε (0.79→0.94);
  EGPT blocks decrease it (0.94→0.81); each iteration decreases ε further.
- Deep GPT (24-layer): ε drops early (0.76→0.44), rises in late layers (0.66).

**Usage:**
```bash
python 02_lperp_trajectory.py --models 410m_hybrid_s8e4,v9 --n_batches 8
```

**Output:** `plots/lperp_trajectory_{models}.{png,pdf}` + JSON stats

---

### `03_lk_alignment.py` — LM-head alignment (top-k and bottom-k)

**What it computes:**
For the full write operator (Π@J for EGPT, W_O@W_V for RecGPT):
```
align_top(k) = ||P_{top-k SVs of W_U}(write_op)||_F / ||write_op||_F
align_bot(k) = ||P_{bottom-k SVs of W_U}(write_op)||_F / ||write_op||_F
excess = align - sqrt(k/d)   # deviation from random baseline
```

**Key results (d=1280, k=32):**
| Model | top-32 excess | bot-32 excess |
|-------|--------------|--------------|
| EGPT Π@J | +0.183 | **+0.329** |
| RecGPT W_OV | +0.020 | +0.052 |

EGPT has a **bimodal** pattern: above random at both extremes (most dominant
and most non-dominant vocab directions), below random in the middle.
RecGPT is near-random everywhere.

**Usage:**
```bash
python 03_lk_alignment.py --models 410m_hybrid_s8e4,410m_recgpt_s8e4 \
                           --output my_alignment_results.json
```

**Output:** JSON with per-block and mean alignment at all k values.

---

### `04_sv_distribution.py` — SV spectrum of W_U

**What it computes:**
- Full singular value spectrum of W_U (plots log-scale)
- k_95, k_99 (effective rank)
- Bottom-k alignment as a function of k

**Key results:**
- Both 410m models: d=1280, k_95≈1110, k_99≈1234. W_U is nearly full-rank.
- Decay ratio ≈63-88×. The "near-null space" is tiny (~46 dimensions).

**Usage:**
```bash
python 04_sv_distribution.py --models 410m_hybrid_s8e4,410m_recgpt_s8e4 \
                              --output_dir ./plots
```

---

## Running all analyses at scale (bsub)

For large models (>400M), submit as bsub jobs with a GPU:

```bash
bsub -q normal -G grp_ebm -J my_analysis \
    -gpu "num=1" -n 1 -M 64G -W 01:00 \
    -o ~/bsub_logs/my_analysis_%J.stdout \
    -e ~/bsub_logs/my_analysis_%J.stderr \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:\$PYTHONPATH
cd /proj/dmfexp/nima/Code/dolomite-engine/experiments/subspace-align
python 01_cross_block_cosim.py --model my_egpt_800m --n_batches 8
python 03_lk_alignment.py --models my_egpt_800m,my_recgpt_800m
python 04_sv_distribution.py --models my_egpt_800m,my_recgpt_800m
EOF
```

## What to look for at larger scale

| Finding at 400M | Prediction at scale |
|----------------|---------------------|
| ρ^J: 0.490 (EGPT) vs -0.001 (RecGPT) | Gap should persist or grow (more iterations = more alignment) |
| δ-h: 0.392 (EGPT) vs 0.027 (RecGPT) | Similar gap expected |
| bot-32 excess: +0.329 (EGPT) vs +0.052 (RecGPT) | EGPT excess should grow with model size |
| k_99 ≈ 1234/1280 | At larger d, k_99/d ratio may change |

If the bimodal alignment pattern (excess at both top-32 and bot-32 for EGPT,
flat for RecGPT) **persists at 800M/1B**, that strongly supports the structural
interpretation that value tying creates a fundamentally different organizational
principle rather than a scale-dependent artifact.

## Reference: NeurIPS 2026 paper

Section 5.2 and Appendix A4-A6.
Code: `experiments/subspace-align/`
Results: `experiments/energy-inference/results/multi-block-ablation/plots/`
Paper: `energy-GPT-neurips2026/neurips_2026/`
