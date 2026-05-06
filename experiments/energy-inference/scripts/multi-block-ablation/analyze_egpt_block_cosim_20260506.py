"""Cross-block cosine similarity for multi-EGPT models (V71, U1, ...).

For models with multiple EGPT blocks (e.g. V71: 4 EGPT blocks with iter=[3,4,6,9]),
this script asks:

  1. Do the EGPT blocks apply similar update directions? (attn_out / ffwd_out cosim)
  2. Do they project onto the same subspace? (weight matrix alignment)
  3. Do repeated iterations within one block converge — and to what?
  4. Are different blocks optimizing the same implicit energy?
     Proxy: if blocks i,j satisfy <update_i, update_j> ≈ 1, they're "co-optimizing"

Produces:
  - egpt_block_update_cosim_{model}.{png,pdf}   — heatmap: per-position cross-block cosim
  - egpt_block_proj_alignment_{model}.{png,pdf}  — cosim of projection matrices
  - egpt_iter_convergence_{model}.{png,pdf}       — within-block iter-to-iter cosim

Usage:
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  python analyze_egpt_block_cosim_20260506.py [--model v71] [--n_batches 8]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

PREFILL_TEXTS = [
    "The tower is part of a complex of buildings that includes the Palace of Westminster. "
    "The tower stands 315 feet tall at the north end of the Palace of Westminster.",
    "Scientists discovered that certain proteins fold into specific shapes that determine "
    "their function in the cell. The process of protein folding is guided by molecular chaperones.",
    "In mathematics, a group is a set equipped with an operation that combines any two elements "
    "to form a third element satisfying closure, associativity, identity, and invertibility.",
    "The stock market experienced significant volatility as investors weighed the impact of "
    "rising interest rates on technology company valuations and future earnings projections.",
    "Deep learning models have achieved remarkable performance on image recognition tasks "
    "by learning hierarchical features from large amounts of labeled training data.",
    "The Amazon rainforest plays a crucial role in regulating the global climate by absorbing "
    "carbon dioxide and releasing oxygen through the process of photosynthesis.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and "
    "entanglement to process information in fundamentally different ways than classical computers.",
    "The human brain contains approximately 86 billion neurons connected by trillions of "
    "synapses that enable complex cognitive functions including memory, language, and reasoning.",
]

MODELS = {
    "v71": BASE / "v71_hybrid_8gpt_4egpt_rmsray_d1280" / "unsharded",
    "v9":  BASE / "v9_gpt_baseline_d1024_lr1e3" / "unsharded",
    # Add more as they become available:
    # "u1": BASE / "u1_2gpt_4egpt3x_rmsray_d1280" / "unsharded",
}


def load_model(path: Path, device="cuda"):
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    return model


def get_blocks(model):
    for attr in ("transformer", "model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "h"):
            return m.h
        if m is not None and hasattr(m, "transformer"):
            return m.transformer.h
    raise AttributeError(f"Cannot find transformer blocks on {type(model)}")


def get_egpt_block_indices(model):
    """Return indices of EGPT blocks.

    For models with EGPT blocks: returns only those blocks.
    For pure GPT models (no EGPT): returns ALL block indices so we can compare
    cross-layer similarity in a standard transformer.
    """
    indices = []
    blocks = get_blocks(model)
    for i, blk in enumerate(blocks):
        mixer = getattr(blk, "sequence_mixer", None) or getattr(blk, "attn", None)
        if mixer is not None and "energy" in type(mixer).__name__.lower():
            indices.append(i)
        elif hasattr(blk, "is_energy_block") and blk.is_energy_block:
            indices.append(i)
    # Check layer_iterations for EGPT blocks
    if not indices:
        cfg = model.config
        if hasattr(cfg, "layer_iterations"):
            for i, itr in enumerate(cfg.layer_iterations):
                if itr > 1:
                    indices.append(i)
    # For pure GPT: return ALL layers (measure cross-layer similarity as baseline)
    if not indices:
        indices = list(range(len(blocks)))
    return indices


def capture_block_updates(model, input_ids, egpt_indices):
    """Capture attn_out and ffwd_out for each EGPT block at each token position.

    Returns dict: block_idx -> {'attn': [T, D], 'ffwd': [T, D], 'update': [T, D]}
    where update = attn + ffwd (the raw gradient before projection).
    """
    updates = {i: {"attn": [], "ffwd": [], "h_in": [], "h_out": []} for i in egpt_indices}
    handles = []
    blocks = get_blocks(model)

    for idx in egpt_indices:
        blk = blocks[idx]

        def make_pre(i):
            def hook(mod, inp):
                x = inp[0] if isinstance(inp, tuple) else inp
                updates[i]["h_in"].append(x.detach().float().cpu())
            return hook

        def make_post(i):
            def hook(mod, inp, out):
                x = out[0] if isinstance(out, tuple) else out
                updates[i]["h_out"].append(x.detach().float().cpu())
            return hook

        handles.append(blk.register_forward_pre_hook(make_pre(idx)))
        handles.append(blk.register_forward_hook(make_post(idx)))

    with torch.no_grad():
        model(input_ids)

    for h in handles:
        h.remove()

    # Compute update = h_out - h_in (net residual added by this block)
    for i in egpt_indices:
        h_in = updates[i]["h_in"][0]   # [1, T, D]
        h_out = updates[i]["h_out"][0]
        updates[i]["update"] = (h_out - h_in).squeeze(0)  # [T, D]
        updates[i]["h_in"] = h_in.squeeze(0)
        updates[i]["h_out"] = h_out.squeeze(0)

    return updates


def cosim_matrix_across_blocks(vecs: dict[int, torch.Tensor]) -> np.ndarray:
    """Compute mean cosine similarity between update vectors of all block pairs.

    vecs: {block_idx: [T, D]} — one vector per token position.
    Returns: n_blocks × n_blocks cosim matrix.
    """
    keys = sorted(vecs.keys())
    n = len(keys)
    mat = np.zeros((n, n))
    for i, ki in enumerate(keys):
        for j, kj in enumerate(keys):
            v1 = F.normalize(vecs[ki], dim=-1)  # [T, D]
            v2 = F.normalize(vecs[kj], dim=-1)
            mat[i, j] = (v1 * v2).sum(-1).mean().item()
    return mat, keys


def flat_cosim_matrix(vecs):
    """Given list of flat tensors, compute n×n cosim matrix."""
    n = len(vecs)
    mat = np.zeros((n, n))
    for a in range(n):
        for b in range(n):
            v1 = F.normalize(vecs[a].float(), dim=0)
            v2 = F.normalize(vecs[b].float(), dim=0)
            mat[a, b] = (v1 * v2).sum().item()
    return mat


def functional_weight_cosim(model, egpt_indices):
    """Cosine similarity of gauge-invariant weight products across EGPT blocks.

    Computes:
      - proj_attn / proj_mlp (raw projection matrices)
      - W_QK = W_Q^T W_K  (effective attention kernel, gauge-invariant)
      - W_ffn = W_down @ W_up  (effective FFN transform at zero, gauge-invariant)

    For attention: W_Q, W_K have the gauge symmetry W_Q -> W_Q g^{-T}, W_K -> W_K g,
    so W_QK = W_Q^T W_K is invariant. This is the matrix that determines which
    directions in h get amplified by attention.

    For FFN (SwiGLU): approximate effective transform ≈ W_down @ W_up (linearization).
    """
    blocks = get_blocks(model)
    results = {}

    for attr in ("proj_attn", "proj_mlp"):
        vecs = []
        for i in egpt_indices:
            proj = getattr(blocks[i], attr, None)
            if proj is not None:
                vecs.append(proj.weight.detach().reshape(-1))
        if vecs:
            results[attr] = flat_cosim_matrix(vecs)

    # W_QK = W_Q^T @ W_K (gauge-invariant attention kernel)
    wqk_vecs = []
    for i in egpt_indices:
        blk = blocks[i]
        # Try EGPT (blk.attn) then GPT (blk.sequence_mixer)
        mixer = getattr(blk, "attn", None) or getattr(blk, "sequence_mixer", None)
        wq = wk = None
        if mixer is not None and hasattr(mixer, "c_attn") and hasattr(mixer.c_attn, "weight"):
            w = mixer.c_attn.weight.detach().float()
            d = w.shape[1]
            # EGPT: c_attn = [2D, D] = [W_Q; W_K]
            # GPT:  c_attn = [3D, D] = [W_Q; W_K; W_V]
            # Either way, W_Q = w[:d], W_K = w[d:2*d]
            wq, wk = w[:d], w[d:2*d]
        if wq is not None and wk is not None:
            wqk = (wq.T @ wk).reshape(-1)
            wqk_vecs.append(wqk)
    if wqk_vecs:
        results["W_QK=W_Q^T@W_K"] = flat_cosim_matrix(wqk_vecs)

    # W_ffn = W_down @ W_up (effective FFN linearization)
    wffn_vecs = []
    for i in egpt_indices:
        blk = blocks[i]
        # EGPT: blk.ffwd.W1 (gate), W2 (up), both [int_s, D]
        # GPT:  blk.mlp_block.c_fc [up_dim, D], c_proj [D, out_dim]
        ffn = getattr(blk, "ffwd", None) or getattr(blk, "mlp_block", None) or getattr(blk, "mlp", None)
        if ffn is None:
            continue
        w1 = getattr(ffn, "W1", None)   # EGPT gate [int_s, D]
        w2 = getattr(ffn, "W2", None)   # EGPT up   [int_s, D]
        if w1 is not None and w2 is not None and hasattr(w1, "weight") and hasattr(w2, "weight"):
            wffn = (w2.weight.detach().float().T @ w1.weight.detach().float()).reshape(-1)
            wffn_vecs.append(wffn)
        else:
            # GPT: c_fc [up_dim, D], c_proj [D, down_dim]
            c_fc   = getattr(ffn, "c_fc",   None)
            c_proj = getattr(ffn, "c_proj", None)
            if c_fc is not None and c_proj is not None and hasattr(c_fc, "weight") and hasattr(c_proj, "weight"):
                # c_proj: [D, out_dim], c_fc: [up_dim, D]
                # For SwiGLU: up_dim = 2*int_s, take second half as "value" path
                wup  = c_fc.weight.detach().float()    # [up_dim, D]
                wdwn = c_proj.weight.detach().float()  # [D, out_dim or 2*int_s]
                # Linearization: if SwiGLU uses first half as gate, second as value
                half = wup.shape[0] // 2
                wffn = (wdwn @ wup[half:]).reshape(-1)  # [D, half] @ [half, D] → [D, D] approx
                if wffn.shape[0] != wdwn.shape[0] * wdwn.shape[0]:
                    # fallback: c_proj.T @ c_fc
                    wffn = (wdwn.T @ wup).reshape(-1)
                wffn_vecs.append(wffn)
    if wffn_vecs:
        results["W_ffn=W2^T@W1"] = flat_cosim_matrix(wffn_vecs)

    # Π·W2^T·W1 — EGPT only: proj_mlp @ W2^T @ W1 (full energy-projected FFN kernel)
    pi_wffn_vecs = []
    for i in egpt_indices:
        blk = blocks[i]
        ffn  = getattr(blk, "ffwd", None) or getattr(blk, "mlp", None)
        proj = getattr(blk, "proj_mlp", None)
        if ffn is None or proj is None:
            continue
        w1 = getattr(ffn, "W1", None)
        w2 = getattr(ffn, "W2", None)
        if w1 is None or w2 is None or not hasattr(w1, "weight"):
            continue
        Pi  = proj.weight.detach().float()             # [D, D]
        W2t = w2.weight.detach().float().T             # [D, int_s]
        W1  = w1.weight.detach().float()               # [int_s, D]
        pi_wffn = (Pi @ W2t @ W1).reshape(-1)          # [D, D] flattened
        pi_wffn_vecs.append(pi_wffn)
    if pi_wffn_vecs:
        results["Pi@W2^T@W1"] = flat_cosim_matrix(pi_wffn_vecs)

    return results


def plot_cosim_heatmap(mat, labels, title, path):
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(mat, vmin=-1, vmax=1, cmap="RdBu_r")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if abs(mat[i, j]) > 0.5 else "black")
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(path) + f".{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="v71")
    parser.add_argument("--n_batches", type=int, default=8)
    args = parser.parse_args()

    ckpt = MODELS.get(args.model)
    if ckpt is None or not ckpt.exists():
        print(f"Model {args.model} not found at {ckpt}. Available: {list(MODELS.keys())}")
        return

    print(f"Loading {args.model} from {ckpt}...")
    tok_path = "/proj/datasets/tokenizers/granite-4.0-tiktoken"
    tok = AutoTokenizer.from_pretrained(tok_path)
    model = load_model(ckpt)

    egpt_idx = get_egpt_block_indices(model)
    print(f"Block indices: {egpt_idx} ({len(egpt_idx)} blocks)")
    cfg = model.config
    if hasattr(cfg, "layer_iterations"):
        iters = cfg.layer_iterations
        block_labels = [f"B{i}(×{iters[i]})" for i in egpt_idx]
    else:
        block_labels = [f"L{i}" for i in egpt_idx]   # GPT: just layer index

    # Accumulate updates over multiple batches
    all_updates = {i: [] for i in egpt_idx}
    for text in PREFILL_TEXTS[:args.n_batches]:
        ids = tok(text, return_tensors="pt").input_ids.cuda()
        upd = capture_block_updates(model, ids, egpt_idx)
        for i in egpt_idx:
            all_updates[i].append(upd[i]["update"])  # [T, D]

    # Concatenate across batches → [total_T, D]
    mean_vecs = {i: torch.cat(all_updates[i], dim=0) for i in egpt_idx}

    # 1. Cross-block update cosim heatmap
    mat, keys = cosim_matrix_across_blocks(mean_vecs)
    tag = f"{args.model}"
    plot_cosim_heatmap(
        mat, block_labels,
        f"{args.model}: Cross-block update cosim\n(update = h_out − h_in per token, avg over {args.n_batches} passages)",
        PLOTS_DIR / f"egpt_block_update_cosim_{tag}"
    )
    print(f"Cross-block update cosim matrix:\n{mat.round(3)}")
    print(f"Off-diagonal mean: {(mat.sum() - np.trace(mat)) / (len(keys)**2 - len(keys)):.3f}")

    # 2. Projection weight alignment
    wmat = functional_weight_cosim(model, egpt_idx)
    for attr, m in wmat.items():
        plot_cosim_heatmap(
            m, block_labels,
            f"{args.model}: proj weight cosim ({attr})",
            PLOTS_DIR / f"egpt_proj_weight_cosim_{attr}_{tag}"
        )
        print(f"\n{attr} weight cosim:\n{m.round(3)}")

    # 3. Save stats
    stats = {
        "model": args.model,
        "egpt_block_indices": egpt_idx,
        "block_labels": block_labels,
        "update_cosim_matrix": mat.tolist(),
        "off_diag_update_cosim_mean": float((mat.sum() - np.trace(mat)) / (len(keys)**2 - len(keys))),
        "proj_weight_cosim": {k: v.tolist() for k, v in wmat.items()},
    }
    stats_path = PLOTS_DIR / f"egpt_block_cosim_stats_{tag}.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"\nStats saved: {stats_path}")
    print(f"Plots saved: {PLOTS_DIR}/egpt_block_*_{tag}.{{png,pdf}}")


if __name__ == "__main__":
    main()
