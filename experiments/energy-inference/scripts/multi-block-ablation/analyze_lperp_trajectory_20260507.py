"""Experiment B: Per-iteration L⊥ trajectory for EGPT vs GPT models.

For the LM head W_U ∈ R^{vocab×d}, compute the top-k right singular vectors
(k=256) forming basis L_k ∈ R^{d×k} that captures ≥95% of vocab energy.

For each token position h at every iteration step inside a recurrent block:
    ε(h) = ||(I - L_k L_k^T) h|| / ||h||

A low ε means h lives mostly in the LM-head subspace (output-ready).
A high ε means h is mostly in a "scratch space" orthogonal to the LM head.

The key question: do EGPT recurrent blocks INCREASE ε (pushing representations
into scratch space) and then reduce it in final iterations (converging back to
the LM-head subspace)?  Contrast with GPT blocks which do one pass each.

Strategy: hook on blk.ln (layer norm inside EnergyBlock, called once per
iteration), capturing its INPUT at each call. For GPT blocks (which have
blk.ln_1, blk.ln_2), hook on ln_1 to get h at each block's start.

Usage:
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
    python analyze_lperp_trajectory_20260507.py [--models v71,v9] [--n_batches 8]

Outputs:
    results/multi-block-ablation/plots/lperp_trajectory_v71_v9.{png,pdf}
    results/multi-block-ablation/plots/lperp_stats_v71_v9.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
import sys

REPO = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: registers model types
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---------------------------------------------------------------------------
# Model paths
# ---------------------------------------------------------------------------
MODELS = {
    "v71": BASE / "v71_hybrid_8gpt_4egpt_rmsray_d1280" / "unsharded",
    "v9":  BASE / "v9_gpt_baseline_d1024_lr1e3" / "unsharded",
}

# ---------------------------------------------------------------------------
# WikiText-style passages (8 passages)
# ---------------------------------------------------------------------------
WIKITEXT_PASSAGES = [
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

# ---------------------------------------------------------------------------
# GSM8K-style arithmetic passages (4 passages)
# ---------------------------------------------------------------------------
GSM8K_PASSAGES = [
    "Janet has 3 dozen eggs. She uses 7 eggs each morning for breakfast. "
    "How many eggs does she have left after 5 days? Answer: 3*12=36 total. 7*5=35 used. 36-35=1.",
    "A store sells apples for $2 each and oranges for $3 each. "
    "If Tom buys 4 apples and 3 oranges, how much does he spend? Answer: 4*2=8 for apples. 3*3=9 for oranges. 8+9=17.",
    "A train travels at 60 miles per hour. How long does it take to travel 240 miles? "
    "Answer: 240 divided by 60 equals 4 hours.",
    "Maria has 5 times as many marbles as John. John has 12 marbles. "
    "How many marbles do they have together? Answer: Maria has 5*12=60 marbles. Together: 60+12=72.",
]


def load_model_and_tokenizer(path: Path, device: str = "cuda"):
    """Load a dolomite model from an unsharded HF checkpoint."""
    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.float32, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(path))
    return model, tokenizer


def get_transformer_blocks(model):
    """Return the nn.ModuleList of transformer blocks."""
    for attr in ("transformer", "model"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "h"):
            return m.h
        if m is not None and hasattr(m, "transformer") and hasattr(m.transformer, "h"):
            return m.transformer.h
    raise AttributeError(f"Cannot find transformer blocks on {type(model)}")


def get_lm_head_weight(model) -> torch.Tensor:
    """Return W_U (the unembedding weight matrix), shape [vocab, d].

    For tied embeddings: W_U = wte.weight.
    For untied:         W_U = lm_head.weight.
    """
    # CausalLMModelMixin: get_output_embeddings returns wte (tied) or lm_head (untied)
    out_emb = model.get_output_embeddings()
    if out_emb is not None:
        return out_emb.weight.detach().float()  # [vocab, d]
    # Fallback: check common attribute names
    for attr in ("lm_head", "transformer.wte"):
        parts = attr.split(".")
        obj = model
        for p in parts:
            obj = getattr(obj, p, None)
            if obj is None:
                break
        if obj is not None and hasattr(obj, "weight"):
            return obj.weight.detach().float()
    raise AttributeError("Cannot find LM head weight")


def compute_lk_basis(W_U: torch.Tensor, k: int = 256, min_energy_fraction: float = 0.95,
                     device: str = "cpu") -> torch.Tensor:
    """Compute top-k right singular vectors of W_U as orthonormal basis L_k.

    W_U has shape [vocab, d]. We want the top-k right singular vectors of W_U,
    i.e. columns of V in W_U = U S V^T. These form an orthonormal basis for the
    subspace of d-space that W_U (the LM head) projects onto most strongly.

    Returns L_k ∈ R^{d×k} (each column is a right singular vector).

    Also checks that the top-k singular values capture ≥ min_energy_fraction
    of the total squared singular value energy (Frobenius norm fraction).
    """
    W = W_U.to(device)
    # SVD: W_U = U @ diag(S) @ V^T,  V has shape [d, min(vocab, d)]
    # We want right singular vectors: columns of V
    # Use torch.linalg.svd with full_matrices=False for efficiency
    # W_U: [vocab, d].  After transpose: [d, vocab].
    # For large vocab, it's cheaper to do SVD on W_U directly.
    # torch.linalg.svd(W, full_matrices=False) returns U[vocab,k], S[k], Vh[k,d]
    # Right singular vectors = rows of Vh = Vh[i, :] for each i.
    # L_k is formed by taking the first k columns of V = Vh^T → shape [d, k]
    k_actual = min(k, W.shape[0], W.shape[1])
    print(f"  Computing SVD of W_U ({W.shape[0]}×{W.shape[1]}), keeping top-{k_actual} singular vectors...")

    # Use truncated SVD via torch (no direct truncated SVD, so use full then slice)
    # For vocab=100352 x d=1280, full_matrices=False gives Vh of shape [1280, 1280]
    # This is memory-feasible.
    _, S, Vh = torch.linalg.svd(W, full_matrices=False)
    # Vh: [min(vocab,d), d]. Right singular vectors are rows of Vh.
    # L_k columns are the right singular vectors for top-k singular values.
    L_k = Vh[:k_actual].T  # [d, k_actual]

    # Check energy fraction
    energy_total = (S ** 2).sum().item()
    energy_topk = (S[:k_actual] ** 2).sum().item()
    frac = energy_topk / (energy_total + 1e-12)
    print(f"  Top-{k_actual} singular vectors capture {frac*100:.1f}% of W_U energy "
          f"(target ≥{min_energy_fraction*100:.0f}%)")

    if frac < min_energy_fraction:
        print(f"  WARNING: only {frac*100:.1f}% energy captured, consider increasing k.")

    return L_k.to(device), frac, S.cpu()


def compute_epsilon(h: torch.Tensor, L_k: torch.Tensor) -> torch.Tensor:
    """Compute ε(h) = ||(I - L_k L_k^T) h|| / ||h|| for each position.

    h:   [T, d]    hidden states at T positions
    L_k: [d, k]    orthonormal basis (columns are orthonormal)

    Returns: [T] tensor of epsilon values (0 = fully in LM-head subspace,
                                            1 = fully orthogonal to it)
    """
    # Project h onto L_k subspace: proj = h @ L_k @ L_k^T = (h @ L_k) @ L_k^T
    # h: [T, d], L_k: [d, k]
    h_proj_coeff = h @ L_k          # [T, k]
    h_in_lk = h_proj_coeff @ L_k.T  # [T, d]
    h_perp = h - h_in_lk            # [T, d]

    norm_h = h.norm(dim=-1).clamp(min=1e-8)         # [T]
    norm_perp = h_perp.norm(dim=-1)                  # [T]
    return norm_perp / norm_h                        # [T]


def analyze_model(model_name: str, model_path: Path, n_batches: int,
                  k_svd: int = 256, device: str = "cuda") -> dict:
    """Run the L-perp trajectory analysis for one model.

    Returns a dict with per-block per-iteration mean epsilon values and
    the correlation between epsilon and hidden state norm.
    """
    print(f"\n=== Analyzing model: {model_name} ===")
    model, tokenizer = load_model_and_tokenizer(model_path, device)

    # ---- Get LM head basis ------------------------------------------------
    W_U = get_lm_head_weight(model)
    L_k, energy_frac, singular_vals = compute_lk_basis(W_U, k=k_svd, device=device)

    # ---- Prepare input texts -----------------------------------------------
    all_texts = WIKITEXT_PASSAGES[:n_batches] + GSM8K_PASSAGES
    all_texts = all_texts[:n_batches]
    if n_batches > len(WIKITEXT_PASSAGES):
        all_texts = WIKITEXT_PASSAGES + GSM8K_PASSAGES
        all_texts = all_texts[:n_batches]

    # ---- Identify blocks and their types -----------------------------------
    blocks = get_transformer_blocks(model)
    layer_iterations = model.config.layer_iterations  # list of int, len=num_layers

    # Classify each block as 'energy' (has .ln, called per-iter) or 'gpt' (has .ln_1)
    block_types = []
    for blk in blocks:
        if hasattr(blk, 'ln') and hasattr(blk, 'attn') and hasattr(blk, 'ffwd'):
            block_types.append('energy')
        elif hasattr(blk, 'ln_1'):
            block_types.append('gpt')
        else:
            block_types.append('unknown')

    print(f"  Block types: {block_types}")
    print(f"  Layer iterations: {layer_iterations}")

    # ---- Per-block, per-iteration epsilon accumulator ----------------------
    # results[block_idx][iter_j] = list of [T] epsilon tensors (one per batch)
    results = defaultdict(lambda: defaultdict(list))
    norm_results = defaultdict(lambda: defaultdict(list))  # for norm correlation

    # ---- Tokenize all texts ------------------------------------------------
    def tokenize(text):
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        return enc["input_ids"].to(device)

    # ---- Process each text -------------------------------------------------
    for text_idx, text in enumerate(all_texts):
        print(f"  Text {text_idx+1}/{len(all_texts)}: {text[:60]}...")
        input_ids = tokenize(text)

        # Collect hidden states at each block-iteration via hooks on ln/ln_1
        # For each block, we register a pre-hook on the block itself and track
        # how many times it has been called (= iteration index for recurrent blocks).
        # For energy blocks: hook on blk.ln (called once per iteration)
        # For GPT blocks:    hook on blk.ln_1 (called once, iteration=0 always)

        # call_counter[block_idx] tracks how many times the ln hook has fired
        call_counter = {}
        handles = []

        for blk_idx, (blk, blk_type, n_iters) in enumerate(
            zip(blocks, block_types, layer_iterations)
        ):
            call_counter[blk_idx] = 0

            def make_ln_hook(bidx, btype, n_it):
                def hook(module, inp):
                    # inp is a tuple; first element is the hidden state tensor
                    x = inp[0] if isinstance(inp, (tuple, list)) else inp
                    iter_j = call_counter[bidx]
                    # Only record up to n_it iterations (handle iter_dropout safety)
                    if iter_j < max(n_it, 1):
                        # [B, T, d] → take batch dim=0, positions 1: (skip BOS)
                        h_seq = x[0, 1:].detach().float()  # [T-1, d]
                        eps = compute_epsilon(h_seq, L_k)   # [T-1]
                        results[bidx][iter_j].append(eps.cpu())
                        norm_h = h_seq.norm(dim=-1).cpu()    # [T-1]
                        norm_results[bidx][iter_j].append(norm_h)
                    call_counter[bidx] += 1
                return hook

            # Hook target: .ln for energy blocks, .ln_1 for GPT blocks
            if blk_type == 'energy' and hasattr(blk, 'ln'):
                h = blk.ln.register_forward_pre_hook(make_ln_hook(blk_idx, blk_type, n_iters))
                handles.append(h)
            elif blk_type == 'gpt' and hasattr(blk, 'ln_1'):
                h = blk.ln_1.register_forward_pre_hook(make_ln_hook(blk_idx, blk_type, 1))
                handles.append(h)
            # For 'unknown' types, skip (no hook)

        with torch.no_grad():
            model(input_ids)

        for h in handles:
            h.remove()

    # ---- Compute mean epsilon per block-iteration --------------------------
    mean_eps = {}   # (block_idx, iter_j) -> float
    mean_norm = {}  # (block_idx, iter_j) -> float

    for blk_idx in sorted(results.keys()):
        for iter_j in sorted(results[blk_idx].keys()):
            eps_list = results[blk_idx][iter_j]
            if eps_list:
                # Each element is [T-1], stack across texts → [N_texts, T-1]
                eps_cat = torch.cat(eps_list)  # [total_tokens]
                mean_eps[(blk_idx, iter_j)] = eps_cat.mean().item()
            norm_list = norm_results[blk_idx][iter_j]
            if norm_list:
                norm_cat = torch.cat(norm_list)
                mean_norm[(blk_idx, iter_j)] = norm_cat.mean().item()

    # ---- Norm-epsilon correlation (scratch-space vs norm) ------------------
    # For blocks with multiple iterations, compute Pearson corr between
    # hidden state norm and epsilon across (token, iter) pairs.
    corr_by_block = {}
    for blk_idx in sorted(results.keys()):
        iter_keys = sorted(results[blk_idx].keys())
        if len(iter_keys) <= 1:
            continue  # need at least 2 iters for meaningful correlation
        eps_vals = []
        norm_vals = []
        for iter_j in iter_keys:
            if results[blk_idx][iter_j]:
                e = torch.cat(results[blk_idx][iter_j]).numpy()
                n = torch.cat(norm_results[blk_idx][iter_j]).numpy()
                eps_vals.append(e)
                norm_vals.append(n)
        if eps_vals:
            eps_all = np.concatenate(eps_vals)
            norm_all = np.concatenate(norm_vals)
            # Pearson correlation
            if eps_all.std() > 1e-8 and norm_all.std() > 1e-8:
                corr = float(np.corrcoef(eps_all, norm_all)[0, 1])
            else:
                corr = 0.0
            corr_by_block[blk_idx] = corr

    # ---- Build unrolled trajectory ----------------------------------------
    # Unrolled index: for block i, iteration j → i * max_iters + j
    # (Makes the plot x-axis continuous across blocks)
    max_iters = max(layer_iterations) if layer_iterations else 1
    trajectory = []  # list of (unrolled_idx, blk_idx, iter_j, mean_eps, blk_type)
    for blk_idx, blk_type, n_iters in zip(range(len(blocks)), block_types, layer_iterations):
        for iter_j in range(max(n_iters, 1)):
            key = (blk_idx, iter_j)
            if key in mean_eps:
                unrolled = blk_idx * max_iters + iter_j
                trajectory.append({
                    "unrolled_idx": unrolled,
                    "block_idx": blk_idx,
                    "iter_j": iter_j,
                    "mean_eps": mean_eps[key],
                    "mean_norm": mean_norm.get(key, None),
                    "block_type": blk_type,
                    "n_iters": n_iters,
                })

    stats = {
        "model_name": model_name,
        "model_path": str(model_path),
        "k_svd": k_svd,
        "energy_frac": energy_frac,
        "n_batches_used": len(all_texts),
        "layer_iterations": layer_iterations,
        "block_types": block_types,
        "trajectory": trajectory,
        "corr_eps_norm_by_block": corr_by_block,
        "mean_eps": {f"{k[0]}_{k[1]}": v for k, v in mean_eps.items()},
        "mean_norm": {f"{k[0]}_{k[1]}": v for k, v in mean_norm.items()},
        "top_singular_values": singular_vals[:20].tolist(),
    }

    del model
    torch.cuda.empty_cache()
    return stats


def plot_trajectories(all_stats: dict[str, dict], out_prefix: Path) -> None:
    """Plot L-perp trajectories for all models on a single figure."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    colors = {
        "v71": ("#e74c3c", "#c0392b"),
        "v9":  ("#2980b9", "#1a5276"),
    }
    default_color = ("#27ae60", "#1e8449")

    # ---- Plot 1: Unrolled L-perp trajectory --------------------------------
    ax = axes[0]
    for model_name, stats in all_stats.items():
        traj = stats["trajectory"]
        if not traj:
            continue
        xs = [t["unrolled_idx"] for t in traj]
        ys = [t["mean_eps"] for t in traj]
        btypes = [t["block_type"] for t in traj]
        col_pair = colors.get(model_name, default_color)

        # Plot GPT and energy points with different markers
        gpt_xs = [x for x, bt in zip(xs, btypes) if bt == "gpt"]
        gpt_ys = [y for y, bt in zip(ys, btypes) if bt == "gpt"]
        egpt_xs = [x for x, bt in zip(xs, btypes) if bt == "energy"]
        egpt_ys = [y for y, bt in zip(ys, btypes) if bt == "energy"]

        ax.plot(xs, ys, "-", color=col_pair[0], alpha=0.7, linewidth=1.5,
                label=f"{model_name}")
        if gpt_xs:
            ax.scatter(gpt_xs, gpt_ys, color=col_pair[0], marker="o", s=40, zorder=5)
        if egpt_xs:
            ax.scatter(egpt_xs, egpt_ys, color=col_pair[1], marker="*", s=80, zorder=5,
                       label=f"{model_name} EGPT")

    ax.set_xlabel("Unrolled iteration index (block × max_iters + iter)")
    ax.set_ylabel("Mean ε(h) = ||h_⊥|| / ||h||")
    ax.set_title("L⊥ trajectory: fraction of h outside LM-head subspace")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # ---- Plot 2: Per-block mean L-perp (grouped by block) ------------------
    ax2 = axes[1]
    for model_name, stats in all_stats.items():
        traj = stats["trajectory"]
        if not traj:
            continue
        col_pair = colors.get(model_name, default_color)
        layer_iters = stats["layer_iterations"]

        # For each block, plot all iterations as separate points
        for blk_idx_val in sorted(set(t["block_idx"] for t in traj)):
            blk_traj = [t for t in traj if t["block_idx"] == blk_idx_val]
            blk_traj.sort(key=lambda x: x["iter_j"])
            n_iters_blk = layer_iters[blk_idx_val] if blk_idx_val < len(layer_iters) else 1

            if len(blk_traj) == 1:
                # Single-iteration block: just a point
                ax2.scatter([blk_idx_val], [blk_traj[0]["mean_eps"]],
                            color=col_pair[0], s=30, alpha=0.7)
            else:
                # Multi-iteration block: show trajectory within block
                for i_t, t in enumerate(blk_traj):
                    alpha = 0.4 + 0.6 * i_t / max(len(blk_traj) - 1, 1)
                    ax2.scatter([blk_idx_val + i_t * 0.15], [t["mean_eps"]],
                                color=col_pair[1], s=50, alpha=alpha, zorder=5)
                # Connect with arrow or line
                x_pts = [blk_idx_val + i * 0.15 for i in range(len(blk_traj))]
                y_pts = [t["mean_eps"] for t in blk_traj]
                ax2.plot(x_pts, y_pts, "-", color=col_pair[1], alpha=0.6, linewidth=1.5,
                         label=f"{model_name} blk{blk_idx_val} ({n_iters_blk} iters)")

    ax2.set_xlabel("Block index (within-block iterations shown as x-offset)")
    ax2.set_ylabel("Mean ε(h)")
    ax2.set_title("Per-block L⊥ (iterations shown as x-spread)")
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(True, alpha=0.3)

    fig.suptitle("L⊥ trajectory: h living outside LM-head subspace L_k", fontsize=12)
    plt.tight_layout()

    for ext in ("png", "pdf"):
        fpath = out_prefix.with_suffix(f".{ext}")
        fig.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fpath}")

    plt.close(fig)

    # ---- Plot 3: Norm vs epsilon scatter (one per model with multiple-iter blocks) ---
    fig2, axes2 = plt.subplots(1, len(all_stats), figsize=(6 * len(all_stats), 5),
                               squeeze=False)
    for ax_i, (model_name, stats) in enumerate(all_stats.items()):
        ax = axes2[0][ax_i]
        corr = stats.get("corr_eps_norm_by_block", {})
        if corr:
            blks = sorted(corr.keys())
            corrs = [corr[b] for b in blks]
            ax.bar(blks, corrs, color="#e74c3c" if model_name == "v71" else "#2980b9",
                   alpha=0.8)
            ax.axhline(0, color="black", linewidth=0.8)
            ax.set_xlabel("Block index")
            ax.set_ylabel("Pearson corr(ε, ||h||)")
            ax.set_title(f"{model_name}: scratch-space vs norm correlation\n"
                         f"(positive = high-norm = more scratch-space)")
            ax.set_ylim(-1, 1)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, f"{model_name}\nNo multi-iter blocks", ha="center",
                    va="center", transform=ax.transAxes)
            ax.set_title(model_name)

    fig2.suptitle("Correlation between ε(h) and ||h|| per recurrent block", fontsize=11)
    plt.tight_layout()
    corr_path = out_prefix.parent / (out_prefix.name.replace("lperp_trajectory", "lperp_norm_corr"))
    for ext in ("png", "pdf"):
        fpath = str(corr_path) + f".{ext}"
        fig2.savefig(fpath, dpi=150, bbox_inches="tight")
        print(f"  Saved: {fpath}")
    plt.close(fig2)


def main():
    parser = argparse.ArgumentParser(description="Analyze L-perp trajectory per block-iteration")
    parser.add_argument("--models", type=str, default="v71,v9",
                        help="Comma-separated model keys to analyze (e.g. v71,v9)")
    parser.add_argument("--n_batches", type=int, default=8,
                        help="Number of text batches to process (uses WikiText + GSM8K)")
    parser.add_argument("--k_svd", type=int, default=256,
                        help="Number of top singular vectors to use for L_k basis")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_keys = [k.strip() for k in args.models.split(",")]
    unknown = [k for k in model_keys if k not in MODELS]
    if unknown:
        print(f"WARNING: Unknown model keys: {unknown}. Available: {list(MODELS.keys())}")
        model_keys = [k for k in model_keys if k in MODELS]

    print(f"Device: {args.device}")
    print(f"Models: {model_keys}")
    print(f"n_batches: {args.n_batches}, k_svd: {args.k_svd}")

    all_stats = {}
    for key in model_keys:
        path = MODELS[key]
        if not path.exists():
            print(f"  SKIP {key}: path not found ({path})")
            continue
        stats = analyze_model(key, path, args.n_batches, k_svd=args.k_svd, device=args.device)
        all_stats[key] = stats

    if not all_stats:
        print("No models analyzed. Exiting.")
        return

    # ---- Save stats JSON ---------------------------------------------------
    # Convert all numpy/torch types for JSON serialization
    def to_jsonable(obj):
        if isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, dict):
            return {k: to_jsonable(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [to_jsonable(v) for v in obj]
        return obj

    json_path = PLOTS_DIR / "lperp_stats_v71_v9.json"
    with open(json_path, "w") as f:
        json.dump(to_jsonable(all_stats), f, indent=2)
    print(f"\nStats saved: {json_path}")

    # ---- Plot ---------------------------------------------------------------
    out_prefix = PLOTS_DIR / "lperp_trajectory_v71_v9"
    plot_trajectories(all_stats, out_prefix)

    # ---- Print summary ------------------------------------------------------
    print("\n=== Summary ===")
    for model_name, stats in all_stats.items():
        traj = stats["trajectory"]
        if not traj:
            continue
        eps_vals = [t["mean_eps"] for t in traj]
        print(f"\n{model_name}:")
        print(f"  k_svd={stats['k_svd']}, energy_frac={stats['energy_frac']:.3f}")
        print(f"  ε range: [{min(eps_vals):.4f}, {max(eps_vals):.4f}]")
        print(f"  Mean ε: {np.mean(eps_vals):.4f}")
        for t in traj:
            if t["block_type"] == "energy":
                print(f"  Block {t['block_idx']} (EGPT, {t['n_iters']} iters), "
                      f"iter {t['iter_j']}: ε={t['mean_eps']:.4f}, "
                      f"||h||={t['mean_norm']:.2f}" if t['mean_norm'] else
                      f"  Block {t['block_idx']} (EGPT), iter {t['iter_j']}: ε={t['mean_eps']:.4f}")
        corr = stats.get("corr_eps_norm_by_block", {})
        if corr:
            print(f"  Norm-eps correlations: {corr}")


if __name__ == "__main__":
    main()
