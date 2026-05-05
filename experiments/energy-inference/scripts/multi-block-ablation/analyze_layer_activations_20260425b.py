"""Same-input delta layer similarity analysis: heatmaps, successive plots, attn/FFN breakdown.

Extends 20260425.py with:
  1. Cleaner same-input delta heatmaps (the primary metric — all blocks see same h)
  2. Attn and FFN sub-output same-input delta cosine similarity matrices + line plots
  3. Combined breakdown bar chart: full / attn / FFN same-input delta cos-sim per model

Same-input delta is defined as:
    Δ_i = Block_i(h) - h    for the SAME fixed base embedding h fed to every block

This isolates the per-block *operator* without any input-distribution confound.
"""

import sys, os, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO)
import lm_engine.hf_models  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE  = Path(f"{REPO}/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"

EVAL_TEXT = """
The transformer architecture has become the dominant paradigm in natural language processing.
Self-attention mechanisms allow models to relate different positions in a sequence to compute
representations. The key insight is that language understanding requires tracking long-range
dependencies between words and phrases. Energy-based models offer an alternative perspective,
where the attention computation corresponds to gradient descent on an energy function. This
connection between energy minimization and attention is not merely an analogy but a precise
mathematical relationship: the output of energy attention equals the gradient of the energy
with respect to the hidden state. Recurrent computation in transformer blocks may thus
implement an iterative optimization process, converging toward a low-energy state that captures
the semantic content of the input. Layer similarity then measures whether different blocks are
performing redundant optimization steps or distinct transformations.
""".strip()

N_TOKENS = 256

MODELS = {
    "V0 GPT\n12×1": {
        "path": BASE / "v0_gpt_baseline_d768" / "unsharded",
        "color": "#4477AA", "n_layers": 12, "arch": "GPT", "linestyle": "-",
    },
    "V1 EGPT\n12×1": {
        "path": BASE / "v1_12x1_d768_lr2e3" / "unsharded",
        "color": "#EE6677", "n_layers": 12, "arch": "EGPT", "linestyle": "-",
    },
    "V39 EGPT\n+act-cosreg": {
        "path": BASE / "v39_egpt_cosreg_12x1_d768_lr2e3" / "unsharded",
        "color": "#AA3377", "n_layers": 12, "arch": "EGPT+cosreg", "linestyle": "--",
    },
    "V41 Sandwich\n2G+8E+2G": {
        "path": BASE / "v41_sandwich_2gpt8e2gpt_d768_lr2e3" / "unsharded",
        "color": "#228833", "n_layers": 12, "arch": "Sandwich", "linestyle": "-.",
    },
    "V42 Naive\nmerge init": {
        "path": BASE / "v42_merged_naive_init",
        "color": "#CCBB44", "n_layers": 6, "arch": "Merged", "linestyle": "--",
    },
    "V42 Naive\n+5k FT": {
        "path": BASE / "v42_egpt_6layer_naive_merge_finetune_d768" / "unsharded",
        "color": "#EE8800", "n_layers": 6, "arch": "Merged+FT", "linestyle": "-",
    },
    "V43 Procrustes\nmerge init": {
        "path": BASE / "v43_merged_procrustes_init",
        "color": "#66CCEE", "n_layers": 6, "arch": "Merged", "linestyle": "--",
    },
    "V43 Procrustes\n+5k FT": {
        "path": BASE / "v43_egpt_6layer_procrustes_merge_finetune_d768" / "unsharded",
        "color": "#0077BB", "n_layers": 6, "arch": "Merged+FT", "linestyle": "-",
    },
}

# V41 block type labels for annotation (0-indexed)
V41_BLOCK_TYPES = ["GPT"] * 2 + ["EGPT"] * 8 + ["GPT"] * 2


def load_model(info):
    path = info["path"]
    if not Path(path).exists():
        print(f"  Missing: {path}")
        return None
    print(f"  Loading from {path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        return model
    except Exception as e:
        print(f"  Error: {e}")
        return None


def get_input_tokens(tokenizer, text, n_tokens):
    ids = tokenizer.encode(text)[:n_tokens]
    return torch.tensor([ids])


def get_embedding(model, input_ids):
    transformer = model.transformer
    with torch.no_grad():
        h = transformer.wte(input_ids).float()
        if hasattr(transformer, "wpe"):
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            h = h + transformer.wpe(pos).float()
    return h


def get_rope_cos_sin(model, input_ids, h_base):
    transformer = model.transformer
    if not hasattr(transformer, "position_embedding_type"):
        return None
    if transformer.position_embedding_type != "rope":
        return None
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        return transformer._get_rope_cos_sin(
            key_length=input_ids.shape[1],
            position_ids=position_ids,
            dtype=h_base.dtype,
        )


def same_input_full_attn_ffn(model, h_base, rope_cos_sin=None):
    """Feed h_base independently to each block; return (full_deltas, attn_outs, ffn_outs).

    full_delta_i = Block_i(h_base) - h_base
    attn_out_i   = attn sub-output of Block_i fed h_base
    ffn_out_i    = FFN sub-output of Block_i fed h_base

    All measured with the SAME h_base — clean operator comparison with no input-shift confound.
    """
    blocks = list(model.transformer.h)
    full_deltas = []
    attn_outs   = []
    ffn_outs    = []

    for blk in blocks:
        # Hook attn and FFN sub-modules
        _attn_buf = []
        _ffn_buf  = []

        def _make_attn_hook(buf):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                buf.append(o.detach().float().squeeze(0))
            return hook

        def _make_ffn_hook(buf):
            def hook(mod, inp, out):
                o = out[0] if isinstance(out, (tuple, list)) else out
                buf.append(o.detach().float().squeeze(0))
            return hook

        hooks = []
        for attr_name in ("attn", "sequence_mixer"):
            if hasattr(blk, attr_name):
                hooks.append(getattr(blk, attr_name).register_forward_hook(_make_attn_hook(_attn_buf)))
                break
        for attr_name in ("ffwd", "mlp", "feed_forward", "ff"):
            if hasattr(blk, attr_name):
                hooks.append(getattr(blk, attr_name).register_forward_hook(_make_ffn_hook(_ffn_buf)))
                break

        with torch.no_grad():
            out = blk(h_base, rope_cos_sin=rope_cos_sin)
        h_out = out[0] if isinstance(out, (tuple, list)) else out

        for hk in hooks:
            hk.remove()

        full_deltas.append((h_out.float() - h_base).squeeze(0))        # (T, d)
        attn_outs.append(_attn_buf[0] if _attn_buf else None)
        ffn_outs.append(_ffn_buf[0] if _ffn_buf else None)

    return full_deltas, attn_outs, ffn_outs


def cos_sim_matrix(act_list):
    """(N, N) cosine similarity matrix; each element is a (T, d) tensor, flattened to a vector.

    Skips None entries — returns a matrix for only the non-None elements.
    Use cos_sim_successive for position-aligned successive pairs (handles sparse lists).
    """
    valid = [a for a in act_list if a is not None]
    if not valid:
        return None
    n = len(valid)
    vecs = [a.reshape(-1) for a in valid]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cs = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            mat[i, j] = cs
            mat[j, i] = cs
    return mat


def cos_sim_successive(act_list):
    """Position-aligned successive cosine similarity, returning NaN for missing pairs.

    act_list has length N (same as number of blocks), with None for blocks lacking the sub-module.
    Returns list of length N-1: cos_sim(act_list[i], act_list[i+1]) or nan.
    """
    n = len(act_list)
    result = []
    for i in range(n - 1):
        a, b = act_list[i], act_list[i + 1]
        if a is not None and b is not None:
            cs = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            result.append(cs)
        else:
            result.append(float("nan"))
    return result


def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


# ── Plotting helpers ────────────────────────────────────────────────────────


def plot_heatmap_single(mat, title, stem, block_types=None):
    n = mat.shape[0]
    labels = [str(i) for i in range(n)]
    if block_types is not None and len(block_types) == n:
        labels = [f"{i}\n({block_types[i][:1]})" for i in range(n)]

    off_vals = [mat[i, j] for i in range(n) for j in range(n) if i != j]
    vmin, vmax = min(off_vals), max(off_vals)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.55 + 1), max(4, n * 0.55)))
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")

    vcenter = (vmin + vmax) / 2
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if val < vcenter else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(5, 7 - n // 4), color=color)
    savefig(fig, stem)


def plot_three_heatmaps(full_mat, attn_mat, ffn_mat, label, stem, block_types=None):
    """Three side-by-side heatmaps: full-block / attn / FFN same-input delta.

    Each matrix may have a different size (e.g. V41: attn=12×12, ffn=8×8 for EGPT blocks only).
    """
    mats  = [full_mat, attn_mat, ffn_mat]
    names = ["Full block Δ", "Attn sub-out Δ", "FFN sub-out Δ"]
    n_full = full_mat.shape[0]

    fig, axes = plt.subplots(1, 3, figsize=(3 * max(4, n_full * 0.5), max(4, n_full * 0.5) + 0.5))
    for ax, mat, name in zip(axes, mats, names):
        if mat is None:
            ax.set_visible(False)
            continue
        n = mat.shape[0]
        lbl = [str(i) for i in range(n)]
        if block_types is not None and n == n_full:
            lbl = [f"{i}({block_types[i][:1]})" for i in range(n)]
        off_vals = [mat[i, j] for i in range(n) for j in range(n) if i != j]
        vmin, vmax = min(off_vals), max(off_vals)
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_xticklabels(lbl, fontsize=max(5, 8 - n//4))
        ax.set_yticks(range(n)); ax.set_yticklabels(lbl, fontsize=max(5, 8 - n//4))
        ax.set_title(name, fontsize=10, fontweight="bold")
        vcenter = (vmin + vmax) / 2
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(4, 6 - n // 4), color="white" if val < vcenter else "black")

    fig.suptitle(f"Same-input delta cosine similarity — {label.replace(chr(10), ' ')}",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, stem)


def plot_successive_comparison(all_results, models_12, models_6):
    """Two-panel line plot: successive same-input delta cos-sim for 12 and 6 layer models."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 4.5))

    for ax, subset, title in [
        (axes[0], models_12, "12-layer models"),
        (axes[1], models_6,  "6-layer merged models"),
    ]:
        if not subset:
            ax.set_visible(False)
            continue
        for label, data in subset.items():
            info  = MODELS.get(label, {})
            color = info.get("color", "#888888")
            ls    = info.get("linestyle", "-")
            lbl   = label.replace("\n", " ")
            n     = len(data["full_succ"])
            x     = np.arange(n)

            ax.plot(x, data["full_succ"], color=color, linewidth=2.2, marker="o",
                    markersize=6, linestyle=ls, label=lbl)
            ax.plot(x, data["attn_succ"], color=color, linewidth=1.4, marker="s",
                    markersize=4, linestyle=ls, alpha=0.65)
            ax.plot(x, data["ffn_succ"],  color=color, linewidth=1.4, marker="^",
                    markersize=4, linestyle=ls, alpha=0.45)

        # Legend: model names + metric legend
        ax.plot([], [], color="gray", linewidth=2.2, marker="o", markersize=6,  label="Full block Δ")
        ax.plot([], [], color="gray", linewidth=1.4, marker="s", markersize=4, alpha=0.65, label="Attn Δ")
        ax.plot([], [], color="gray", linewidth=1.4, marker="^", markersize=4, alpha=0.45, label="FFN Δ")
        ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6)
        ax.set_xlabel("Layer pair (i, i+1)", fontsize=11)
        ax.set_ylabel("Same-input delta cosine similarity", fontsize=11)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Layer functional similarity: same-input delta cos-sim\n"
                 "(solid=full block, semi=attn, faint=FFN)", fontsize=12, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "same_delta_successive_full_attn_ffn")


def plot_breakdown_bar(all_results):
    """Grouped bar chart: full / attn / FFN same-input delta (off-diagonal mean) per model."""
    labels_list   = []
    full_avgs, attn_avgs, ffn_avgs = [], [], []
    rand_avgs     = []
    colors_list   = []

    for label, data in all_results.items():
        n = data["n_layers"]
        mask_n = ~np.eye(n, dtype=bool)
        full_avgs.append(float(data["full_mat"][mask_n].mean()))
        rand_avgs.append(float(np.nanmean(data["rand_succ"])))
        # attn/ffn matrices may be smaller than n×n (V41: GPT blocks lack FFN)
        for mat_key, lst in [("attn_mat", attn_avgs), ("ffn_mat", ffn_avgs)]:
            m = data[mat_key]
            if m is not None:
                m_mask = ~np.eye(m.shape[0], dtype=bool)
                lst.append(float(m[m_mask].mean()))
            else:
                lst.append(float("nan"))
        labels_list.append(label.replace("\n", " "))
        colors_list.append(MODELS.get(label, {}).get("color", "#888888"))

    x     = np.arange(len(labels_list))
    width = 0.22
    fig, ax = plt.subplots(figsize=(max(8, len(labels_list) * 1.5), 5))

    ax.bar(x - 1.5*width, full_avgs, width, label="Full block Δ",   color=colors_list, edgecolor="white", alpha=0.95)
    ax.bar(x - 0.5*width, attn_avgs, width, label="Attn sub-out Δ", color=colors_list, edgecolor="white", alpha=0.65)
    ax.bar(x + 0.5*width, ffn_avgs,  width, label="FFN sub-out Δ",  color=colors_list, edgecolor="white", alpha=0.40)
    ax.bar(x + 1.5*width, rand_avgs, width, label="Random init Δ",  color="#AAAAAA",   edgecolor="white", alpha=0.50)

    for i, (full, attn, ffn) in enumerate(zip(full_avgs, attn_avgs, ffn_avgs)):
        ax.text(x[i] - 1.5*width, full + 0.004, f"{full:.2f}", ha="center", va="bottom", fontsize=8)
        if not np.isnan(attn):
            ax.text(x[i] - 0.5*width, attn + 0.004, f"{attn:.2f}", ha="center", va="bottom", fontsize=7, alpha=0.8)
        if not np.isnan(ffn):
            ax.text(x[i] + 0.5*width, ffn  + 0.004, f"{ffn:.2f}",  ha="center", va="bottom", fontsize=7, alpha=0.6)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg same-input delta cosine similarity (off-diagonal)", fontsize=11)
    ax.set_title("Same-input delta layer similarity breakdown: full / attn / FFN",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "same_delta_breakdown_bar")


def plot_v41_annotated(data):
    """Special V41 plot annotating GPT vs EGPT block boundaries."""
    full_succ = data["full_succ"]
    attn_succ = data["attn_succ"]
    ffn_succ  = data["ffn_succ"]
    n = len(full_succ)
    x = np.arange(n)

    fig, ax = plt.subplots(figsize=(10, 4))

    ax.fill_betweenx([-0.15, 1.0], -0.5, 1.5, alpha=0.07, color="steelblue", label="GPT region")
    ax.fill_betweenx([-0.15, 1.0],  1.5, 8.5, alpha=0.07, color="crimson",   label="EGPT core region")
    ax.fill_betweenx([-0.15, 1.0],  8.5, 10.5, alpha=0.07, color="steelblue")

    ax.plot(x, full_succ, color="#228833", linewidth=2.5, marker="o", markersize=7, label="Full block Δ")
    ax.plot(x, attn_succ, color="#228833", linewidth=1.5, marker="s", markersize=5, alpha=0.65, label="Attn Δ")
    ax.plot(x, ffn_succ,  color="#228833", linewidth=1.5, marker="^", markersize=5, alpha=0.45, label="FFN Δ")

    ax.axvline(1.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.axvline(8.5, color="gray", linestyle="--", linewidth=1.2, alpha=0.7)
    ax.text(0.5, 0.93, "GPT→EGPT\nboundary", ha="center", fontsize=8, transform=ax.transAxes,
            color="gray")

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer pair (i, i+1)", fontsize=11)
    ax.set_ylabel("Same-input delta cosine similarity", fontsize=11)
    ax.set_title("V41 Sandwich 2G+8E+2G: same-input delta per layer pair\n"
                 "(shaded: GPT=blue, EGPT=red)", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right")
    ax.set_xlim(-0.5, n - 0.5)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig(fig, "v41_sandwich_same_delta_annotated")


# ── Main analysis ────────────────────────────────────────────────────────────


def analyze_model(label, info, input_ids, h_base_fn, rope_fn):
    print(f"\n=== {label.replace(chr(10), ' ')} ===")
    model = load_model(info)
    if model is None:
        return None

    h_base      = h_base_fn(model, input_ids)
    rope_cos_sin = rope_fn(model, input_ids, h_base)

    # Same-input delta with attn/FFN sub-output capture
    full_deltas, attn_outs, ffn_outs = same_input_full_attn_ffn(model, h_base, rope_cos_sin)
    n = len(full_deltas)
    print(f"  {n} blocks, {sum(a is not None for a in attn_outs)} attn, {sum(f is not None for f in ffn_outs)} ffn")

    full_mat = cos_sim_matrix(full_deltas)
    attn_mat = cos_sim_matrix([a for a in attn_outs if a is not None]) if any(a is not None for a in attn_outs) else None
    ffn_mat  = cos_sim_matrix([f for f in ffn_outs  if f is not None]) if any(f is not None for f in ffn_outs)  else None

    # Random init baseline (same architecture)
    rand_model  = AutoModelForCausalLM.from_config(model.config)
    rand_model.eval()
    rand_rope   = rope_fn(rand_model, input_ids, h_base)
    rand_deltas, _, _ = same_input_full_attn_ffn(rand_model, h_base, rand_rope)
    rand_mat = cos_sim_matrix(rand_deltas)
    del rand_model

    # Successive pairs: position-aligned (handles V41 where FFN list has Nones for GPT blocks)
    full_succ = [full_mat[i, i+1] for i in range(n - 1)]
    attn_succ = cos_sim_successive(attn_outs)
    ffn_succ  = cos_sim_successive(ffn_outs)
    rand_succ = [rand_mat[i, i+1] for i in range(n - 1)]

    print(f"  Full-block succ   (mean={np.nanmean(full_succ):.4f}): {[f'{v:.3f}' for v in full_succ]}")
    print(f"  Attn sub-out succ (mean={np.nanmean(attn_succ):.4f}): {[f'{v:.3f}' for v in attn_succ]}")
    print(f"  FFN sub-out succ  (mean={np.nanmean(ffn_succ):.4f}):  {[f'{v:.3f}' for v in ffn_succ]}")
    print(f"  Random init succ  (mean={np.nanmean(rand_succ):.4f}): {[f'{v:.3f}' for v in rand_succ]}")

    del model
    return {
        "n_layers":  n,
        "full_mat":  full_mat,
        "attn_mat":  attn_mat,
        "ffn_mat":   ffn_mat,
        "rand_mat":  rand_mat,
        "full_succ": full_succ,
        "attn_succ": attn_succ,
        "ffn_succ":  ffn_succ,
        "rand_succ": rand_succ,
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    input_ids = get_input_tokens(tokenizer, EVAL_TEXT, N_TOKENS)
    print(f"Using {input_ids.shape[1]} tokens.")

    # Pre-bind helpers that take (model, input_ids, ...) to avoid re-instantiation
    def _h_base(model, ids): return get_embedding(model, ids)
    def _rope(model, ids, h): return get_rope_cos_sin(model, ids, h)

    all_results = {}
    for label, info in MODELS.items():
        r = analyze_model(label, info, input_ids, _h_base, _rope)
        if r is not None:
            all_results[label] = r

    # 1. Per-model three-panel heatmaps (full / attn / FFN same-input delta)
    for label, data in all_results.items():
        stem = (label.replace("\n","_").replace(" ","").replace("/","")
                .replace("(","").replace(")","").replace("+","p").replace("×","x"))
        bt = V41_BLOCK_TYPES if "V41" in label else None
        plot_three_heatmaps(data["full_mat"], data["attn_mat"], data["ffn_mat"],
                            label, f"same_delta_3panel_{stem}", block_types=bt)

    # 2. Also save standalone same-input delta heatmap per model
    for label, data in all_results.items():
        stem = (label.replace("\n","_").replace(" ","").replace("/","")
                .replace("(","").replace(")","").replace("+","p").replace("×","x"))
        bt = V41_BLOCK_TYPES if "V41" in label else None
        plot_heatmap_single(data["full_mat"],
                            f"Same-input delta cosine sim — {label.replace(chr(10), ' ')}",
                            f"same_delta_heatmap2_{stem}", block_types=bt)

    # 3. Successive line plots (full / attn / FFN same-input delta)
    models_12 = {k: v for k, v in all_results.items() if v["n_layers"] == 12}
    models_6  = {k: v for k, v in all_results.items() if v["n_layers"] == 6}
    plot_successive_comparison(all_results, models_12, models_6)

    # 4. V41 annotated plot (GPT vs EGPT regions)
    if "V41 Sandwich\n2G+8E+2G" in all_results:
        plot_v41_annotated(all_results["V41 Sandwich\n2G+8E+2G"])

    # 5. Breakdown bar chart: full / attn / FFN / random per model
    plot_breakdown_bar(all_results)

    # 6. Summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Model':<30} {'N':>3} {'Full Δ succ':>12} {'Attn Δ succ':>12} {'FFN Δ succ':>12} {'Rand succ':>10}")
    print("-" * 85)
    for label, data in all_results.items():
        print(f"{label.replace(chr(10),' '):<30} {data['n_layers']:>3} "
              f"{np.nanmean(data['full_succ']):>12.4f} "
              f"{np.nanmean(data['attn_succ']):>12.4f} "
              f"{np.nanmean(data['ffn_succ']):>12.4f} "
              f"{np.nanmean(data['rand_succ']):>10.4f}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
