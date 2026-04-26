"""Layer merger analysis for V39-V43 experiments.

Computes:
1. Weight-space cosine similarity between all layer pairs
2. Activation-space cosine similarity on sample data
3. Plots: pairwise heatmaps, successive-pair line plots, avg-sim bar charts

Models analyzed:
- V0  GPT 12x1 d768 (baseline)
- V1  EGPT 12x1 d768 (baseline EGPT)
- V39 EGPT 12x1 d768 + cosreg (λ=0.01, 30k)
- V41 Sandwich 2GPT+8EGPT+2GPT d768 (30k)
- V42_init  naive merge 12→6 (before FT)
- V42_ft    naive merge + 5k FT
- V43_init  Procrustes merge 12→6 (before FT)
- V43_ft    Procrustes merge + 5k FT
"""

import sys, os, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from safetensors import safe_open

REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
sys.path.insert(0, REPO)
import lm_engine.hf_models  # register energy model type

from transformers import AutoModelForCausalLM

BASE = Path(f"{REPO}/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

# ── Model catalog ────────────────────────────────────────────────────────────
MODELS = {
    "V0 GPT\n12×1 d768": {
        "path": BASE / "v0_gpt_baseline_d768" / "unsharded",
        "color": "#4477AA", "n_layers": 12, "arch": "GPT",
    },
    "V1 EGPT\n12×1 d768": {
        "path": BASE / "v1_12x1_d768_lr2e3" / "unsharded",
        "color": "#EE6677", "n_layers": 12, "arch": "EGPT",
    },
    "V39 EGPT\n+cosreg": {
        "path": BASE / "v39_egpt_cosreg_12x1_d768_lr2e3" / "unsharded",
        "color": "#AA3377", "n_layers": 12, "arch": "EGPT+cosreg",
    },
    "V41 Sandwich\n2G+8E+2G": {
        "path": BASE / "v41_sandwich_2gpt8e2gpt_d768_lr2e3" / "unsharded",
        "color": "#228833", "n_layers": 12, "arch": "Sandwich",
    },
    "V42 Naive\nmerge (init)": {
        "path": BASE / "v42_merged_naive_init",
        "color": "#CCBB44", "n_layers": 6, "arch": "Merged",
    },
    "V42 Naive\n+5k FT": {
        "path": None, "dolomite": BASE / "v42_egpt_6layer_naive_merge_finetune_d768",
        "color": "#EE8800", "n_layers": 6, "arch": "Merged+FT",
    },
    "V43 Procrustes\nmerge (init)": {
        "path": BASE / "v43_merged_procrustes_init",
        "color": "#66CCEE", "n_layers": 6, "arch": "Merged",
    },
    "V43 Procrustes\n+5k FT": {
        "path": None, "dolomite": BASE / "v43_egpt_6layer_procrustes_merge_finetune_d768",
        "color": "#0077BB", "n_layers": 6, "arch": "Merged+FT",
    },
}

# ── Load helpers ─────────────────────────────────────────────────────────────

def load_model(info):
    path = info.get("path")
    if path is None:
        # Need to load from dolomite checkpoint
        dol_path = info["dolomite"]
        # lm_engine.unshard writes to top-level <save_path>/unsharded
        top_unsharded = dol_path / "unsharded"
        if top_unsharded.exists():
            path = top_unsharded
        else:
            latest_json = dol_path / "latest_checkpointed_iteration.json"
            step = json.load(open(latest_json))["latest_checkpointed_iteration"]
            step_dir = dol_path / f"global_step{step}"
            # Check if unsharded exists inside step dir
            step_unsharded = step_dir / "unsharded"
            if step_unsharded.exists():
                path = step_unsharded
            else:
                print(f"  WARNING: no unsharded at {dol_path}, skipping")
                return None
    if not Path(path).exists():
        print(f"  Missing: {path}")
        return None
    print(f"  Loading from {path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32, device_map="cpu"
        )
        return model
    except Exception as e:
        print(f"  Error loading {path}: {e}")
        return None


def get_blocks(model):
    """Return list of layer blocks from model."""
    # EnergyForCausalLM / GPTForCausalLM: model.transformer.h
    if hasattr(model, "transformer") and hasattr(model.transformer, "h"):
        return list(model.transformer.h)
    # Fallback
    for attr in ["model", "transformer"]:
        if hasattr(model, attr):
            inner = getattr(model, attr)
            if hasattr(inner, "h"):
                return list(inner.h)
    return []


# ── Weight-space cosine similarity ──────────────────────────────────────────

def block_weight_vector(block):
    """Flatten all parameters of a block into a single vector (keyed by name)."""
    return {k: p.data.float().reshape(-1) for k, p in block.named_parameters()}


def _cosine_sim_dicts(d1, d2):
    """Cosine similarity between two param-name dicts on their common keys."""
    common = sorted(set(d1) & set(d2))
    if not common:
        return 0.0
    v1 = torch.cat([d1[k] for k in common])
    v2 = torch.cat([d2[k] for k in common])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def weight_cos_sim_matrix(blocks):
    """Return (n_layers x n_layers) cosine-similarity matrix over weight vectors.
    Uses only parameter names common to each pair, handling mixed-arch models."""
    vecs = [block_weight_vector(b) for b in blocks]
    n = len(vecs)
    mat = torch.zeros(n, n)
    for i in range(n):
        for j in range(i, n):
            cs = _cosine_sim_dicts(vecs[i], vecs[j])
            mat[i, j] = cs
            mat[j, i] = cs
    return mat.numpy()


def attn_weight_vector(block):
    """Attn-only weight dict (params with 'attn'/'W_Q'/'c_proj' in name)."""
    return {k: p.data.float().reshape(-1) for k, p in block.named_parameters()
            if "attn" in k or "c_attn" in k or "W_Q" in k or "c_proj" in k}


def ffn_weight_vector(block):
    """FFN-only weight dict."""
    return {k: p.data.float().reshape(-1) for k, p in block.named_parameters()
            if "ffwd" in k or "mlp" in k or "ff" in k}


# ── Plotting ─────────────────────────────────────────────────────────────────

def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


def plot_heatmap(mat, title, stem, labels=None):
    n = mat.shape[0]
    labels = labels or [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.55 + 1), max(4, n * 0.55)))
    # Use diverging colormap centered at 0
    vmax = max(abs(mat[i, j]) for i in range(n) for j in range(n) if i != j)
    vmax = min(vmax, 0.99)
    im = ax.imshow(mat, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if abs(val) > 0.5 * vmax else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(5, 7 - n // 4), color=color)
    savefig(fig, stem)


def plot_successive_line(results, stem, title, key="successive"):
    """Line plot: cosine similarity of consecutive layer pairs."""
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in results.items():
        cs = data.get(key)
        if cs is None: continue
        x = np.arange(len(cs))
        color = MODELS[label]["color"] if label in MODELS else None
        ax.plot(x, cs, marker="o", label=label.replace("\n", " "),
                color=color, linewidth=2, markersize=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer pair (i, i+1)", fontsize=11)
    ax.set_ylabel("Cosine similarity (weights)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.set_ylim(-0.15, 1.05)
    ax.grid(True, alpha=0.3)
    savefig(fig, stem)


def plot_avg_sim_bar(results, stem, title):
    """Bar chart of average off-diagonal cosine similarity per model."""
    labels, avgs, colors = [], [], []
    for label, data in results.items():
        mat = data.get("matrix")
        if mat is None: continue
        n = mat.shape[0]
        # avg of all off-diagonal pairs
        mask = ~np.eye(n, dtype=bool)
        avgs.append(float(mat[mask].mean()))
        labels.append(label.replace("\n", " "))
        colors.append(MODELS[label]["color"] if label in MODELS else "#888888")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avgs, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg weight cosine sim (off-diagonal)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, stem)


def plot_avg_sim_bar_attn_ffn(results, stem, title):
    """Grouped bar: avg sim for full / attn / ffn weights."""
    labels = []
    full_avgs, attn_avgs, ffn_avgs = [], [], []
    colors = []
    for label, data in results.items():
        if data.get("matrix") is None: continue
        n = data["matrix"].shape[0]
        mask = ~np.eye(n, dtype=bool)
        full_avgs.append(float(data["matrix"][mask].mean()))
        attn_avgs.append(float(data["attn_matrix"][mask].mean()) if data.get("attn_matrix") is not None else float("nan"))
        ffn_avgs.append(float(data["ffn_matrix"][mask].mean()) if data.get("ffn_matrix") is not None else float("nan"))
        labels.append(label.replace("\n", " "))
        colors.append(MODELS[label]["color"] if label in MODELS else "#888888")

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    b1 = ax.bar(x - width, full_avgs, width, label="Full block", color=colors, edgecolor="white", alpha=0.9)
    b2 = ax.bar(x,         attn_avgs, width, label="Attn only",  color=colors, edgecolor="white", alpha=0.6)
    b3 = ax.bar(x + width, ffn_avgs,  width, label="FFN only",   color=colors, edgecolor="white", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg weight cosine sim", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, stem)


# ── Main analysis ─────────────────────────────────────────────────────────────

def analyze_model(label, info):
    print(f"\n=== {label.replace(chr(10), ' ')} ===")
    model = load_model(info)
    if model is None:
        return None
    blocks = get_blocks(model)
    if not blocks:
        print("  No blocks found!")
        return None
    n = len(blocks)
    print(f"  {n} blocks")

    # Full weight cos-sim matrix
    mat = weight_cos_sim_matrix(blocks)

    # Attn and FFN matrices
    attn_vecs = [attn_weight_vector(b) for b in blocks]
    ffn_vecs = [ffn_weight_vector(b) for b in blocks]

    def vec_matrix(vecs):
        n = len(vecs)
        if any(not v for v in vecs): return None
        m = np.zeros((n, n))
        for i in range(n):
            for j in range(i, n):
                cs = _cosine_sim_dicts(vecs[i], vecs[j])
                m[i, j] = cs; m[j, i] = cs
        return m

    attn_mat = vec_matrix(attn_vecs)
    ffn_mat = vec_matrix(ffn_vecs)

    # Successive pair similarities (i, i+1)
    successive = [mat[i, i+1] for i in range(n - 1)]
    # Also compute every-other (i, i+2)
    skip1 = [mat[i, i+2] for i in range(n - 2)]

    print(f"  Successive cos-sim (mean={np.mean(successive):.4f}): {[f'{v:.3f}' for v in successive]}")

    result = {
        "matrix": mat,
        "attn_matrix": attn_mat,
        "ffn_matrix": ffn_mat,
        "successive": successive,
        "skip1": skip1,
        "n_layers": n,
    }
    del model
    return result


def main():
    all_results = {}

    for label, info in MODELS.items():
        r = analyze_model(label, info)
        if r is not None:
            all_results[label] = r

    # 1. Per-model heatmaps
    for label, data in all_results.items():
        stem_name = label.replace("\n", "_").replace(" ", "").replace("/", "").replace("(", "").replace(")", "").replace("+", "p")
        n = data["n_layers"]
        layer_labels = [str(i) for i in range(n)]
        plot_heatmap(data["matrix"],
                     f"Weight Cosine Similarity — {label.replace(chr(10), ' ')}",
                     f"merger_weight_heatmap_{stem_name}",
                     labels=layer_labels)
        if data["attn_matrix"] is not None:
            plot_heatmap(data["attn_matrix"],
                         f"Attn Weight Cosine Sim — {label.replace(chr(10), ' ')}",
                         f"merger_attn_heatmap_{stem_name}",
                         labels=layer_labels)

    # 2. Successive-pair line plot for 12-layer models
    models_12 = {k: v for k, v in all_results.items() if v["n_layers"] == 12}
    if models_12:
        plot_successive_line(models_12,
                             "merger_successive_cosim_12layer",
                             "Weight Cosine Sim of Consecutive Layers (12-layer models)")

    # 3. Successive-pair line plot for 6-layer merged models
    models_6 = {k: v for k, v in all_results.items() if v["n_layers"] == 6}
    if models_6:
        plot_successive_line(models_6,
                             "merger_successive_cosim_6layer",
                             "Weight Cosine Sim of Consecutive Layers (6-layer merged models)")

    # 4. Combined successive plot (all models normalized to [0,1] x-axis)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, data in all_results.items():
        cs = data["successive"]
        n = len(cs)
        x = np.linspace(0, 1, n)
        color = MODELS[label]["color"] if label in MODELS else None
        ls = "--" if data["n_layers"] == 6 else "-"
        ax.plot(x, cs, marker="o", markersize=4, label=label.replace("\n", " "),
                color=color, linewidth=2, linestyle=ls)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Normalized layer position", fontsize=11)
    ax.set_ylabel("Weight cosine sim (i, i+1)", fontsize=11)
    ax.set_title("Consecutive-Layer Weight Similarity — All Merger Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "merger_successive_cosim_all")

    # 5. Avg sim bar chart (all models)
    plot_avg_sim_bar(all_results,
                     "merger_avg_cosim_bar",
                     "Avg Off-Diagonal Weight Cosine Similarity per Model")

    # 6. Grouped bar: full / attn / ffn
    plot_avg_sim_bar_attn_ffn(all_results,
                              "merger_avg_cosim_attn_ffn_bar",
                              "Avg Cosine Similarity: Full Block vs Attn vs FFN")

    # 7. Summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Model':<35} {'N':>3} {'AvgSim':>8} {'SuccMean':>10} {'SuccMax':>9} {'SuccMin':>9} {'AttnAvg':>9} {'FFNAvg':>8}")
    print("-" * 95)
    for label, data in all_results.items():
        mat = data["matrix"]
        n = mat.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avg = float(mat[mask].mean())
        suc = data["successive"]
        attn_avg = float(data["attn_matrix"][mask].mean()) if data["attn_matrix"] is not None else float("nan")
        ffn_avg = float(data["ffn_matrix"][mask].mean()) if data["ffn_matrix"] is not None else float("nan")
        print(f"{label.replace(chr(10),' '):<35} {n:>3} {avg:>8.4f} {np.mean(suc):>10.4f} "
              f"{np.max(suc):>9.4f} {np.min(suc):>9.4f} {attn_avg:>9.4f} {ffn_avg:>8.4f}")

    # 8. Special: merged models — adjacent pair similarity (were these identical after merge?)
    print("\n\n=== MERGED MODEL: ADJACENT PAIR WEIGHT SIM ===")
    for label in ["V42 Naive\nmerge (init)", "V43 Procrustes\nmerge (init)",
                  "V42 Naive\n+5k FT", "V43 Procrustes\n+5k FT"]:
        if label not in all_results: continue
        mat = all_results[label]["matrix"]
        suc = all_results[label]["successive"]
        print(f"{label.replace(chr(10), ' ')}: consecutive sims = {[f'{v:.4f}' for v in suc]}")
        print(f"  Mean={np.mean(suc):.4f}, Max={np.max(suc):.4f}, Min={np.min(suc):.4f}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
