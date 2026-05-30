"""
make_moe_scatter.py  —  MoE comparison scatter plots
Saves to experiments/boltzmann-moe/paper/figs/
Run on CPU, uses Agg backend (no X11 needed).
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────────────
# name, total_params_M, active_params_M, avg_acc, wiki_ppl, gsm8k_pct, tokens_B, routing_type, series
MODELS = [
    ("V9 GPT d=1024",       354, 251, 0.513, 29.8, 2.9,  7.86, "none",         "baseline"),
    ("V1-400M EGPT d=1024", 354, 251, 0.494, 38.6, 1.7,  7.86, "none",         "baseline"),
    ("V1 EGPT d=768",       143,  66, 0.481, 47.7, 1.7,  7.86, "none",         "baseline"),
    ("V58 EGPT rec 1×24",   113,  10, 0.459, 65.7, 1.7,  7.86, "none",         "baseline"),
    ("B1 BoltzMoE (no reg)",407, 330, 0.474, 51.9, 1.4,  7.86, "boltzmann",    "B-series"),
    ("B4 BoltzMoE rep0.1",  407, 330, 0.466, 51.9, 1.7,  7.86, "boltzmann",    "B-series"),
    ("C1 TopK EnergyMoE",   165,  88, 0.474, 47.3, 2.0,  7.86, "topk",         "C-series"),
    ("h1_boltz iso-param",  145,  68, 0.464, 46.1, 2.4,  7.86, "boltzmann",    "H-series"),
    ("h1_topk_egpt_moe",    145,  68, 0.499, 39.8, 2.2,  7.86, "topk",         "H-series"),
    ("h1_topk_r128",        145,  68, 0.484, 39.6, 2.3,  7.86, "topk",         "H-series"),
    ("h1_boltz_fullsize",   145,  68, 0.501, 36.5, 2.0,  7.86, "boltzmann",    "H-series"),
    ("h1_gptmoe_boltz",     145,  68, 0.486, 35.5, 1.8,  7.86, "switch+boltz", "H-series"),
]

# Short display names for labels
SHORT_NAMES = {
    "V9 GPT d=1024":       "V9 GPT",
    "V1-400M EGPT d=1024": "V1-400M EGPT",
    "V1 EGPT d=768":       "V1 EGPT",
    "V58 EGPT rec 1×24":   "V58 rec",
    "B1 BoltzMoE (no reg)":"B1 Boltz",
    "B4 BoltzMoE rep0.1":  "B4 rep0.1",
    "C1 TopK EnergyMoE":   "C1 TopK",
    "h1_boltz iso-param":  "h1 boltz-iso",
    "h1_topk_egpt_moe":    "h1 topK",
    "h1_topk_r128":        "h1 topK+r128",
    "h1_boltz_fullsize":   "h1 boltz-full",
    "h1_gptmoe_boltz":     "h1 gpt+boltz",
}

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline":   "#888888",   # gray
    "boltzmann":  "#2566c8",   # blue
    "topk":       "#e07b20",   # orange
    "switch+boltz": "#8e44ad", # purple
}

SERIES_MARKERS = {
    "baseline": "o",   # circle
    "B-series": "s",   # square
    "C-series": "D",   # diamond
    "H-series": "*",   # star
}

SERIES_SIZES = {
    "baseline": 90,
    "B-series": 80,
    "C-series": 80,
    "H-series": 130,
}

def get_color(routing, series):
    if series == "baseline":
        return COLORS["baseline"]
    return COLORS.get(routing, "#555555")


def make_scatter(use_active_params=False):
    suffix = "active_params" if use_active_params else "total_params"
    xlabel = "Active (non-embedding) params (M)" if use_active_params else "Total params (M)"
    outfile = f"moe_scatter_{suffix}.pdf"

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "MoE variants vs baselines (7.86B tokens, same architecture family)",
        fontsize=11, y=1.01
    )

    ylabels = ["Avg zero-shot acc", "WikiText PPL (↓ better)", "GSM8k (%)"]
    ykeys   = [3, 4, 5]   # indices into MODELS tuple

    for ax, ykey, ylabel in zip(axes, ykeys, ylabels):
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        xs, ys = [], []
        for row in MODELS:
            name, total, active, avg_acc, wiki_ppl, gsm8k, _, routing, series = row
            x = active if use_active_params else total
            y = row[ykey]
            color  = get_color(routing, series)
            marker = SERIES_MARKERS[series]
            size   = SERIES_SIZES[series]

            sc = ax.scatter(x, y, c=color, marker=marker, s=size,
                            edgecolors="white", linewidths=0.6, zorder=3)
            xs.append(x); ys.append(y)

            # Label
            label = SHORT_NAMES[name]
            ax.annotate(label, (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=color, zorder=4)

        # Connecting lines: h1_topk_egpt_moe ↔ h1_boltz_fullsize
        h_topk  = next(r for r in MODELS if r[0] == "h1_topk_egpt_moe")
        h_boltz = next(r for r in MODELS if r[0] == "h1_boltz_fullsize")
        x0 = (h_topk[1]  if not use_active_params else h_topk[2])
        x1 = (h_boltz[1] if not use_active_params else h_boltz[2])
        y0 = h_topk[ykey]; y1 = h_boltz[ykey]
        ax.plot([x0, x1], [y0, y1], color="#555555", linewidth=1.0,
                linestyle="--", alpha=0.6, zorder=2)

        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Legend — routing type colors
    routing_patches = [
        mpatches.Patch(color=COLORS["baseline"],     label="Baseline (no MoE)"),
        mpatches.Patch(color=COLORS["boltzmann"],    label="Boltzmann routing†"),
        mpatches.Patch(color=COLORS["topk"],         label="TopK routing"),
        mpatches.Patch(color=COLORS["switch+boltz"], label="Switch+Boltzmann"),
    ]
    # Legend — series markers
    series_handles = [
        plt.scatter([], [], marker="o", s=80, c="gray", label="Baseline"),
        plt.scatter([], [], marker="s", s=70, c="gray", label="B-series"),
        plt.scatter([], [], marker="D", s=70, c="gray", label="C-series"),
        plt.scatter([], [], marker="*", s=100, c="gray", label="H-series"),
    ]

    axes[1].legend(
        handles=routing_patches + series_handles,
        loc="upper center", bbox_to_anchor=(0.5, -0.15),
        ncol=4, fontsize=7.5, framealpha=0.8
    )

    fig.text(
        0.5, -0.04,
        "† Boltzmann rows: routing weights $p_i = \\mathrm{softmax}(E_i/\\tau)$ derived from energy scores (no learned router).\n"
        "Dashed line connects h1\\_topk\\_egpt\\_moe ↔ h1\\_boltz\\_fullsize (same H-series architecture, different routing).",
        ha="center", fontsize=7, style="italic", color="#444444"
    )

    plt.tight_layout(rect=[0, 0.05, 1, 1])

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    outpath = os.path.join(figs_dir, outfile)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    make_scatter(use_active_params=False)
    make_scatter(use_active_params=True)
    print("Done.")
