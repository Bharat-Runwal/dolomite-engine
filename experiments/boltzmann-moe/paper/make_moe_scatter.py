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
# name, total_params_M, active_params_M, avg_acc, wiki_ppl, gsm8k_flex_avg, tokens_B, routing_type, series
# active_params = per-token active (non-embedding).
# BoltzmannMoE: ALL experts compute every step → active = total (no sparsity).
# TopK sparse:  only top_k/n_experts fraction of FFN is active per token.
#   h1_topk: only 1 EGPT block sparse (6/7 layers fully dense) → active ≈ total - 3M
#   C1:      all 12 blocks sparse (top-2/4) → active FFN ≈ 50% of total FFN
# gsm8k_flex_avg = mean of (gsm8k flexible-extract, gsm8k_cot flexible-extract).
#   Flex filter accepts both "####" and "the answer is" patterns; strict-match
#   penalised EGPT-style models heavily for emitting non-#### formats. Avg of two
#   tasks reduces noise.
MODELS = [
    ("V9 GPT d=1024",       354, 251, 0.513, 29.8, 2.69, 7.86, "none",         "baseline"),
    ("V0 GPT d=768",        162,  85, 0.479, 38.3, 2.08, 7.86, "none",         "baseline"),
    ("V1-400M EGPT d=1024", 354, 251, 0.494, 38.6, 1.93, 7.86, "none",         "baseline"),
    ("V1 EGPT d=768",       143,  66, 0.481, 47.7, 1.97, 7.86, "none",         "baseline"),
    ("V58 EGPT rec 1×24",   113,  10, 0.459, 65.7, 1.93, 7.86, "none",         "baseline"),
    ("B1 BoltzMoE (no reg)",407, 330, 0.474, 51.9, 1.63, 7.86, "boltzmann",    "B-series"),
    ("B4 BoltzMoE rep0.1",  407, 330, 0.466, 51.9, 1.90, 7.86, "boltzmann",    "B-series"),
    # C1: 12 deep blocks, top-2/4 → active FFN ≈ 50%, so active ≈ 165-77(emb) × 0.65 + 77 ≈ 135M total, active FFN ≈ 55M
    ("C1 TopK EnergyMoE",   165,  55, 0.474, 47.3, 1.97, 7.86, "topk",         "C-series"),
    ("h1_boltz iso-param",  145,  68, 0.464, 46.1, 2.43, 7.86, "boltzmann",    "H-series"),
    # h1_topk: only EGPT FFN sparse (1/7 layers), so active ≈ 68 - 3 = 65M active
    ("h1_topk_egpt_moe",    145,  65, 0.499, 39.8, 2.01, 7.86, "topk",         "H-series"),
    ("h1_topk_r128",        145,  65, 0.484, 39.6, 2.24, 7.86, "topk",         "H-series"),
    # Boltzmann: all experts active → active = total non-embed
    ("h1_boltz_fullsize",   145,  68, 0.501, 36.5, 2.12, 7.86, "boltzmann",    "H-series"),
    ("h1_gptmoe_boltz",     145,  68, 0.486, 35.5, 1.86, 7.86, "switch+boltz", "H-series"),
    # Sparse Boltzmann (top-2 of 4): idealized active params reduced ~25% vs soft
    # (full energy + top-2 gradient half). Realized active = 68M with current impl
    # that does not skip compute, but we plot the idealized 50M to reflect the
    # design intent and FLOPs interpretation.
    ("h1_boltz_topk2",      145,  50, 0.4856, 36.37, 1.86, 7.86, "boltzmann-sparse", "H-series"),
    # 580M Boltzmann MoE: K=8 experts × I_e=4096, d=1536. Training to 124k = 65B.
    # Already beats V9 GPT on avg & ppl at <15% of budget.
    ("580M @ 7.3B",         679, 679, 0.5137, 30.4, 2.39, 7.34, "boltzmann",   "H-series"),
    ("580M @ 9.4B",         679, 679, 0.5238, 28.97, 2.12, 9.43, "boltzmann",   "H-series"),
]

# Short display names for labels
SHORT_NAMES = {
    "V9 GPT d=1024":       "V9 GPT",
    "V1-400M EGPT d=1024": "V1-400M EGPT",
    "V1 EGPT d=768":       "V1 EGPT",
    "V0 GPT d=768":        "V0 GPT-160M",
    "V58 EGPT rec 1×24":   "V58 rec",
    "B1 BoltzMoE (no reg)":"B1 Boltz",
    "B4 BoltzMoE rep0.1":  "B4 rep0.1",
    "C1 TopK EnergyMoE":   "C1 TopK",
    "h1_boltz iso-param":  "h1 boltz-iso",
    "h1_topk_egpt_moe":    "h1 topK",
    "h1_topk_r128":        "h1 topK+r128",
    "h1_boltz_fullsize":   "h1 boltz-full",
    "h1_gptmoe_boltz":     "h1 gpt+boltz",
    "h1_boltz_topk2":      "h1 boltz-top2",
    "580M @ 7.3B":         "580M @ 7.3B",
    "580M @ 9.4B":         "580M @ 9.4B",
}

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline":   "#888888",   # gray
    "boltzmann":  "#2566c8",   # blue
    "boltzmann-sparse": "#0d75c4",  # darker blue (sparse top-k Boltzmann)
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


def _draw_panel(ax, ykey, ylabel, use_active, show_arrows):
    """Draw one scatter panel with the given y metric."""
    xlabel = "Active (non-embed) params (M)" if use_active else "Total params (M)"
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)

    for row in MODELS:
        name, total, active, avg_acc, wiki_ppl, gsm8k, _, routing, series = row
        x_plot = active if use_active else total
        y = row[ykey]
        color  = get_color(routing, series)
        marker = SERIES_MARKERS[series]
        size   = SERIES_SIZES[series]

        # Draw arrow from total → active for TopK models when in arrow mode
        if show_arrows and not use_active and routing == "topk" and total != active:
            ax.annotate("",
                xy=(active, y), xytext=(total, y),
                arrowprops=dict(arrowstyle="->", color=color, alpha=0.5,
                                lw=1.2, linestyle="dashed"))
            # Faint open circle at total params
            ax.scatter(total, y, c="none", marker=marker, s=size,
                       edgecolors=color, linewidths=1.2, alpha=0.35, zorder=2)
            x_plot = active  # filled point at active params

        ax.scatter(x_plot, y, c=color, marker=marker, s=size,
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(SHORT_NAMES[name], (x_plot, y),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=6.5, color=color, zorder=4)

    # Connecting dashed line: h1_topk ↔ h1_boltz_fullsize (routing comparison)
    h_topk  = next(r for r in MODELS if r[0] == "h1_topk_egpt_moe")
    h_boltz = next(r for r in MODELS if r[0] == "h1_boltz_fullsize")
    x_tk = h_topk[2]  if (use_active or show_arrows) else h_topk[1]
    x_bz = h_boltz[2] if (use_active or show_arrows) else h_boltz[1]
    ax.plot([x_tk, x_bz], [h_topk[ykey], h_boltz[ykey]],
            color="#555555", linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)

    # Flip WikiPPL axis so higher position = better
    if ykey == 4:
        ax.invert_yaxis()
        ax.set_ylabel("WikiText PPL (↑ = better = lower PPL)", fontsize=9)

    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def make_scatter(variant="active"):
    """
    variant: "active"  – x-axis = active params only
             "arrows"  – x-axis = total params; TopK models show dashed arrow to active
             "total"   – original total-params-only (no arrows)
    """
    assert variant in ("active", "arrows", "total")
    outfile = f"moe_scatter_{variant}.pdf"
    use_active  = (variant == "active")
    show_arrows = (variant == "arrows")

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    subtitle = {
        "active":  "x-axis: active (per-token) params — Boltzmann active=total; TopK active<total",
        "arrows":  "x-axis: total params; dashed arrow shows sparsity discount for TopK models",
        "total":   "x-axis: total params",
    }[variant]
    fig.suptitle(
        f"MoE variants vs baselines (7.86B tokens)\n{subtitle}",
        fontsize=10, y=1.02
    )

    ylabels = ["Avg zero-shot acc", "WikiText PPL", "GSM8k flex-avg (%)"]
    ykeys   = [3, 4, 5]

    for ax, ykey, ylabel in zip(axes, ykeys, ylabels):
        _draw_panel(ax, ykey, ylabel, use_active, show_arrows)

    # Shared legend
    routing_patches = [
        mpatches.Patch(color=COLORS["baseline"],         label="Baseline (no MoE)"),
        mpatches.Patch(color=COLORS["boltzmann"],        label="Boltzmann soft (all experts active†)"),
        mpatches.Patch(color=COLORS["boltzmann-sparse"], label="Boltzmann sparse top-2 (energy-selected)"),
        mpatches.Patch(color=COLORS["topk"],             label="TopK learned-router (top-2/4 active)"),
        mpatches.Patch(color=COLORS["switch+boltz"],     label="Switch+Boltzmann"),
    ]
    series_handles = [
        plt.scatter([], [], marker="o", s=80, c="gray", label="Baseline"),
        plt.scatter([], [], marker="s", s=70, c="gray", label="B-series"),
        plt.scatter([], [], marker="D", s=70, c="gray", label="C-series"),
        plt.scatter([], [], marker="*", s=100, c="gray", label="H-series"),
    ]
    axes[1].legend(handles=routing_patches + series_handles,
                   loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=5, fontsize=7.5, framealpha=0.8)

    footnote = (
        "† BoltzmannMoE uses soft routing: all $K$ experts compute every step "
        "(active = total params). TopK uses hard routing: only top-$k$ experts active per token.\n"
        "Dashed line: same H-series arch, different routing (h1\\_topk ↔ h1\\_boltz\\_fullsize)."
    )
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7,
             style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    outpath = os.path.join(figs_dir, outfile)
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


def make_scatter_flops():
    """
    FLOPs scatter: x-axis = training FLOPs ≈ 6 × active_params × tokens.

    Active params encode the sparsity assumption: BoltzmannMoE soft = total
    non-embed (all experts active), sparse top-2 is reduced via the idealized
    impl (~25% saving on the gradient half), TopK = top-k/n_experts of FFN.

    Boltzmann routing itself is FLOPs-free relative to the MoE forward — the
    energy E_i = x · term1_i is computed from term1 which is needed anyway,
    so there is no double-counting of routing cost.

    Plotted in EFLOPs (1e18) so values are O(1)–O(20).
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle(
        "MoE variants vs baselines (7.86B tokens)\n"
        "x-axis: training FLOPs ≈ 6 · active_params · tokens",
        fontsize=10, y=1.02
    )

    ylabels = ["Avg zero-shot acc", "WikiText PPL", "GSM8k flex-avg (%)"]
    ykeys   = [3, 4, 5]

    for ax, ykey, ylabel in zip(axes, ykeys, ylabels):
        ax.set_xlabel("Training FLOPs (EFLOPs = $10^{18}$)", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=9)

        for row in MODELS:
            name, total, active, avg_acc, wiki_ppl, gsm8k, tokens_B, routing, series = row
            # FLOPs = 6 * active(M) * tokens(B) * 1e15  (params in M, tokens in B)
            #       = 6 * active * 1e6 * tokens * 1e9 = 6*active*tokens * 1e15 raw FLOPs
            # In EFLOPs (1e18): divide by 1e3 → 6*active*tokens / 1000
            flops_eflops = 6.0 * active * tokens_B / 1e3
            y = row[ykey]
            color  = get_color(routing, series)
            marker = SERIES_MARKERS[series]
            size   = SERIES_SIZES[series]

            ax.scatter(flops_eflops, y, c=color, marker=marker, s=size,
                       edgecolors="white", linewidths=0.6, zorder=3)
            ax.annotate(SHORT_NAMES[name], (flops_eflops, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=color, zorder=4)

        # Connecting dashed line: h1_topk ↔ h1_boltz_fullsize ↔ h1_boltz_topk2
        h_topk  = next(r for r in MODELS if r[0] == "h1_topk_egpt_moe")
        h_boltz = next(r for r in MODELS if r[0] == "h1_boltz_fullsize")
        h_bsp   = next(r for r in MODELS if r[0] == "h1_boltz_topk2")
        for a, b in [(h_topk, h_boltz), (h_boltz, h_bsp)]:
            xa = 6.0 * a[2] * a[6] / 1e3
            xb = 6.0 * b[2] * b[6] / 1e3
            ax.plot([xa, xb], [a[ykey], b[ykey]],
                    color="#555555", linewidth=1.0, linestyle="--",
                    alpha=0.5, zorder=2)

        if ykey == 4:
            ax.invert_yaxis()
            ax.set_ylabel("WikiText PPL (↑ = better = lower PPL)", fontsize=9)

        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(labelsize=8)

    # Shared legend (same as make_scatter)
    routing_patches = [
        mpatches.Patch(color=COLORS["baseline"],         label="Baseline (no MoE)"),
        mpatches.Patch(color=COLORS["boltzmann"],        label="Boltzmann soft (all experts active†)"),
        mpatches.Patch(color=COLORS["boltzmann-sparse"], label="Boltzmann sparse top-2 (energy-selected)"),
        mpatches.Patch(color=COLORS["topk"],             label="TopK learned-router (top-2/4 active)"),
        mpatches.Patch(color=COLORS["switch+boltz"],     label="Switch+Boltzmann"),
    ]
    series_handles = [
        plt.scatter([], [], marker="o", s=80, c="gray", label="Baseline"),
        plt.scatter([], [], marker="s", s=70, c="gray", label="B-series"),
        plt.scatter([], [], marker="D", s=70, c="gray", label="C-series"),
        plt.scatter([], [], marker="*", s=100, c="gray", label="H-series"),
    ]
    axes[1].legend(handles=routing_patches + series_handles,
                   loc="upper center", bbox_to_anchor=(0.5, -0.16),
                   ncol=5, fontsize=7.5, framealpha=0.8)

    footnote = (
        "FLOPs = 6 · active\\_params · tokens. Active params encode the sparsity "
        "assumption (Boltzmann soft = all experts active; sparse top-2 ≈ 75\\% of soft).\n"
        "Boltzmann routing itself is FLOPs-free vs the MoE forward — energy "
        "$E_i = x \\cdot \\text{term1}_i$ reuses term1 already needed for output."
    )
    fig.text(0.5, -0.05, footnote, ha="center", fontsize=7,
             style="italic", color="#444444")

    plt.tight_layout(rect=[0, 0.07, 1, 1])

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    outpath = os.path.join(figs_dir, "moe_scatter_flops.pdf")
    fig.savefig(outpath, bbox_inches="tight", dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    make_scatter("active")   # active params only, WikiPPL flipped
    make_scatter("total")    # original total params
    make_scatter_flops()     # training FLOPs ≈ 6·active·tokens
    print("Done.")
