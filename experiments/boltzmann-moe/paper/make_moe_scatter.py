"""
make_moe_scatter.py  —  MoE comparison scatter plots
Saves to experiments/boltzmann-moe/paper/figs/
Run on CPU, uses Agg backend (no X11 needed).

Models shown: baselines (V0/V1/V1-400M/V58/V9), the H-series MoE variants we
care about (h1_topk, h1_boltz_fullsize, h1_boltz_topk2, h1_gptmoe_boltz, plus
h1_boltz iso-param for the failure-mode reference), and the 580M scale points.
B-series, C-series, and the r128 variant are excluded — they're in the paper
appendix table for reference but clutter the scatter without changing the
story.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# ── Data ──────────────────────────────────────────────────────────────────────
# name, total_params_M, active_params_M, avg_acc, wiki_ppl, gsm8k_flex_avg,
# tokens_B, routing_type, series
#
# gsm8k_flex_avg = mean of (gsm8k flexible-extract, gsm8k_cot flexible-extract).
# Flex filter accepts both "####" and "the answer is" patterns; strict-match
# was unfair to EGPT-style models that emit non-#### formats.
MODELS = [
    ("V9 GPT d=1024",       354, 251, 0.513, 29.8, 2.69, 7.86, "none",         "baseline"),
    # Same architecture as V9 GPT, scaled to 126B tokens (16× longer training).
    ("scale_v9 GPT @ 126B", 354, 251, 0.5413, 26.17, 1.82, 126.0, "none",       "baseline"),
    ("V0 GPT d=768",        162,  85, 0.479, 38.3, 2.08, 7.86, "none",         "baseline"),
    ("V1-400M EGPT d=1024", 354, 251, 0.494, 38.6, 1.93, 7.86, "none",         "baseline"),
    ("V1 EGPT d=768",       143,  66, 0.481, 47.7, 1.97, 7.86, "none",         "baseline"),
    ("V58 EGPT rec 1×24",   113,  10, 0.459, 65.7, 1.93, 7.86, "none",         "baseline"),
    # H-series at d=768, ~145M total params
    ("h1_egpt (no MoE)",    145,  68, 0.4892, 39.55, 2.20, 7.86, "none",         "H-series"),
    ("h1_boltz iso-param",  145,  68, 0.464, 46.1, 2.43, 7.86, "boltzmann",    "H-series"),
    ("h1_topk_egpt_moe",    145,  65, 0.499, 39.8, 2.01, 7.86, "topk",         "H-series"),
    ("h1_boltz_fullsize",   145,  68, 0.501, 36.5, 2.12, 7.86, "boltzmann",    "H-series"),
    ("h1_gptmoe_boltz",     145,  68, 0.486, 35.5, 1.86, 7.86, "switch+boltz", "H-series"),
    # Sparse top-2 of 4: idealized active params reduced ~25% vs soft.
    ("h1_boltz_topk2",      145,  50, 0.4856, 36.37, 1.86, 7.86, "boltzmann-sparse", "H-series"),
    # B4 rerun under current code: deep iso-param BoltzMoE 16x1024, FFN-heavy but
    # competitive after the 1/sqrt(I_e) routing-scale fix.
    ("B4 rerun",            407, 330, 0.4935, 38.04, 2.20, 7.86, "boltzmann",    "B-series"),
    # h2: 6 GPT + 2 distinct EGPT blocks × 6 iters each (18 effective layers).
    # Adds a second unique EGPT block to h1's pattern; lost to h1_boltz_fullsize.
    ("h2_6gpt_2egpt6x",     155,  78, 0.4871, 37.71, 1.74, 7.86, "boltzmann",    "H-series"),
    # 580M Boltzmann MoE (K=8 × I_e=4096, d=1536) — training to 65B.
    ("580M @ 7.3B",         679, 679, 0.5137, 30.4, 2.39, 7.34, "boltzmann",   "H-series"),
    ("580M @ 9.4B",         679, 679, 0.5238, 28.97, 2.12, 9.43, "boltzmann",   "H-series"),
    ("580M @ 15.7B",        679, 679, 0.5366, 26.84, 1.90, 15.73, "boltzmann", "H-series"),
    ("580M @ 39.8B",        679, 679, 0.5593, 22.41, 2.39, 39.83, "boltzmann", "H-series"),
    # scale_h3_boltz: 8gpt+4egpt no-recursion, d=1280, ~620M, BoltzMoE in EGPT.
    ("scale_h3_boltz @ 54.5B", 620, 620, 0.5563, 22.67, 2.39, 54.53, "boltzmann", "H-series"),
]

SHORT_NAMES = {
    "V9 GPT d=1024":       "V9 GPT",
    "scale_v9 GPT @ 126B": "scale_v9 GPT @126B",
    "V1-400M EGPT d=1024": "V1-400M EGPT",
    "V1 EGPT d=768":       "V1 EGPT",
    "V0 GPT d=768":        "V0 GPT-160M",
    "V58 EGPT rec 1×24":   "V58 rec",
    "h1_egpt (no MoE)":    "h1 EGPT (no MoE)",
    "h1_boltz iso-param":  "h1 boltz-iso",
    "h1_topk_egpt_moe":    "h1 topK",
    "h1_boltz_fullsize":   "h1 boltz-full",
    "h1_gptmoe_boltz":     "h1 gpt+boltz",
    "h1_boltz_topk2":      "h1 boltz-top2",
    "B4 rerun":            "B4 rerun",
    "h2_6gpt_2egpt6x":     "h2 (2 EGPT)",
    "580M @ 7.3B":         "580M @ 7.3B",
    "580M @ 9.4B":         "580M @ 9.4B",
    "580M @ 15.7B":        "580M @ 15.7B",
    "580M @ 39.8B":        "580M @ 39.8B",
    "scale_h3_boltz @ 54.5B": "scale_h3_boltz",
}

# ── Style ──────────────────────────────────────────────────────────────────────
COLORS = {
    "baseline":         "#888888",
    "boltzmann":        "#2566c8",
    "boltzmann-sparse": "#0d75c4",
    "topk":             "#e07b20",
    "switch+boltz":     "#8e44ad",
}
SERIES_MARKERS = {"baseline": "o", "H-series": "*", "B-series": "s"}
SERIES_SIZES   = {"baseline": 70,  "H-series": 130, "B-series": 80}


def _get_color(routing, series):
    return COLORS["baseline"] if series == "baseline" else COLORS.get(routing, "#555555")


def _draw_panel(ax, ykey, ylabel, xs):
    """Draw one scatter panel. xs = list of x-values (one per MODELS row)."""
    for row, x in zip(MODELS, xs):
        name, _, _, _, _, _, _, routing, series = row
        y = row[ykey]
        c = _get_color(routing, series)
        ax.scatter(x, y, c=c, marker=SERIES_MARKERS[series], s=SERIES_SIZES[series],
                   edgecolors="white", linewidths=0.6, zorder=3)
        ax.annotate(SHORT_NAMES[name], (x, y),
                    textcoords="offset points", xytext=(5, 3),
                    fontsize=7, color=c, zorder=4)

    # Routing comparison: dashed line h1_topk ↔ h1_boltz_fullsize
    idx_tk  = next(i for i, r in enumerate(MODELS) if r[0] == "h1_topk_egpt_moe")
    idx_bz  = next(i for i, r in enumerate(MODELS) if r[0] == "h1_boltz_fullsize")
    ax.plot([xs[idx_tk], xs[idx_bz]], [MODELS[idx_tk][ykey], MODELS[idx_bz][ykey]],
            color="#555555", linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)

    if ykey == 4:  # WikiPPL — flip so up = better
        ax.invert_yaxis()
        ax.set_ylabel("WikiText PPL (↓ = better)", fontsize=9)
    else:
        ax.set_ylabel(ylabel, fontsize=9)

    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)


def _add_compact_legend(fig):
    handles = [
        mpatches.Patch(color=COLORS["baseline"],         label="Baseline (no MoE)"),
        mpatches.Patch(color=COLORS["boltzmann"],        label="Boltzmann (soft)"),
        mpatches.Patch(color=COLORS["boltzmann-sparse"], label="Boltzmann (sparse top-2)"),
        mpatches.Patch(color=COLORS["topk"],             label="TopK learned-router"),
        mpatches.Patch(color=COLORS["switch+boltz"],     label="Switch+Boltzmann"),
    ]
    # Place below subplots; constrained_layout handles the reserved space.
    fig.legend(handles=handles, loc="outside lower center",
               ncol=5, fontsize=8, frameon=False)


def make_scatter(variant="active"):
    """variant: 'active' (per-token active params) | 'total' (total params)."""
    assert variant in ("active", "total")
    use_active = (variant == "active")
    xlabel = "Active (non-embed) params (M)" if use_active else "Total params (M)"
    xs = [r[2] if use_active else r[1] for r in MODELS]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ylabels = ["WikiText PPL", "Avg zero-shot acc", "GSM8k flex-avg (%)"]
    for ax, ykey, ylabel in zip(axes, [4, 3, 5], ylabels):
        _draw_panel(ax, ykey, ylabel, xs)
        ax.set_xlabel(xlabel, fontsize=9)

    _add_compact_legend(fig)

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    os.makedirs(figs_dir, exist_ok=True)
    outpath = os.path.join(figs_dir, f"moe_scatter_{variant}.pdf")
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


def make_scatter_flops():
    """x-axis = forward-pass FLOPs per token ≈ 2 × active_params (GFLOPs).

    Per-forward-pass (not per-training-run): two snapshots of the same model
    at different token counts collapse onto a single x value. This is a
    model-size / inference-cost axis, distinct from the training-FLOPs
    interpretation (which would be 6·active·tokens).
    """
    # 2 · (active_M · 1e6) FLOPs/token = 2·active_M / 1000 GFLOPs/token.
    xs = [2.0 * r[2] / 1000.0 for r in MODELS]  # GFLOPs / token (forward)

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), constrained_layout=True)
    ylabels = ["WikiText PPL", "Avg zero-shot acc", "GSM8k flex-avg (%)"]
    for ax, ykey, ylabel in zip(axes, [4, 3, 5], ylabels):
        _draw_panel(ax, ykey, ylabel, xs)
        ax.set_xlabel("Forward FLOPs / token (GFLOPs)", fontsize=9)

    # Extra dashed: h1_boltz_fullsize ↔ h1_boltz_topk2 (sparse routing comparison)
    for ax, ykey in zip(axes, [4, 3, 5]):
        idx_bz = next(i for i, r in enumerate(MODELS) if r[0] == "h1_boltz_fullsize")
        idx_sp = next(i for i, r in enumerate(MODELS) if r[0] == "h1_boltz_topk2")
        ax.plot([xs[idx_bz], xs[idx_sp]], [MODELS[idx_bz][ykey], MODELS[idx_sp][ykey]],
                color="#555555", linewidth=1.0, linestyle="--", alpha=0.5, zorder=2)

    _add_compact_legend(fig)

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    outpath = os.path.join(figs_dir, "moe_scatter_flops.pdf")
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    make_scatter("active")
    make_scatter("total")
    make_scatter_flops()
    print("Done.")
