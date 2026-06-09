"""
make_moe_scatter.py  —  MoE comparison scatter plots
Saves to experiments/boltzmann-moe/paper/figs/
Run on CPU, uses Agg backend (no X11 needed).

Models shown: baselines (V0/V1/V1-400M/V58/V9), the H-series MoE variants we
care about (h1_topk, h1_boltz_fullsize, h1_boltz_topk2, h1_gptmoe_boltz, plus
h1_boltz iso-param for the failure-mode reference), and the 680M scale points.
B-series, C-series, and the r128 variant are excluded — they're in the paper
appendix table for reference but clutter the scatter without changing the
story.
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
try:
    from adjustText import adjust_text
    _HAS_ADJUST_TEXT = True
except ImportError:
    _HAS_ADJUST_TEXT = False

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
    ("scale_v9 GPT @ 100.7B", 354, 251, 0.5413, 26.17, 1.82, 100.7, "none",     "baseline"),
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
    # 680M Boltzmann MoE (K=8 × I_e=4096, d=1536) — training to 65B.
    ("680M @ 7.3B",         679, 679, 0.5137, 30.4, 2.39, 7.34, "boltzmann",   "H-series"),
    ("680M @ 9.4B",         679, 679, 0.5238, 28.97, 2.12, 9.43, "boltzmann",   "H-series"),
    ("680M @ 15.7B",        679, 679, 0.5366, 26.84, 1.90, 15.73, "boltzmann", "H-series"),
    ("680M @ 39.8B",        679, 679, 0.5593, 22.41, 2.39, 39.83, "boltzmann", "H-series"),
    ("680M @ 53.5B",        679, 679, 0.5801, 20.23, 2.31, 53.48, "boltzmann", "H-series"),
    ("680M @ 57.7B",        679, 679, 0.5759, 19.70, 1.86, 57.67, "boltzmann", "H-series"),
    ("680M @ 61.9B",        679, 679, 0.5781, 19.48, 2.12, 61.87, "boltzmann", "H-series"),
    ("680M @ 65.0B (FINAL)", 679, 679, 0.5847, 19.33, 2.08, 65.01, "boltzmann", "H-series"),
    # scale_h3_boltz: 8gpt+4egpt no-recursion, d=1280, ~620M, BoltzMoE in EGPT.
    ("scale_h3_boltz @ 54.5B", 620, 620, 0.5563, 22.67, 2.39, 54.53, "boltzmann", "H-series"),
    ("scale_h3_boltz @ 62.9B", 620, 620, 0.5694, 21.89, 1.74, 62.91, "boltzmann", "H-series"),
    ("scale_h3_boltz @ 65.0B FINAL", 620, 620, 0.5754, 21.82, 2.08, 65.01, "boltzmann", "H-series"),
    # Pure-GPT + MoE comparison set (d=1280, in-progress). Numbers (total /
    # active M) are model.numel() / model_wrapper.calculate_num_parameters().
    # Switch top-1 of 4 routes ~78% active (8gpt+4sw); the 12-layer K=4 MoEs
    # report ~68% (I=2048) and ~61% (I=4096) of total as "active".
    ("8gpt+4sw @ 12.6B",    585, 459, 0.5081, 30.48, 2.46, 12.58, "switch+boltz", "H-series"),
    ("8gpt+4sw @ 31.5B",    585, 459, 0.5474, 26.23, 2.50, 31.46, "switch+boltz", "H-series"),
    ("8gpt+4sw @ 56.6B",    585, 459, 0.5725, 21.49, 2.01, 56.62, "switch+boltz", "H-series"),
    ("8gpt+4sw @ 65.0B FINAL", 585, 459, 0.5797, 20.90, 1.74, 65.01, "switch+boltz", "H-series"),
    ("12moe-I2k @ 13.6B",   585, 396, 0.5357, 30.68, 2.35, 13.63, "boltzmann",    "H-series"),
    ("12moe-I2k @ 34.1B",   585, 396, 0.5401, 26.39, 2.35, 34.08, "boltzmann",    "H-series"),
    ("12moe-I2k @ 57.7B",   585, 396, 0.5679, 22.00, 2.05, 57.67, "boltzmann",    "H-series"),
    ("12moe-I2k @ 65.0B FINAL", 585, 396, 0.5612, 21.64, 2.39, 65.01, "boltzmann", "H-series"),
    ("12moe-I4k @ 7.3B",    962, 585, 0.5299, 31.30, 2.01,  7.34, "boltzmann",    "H-series"),
    ("12moe-I4k @ 23.6B",   962, 585, 0.5436, 26.35, 2.24, 23.59, "boltzmann",    "H-series"),
    ("12moe-I4k @ 38.3B",   962, 585, 0.5572, 23.66, 2.58, 38.27, "boltzmann",    "H-series"),
    ("12moe-I4k @ 50.3B",   962, 585, 0.5732, 21.27, 2.50, 50.33, "boltzmann",    "H-series"),
    ("12moe-I4k @ 65.0B FINAL", 962, 585, 0.5802, 19.73, 1.67, 65.01, "boltzmann", "H-series"),
    # Matched-structure baseline: same 11+1×6 layout as 680M Boltz, but uses
    # softmax_attention + Switch-MoE in the recurrent block instead of
    # energy_attention + BoltzmannMoE. Tests whether the 680M Boltz advantage
    # comes from the structural template or from energy-attn + Boltz routing.
    ("gptswitchmoe-680M @ 15.2B", 730, 730, 0.5208, 27.73, 1.93, 15.21, "switch+boltz", "H-series"),
    ("gptswitchmoe-680M @ 25.2B", 730, 730, 0.5417, 25.66, 1.86, 25.17, "switch+boltz", "H-series"),
    ("gptswitchmoe-680M @ 42.5B", 730, 730, 0.5495, 22.52, 1.90, 42.47, "switch+boltz", "H-series"),
    ("gptswitchmoe-680M @ 46.1B", 730, 730, 0.5602, 21.90, 2.69, 46.13, "switch+boltz", "H-series"),
]

SHORT_NAMES = {
    "V9 GPT d=1024":       "V9 GPT",
    "scale_v9 GPT @ 100.7B": "scale_v9 GPT @100B",
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
    "680M @ 7.3B":         "680M @ 7.3B",
    "680M @ 9.4B":         "680M @ 9.4B",
    "680M @ 15.7B":        "680M @ 15.7B",
    "680M @ 39.8B":        "680M @ 39.8B",
    "680M @ 53.5B":        "680M @ 53.5B",
    "680M @ 57.7B":        "680M @ 57.7B",
    "680M @ 61.9B":        "680M @ 61.9B",
    "680M @ 65.0B (FINAL)": "680M @ 65B (final)",
    "scale_h3_boltz @ 54.5B": "scale_h3_boltz @54B",
    "scale_h3_boltz @ 62.9B": "scale_h3_boltz @63B",
    "scale_h3_boltz @ 65.0B FINAL": "scale_h3_boltz @65B (final)",
    "8gpt+4sw @ 12.6B":    "8gpt+4sw @13B",
    "8gpt+4sw @ 31.5B":    "8gpt+4sw @31B",
    "8gpt+4sw @ 56.6B":    "8gpt+4sw @57B",
    "8gpt+4sw @ 65.0B FINAL": "8gpt+4sw @65B (final)",
    "12moe-I2k @ 13.6B":   "12moe-I2k @14B",
    "12moe-I2k @ 34.1B":   "12moe-I2k @34B",
    "12moe-I2k @ 57.7B":   "12moe-I2k @58B",
    "12moe-I2k @ 65.0B FINAL": "12moe-I2k @65B (final)",
    "12moe-I4k @ 7.3B":    "12moe-I4k @7B",
    "12moe-I4k @ 23.6B":   "12moe-I4k @24B",
    "12moe-I4k @ 38.3B":   "12moe-I4k @38B",
    "12moe-I4k @ 50.3B":   "12moe-I4k @50B",
    "12moe-I4k @ 65.0B FINAL": "12moe-I4k @65B (final)",
    "gptswitchmoe-680M @ 15.2B": "gptswitch-680M @15B",
    "gptswitchmoe-680M @ 25.2B": "gptswitch-680M @25B",
    "gptswitchmoe-680M @ 42.5B": "gptswitch-680M @42B",
    "gptswitchmoe-680M @ 46.1B": "gptswitch-680M @46B",
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


def _wrap_label(s: str, max_len: int = 14) -> str:
    """Break long labels onto two lines at a sensible separator (space, '+', '@')."""
    if len(s) <= max_len:
        return s
    # Prefer to split at " @", " +", or last space before max_len
    for sep in (" @ ", " + ", " "):
        idx = s.rfind(sep, 0, max_len + 1)
        if idx > 0:
            return s[:idx].rstrip() + "\n" + s[idx + len(sep) - 1:].lstrip(" ")
    return s


def _draw_panel(ax, ykey, ylabel, xs):
    """Draw one scatter panel. xs = list of x-values (one per MODELS row)."""
    texts = []
    for row, x in zip(MODELS, xs):
        name, _, _, _, _, _, _, routing, series = row
        y = row[ykey]
        c = _get_color(routing, series)
        ax.scatter(x, y, c=c, marker=SERIES_MARKERS[series], s=SERIES_SIZES[series],
                   edgecolors="white", linewidths=0.6, zorder=3)
        label = _wrap_label(SHORT_NAMES[name])
        if _HAS_ADJUST_TEXT:
            # adjust_text needs Text objects (not Annotation), placed at the data point.
            t = ax.text(x, y, label, fontsize=6.5, color=c, zorder=4,
                        ha="left", va="bottom")
            texts.append(t)
        else:
            ax.annotate(label, (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=6.5, color=c, zorder=4)

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

    # Apply repulsion. Must be after axes limits are set, so the algorithm
    # knows which directions are "free."
    if _HAS_ADJUST_TEXT and texts:
        adjust_text(
            texts,
            ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, alpha=0.6),
            expand_text=(1.05, 1.15),
            expand_points=(1.2, 1.4),
            force_text=(0.4, 0.6),
            force_points=(0.3, 0.4),
            only_move={"text": "xy", "points": "xy"},
        )


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


# ── Simplified scatter (final-only, 4-class color scheme) ─────────────────────
# One canonical point per model family per scale. Used for the main-talk slide.
# Family colors: Baseline GPT = red, EGPT no-MoE = blue, Boltz-MoE = orange,
# GPT-MoE = green. Marker size scales with total params.
SIMPLE_FAMILIES = {
    "baseline_gpt": ("#d62728", "Baseline GPT"),
    "egpt_no_moe":  ("#1f77b4", "EGPT hybrid (no MoE)"),
    "boltz_moe":    ("#ff7f0e", "Boltz-MoE (this work)"),
    "gpt_moe":      ("#2ca02c", "GPT + standard MoE"),
}

# (name, total_M, active_M, avg, ppl, gsm8k_flex_avg, tokens_B, family)
# Selection rule: latest/canonical evaluated checkpoint per (arch, scale).
# Names kept short — token counts are reported in tables, not labels.
MODELS_SIMPLE = [
    # Baseline GPT (red) — 160M, 354M, 354M-extended.
    ("V0 GPT",                  162,  85, 0.479,  38.30, 2.08,   7.86, "baseline_gpt"),
    ("V9 GPT",                  354, 251, 0.513,  29.80, 2.69,   7.86, "baseline_gpt"),
    ("scale_v9 GPT",            354, 251, 0.5413, 26.17, 1.82, 100.70, "baseline_gpt"),

    # EGPT non-MoE hybrids (blue) — 145M, 354M, 620M.
    ("V1 EGPT",                 143,  66, 0.481,  47.70, 1.97,   7.86, "egpt_no_moe"),
    ("h1 EGPT",                 145,  68, 0.4892, 39.55, 2.20,   7.86, "egpt_no_moe"),
    ("V1-400M EGPT",            354, 251, 0.494,  38.60, 1.93,   7.86, "egpt_no_moe"),
    ("scale_h3 EGPT",           620, 620, 0.542,  25.00, 2.50,  63.00, "egpt_no_moe"),
    ("scale_r3 EGPT-rec",       620, 620, 0.546,  24.20, 3.26,  63.00, "egpt_no_moe"),

    # Boltz-MoE (orange) — 145M, 620M, 679M.
    # NB: B-series points (407M, 12 deep EGPT-MoE blocks) excluded from the
    # simplified plot — those land at avg ~49 / PPL ~38 due to a 21:1 FFN:Attn
    # imbalance in the all-MoE arch, not the Boltz routing itself. The
    # canonical Boltz-MoE line is the hybrid (GPT prefix + late MoE) at
    # h1 145M → scale_h3 620M → 680M 679M.
    ("h1 Boltz-MoE",            145,  68, 0.501,  36.50, 2.12,   7.86, "boltz_moe"),
    ("scale_h3 Boltz-MoE",      620, 620, 0.5754, 21.82, 2.08,  65.01, "boltz_moe"),
    ("680M Boltz-MoE",          679, 679, 0.5847, 19.33, 2.08,  65.01, "boltz_moe"),

    # GPT + standard MoE (green) — 145M, 585M, 585M, 962M.
    ("h1 GPT+Switch-MoE",       145,  68, 0.486,  35.50, 1.86,   7.86, "gpt_moe"),
    ("8gpt+4switch",            585, 459, 0.5797, 20.90, 1.74,  65.01, "gpt_moe"),
    ("12moe Boltz I=2k",        585, 396, 0.5612, 21.64, 2.39,  65.01, "gpt_moe"),
    ("12moe Boltz I=4k",        962, 585, 0.5802, 19.73, 1.67,  65.01, "gpt_moe"),
]


def _draw_simple_panel(ax, ykey, ylabel, xs):
    texts = []
    for row, x in zip(MODELS_SIMPLE, xs):
        name = row[0]; tokens_B = row[6]; family = row[7]
        y = row[ykey]
        color = SIMPLE_FAMILIES[family][0]
        # Size scales with training tokens (linear): 8B → ~50; 65B → ~225;
        # 100B → ~330. Highlights long-trained models like scale_v9 @ 100B
        # and the 65B-token MoE finals against the 7.86B short-budget runs.
        size = 30 + 3.0 * tokens_B
        ax.scatter(x, y, c=color, marker="o", s=size,
                   edgecolors="white", linewidths=0.8, alpha=0.9, zorder=3)
        if _HAS_ADJUST_TEXT:
            t = ax.text(x, y, name, fontsize=7, color=color, zorder=4,
                        ha="center", va="center")
            texts.append(t)
        else:
            ax.annotate(name, (x, y),
                        textcoords="offset points", xytext=(5, 3),
                        fontsize=7, color=color, zorder=4)

    if ykey == 4:
        ax.invert_yaxis()
        ax.set_ylabel("WikiText PPL (↓ = better)", fontsize=9)
    else:
        ax.set_ylabel(ylabel, fontsize=9)

    ax.grid(True, alpha=0.25, linewidth=0.5)
    ax.tick_params(labelsize=8)

    if _HAS_ADJUST_TEXT and texts:
        # Run after axis limits are settled. Stronger repulsion between texts
        # to keep the cluster on the right side from collapsing onto each
        # other; modest repulsion from the points so labels stay close to
        # their dots when there's room.
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#888888", lw=0.4, alpha=0.5),
            expand_text=(1.20, 1.30),
            expand_points=(1.30, 1.30),
            force_text=(0.7, 1.0),
            force_points=(0.4, 0.5),
            only_move={"text": "xy", "static": "xy", "explode": "xy", "pull": "xy"},
        )


def _add_simple_legend(fig):
    family_handles = [mpatches.Patch(color=c, label=lbl)
                      for c, lbl in SIMPLE_FAMILIES.values()]
    # Size legend: three reference dots for token count.
    size_handles = []
    for tok in (8, 65, 100):
        s = 30 + 3.0 * tok
        size_handles.append(plt.scatter([], [], s=s, c="#888888",
                                         edgecolors="white", linewidths=0.8,
                                         label=f"{tok} B tokens"))
    fig.legend(handles=family_handles + size_handles,
               loc="outside lower center",
               ncol=7, fontsize=9, frameon=False,
               handletextpad=0.4, columnspacing=1.2)


def make_scatter_simple(variant="total"):
    """Simplified scatter: final/canonical points only, 4-color family scheme.

    Param axes (total/active) span ~1 decade (~140M-960M) so we keep linear x
    — log-scale concentrates the labels in a tight high-x band that adjustText
    can't separate cleanly. FLOPs span ~3 decades, so we keep log-x there.
    """
    assert variant in ("active", "total", "flops")
    if variant == "total":
        xlabel = "Total params (M)"
        xs = [r[1] for r in MODELS_SIMPLE]
        log_x = False
    elif variant == "active":
        xlabel = "Active (non-embed) params (M)"
        xs = [r[2] for r in MODELS_SIMPLE]
        log_x = False
    else:  # flops — training FLOPs ≈ 6 · active · tokens, in EFLOPs (10^18)
        xlabel = "Training FLOPs (EFLOPs)"
        xs = [6.0 * r[2] * 1e6 * r[6] * 1e9 / 1e18 for r in MODELS_SIMPLE]
        log_x = True

    fig, axes = plt.subplots(1, 3, figsize=(13, 4.0), constrained_layout=True)
    ylabels = ["WikiText PPL", "Avg zero-shot acc", "GSM8k flex-avg (%)"]
    for ax, ykey, ylabel in zip(axes, [4, 3, 5], ylabels):
        if log_x:
            ax.set_xscale("log")
        # Add 5% horizontal padding so labels at the right edge have room.
        if not log_x:
            xmin, xmax = min(xs), max(xs)
            pad = (xmax - xmin) * 0.10
            ax.set_xlim(xmin - pad, xmax + pad)
        _draw_simple_panel(ax, ykey, ylabel, xs)
        ax.set_xlabel(xlabel, fontsize=9)

    _add_simple_legend(fig)

    figs_dir = os.path.join(os.path.dirname(__file__), "figs")
    outpath = os.path.join(figs_dir, f"moe_scatter_simple_{variant}.pdf")
    fig.savefig(outpath, dpi=150)
    print(f"Saved: {outpath}")
    plt.close(fig)


if __name__ == "__main__":
    make_scatter("active")
    make_scatter("total")
    make_scatter_flops()
    make_scatter_simple("total")
    make_scatter_simple("active")
    make_scatter_simple("flops")
    print("Done.")
