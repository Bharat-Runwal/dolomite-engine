"""plot_scatter_alignment_20260428.py

Scatter plots (WikiPPL vs params, Avg acc vs params, same vs FLOPs) for the
alignment-strategy sweep, extending plot_scatter_v2_20260421.py with:
  - V39/V52/V53 (activation cosreg)
  - V48 (weight cosreg)
  - V57/V59/V63 (single-block recurrent baselines)

Models are grouped by alignment strategy; GPT and EGPT baselines are shown
in the background for reference.

Outputs:
  scatter_ppl_acc_alignment.{pdf,png}   — 2-panel: PPL vs params + acc vs params
  scatter_ppl_flops_alignment.{pdf,png} — 2-panel: PPL vs FLOPs + acc vs FLOPs
"""

import sys, json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_tables import AVG_TASKS_10

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"


# ── Model registry ─────────────────────────────────────────────────────────
# (label, subdir, params_M_non_emb, unique_layers, iters, d, int_size, group)
# FLOPs computed via compute_mflops; params are non-embedding (for x-axis use total)
MODELS = {
    # ── GPT baselines ──
    "V0":    ("V0 GPT 12×1",         "v0_gpt_baseline_d768",              162, 12, 1, 768,  2048, "gpt"),
    "V9":    ("V9 GPT 24×1 d=1024",  "v9_gpt_baseline_d1024_lr1e3",       354, 24, 1, 1024, 2730, "gpt"),
    # ── EGPT baselines ──
    "V1":    ("V1 EGPT 12×1",        "v1_12x1_d768_lr2e3",                176, 12, 1, 768,  2048, "energy"),
    "V1_4m": ("V1 EGPT 24×1",        "v1_400m_d1024_lr7e4",               354, 24, 1, 1024, 2730, "energy"),
    # ── Shared backbone ──
    "V3":    ("V3 shared-bb d=1152",      "v3_shared_d1152",              163, 12, 1, 1152, 3072, "shared"),
    "V4":    ("V4 shared-bb wide d=1152", "v4_shared_wide_d1152_lr2e3",   174, 12, 1, 1152, 6144, "shared"),
    # ── Recurrent (exact weight sharing) ──
    "V2":    ("V2 EGPT 6×2",         "v2_6x2_d768",                       110,  6, 2, 768,  2048, "recurrent"),
    "V57":   ("V57 1×6 d=768",       "v57_egpt_1x6_d768_lr2e3",            85,  1, 6, 768,  2048, "recurrent"),
    "V59":   ("V59 1×12 d=1024",     "v59_egpt_1x12_d1024_lr1e3",         117,  1,12, 1024, 2730, "recurrent"),
    "V63":   ("V63 1×12 d=1408",     "v63_egpt_1x12_d1408_lr2e3",         160,  1,12, 1408, 3840, "recurrent"),
    # ── Cosreg ──
    "V39":   ("V39 act-cosreg λ=0.01", "v39_egpt_cosreg_12x1_d768_lr2e3",176, 12, 1, 768,  2048, "cosreg"),
    "V52":   ("V52 act-cosreg λ=0.1",  "v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3", 176, 12, 1, 768, 2048, "cosreg"),
    "V53":   ("V53 cosreg ramp→1.0",   "v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3",176, 12, 1, 768, 2048, "cosreg"),
    "V48":   ("V48 wt-cosreg λ=0.01",  "v48_egpt_weight_cosreg_12x1_d768_lr2e3", 176, 12, 1, 768, 2048, "cosreg"),
}

GROUP_COLORS = {
    "gpt":       "#2196F3",
    "energy":    "#FF9800",
    "shared":    "#9C27B0",
    "recurrent": "#4CAF50",
    "cosreg":    "#F44336",
}
GROUP_LABELS = {
    "gpt":       "GPT baseline",
    "energy":    "EGPT baseline",
    "shared":    "Shared backbone (V3/V4)",
    "recurrent": "Recurrent (1-block)",
    "cosreg":    "Cosreg (activation/weight)",
}
GROUP_MARKERS = {
    "gpt": "s", "energy": "o", "shared": "D",
    "recurrent": "^", "cosreg": "*",
}


def compute_mflops(unique_layers, iters, d, int_size):
    return 2 * unique_layers * iters * (4 * d * d + 3 * d * int_size) / 1e6


def canon_avg(res):
    return sum(res.get(t, {}).get(m, 0) for t, m in AVG_TASKS_10) / len(AVG_TASKS_10) * 100


def load(subdir):
    files = sorted(glob.glob(str(BASE / subdir / "unsharded" / "harness_results_*.json")))
    if not files:
        return None, None, None
    res = json.load(open(files[-1]))["results"]
    ppl = res.get("wikitext", {}).get("word_perplexity,none")
    avg = canon_avg(res)
    return ppl, avg, res


# ── Collect data ────────────────────────────────────────────────────────────
data = {}
for key, (label, subdir, params, ul, iters, d, int_s, group) in MODELS.items():
    ppl, avg, res = load(subdir)
    if ppl is None:
        print(f"  {key} ({label}): no harness, skipped")
        continue
    mflops = compute_mflops(ul, iters, d, int_s)
    data[key] = dict(label=label, params=params, mflops=mflops,
                     ppl=ppl, avg=avg, group=group)
    print(f"  {key}: params={params}M  FLOPs={mflops:.0f}  PPL={ppl:.1f}  Avg={avg:.1f}%")

print(f"\nLoaded {len(data)} models.")


# ── Plot helpers ─────────────────────────────────────────────────────────────
def scatter_panel(ax, xkey, xlabel, ykey, ylabel, invert_y=True):
    texts = []
    for key, d in data.items():
        x, y = d[xkey], d[ykey]
        grp  = d["group"]
        col  = GROUP_COLORS[grp]
        mkr  = GROUP_MARKERS[grp]
        size = 120 if grp in ("shared", "cosreg") else 80
        alpha = 1.0
        ax.scatter(x, y, c=col, marker=mkr, s=size, alpha=alpha,
                   edgecolors="white", linewidths=0.5, zorder=3)
        texts.append(ax.text(x, y, f" {d['label']}", fontsize=6.5,
                              color=col, alpha=0.85))
    try:
        from adjustText import adjust_text
        adjust_text(texts, ax=ax, arrowprops=dict(arrowstyle="-", color="gray",
                                                   lw=0.4, alpha=0.6),
                    expand_text=(1.05, 1.2), force_text=0.3)
    except ImportError:
        pass
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    if invert_y:
        ax.invert_yaxis()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(alpha=0.3, lw=0.5)


def add_legend(fig):
    handles = [
        mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
        for g in GROUP_COLORS
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, fontsize=8,
               frameon=False, bbox_to_anchor=(0.5, -0.04))


# ── Figure 1: params axis ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Alignment strategies — WikiPPL and Avg accuracy vs.\ total parameters",
             fontsize=11, fontweight="bold")
scatter_panel(ax1, "params", "Total parameters (M)", "ppl",  "WikiText PPL (↓)", invert_y=True)
scatter_panel(ax2, "params", "Total parameters (M)", "avg",  "Avg accuracy % (↑)", invert_y=False)
add_legend(fig)
plt.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ["pdf", "png"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_acc_alignment.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_acc_alignment")

# ── Figure 2: FLOPs axis ───────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
fig.suptitle("Alignment strategies — WikiPPL and Avg accuracy vs.\ FLOPs/token",
             fontsize=11, fontweight="bold")
scatter_panel(ax1, "mflops", "FLOPs/token (M)", "ppl",  "WikiText PPL (↓)", invert_y=True)
scatter_panel(ax2, "mflops", "FLOPs/token (M)", "avg",  "Avg accuracy % (↑)", invert_y=False)
add_legend(fig)
plt.tight_layout(rect=[0, 0.06, 1, 1])
for ext in ["pdf", "png"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_flops_alignment.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_flops_alignment")
print("\nAll done.")
