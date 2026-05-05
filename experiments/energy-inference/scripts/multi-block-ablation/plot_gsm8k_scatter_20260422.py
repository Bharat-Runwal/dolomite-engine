"""
GSM8K scatter plots: score vs params and vs FLOPs (log-x, v2 style).
Saves scatter_gsm8k_vs_params.{pdf,png} and scatter_gsm8k_vs_flops.{pdf,png}
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"

MODELS = {
    "V0":      ("V0",    "v0_gpt_baseline_d768",                       143.1, 12, 1, 768,  2048, "gpt"),
    "V1":      ("V1",    "v1_12x1_d768_lr2e3",                         143.1, 12, 1, 768,  2048, "energy"),
    "V2":      ("V2",    "v2_6x2_d768",                                110.1,  6, 2, 768,  2048, "energy"),
    "V3":      ("V3",    "v3_shared_d1152",                            157.2, 12, 1, 1152, 3072, "energy"),
    "V4":      ("V4",    "v4_shared_wide_d1152_lr2e3",                 164.3, 12, 1, 1152, 6144, "energy"),
    "V5":      ("V5",    "v5_12x1_d768_attn_only_energy_lr2e3",        136.1, 12, 1, 768,  2048, "energy"),
    "V9":      ("V9",    "v9_gpt_baseline_d1024_lr1e3",                354.5, 24, 1, 1024, 2048, "gpt"),
    "V10":     ("V10",   "v10_mixed_12x1_d768_lr2e3",                  144.3, 12, 1, 768,  1536, "mixed"),
    "V11":     ("V11",   "v11_mixed_6x2_d768_lr2e3",                   117.8,  6, 2, 768,  2048, "mixed"),
    "V12":     ("V12",   "v12_gpt_6x2_d768_lr2e3",                     119.5,  6, 2, 768,  2048, "gpt"),
    "V13":     ("V13",   "v13_mixed_8e4g_12x1_d768_lr2e3",             143.1, 12, 1, 768,  1536, "mixed"),
    "V14":     ("V14",   "v14_mixed_10e2g_12x1_d768_lr2e3",            142.0, 12, 1, 768,  1536, "mixed"),
    "V15":     ("V15",   "v15_energy_grad_mixed_12x1_d768_lr2e3",       144.3, 12, 1, 768,  1536, "egrad"),
    "V16":     ("V16",   "v16_mixed_energy_descent_12x1_d768_lr2e3",    144.3, 12, 1, 768,  1536, "edesc"),
    "V17":     ("V17",   "v17_energy_grad_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "egrad"),
    "V18":     ("V18",   "v18_energy_desc_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "edesc"),
    "V1_400m": ("V1_400m","v1_400m_d1024_lr7e4",                       354.4, 24, 1, 1024, 3072, "energy"),
    "V19":     ("V19",   "v19_energy_grad_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "egrad"),
    "V20":     ("V20",   "v20_energy_desc_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "edesc"),
    "V21":     ("V21",   "v21_energy_grad_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "egrad"),
    "V22":     ("V22",   "v22_energy_desc_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "edesc"),
    "V23":     ("V23",   "v23_energy_grad_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "egrad"),
    "V24":     ("V24",   "v24_energy_desc_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "edesc"),
}

HELMHOLTZ = {"V6", "V7", "V8"}

GROUP_COLORS = {
    "gpt":    "#2196F3",
    "energy": "#FF9800",
    "egrad":  "#F44336",
    "edesc":  "#4CAF50",
    "mixed":  "#9C27B0",
}
GROUP_LABELS = {
    "gpt": "GPT", "energy": "EGPT", "egrad": "EGrad",
    "edesc": "EDesc", "mixed": "Mixed",
}


def compute_mflops(ul, itr, d, int_s):
    return 2 * ul * itr * (4 * d * d + 3 * d * int_s) / 1e6


def load_gsm8k(subdir):
    files = sorted(glob.glob(str(BASE / subdir / "unsharded" / "harness_results_*.json")))
    if not files:
        return None
    d = json.load(open(files[-1]))["results"]
    return d.get("gsm8k", {}).get("exact_match,flexible-extract")


data = {}
for key, (disp, subdir, params, ul, itr, d, int_s, grp) in MODELS.items():
    if key in HELMHOLTZ:
        continue
    gsm = load_gsm8k(subdir)
    data[key] = dict(
        label=key, params=params,
        mflops=compute_mflops(ul, itr, d, int_s),
        gsm8k=gsm,
        group=grp,
        recurrent=(itr > 1),
    )

keys = [k for k in data if data[k]["gsm8k"] is not None]
entries = [data[k] for k in keys]
xs_p = [e["params"]  for e in entries]
xs_f = [e["mflops"]  for e in entries]
ys   = [e["gsm8k"] * 100 for e in entries]

legend_patches = [mpatches.Patch(color=c, label=GROUP_LABELS[g])
                  for g, c in GROUP_COLORS.items()
                  if g in {e["group"] for e in entries}]
rec_legend = [
    plt.scatter([], [], marker="o", color="#888", s=60, label="non-recurrent"),
    plt.scatter([], [], marker="s", color="#888", s=60, label="recurrent (6×2 / 12×2)"),
]


def make_scatter(ax, xs, log_x=True):
    texts = []
    for x, y, lbl, e in zip(xs, ys, keys, entries):
        c = GROUP_COLORS[e["group"]]
        marker = "s" if e["recurrent"] else "o"
        ax.scatter(x, y, color=c, s=90, marker=marker, alpha=0.9,
                   linewidths=0.6, edgecolors="white", zorder=3)
        texts.append(ax.text(x, y, lbl, fontsize=7, color=c, zorder=4))
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        expand=(1.3, 1.5), force_text=(0.5, 0.8), force_points=(0.2, 0.3),
        only_move={"texts": "xy", "points": ""},
    )
    ax.grid(alpha=0.25, which="both" if log_x else "major")
    ax.set_ylabel("GSM8K Exact Match ↑ (%)")


# Plot A: GSM8K vs params
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_p)
ax.set_xlabel("Parameters (M, log scale)")
ax.set_title("GSM8K vs Model Size (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_gsm8k_vs_params.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_gsm8k_vs_params")

# Plot B: GSM8K vs FLOPs
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_f)
ax.set_xlabel("FLOPs/token (M, log scale, 2×matmul approx)")
ax.set_title("GSM8K vs Compute (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="upper left", ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_gsm8k_vs_flops.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_gsm8k_vs_flops")
