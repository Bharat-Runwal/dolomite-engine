"""
V2 scatter plots: log-x, squares for recurrent models, adjustText labels, no Helmholtz.
Saves as scatter_*_v2.{pdf,png} alongside the originals.
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from adjustText import adjust_text
from pathlib import Path

BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
PLOTS_DIR = BASE / "plots"

MODELS = {
    "V0":     ("V0: GPT 12×1",          "v0_gpt_baseline_d768",                       143.1, 12, 1, 768,  2048, "gpt"),
    "V1":     ("V1: EGPT 12×1",         "v1_12x1_d768_lr2e3",                         143.1, 12, 1, 768,  2048, "energy"),
    "V2":     ("V2: EGPT 6×2",          "v2_6x2_d768",                                110.1,  6, 2, 768,  2048, "energy"),
    "V3":     ("V3: EGPT d=1152",       "v3_shared_d1152",                            157.2, 12, 1, 1152, 3072, "energy"),
    "V4":     ("V4: EGPT wide d=1152",  "v4_shared_wide_d1152_lr2e3",                 164.3, 12, 1, 1152, 6144, "energy"),
    "V5":     ("V5: EAttn-only",        "v5_12x1_d768_attn_only_energy_lr2e3",        136.1, 12, 1, 768,  2048, "energy"),
    "V9":     ("V9: GPT 24×1 d=1024",  "v9_gpt_baseline_d1024_lr1e3",                354.5, 24, 1, 1024, 2048, "gpt"),
    "V10":    ("V10: Mixed 12×1",       "v10_mixed_12x1_d768_lr2e3",                  144.3, 12, 1, 768,  1536, "mixed"),
    "V11":    ("V11: Mixed 6×2",        "v11_mixed_6x2_d768_lr2e3",                   117.8,  6, 2, 768,  2048, "mixed"),
    "V12":    ("V12: GPT 6×2",          "v12_gpt_6x2_d768_lr2e3",                     119.5,  6, 2, 768,  2048, "gpt"),
    "V13":    ("V13: 8E+4G 12×1",       "v13_mixed_8e4g_12x1_d768_lr2e3",             143.1, 12, 1, 768,  1536, "mixed"),
    "V14":    ("V14: 10E+2G 12×1",      "v14_mixed_10e2g_12x1_d768_lr2e3",            142.0, 12, 1, 768,  1536, "mixed"),
    "V15":    ("V15: EGrad 12×1",       "v15_energy_grad_mixed_12x1_d768_lr2e3",       144.3, 12, 1, 768,  1536, "egrad"),
    "V16":    ("V16: EDesc 12×1",       "v16_mixed_energy_descent_12x1_d768_lr2e3",    144.3, 12, 1, 768,  1536, "edesc"),
    "V17":    ("V17: EGrad 6×2",        "v17_energy_grad_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "egrad"),
    "V18":    ("V18: EDesc 6×2",        "v18_energy_desc_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "edesc"),
    "V1_400m":("V1: EGPT 24×1 d=1024", "v1_400m_d1024_lr7e4",                        354.4, 24, 1, 1024, 3072, "energy"),
    "V19":    ("V19: EGrad 24×1 1024",  "v19_energy_grad_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "egrad"),
    "V20":    ("V20: EDesc 24×1 1024",  "v20_energy_desc_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "edesc"),
    "V21":    ("V21: EGrad 6×2 1024",   "v21_energy_grad_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "egrad"),
    "V22":    ("V22: EDesc 6×2 1024",   "v22_energy_desc_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "edesc"),
    "V23":    ("V23: EGrad 12×2 1024",  "v23_energy_grad_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "egrad"),
    "V24":    ("V24: EDesc 12×2 1024",  "v24_energy_desc_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "edesc"),
}

HELMHOLTZ = {"V6", "V7", "V8"}   # excluded from v2

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

BENCH_TASKS = [
    ("arc_challenge", "acc_norm,none"),
    ("arc_easy",      "acc_norm,none"),
    ("boolq",         "acc,none"),
    ("copa",          "acc,none"),
    ("hellaswag",     "acc_norm,none"),
    ("mmlu",          "acc,none"),
    ("openbookqa",    "acc_norm,none"),
    ("piqa",          "acc_norm,none"),
    ("sciq",          "acc_norm,none"),
    ("winogrande",    "acc,none"),
]


def compute_mflops(ul, itr, d, int_s):
    return 2 * ul * itr * (4 * d * d + 3 * d * int_s) / 1e6


def load_results(subdir):
    files = sorted(glob.glob(str(BASE / subdir / "unsharded" / "harness_results_*.json")))
    if not files:
        return None, None
    d = json.load(open(files[-1]))
    res = d["results"]
    ppl = res.get("wikitext", {}).get("word_perplexity,none")
    return res, ppl


def avg_acc(res):
    scores = [res.get(t, {}).get(m) for t, m in BENCH_TASKS]
    scores = [s for s in scores if s is not None]
    return np.mean(scores) if scores else None


# ── load ──────────────────────────────────────────────────────────────────────
data = {}
for key, (disp, subdir, params, ul, itr, d, int_s, grp) in MODELS.items():
    if key in HELMHOLTZ:
        continue
    res, ppl = load_results(subdir)
    data[key] = dict(
        label=key, display=disp,
        params=params, mflops=compute_mflops(ul, itr, d, int_s),
        ppl=ppl, avg_acc=avg_acc(res) if res else None,
        group=grp, pending=(res is None),
        recurrent=(itr > 1),   # 6×2, 12×2 → square marker
    )

# ── scatter helper ─────────────────────────────────────────────────────────────
def make_scatter(ax, xs, ys, labels, entries, log_x=True):
    texts = []
    for x, y, lbl, e in zip(xs, ys, labels, entries):
        if x is None or y is None:
            continue
        c = GROUP_COLORS[e["group"]]
        marker = "s" if e["recurrent"] else "o"
        alpha = 0.45 if e["pending"] else 0.9
        size = 90
        ax.scatter(x, y, color=c, s=size, marker=marker, alpha=alpha,
                   linewidths=0.6, edgecolors="white", zorder=3)
        txt = ax.text(x, y, lbl, fontsize=7, color=c, zorder=4)
        texts.append(txt)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    adjust_text(
        texts, ax=ax,
        arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
        expand=(1.3, 1.5),
        force_text=(0.5, 0.8),
        force_points=(0.2, 0.3),
        only_move={"texts": "xy", "points": ""},
    )
    ax.grid(alpha=0.25, which="both" if log_x else "major")

# collect lists (skip pending for cleaner plots)
keys = [k for k in data if not data[k]["pending"]]
entries = [data[k] for k in keys]
xs_p = [e["params"]  for e in entries]
xs_f = [e["mflops"]  for e in entries]
ys_p = [e["ppl"]     for e in entries]
ys_a = [e["avg_acc"] for e in entries]

legend_patches = [mpatches.Patch(color=c, label=GROUP_LABELS[g])
                  for g, c in GROUP_COLORS.items()
                  if g in {e["group"] for e in entries}]
rec_legend = [
    plt.scatter([], [], marker="o", color="#888", s=60, label="12×1 (non-recurrent)"),
    plt.scatter([], [], marker="s", color="#888", s=60, label="6×2 / 12×2 (recurrent)"),
]

# ── Plot A: PPL vs params ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_p, ys_p, keys, entries, log_x=True)
ax.invert_yaxis()
ax.set_xlabel("Parameters (M, log scale)")
ax.set_ylabel("WikiText PPL ↓")
ax.set_title("Perplexity vs Model Size (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="upper left",
          ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_vs_params_v2.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_vs_params_v2")

# ── Plot B: avg acc vs params ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_p, ys_a, keys, entries, log_x=True)
ax.set_xlabel("Parameters (M, log scale)")
ax.set_ylabel("Avg Accuracy ↑")
ax.set_title("Avg Benchmark Accuracy vs Model Size (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="lower right",
          ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_acc_vs_params_v2.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_acc_vs_params_v2")

# ── Plot C: avg acc vs FLOPs ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_f, ys_a, keys, entries, log_x=True)
ax.set_xlabel("FLOPs/token (M, log scale, 2×matmul approx)")
ax.set_ylabel("Avg Accuracy ↑")
ax.set_title("Avg Accuracy vs Compute (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="lower right",
          ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_acc_vs_flops_v2.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_acc_vs_flops_v2")

# ── Plot D: PPL vs FLOPs ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
make_scatter(ax, xs_f, ys_p, keys, entries, log_x=True)
ax.invert_yaxis()
ax.set_xlabel("FLOPs/token (M, log scale, 2×matmul approx)")
ax.set_ylabel("WikiText PPL ↓")
ax.set_title("Perplexity vs Compute (no Helmholtz)")
ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="upper left",
          ncol=2, framealpha=0.8)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_vs_flops_v2.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_vs_flops_v2")
