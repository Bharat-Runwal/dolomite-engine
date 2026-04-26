"""
Comprehensive plots for multi-block ablation variants.
Generates: accuracy heatmap, 400M bar chart, PPL vs params, avg acc vs params, avg acc vs FLOPs.
"""
import json, glob, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Model registry ─────────────────────────────────────────────────────────────
# (display_name, results_subdir, params_M, unique_layers, iters_per_layer, d, int_size, group)
# params_M from training-log measurements; group drives colour.
MODELS = {
    "V0":     ("V0: GPT 12×1",           "v0_gpt_baseline_d768",                       143.1, 12, 1, 768,  2048, "gpt"),
    "V1":     ("V1: EGPT 12×1",          "v1_12x1_d768_lr2e3",                         143.1, 12, 1, 768,  2048, "energy"),
    "V2":     ("V2: EGPT 6×2",           "v2_6x2_d768",                                110.1,  6, 2, 768,  2048, "energy"),
    "V3":     ("V3: EGPT d=1152",        "v3_shared_d1152",                            157.2, 12, 1, 1152, 3072, "energy"),
    "V4":     ("V4: EGPT wide d=1152",   "v4_shared_wide_d1152_lr2e3",                 164.3, 12, 1, 1152, 6144, "energy"),
    "V5":     ("V5: EAttn-only 12×1",    "v5_12x1_d768_attn_only_energy_lr2e3",        136.1, 12, 1, 768,  2048, "energy"),
    "V6":     ("V6: Helmholtz-fac 12×1", "v6_12x1_d768_helmholtz_factored_lr2e3",      129.7, 12, 1, 768,  2048, "helmholtz"),
    "V7":     ("V7: Helmholtz-dual 12×1","v7_12x1_d768_helmholtz_dual_lr2e3",          129.7, 12, 1, 768,  2048, "helmholtz"),
    "V8":     ("V8: Helmholtz-rev 12×1", "v8_12x1_d768_helmholtz_dual_reversed_lr2e3", 129.7, 12, 1, 768,  2048, "helmholtz"),
    "V9":     ("V9: GPT 24×1 d=1024",   "v9_gpt_baseline_d1024_lr1e3",                354.5, 24, 1, 1024, 2048, "gpt"),
    "V10":    ("V10: Mixed 12×1",        "v10_mixed_12x1_d768_lr2e3",                  144.3, 12, 1, 768,  1536, "mixed"),
    "V11":    ("V11: Mixed 6×2",         "v11_mixed_6x2_d768_lr2e3",                   117.8,  6, 2, 768,  2048, "mixed"),
    "V12":    ("V12: GPT 6×2",           "v12_gpt_6x2_d768_lr2e3",                     119.5,  6, 2, 768,  2048, "gpt"),
    "V13":    ("V13: Mixed 8E+4G 12×1",  "v13_mixed_8e4g_12x1_d768_lr2e3",             143.1, 12, 1, 768,  1536, "mixed"),
    "V14":    ("V14: Mixed 10E+2G 12×1", "v14_mixed_10e2g_12x1_d768_lr2e3",            142.0, 12, 1, 768,  1536, "mixed"),
    "V15":    ("V15: EGrad 12×1",        "v15_energy_grad_mixed_12x1_d768_lr2e3",       144.3, 12, 1, 768,  1536, "egrad"),
    "V16":    ("V16: EDesc 12×1",        "v16_mixed_energy_descent_12x1_d768_lr2e3",    144.3, 12, 1, 768,  1536, "edesc"),
    "V17":    ("V17: EGrad 6×2",         "v17_energy_grad_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "egrad"),
    "V18":    ("V18: EDesc 6×2",         "v18_energy_desc_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "edesc"),
    "V1_400m":("V1: EGPT 24×1 d=1024",  "v1_400m_d1024_lr7e4",                        354.4, 24, 1, 1024, 3072, "energy"),
    "V19":    ("V19: EGrad 24×1 d=1024", "v19_energy_grad_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "egrad"),
    "V20":    ("V20: EDesc 24×1 d=1024", "v20_energy_desc_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "edesc"),
    "V21":    ("V21: EGrad 6×2 d=1024",  "v21_energy_grad_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "egrad"),
    "V22":    ("V22: EDesc 6×2 d=1024",  "v22_energy_desc_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "edesc"),
    "V23":    ("V23: EGrad 12×2 d=1024", "v23_energy_grad_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "egrad"),
    "V24":    ("V24: EDesc 12×2 d=1024", "v24_energy_desc_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "edesc"),
}

GROUP_COLORS = {
    "gpt":       "#2196F3",   # blue
    "energy":    "#FF9800",   # orange
    "egrad":     "#F44336",   # red
    "edesc":     "#4CAF50",   # green
    "mixed":     "#9C27B0",   # purple
    "helmholtz": "#795548",   # brown
}
GROUP_LABELS = {
    "gpt": "GPT baseline", "energy": "EGPT", "egrad": "EGrad",
    "edesc": "EDesc", "mixed": "Mixed", "helmholtz": "Helmholtz",
}

BENCH_TASKS = [
    ("arc_challenge", "acc_norm,none", "ARC-C"),
    ("arc_easy",      "acc_norm,none", "ARC-E"),
    ("boolq",         "acc,none",      "BoolQ"),
    ("copa",          "acc,none",      "COPA"),
    ("hellaswag",     "acc_norm,none", "HellaSwag"),
    ("mmlu",          "acc,none",      "MMLU"),
    ("openbookqa",    "acc_norm,none", "OBQA"),
    ("piqa",          "acc_norm,none", "PIQA"),
    ("sciq",          "acc_norm,none", "SciQ"),
    ("winogrande",    "acc,none",      "Winogrande"),
]

# ── Helpers ───────────────────────────────────────────────────────────────────

def compute_mflops(unique_layers, iters, d, int_size):
    """Forward-pass FLOPs/token (millions), 2×matmul approximation."""
    per_layer = 4 * d * d + 3 * d * int_size   # attn (Q,K,V,O) + SwiGLU
    eff_non_emb = unique_layers * iters * per_layer
    return 2 * eff_non_emb / 1e6


def load_results(subdir):
    """Return (results_dict, ppl) from latest harness_results_*.json, or (None, None)."""
    pattern = str(BASE / subdir / "unsharded" / "harness_results_*.json")
    files = sorted(glob.glob(pattern))
    if not files:
        return None, None
    d = json.load(open(files[-1]))
    res = d["results"]
    ppl = res.get("wikitext", {}).get("word_perplexity,none", None)
    return res, ppl


def avg_acc(res):
    scores = []
    for task, metric, _ in BENCH_TASKS:
        v = res.get(task, {}).get(metric)
        if v is not None:
            scores.append(v)
    return np.mean(scores) if scores else None


# ── Load all data ─────────────────────────────────────────────────────────────
data = {}   # key -> dict with display, params, mflops, ppl, avg_acc, per_task, group, pending
for key, (disp, subdir, params, ul, itr, d, int_s, grp) in MODELS.items():
    res, ppl = load_results(subdir)
    pending = (res is None)
    acc_mean = avg_acc(res) if res else None
    per_task = {}
    if res:
        for task, metric, short in BENCH_TASKS:
            v = res.get(task, {}).get(metric)
            if v is not None:
                per_task[short] = v
    data[key] = dict(
        display=disp, params=params,
        mflops=compute_mflops(ul, itr, d, int_s),
        ppl=ppl, avg_acc=acc_mean,
        per_task=per_task, group=grp, pending=pending,
    )

# ── Plot 1: Heatmap of benchmark scores — 768d small models ───────────────────
small_keys = [k for k in ["V0","V1","V2","V3","V4","V5","V6","V7","V8","V10","V11","V12","V13","V14","V15","V16","V17","V18"]
              if not data[k]["pending"]]
task_shorts = [s for _, _, s in BENCH_TASKS]

fig, ax = plt.subplots(figsize=(14, 6))
mat = np.array([[data[k]["per_task"].get(s, np.nan) for s in task_shorts] for k in small_keys])
im = ax.imshow(mat, aspect="auto", cmap="RdYlGn", vmin=0.2, vmax=0.7)
ax.set_xticks(range(len(task_shorts))); ax.set_xticklabels(task_shorts, rotation=45, ha="right")
ax.set_yticks(range(len(small_keys))); ax.set_yticklabels([data[k]["display"] for k in small_keys], fontsize=8)
for i in range(len(small_keys)):
    for j in range(len(task_shorts)):
        v = mat[i, j]
        if not np.isnan(v):
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                    color="black" if 0.3 < v < 0.65 else "white")
plt.colorbar(im, ax=ax, label="Accuracy")
ax.set_title("Benchmark Accuracy — Small Models (~110-165M params)", fontsize=12)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"heatmap_small_models.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap_small_models")

# ── Plot 2: 400M models bar chart ─────────────────────────────────────────────
keys_400 = ["V9", "V1_400m", "V21", "V22", "V23", "V24", "V19", "V20"]
labels_400 = [data[k]["display"].replace(" d=1024", "").replace(" ~", "\n~") for k in keys_400]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, metric, title in zip(axes, ["ppl", "avg_acc"], ["WikiText PPL (↓)", "Avg Accuracy (↑)"]):
    vals = [data[k][metric] for k in keys_400]
    colors_bar = [GROUP_COLORS[data[k]["group"]] for k in keys_400]
    alphas = [0.35 if data[k]["pending"] else 0.85 for k in keys_400]
    x = np.arange(len(keys_400))
    bars = ax.bar(x, [v if v is not None else 0 for v in vals], color=colors_bar)
    for bar, a in zip(bars, alphas):
        bar.set_alpha(a)
    ax.set_xticks(x); ax.set_xticklabels(labels_400, fontsize=8, rotation=20, ha="right")
    ax.set_title(title); ax.set_ylabel(title)
    if metric == "ppl":
        ax.set_ylim(0, max(v for v in vals if v) * 1.2)
    else:
        ax.set_ylim(0.3, 0.5)
    ylo, yhi = ax.get_ylim()
    for i, (v, k) in enumerate(zip(vals, keys_400)):
        if v is None:
            ax.text(i, ylo + 0.01 * (yhi - ylo), "pending",
                    ha="center", va="bottom", fontsize=7, color="#888")
        else:
            ax.text(i, v + 0.01 * (yhi - ylo),
                    f"{v:.3f}" if metric == "avg_acc" else f"{v:.1f}",
                    ha="center", va="bottom", fontsize=7)
# Legend
patches = [mpatches.Patch(color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()
           if g in {data[k]["group"] for k in keys_400}]
axes[1].legend(handles=patches, fontsize=8, loc="lower right")
fig.suptitle("400M-scale Models: V9/V1_400m vs EGrad/EDesc recurrent variants", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"bar_400m_models.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved bar_400m_models")

# ── Scatter helpers ────────────────────────────────────────────────────────────
def scatter_plot(ax, xs, ys, labels, groups, pending_mask, xlabel, ylabel, title, invert_y=False):
    for x, y, lbl, grp, pend in zip(xs, ys, labels, groups, pending_mask):
        if x is None or y is None:
            continue
        c = GROUP_COLORS[grp]
        marker = "x" if pend else "o"
        ax.scatter(x, y, color=c, s=80, marker=marker, alpha=0.4 if pend else 0.85, zorder=3)
        ax.annotate(lbl, (x, y), fontsize=6.5, xytext=(4, 2), textcoords="offset points")
    ax.set_xlabel(xlabel); ax.set_ylabel(ylabel); ax.set_title(title)
    if invert_y:
        ax.invert_yaxis()
    patches = [mpatches.Patch(color=c, label=GROUP_LABELS[g]) for g, c in GROUP_COLORS.items()
               if g in set(groups)]
    ax.legend(handles=patches, fontsize=7, loc="best")
    ax.grid(alpha=0.3)

all_keys = list(data.keys())
xs_params  = [data[k]["params"]  for k in all_keys]
xs_mflops  = [data[k]["mflops"]  for k in all_keys]
ys_ppl     = [data[k]["ppl"]     for k in all_keys]
ys_acc     = [data[k]["avg_acc"] for k in all_keys]
labels_all = [k for k in all_keys]   # short keys as point labels
groups_all = [data[k]["group"]   for k in all_keys]
pending_all= [data[k]["pending"] for k in all_keys]

# ── Plot 3: PPL vs params ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter_plot(ax, xs_params, ys_ppl, labels_all, groups_all, pending_all,
             "Params (M)", "WikiText PPL (↓)", "Perplexity vs Model Size", invert_y=False)
ax.invert_yaxis()
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_vs_params.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_vs_params")

# ── Plot 4: Avg acc vs params ─────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter_plot(ax, xs_params, ys_acc, labels_all, groups_all, pending_all,
             "Params (M)", "Avg Accuracy (↑)", "Avg Benchmark Accuracy vs Model Size")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_acc_vs_params.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_acc_vs_params")

# ── Plot 5: Avg acc vs FLOPs ──────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter_plot(ax, xs_mflops, ys_acc, labels_all, groups_all, pending_all,
             "FLOPs/token (M, 2×matmul approx)", "Avg Accuracy (↑)", "Avg Accuracy vs Compute (FLOPs/token)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_acc_vs_flops.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_acc_vs_flops")

# ── Plot 6: PPL vs FLOPs ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 6))
scatter_plot(ax, xs_mflops, ys_ppl, labels_all, groups_all, pending_all,
             "FLOPs/token (M, 2×matmul approx)", "WikiText PPL (↓)", "Perplexity vs Compute (FLOPs/token)", invert_y=False)
ax.invert_yaxis()
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"scatter_ppl_vs_flops.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved scatter_ppl_vs_flops")

# ── Summary table ──────────────────────────────────────────────────────────────
print("\n=== Summary Table ===")
header = f"{'Key':<8} {'Display':<28} {'Params':>8} {'MFLOPs':>8} {'PPL':>7} {'AvgAcc':>7}  ARC-C  HellaS  MMLU"
print(header); print("-" * len(header))
for k in all_keys:
    d_ = data[k]
    ppl_s = f"{d_['ppl']:.1f}" if d_['ppl'] else "—"
    acc_s = f"{d_['avg_acc']:.3f}" if d_['avg_acc'] else "—"
    arc_s = f"{d_['per_task'].get('ARC-C', 0):.3f}" if d_['per_task'] else "—"
    hs_s  = f"{d_['per_task'].get('HellaSwag', 0):.3f}" if d_['per_task'] else "—"
    mm_s  = f"{d_['per_task'].get('MMLU', 0):.3f}" if d_['per_task'] else "—"
    pend = " (pending)" if d_["pending"] else ""
    print(f"{k:<8} {d_['display']:<28} {d_['params']:>8.1f} {d_['mflops']:>8.0f} {ppl_s:>7} {acc_s:>7}  {arc_s}  {hs_s}  {mm_s}{pend}")

print(f"\nPlots saved to {PLOTS_DIR}")
