"""
Scatter plots and tables for true EGrad/EDesc runs (V27-V36).

Two variants of each scatter:
  - *_with_mixhead: includes V15-V24 (relabeled MixH†) for comparison
  - *_egrad_only:  only V27-V36 (true EGrad/EDesc) + GPT/EGPT/Mixed baselines

Old plots from plot_scatter_v2_20260421.py are left untouched.
New outputs: scatter_*_v3_{with_mixhead,egrad_only}.{pdf,png}
             table_egrad_edesc_v27_v36.txt
             bar_egrad_edesc_v27_v36.{pdf,png}

Run after harness_results_*.json files exist for V27-V30 and V33-V34.
V31/V32/V35/V36 (400M, longer runs) are included if available, skipped if not.
"""

import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.ticker
from adjustText import adjust_text
from pathlib import Path

BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
# (display, subdir, params_M, num_unique_layers, iterations, hidden, intermediate, group)
MODELS = {
    # GPT baselines
    "V0":      ("V0: GPT 12×1",          "v0_gpt_baseline_d768",                       162.1, 12, 1, 768,  3072, "gpt"),
    "V9":      ("V9: GPT 24×1 d=1024",   "v9_gpt_baseline_d1024_lr1e3",                354.5, 24, 1, 1024, 4096, "gpt"),
    # EGPT
    "V1":      ("V1: EGPT 12×1",         "v1_12x1_d768_lr2e3",                         143.1, 12, 1, 768,  2048, "energy"),
    "V2":      ("V2: EGPT 6×2",          "v2_6x2_d768",                                110.1,  6, 2, 768,  2048, "energy"),
    "V1_400m": ("V1: EGPT 24×1 d=1024",  "v1_400m_d1024_lr7e4",                        354.4, 24, 1, 1024, 3072, "energy"),
    # Mixed head baselines
    "V10":     ("V10: Mixed 12×1",        "v10_mixed_12x1_d768_lr2e3",                  144.3, 12, 1, 768,  1536, "mixed"),
    "V11":     ("V11: Mixed 6×2",         "v11_mixed_6x2_d768_lr2e3",                   117.8,  6, 2, 768,  2048, "mixed"),
    # MixHead† (V15-V24: trained as plain MixedHead due to config bug)
    "V15":     ("V15: MixH† 12×1",        "v15_energy_grad_mixed_12x1_d768_lr2e3",      144.3, 12, 1, 768,  1536, "mixhead_dag"),
    "V16":     ("V16: MixH† 12×1",        "v16_mixed_energy_descent_12x1_d768_lr2e3",   144.3, 12, 1, 768,  1536, "mixhead_dag"),
    "V17":     ("V17: MixH† 6×2",         "v17_energy_grad_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "mixhead_dag"),
    "V18":     ("V18: MixH† 6×2",         "v18_energy_desc_6x2_d768_lr2e3",             117.8,  6, 2, 768,  2048, "mixhead_dag"),
    "V19":     ("V19: MixH† 24×1 1024",   "v19_energy_grad_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "mixhead_dag"),
    "V20":     ("V20: MixH† 24×1 1024",   "v20_energy_desc_24x1_d1024_lr1e3",           341.9, 24, 1, 1024, 2048, "mixhead_dag"),
    "V21":     ("V21: MixH† 6×2 1024",    "v21_energy_grad_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "mixhead_dag"),
    "V22":     ("V22: MixH† 6×2 1024",    "v22_energy_desc_6x2_d1024_lr1e3",            162.5,  6, 2, 1024, 2048, "mixhead_dag"),
    "V23":     ("V23: MixH† 12×2 1024",   "v23_energy_grad_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "mixhead_dag"),
    "V24":     ("V24: MixH† 12×2 1024",   "v24_energy_desc_12x2_d1024_lr1e3",           228.0, 12, 2, 1024, 2048, "mixhead_dag"),
    # True EGrad/EDesc (V27-V36)
    "V27":     ("V27: EGrad 12×1 d768",   "v27_egrad_attn_12x1_d768_lr2e3",             140.8, 12, 1, 768,  1536, "egrad"),
    "V28":     ("V28: EDesc 12×1 d768",   "v28_edesc_12x1_d768_lr2e3",                  140.8, 12, 1, 768,  1536, "edesc"),
    "V29":     ("V29: EGrad 6×2 d768",    "v29_egrad_attn_6x2_d768_lr2e3",              114.5,  6, 2, 768,  2048, "egrad"),
    "V30":     ("V30: EDesc 6×2 d768",    "v30_edesc_6x2_d768_lr2e3",                   114.5,  6, 2, 768,  2048, "edesc"),
    "V31":     ("V31: EGrad 24×1 1024",   "v31_egrad_attn_24x1_d1024_lr1e3",            341.9, 24, 1, 1024, 2048, "egrad"),
    "V32":     ("V32: EDesc 24×1 1024",   "v32_edesc_24x1_d1024_lr1e3",                 341.9, 24, 1, 1024, 2048, "edesc"),
    "V33":     ("V33: EGrad 6×2 1024",    "v33_egrad_attn_6x2_d1024_lr1e3",             162.5,  6, 2, 1024, 2048, "egrad"),
    "V34":     ("V34: EDesc 6×2 1024",    "v34_edesc_6x2_d1024_lr1e3",                  162.5,  6, 2, 1024, 2048, "edesc"),
    "V35":     ("V35: EGrad 12×2 1024",   "v35_egrad_attn_12x2_d1024_lr1e3",            228.0, 12, 2, 1024, 2048, "egrad"),
    "V36":     ("V36: EDesc 12×2 1024",   "v36_edesc_12x2_d1024_lr1e3",                 228.0, 12, 2, 1024, 2048, "edesc"),
}

GROUP_COLORS = {
    "gpt":         "#2196F3",
    "energy":      "#FF9800",
    "mixed":       "#9C27B0",
    "mixhead_dag": "#BDBDBD",   # grey — mislabeled runs
    "egrad":       "#F44336",
    "edesc":       "#4CAF50",
}
GROUP_LABELS = {
    "gpt": "GPT", "energy": "EGPT", "mixed": "Mixed",
    "mixhead_dag": "MixH† (bug)", "egrad": "EGrad (true)", "edesc": "EDesc (true)",
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
    ("sciq",          "acc,none"),
    ("winogrande",    "acc,none"),
]
TASK_LABELS = {
    "arc_challenge": "ARC-C", "arc_easy": "ARC-E", "boolq": "BoolQ",
    "copa": "COPA", "hellaswag": "HellaSwag", "mmlu": "MMLU",
    "openbookqa": "ObQA", "piqa": "PIQA", "sciq": "SciQ", "winogrande": "Wino.",
}


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
    if res is None:
        return None
    scores = []
    for t, m in BENCH_TASKS:
        v = res.get(t, {}).get(m)
        if v is None:
            v = res.get(t, {}).get("acc,none")
        if v is not None:
            scores.append(v)
    return float(np.mean(scores)) if scores else None


# ── Load all ──────────────────────────────────────────────────────────────────
data = {}
for key, (disp, subdir, params, ul, itr, d, int_s, grp) in MODELS.items():
    res, ppl = load_results(subdir)
    acc = avg_acc(res)
    gsm = None
    if res is not None:
        g = res.get("gsm8k", {})
        gsm = g.get("exact_match,flexible-extract", g.get("exact_match,none"))
        if gsm is not None:
            gsm *= 100
    data[key] = dict(
        label=key, display=disp,
        params=params,
        mflops=compute_mflops(ul, itr, d, int_s),
        ppl=ppl, avg_acc=acc if acc is not None else None,
        gsm=gsm,
        group=grp,
        pending=(res is None),
        recurrent=(itr > 1),
        task_accs={t: (res.get(t, {}).get(m) or res.get(t, {}).get("acc,none")) for t, m in BENCH_TASKS} if res else {},
    )

# ── Print table ───────────────────────────────────────────────────────────────
print(f"\n{'Model':<30} {'Params':>7} {'PPL':>7} {'AvgAcc':>8} {'GSM8K':>7}")
print("-" * 65)

GROUPS_ORDER = ["gpt", "energy", "mixed", "mixhead_dag", "egrad", "edesc"]
for grp in GROUPS_ORDER:
    keys_in_grp = [k for k in MODELS if data[k]["group"] == grp]
    for k in keys_in_grp:
        e = data[k]
        ppl_str  = f"{e['ppl']:.2f}"  if e['ppl']  is not None else "pending"
        acc_str  = f"{e['avg_acc']*100:.1f}%" if e['avg_acc'] is not None else "pending"
        gsm_str  = f"{e['gsm']:.2f}%" if e['gsm']  is not None else "pending"
        print(f"{e['display']:<30} {e['params']:>7.1f} {ppl_str:>7} {acc_str:>8} {gsm_str:>7}")

# Save table to file
table_path = PLOTS_DIR / "table_egrad_edesc_v27_v36.txt"
with open(table_path, "w") as f:
    f.write(f"{'Model':<30} {'Params':>7} {'PPL':>7} {'AvgAcc':>8} {'GSM8K':>7}\n")
    f.write("-" * 65 + "\n")
    for grp in GROUPS_ORDER:
        keys_in_grp = [k for k in MODELS if data[k]["group"] == grp]
        for k in keys_in_grp:
            e = data[k]
            ppl_str  = f"{e['ppl']:.2f}"  if e['ppl']  is not None else "pending"
            acc_str  = f"{e['avg_acc']*100:.1f}%" if e['avg_acc'] is not None else "pending"
            gsm_str  = f"{e['gsm']:.2f}%" if e['gsm']  is not None else "pending"
            f.write(f"{e['display']:<30} {e['params']:>7.1f} {ppl_str:>7} {acc_str:>8} {gsm_str:>7}\n")
print(f"\nTable saved to {table_path}")


# ── Scatter helper ────────────────────────────────────────────────────────────
def make_scatter(ax, entries, x_key, y_key, log_x=True, invert_y=False):
    texts = []
    for e in entries:
        x, y = e[x_key], e[y_key]
        if x is None or y is None:
            continue
        c = GROUP_COLORS[e["group"]]
        marker = "s" if e["recurrent"] else "o"
        alpha  = 0.4 if e["pending"] else 0.9
        ax.scatter(x, y, color=c, s=90, marker=marker, alpha=alpha,
                   linewidths=0.6, edgecolors="white", zorder=3)
        txt = ax.text(x, y, e["label"], fontsize=7, color=c, zorder=4)
        texts.append(txt)
    if log_x:
        ax.set_xscale("log")
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if invert_y:
        ax.invert_yaxis()
    try:
        adjust_text(
            texts, ax=ax,
            arrowprops=dict(arrowstyle="-", color="#aaaaaa", lw=0.6),
            expand=(1.3, 1.5),
            force_text=(0.5, 0.8),
            force_points=(0.2, 0.3),
            only_move={"texts": "xy", "points": ""},
        )
    except Exception:
        pass
    ax.grid(alpha=0.25, which="both" if log_x else "major")


def save_scatter(entries, stem, title_suffix, x_key, x_label, y_key, y_label, invert_y=False):
    groups_present = {e["group"] for e in entries if e[x_key] and e[y_key]}
    legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
                      for g in GROUPS_ORDER if g in groups_present]
    rec_legend = [
        plt.scatter([], [], marker="o", color="#888", s=60, label="non-recurrent"),
        plt.scatter([], [], marker="s", color="#888", s=60, label="recurrent (6×2 / 12×2)"),
    ]
    fig, ax = plt.subplots(figsize=(9, 6))
    make_scatter(ax, entries, x_key, y_key, log_x=True, invert_y=invert_y)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(f"{y_label} vs {x_label} — {title_suffix}")
    ax.legend(handles=legend_patches + rec_legend, fontsize=7, loc="best", ncol=2, framealpha=0.8)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {stem}")


# ── Variant A: with MixHead† for comparison ───────────────────────────────────
entries_all = [e for e in data.values() if not e["pending"]]

save_scatter(
    entries_all, "scatter_ppl_vs_params_v3_with_mixhead",
    "EGrad/EDesc (V27-V36) + MixH† comparison",
    "params", "Parameters (M, log scale)",
    "ppl",    "WikiText PPL ↓", invert_y=True,
)
save_scatter(
    entries_all, "scatter_acc_vs_params_v3_with_mixhead",
    "EGrad/EDesc (V27-V36) + MixH† comparison",
    "params",   "Parameters (M, log scale)",
    "avg_acc",  "Avg Accuracy ↑",
)
save_scatter(
    entries_all, "scatter_acc_vs_flops_v3_with_mixhead",
    "EGrad/EDesc (V27-V36) + MixH† comparison",
    "mflops",  "FLOPs/token (M, log scale)",
    "avg_acc", "Avg Accuracy ↑",
)
save_scatter(
    entries_all, "scatter_ppl_vs_flops_v3_with_mixhead",
    "EGrad/EDesc (V27-V36) + MixH† comparison",
    "mflops", "FLOPs/token (M, log scale)",
    "ppl",    "WikiText PPL ↓", invert_y=True,
)

# ── Variant B: EGrad/EDesc only (no MixHead†) ─────────────────────────────────
EXCLUDE_MIXHEAD = {"mixhead_dag"}
entries_clean = [e for e in data.values()
                 if not e["pending"] and e["group"] not in EXCLUDE_MIXHEAD]

save_scatter(
    entries_clean, "scatter_ppl_vs_params_v3_egrad_only",
    "True EGrad/EDesc (V27-V36)",
    "params", "Parameters (M, log scale)",
    "ppl",    "WikiText PPL ↓", invert_y=True,
)
save_scatter(
    entries_clean, "scatter_acc_vs_params_v3_egrad_only",
    "True EGrad/EDesc (V27-V36)",
    "params",  "Parameters (M, log scale)",
    "avg_acc", "Avg Accuracy ↑",
)
save_scatter(
    entries_clean, "scatter_acc_vs_flops_v3_egrad_only",
    "True EGrad/EDesc (V27-V36)",
    "mflops",  "FLOPs/token (M, log scale)",
    "avg_acc", "Avg Accuracy ↑",
)
save_scatter(
    entries_clean, "scatter_ppl_vs_flops_v3_egrad_only",
    "True EGrad/EDesc (V27-V36)",
    "mflops", "FLOPs/token (M, log scale)",
    "ppl",    "WikiText PPL ↓", invert_y=True,
)

# ── Bar chart: avg accuracy by model (EGrad/EDesc + baselines, no MixH†) ─────
bar_entries = [e for e in entries_clean if e["avg_acc"] is not None]
bar_entries.sort(key=lambda e: e["avg_acc"])

fig, ax = plt.subplots(figsize=(12, 5))
bar_labels = [e["label"] for e in bar_entries]
bar_accs   = [e["avg_acc"] * 100 for e in bar_entries]
bar_colors = [GROUP_COLORS[e["group"]] for e in bar_entries]
bars = ax.bar(range(len(bar_entries)), bar_accs, color=bar_colors, edgecolor="white", linewidth=0.5)
ax.set_xticks(range(len(bar_entries)))
ax.set_xticklabels(bar_labels, rotation=45, ha="right", fontsize=8)
ax.set_ylabel("Avg Accuracy (10 tasks) ↑")
ax.set_title("Average Benchmark Accuracy — True EGrad/EDesc vs Baselines")
ax.set_ylim(min(bar_accs) * 0.97, max(bar_accs) * 1.02)
ax.grid(axis="y", alpha=0.3)
legend_patches = [mpatches.Patch(color=GROUP_COLORS[g], label=GROUP_LABELS[g])
                  for g in GROUPS_ORDER if g in {e["group"] for e in bar_entries} and g != "mixhead_dag"]
ax.legend(handles=legend_patches, fontsize=8, loc="upper left")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"bar_egrad_edesc_v27_v36.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved bar_egrad_edesc_v27_v36")

# ── Per-task heatmap: EGrad/EDesc vs baselines ────────────────────────────────
heatmap_entries = [e for e in entries_clean if e["task_accs"]]
heatmap_entries.sort(key=lambda e: e.get("avg_acc") or 0)
task_names = [t for t, _ in BENCH_TASKS]

matrix = np.full((len(heatmap_entries), len(task_names)), np.nan)
for i, e in enumerate(heatmap_entries):
    for j, t in enumerate(task_names):
        v = e["task_accs"].get(t)
        if v is not None:
            matrix[i, j] = v * 100

fig, ax = plt.subplots(figsize=(12, max(4, len(heatmap_entries) * 0.45)))
im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=40, vmax=85)
ax.set_xticks(range(len(task_names)))
ax.set_xticklabels([TASK_LABELS.get(t, t) for t in task_names], rotation=45, ha="right", fontsize=9)
ax.set_yticks(range(len(heatmap_entries)))
ax.set_yticklabels([e["label"] for e in heatmap_entries], fontsize=9)
for i in range(len(heatmap_entries)):
    for j in range(len(task_names)):
        if not np.isnan(matrix[i, j]):
            ax.text(j, i, f"{matrix[i,j]:.1f}", ha="center", va="center", fontsize=7,
                    color="black" if 45 < matrix[i, j] < 80 else "white")
plt.colorbar(im, ax=ax, label="Accuracy (%)")
ax.set_title("Per-Task Accuracy — True EGrad/EDesc vs Baselines (V27-V36)")
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"heatmap_egrad_edesc_v27_v36.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved heatmap_egrad_edesc_v27_v36")

print("\nAll plots saved to", PLOTS_DIR)
