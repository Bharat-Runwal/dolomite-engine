"""
Plots for V13-V16 ablation: head ratio and output rule variants (all 12×1, d=768).

V0  GPT baseline      162M / 321 MFLOPs
V1  EGPT 12×1         143M / 359 MFLOPs
V10 Mixed 6E+6G       144M / 285 MFLOPs
V13 Mixed 8E+4G       143M / 313 MFLOPs   (eval pending)
V14 Mixed 10E+2G      142M / 341 MFLOPs   (eval pending)
V15 EGrad 6E+6G       144M / 285 MFLOPs   (energy grad output)
V16 EDesc 6E+6G       144M / 285 MFLOPs   (energy descent update)
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RESULT_FILES = {
    "V0 GPT":  BASE / "v0_gpt_baseline_d768/unsharded/harness_results_2026-04-19T09-04-34.326698.json",
    "V1 EGPT": BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_2026-04-19T15-07-35.156466.json",
    "V10 Mix\n6E+6G": BASE / "v10_mixed_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T03-56-05.097549.json",
    "V15 EGrad\n6E+6G": BASE / "v15_energy_grad_mixed_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T14-31-26.429561.json",
    "V16 EDesc\n6E+6G": BASE / "v16_mixed_energy_descent_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T14-30-44.691975.json",
}

PARAMS = {"V0 GPT": 162, "V1 EGPT": 143, "V10 Mix\n6E+6G": 144,
          "V15 EGrad\n6E+6G": 144, "V16 EDesc\n6E+6G": 144}
FLOPS  = {"V0 GPT": 321, "V1 EGPT": 359, "V10 Mix\n6E+6G": 285,
          "V15 EGrad\n6E+6G": 285, "V16 EDesc\n6E+6G": 285}

TASKS = ["arc_challenge","arc_easy","boolq","copa","hellaswag","openbookqa","piqa","sciq","winogrande","mmlu"]
TASK_KEYS = {
    "arc_challenge":"acc_norm,none","arc_easy":"acc,none","boolq":"acc,none",
    "copa":"acc,none","hellaswag":"acc_norm,none","openbookqa":"acc_norm,none",
    "piqa":"acc_norm,none","sciq":"acc,none","winogrande":"acc,none","mmlu":"acc,none",
}
TASK_LABELS = {
    "arc_challenge":"ARC-C","arc_easy":"ARC-E","boolq":"BoolQ","copa":"COPA",
    "hellaswag":"HellaSwag","openbookqa":"ObQA","piqa":"PIQA","sciq":"SciQ",
    "winogrande":"Wino.","mmlu":"MMLU",
}

COLORS = {
    "V0 GPT":          "#1f77b4",
    "V1 EGPT":         "#ff7f0e",
    "V10 Mix\n6E+6G":  "#9467bd",
    "V15 EGrad\n6E+6G":"#e377c2",
    "V16 EDesc\n6E+6G":"#17becf",
}
COLORS_LIST = ["#1f77b4","#ff7f0e","#9467bd","#e377c2","#17becf"]

def load(path):
    with open(path) as f:
        data = json.load(f)
    accs = {}
    for task, key in TASK_KEYS.items():
        if task in data["results"]:
            v = data["results"][task].get(key, data["results"][task].get("acc,none"))
            if v is not None:
                accs[task] = v * 100
    avg = sum(accs.values()) / len(accs) if accs else 0
    ppl = data["results"].get("wikitext", {}).get("word_perplexity,none")
    gsm = data["results"].get("gsm8k", {})
    gsm_score = gsm.get("exact_match,flexible-extract", gsm.get("exact_match,none", 0)) * 100
    return accs, avg, ppl, gsm_score

results = {name: load(path) for name, path in RESULT_FILES.items()}
model_names = list(results.keys())

# ── Print summary ─────────────────────────────────────────────────────────────
print(f"{'Model':<20} {'PPL':>7} {'Avg':>7} {'GSM8K':>7}")
for name in model_names:
    accs, avg, ppl, gsm = results[name]
    print(f"{name.replace(chr(10),' '):<20} {ppl:>7.2f} {avg:>7.1f}% {gsm:>6.2f}%")

# ── Fig 1: Grouped bar — avg acc ──────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
x = np.arange(len(model_names))
avgs = [results[n][1] for n in model_names]
bars = ax.bar(x, avgs, color=COLORS_LIST, width=0.55, edgecolor="white", linewidth=1.2)
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n", "\n") for n in model_names], fontsize=9)
ax.set_ylabel("Avg zero-shot accuracy (%)")
ax.set_title("Avg accuracy: output rule ablation (all 12×1, d=768)", fontsize=11)
ax.set_ylim(44, 50)
for bar, v, n in zip(bars, avgs, model_names):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.05,
            f"{v:.1f}%\n{PARAMS[n]}M/{FLOPS[n]}M", ha="center", va="bottom", fontsize=7.5)
ax.grid(axis="y", alpha=0.3)
# Mark V0 GPT baseline with horizontal line
ax.axhline(avgs[0], color=COLORS["V0 GPT"], linewidth=1.0, linestyle="--", alpha=0.5)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_avg_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_avg_acc")

# ── Fig 2: Grouped bar — PPL ───────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(9, 5))
ppls = [results[n][2] for n in model_names]
bars = ax.bar(x, ppls, color=COLORS_LIST, width=0.55, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels([n.replace("\n", "\n") for n in model_names], fontsize=9)
ax.set_ylabel("WikiText Word Perplexity ↓")
ax.set_title("Perplexity: output rule ablation (all 12×1, d=768)", fontsize=11)
for bar, v in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.2f}", ha="center", va="bottom", fontsize=9)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_ppl.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_ppl")

# ── Fig 3: Heatmap (absolute accuracy) ────────────────────────────────────────
all_cols = TASKS + ["AVG", "GSM8K"]
col_labels = [TASK_LABELS[t] for t in TASKS] + ["AVG", "GSM8K"]

data_matrix = np.zeros((len(model_names), len(all_cols)))
for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    for j, col in enumerate(TASKS):
        data_matrix[i, j] = accs.get(col, 0)
    data_matrix[i, -2] = avg
    data_matrix[i, -1] = gsm

v0_row = data_matrix[0].copy()
delta_matrix = data_matrix - v0_row[np.newaxis, :]

fig, ax = plt.subplots(figsize=(13, 3.5))
norm = mcolors.Normalize(vmin=20, vmax=85)
im = ax.imshow(data_matrix, cmap="RdYlGn", norm=norm, aspect="auto")
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([n.replace("\n", " ") for n in model_names], fontsize=8.5)
ax.axhline(0.5, color="white", linewidth=2)   # V0 separator
ax.axhline(2.5, color="white", linewidth=1.5) # EGPT separator
ax.axvline(len(TASKS) - 0.5, color="white", linewidth=2)
for i in range(len(model_names)):
    for j in range(len(col_labels)):
        val = data_matrix[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5,
                color="black" if 30 < val < 75 else "white")
plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Accuracy (%)")
ax.set_title("Output-rule ablation: V0/V1/V10/V15/V16 (all 12×1, d=768)", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_heatmap.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_heatmap")

# ── Fig 4: Delta vs V0 heatmap ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 3.5))
max_delta = max(abs(delta_matrix[1:]).max(), 5)
norm2 = mcolors.TwoSlopeNorm(vmin=-max_delta, vcenter=0, vmax=max_delta)
im2 = ax.imshow(delta_matrix, cmap="RdYlGn", norm=norm2, aspect="auto")
ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([n.replace("\n", " ") for n in model_names], fontsize=8.5)
ax.axhline(0.5, color="white", linewidth=2)
ax.axhline(2.5, color="white", linewidth=1.5)
ax.axvline(len(TASKS) - 0.5, color="white", linewidth=2)
for i in range(len(model_names)):
    for j in range(len(col_labels)):
        val = delta_matrix[i, j]
        txt = "±0" if abs(val) < 0.05 else f"{val:+.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5)
plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.01, label="Δ vs V0 GPT (pp)")
ax.set_title("Accuracy delta vs V0 GPT baseline — output-rule ablation", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_delta_heatmap.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_delta_heatmap")

# ── Fig 5: Per-task grouped bar ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 5))
x_tasks = np.arange(len(TASKS))
n = len(model_names)
width = 0.14
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
for i, name in enumerate(model_names):
    accs, _, _, _ = results[name]
    vals = [accs.get(t, 0) for t in TASKS]
    label = name.replace("\n", " ")
    ax.bar(x_tasks + offsets[i], vals, width=width, label=label,
           color=COLORS_LIST[i], alpha=0.88, edgecolor="white", linewidth=0.5)
# Outline V15 bars
i_v15 = model_names.index("V15 EGrad\n6E+6G")
accs_v15, _, _, _ = results["V15 EGrad\n6E+6G"]
for j, t in enumerate(TASKS):
    ax.bar(x_tasks[j] + offsets[i_v15], accs_v15.get(t, 0), width=width,
           color=COLORS_LIST[i_v15], edgecolor="black", linewidth=1.2, zorder=4)
ax.set_xticks(x_tasks)
ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-task: output-rule ablation — V15 (EGrad) outlined for comparison", fontsize=10)
ax.legend(fontsize=8, loc="upper left", ncol=n)
ax.set_ylim(15, 90)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_per_task.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_per_task")

# ── Fig 6: PPL vs avg acc scatter ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    marker = "*" if "V0" in name else "o"
    sz = 220 if "V0" in name else 130
    ax.scatter(ppl, avg, color=COLORS_LIST[i], s=sz, marker=marker,
               zorder=5, edgecolors="black" if "V0" in name else "white", linewidth=1.0)
    label = name.replace("\n", " ")
    ax.annotate(label, (ppl, avg), textcoords="offset points",
                xytext=(7, 3), fontsize=8.5, color=COLORS_LIST[i], fontweight="bold")
ax.set_xlabel("WikiText Word Perplexity ↓", fontsize=10)
ax.set_ylabel("Avg zero-shot accuracy (%)", fontsize=10)
ax.set_title("Accuracy vs Perplexity: output-rule ablation\n(all 12×1, 6E+6G heads unless labelled)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"new_variants_ppl_vs_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved new_variants_ppl_vs_acc")
