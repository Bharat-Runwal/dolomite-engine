"""
Heatmap visualization of the iso-family results table (V0, V1, V2, V10, V11, V12).
Rows = models (grouped by family), columns = tasks + aggregate metrics.
Colour encodes accuracy relative to V0 GPT baseline.
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RESULT_FILES = {
    "V0 GPT\n162M/321MF":       BASE / "v0_gpt_baseline_d768/unsharded/harness_results_2026-04-19T09-04-34.326698.json",
    "V1 EGPT 12×1\n143M/359MF": BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_2026-04-19T15-07-35.156466.json",
    "V10 Mix 12×1\n144M/285MF": BASE / "v10_mixed_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T03-56-05.097549.json",
    "V2 EGPT 6×2\n110M/359MF":  BASE / "v2_6x2_d768/unsharded/harness_results_2026-04-19T09-24-43.813924.json",
    "V11 Mix 6×2\n118M/314MF":  BASE / "v11_mixed_6x2_d768_lr2e3/unsharded/harness_results_2026-04-21T03-36-11.074064.json",
    "V12 GPT 6×2\n120M/321MF":  BASE / "v12_gpt_6x2_d768_lr2e3/unsharded/harness_results_2026-04-21T11-24-43.182905.json",
}

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

# ── Build data matrix: rows=models, cols=tasks + AVG + GSM8K ─────────────────
all_cols = TASKS + ["AVG", "GSM8K"]
col_labels = [TASK_LABELS[t] for t in TASKS] + ["AVG", "GSM8K"]

data_matrix = np.zeros((len(model_names), len(all_cols)))
for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    for j, col in enumerate(TASKS):
        data_matrix[i, j] = accs.get(col, 0)
    data_matrix[i, -2] = avg
    data_matrix[i, -1] = gsm

# ── Delta vs V0 GPT (first row) ───────────────────────────────────────────────
v0_row = data_matrix[0].copy()
delta_matrix = data_matrix - v0_row[np.newaxis, :]

# ── Fig 1: Absolute accuracy heatmap ─────────────────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))

cmap = plt.cm.RdYlGn
norm = mcolors.Normalize(vmin=20, vmax=85)
im = ax.imshow(data_matrix, cmap=cmap, norm=norm, aspect="auto")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([n.replace("\n", " ") for n in model_names], fontsize=8)

# Draw separator between 12×1 group and 6×2 group (3 models in 12x1, 3 in 6x2)
ax.axhline(2.5, color="white", linewidth=3)
ax.axvline(len(TASKS) - 0.5, color="white", linewidth=2)  # separate per-task from aggregates

for i in range(len(model_names)):
    for j in range(len(col_labels)):
        val = data_matrix[i, j]
        ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=7.5,
                color="black" if 30 < val < 75 else "white")

plt.colorbar(im, ax=ax, fraction=0.02, pad=0.01, label="Accuracy (%)")
ax.set_title("Iso-family results: V0/V1/V10 (12×1) and V2/V11/V12 (6×2)", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_heatmap.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_heatmap")

# ── Fig 2: Delta vs V0 GPT (diverging colormap) ───────────────────────────────
fig, ax = plt.subplots(figsize=(13, 4))

max_delta = max(abs(delta_matrix[1:]).max(), 5)
norm2 = mcolors.TwoSlopeNorm(vmin=-max_delta, vcenter=0, vmax=max_delta)
im2 = ax.imshow(delta_matrix, cmap="RdYlGn", norm=norm2, aspect="auto")

ax.set_xticks(range(len(col_labels)))
ax.set_xticklabels(col_labels, rotation=30, ha="right", fontsize=9)
ax.set_yticks(range(len(model_names)))
ax.set_yticklabels([n.replace("\n", " ") for n in model_names], fontsize=8)
ax.axhline(2.5, color="white", linewidth=3)
ax.axvline(len(TASKS) - 0.5, color="white", linewidth=2)
ax.set_title("Accuracy delta vs V0 GPT baseline", fontsize=11)

for i in range(len(model_names)):
    for j in range(len(col_labels)):
        val = delta_matrix[i, j]
        txt = "±0" if abs(val) < 0.05 else f"{val:+.1f}"
        ax.text(j, i, txt, ha="center", va="center", fontsize=7.5)

plt.colorbar(im2, ax=ax, fraction=0.02, pad=0.01, label="Δ vs V0 GPT (pp)")
ax.set_title("Accuracy delta vs V0 GPT baseline", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_delta_heatmap.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_delta_heatmap")

# ── Fig 3: PPL + accuracy scatter (iso-family) ────────────────────────────────
GROUP_MARKER = {"12×1": "o", "6×2": "s"}
GROUP_COLOR  = {
    "V0 GPT\n162M/321MF":       "#1f77b4",
    "V1 EGPT 12×1\n143M/359MF": "#ff7f0e",
    "V10 Mix 12×1\n144M/285MF": "#9467bd",
    "V2 EGPT 6×2\n110M/359MF":  "#2ca02c",
    "V11 Mix 6×2\n118M/314MF":  "#8c564b",
    "V12 GPT 6×2\n120M/321MF":  "#d62728",
}

fig, ax = plt.subplots(figsize=(7, 5))
for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    marker = "s" if "6×2" in name else "o"
    ax.scatter(ppl, avg, color=GROUP_COLOR[name], s=120, marker=marker,
               zorder=5, edgecolors="white", linewidth=0.8)
    label = name.split("\n")[0]
    ax.annotate(label, (ppl, avg), textcoords="offset points",
                xytext=(6, 3), fontsize=8.5, color=GROUP_COLOR[name])

ax.set_xlabel("WikiText Word Perplexity ↓", fontsize=10)
ax.set_ylabel("Avg zero-shot accuracy (%)", fontsize=10)
ax.set_title("Accuracy vs Perplexity: iso-family\n(circles=12×1, squares=6×2)", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_ppl_vs_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_ppl_vs_acc")

# ── Fig 4: Merged all-5 per-task grouped bar chart ────────────────────────────
TASKS_ORDERED = ["arc_challenge","arc_easy","boolq","copa","hellaswag",
                 "openbookqa","piqa","sciq","winogrande","mmlu"]
TASK_LABELS_SHORT = ["ARC-C","ARC-E","BoolQ","COPA","HellaSwag","ObQA","PIQA","SciQ","Wino.","MMLU"]
COLORS_6 = ["#1f77b4","#ff7f0e","#9467bd","#2ca02c","#8c564b","#d62728"]

fig, ax = plt.subplots(figsize=(16, 5))
x = np.arange(len(TASKS_ORDERED))
n = len(model_names)
width = 0.12
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    vals = [accs.get(t, 0) for t in TASKS_ORDERED]
    label = name.split("\n")[0]
    ax.bar(x + offsets[i], vals, width=width, label=label,
           color=COLORS_6[i], alpha=0.88,
           edgecolor="white", linewidth=0.5)

# Highlight V11 bars with a thick border to make comparison easy
i_v11 = model_names.index("V11 Mix 6×2\n118M/314MF")
accs_v11, _, _, _ = results["V11 Mix 6×2\n118M/314MF"]
for j, t in enumerate(TASKS_ORDERED):
    ax.bar(x[j] + offsets[i_v11], accs_v11.get(t, 0), width=width,
           color=COLORS_6[i_v11], edgecolor="black", linewidth=1.2, zorder=4)

ax.set_xticks(x)
ax.set_xticklabels(TASK_LABELS_SHORT, fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("All-6 per-task comparison — V11 outlined for easy comparison with V0/V1/V12", fontsize=10)
ax.legend(fontsize=8, loc="upper left", ncol=6)
ax.set_ylim(15, 90)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_all5_per_task.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_all5_per_task")

# ── Fig 5: Merged PPL vs acc with connecting lines per family ──────────────────
fig, ax = plt.subplots(figsize=(8, 5.5))

# Draw connecting lines within each family (dashed arrows)
family_12x1 = ["V1 EGPT 12×1\n143M/359MF", "V10 Mix 12×1\n144M/285MF"]
family_6x2  = ["V2 EGPT 6×2\n110M/359MF",  "V11 Mix 6×2\n118M/314MF", "V12 GPT 6×2\n120M/321MF"]

for fam, ls in [(family_12x1, "--"), (family_6x2, ":")]:
    ppls = [results[n][2] for n in fam]
    avgs = [results[n][1] for n in fam]
    ax.plot(ppls, avgs, color="gray", linestyle=ls, linewidth=1.2, zorder=1)

for i, name in enumerate(model_names):
    accs, avg, ppl, gsm = results[name]
    marker = "s" if "6×2" in name else "o"
    # V0 gets a star
    if "V0" in name:
        marker = "*"
        sz = 220
    else:
        sz = 130
    ax.scatter(ppl, avg, color=COLORS_6[i], s=sz, marker=marker,
               zorder=5, edgecolors="white" if "V0" not in name else "black",
               linewidth=1.0)
    label = name.split("\n")[0]
    ax.annotate(label, (ppl, avg), textcoords="offset points",
                xytext=(7, 3), fontsize=8.5, color=COLORS_6[i], fontweight="bold")

# Add explanatory annotation for V11 vs V0
ppl_v11, avg_v11 = results["V11 Mix 6×2\n118M/314MF"][2], results["V11 Mix 6×2\n118M/314MF"][1]
ppl_v0,  avg_v0  = results["V0 GPT\n162M/321MF"][2],      results["V0 GPT\n162M/321MF"][1]
ax.annotate("", xy=(ppl_v11, avg_v11), xytext=(ppl_v0, avg_v0),
            arrowprops=dict(arrowstyle="->", color="#8c564b", lw=1.5,
                            connectionstyle="arc3,rad=0.15"))

ax.set_xlabel("WikiText Word Perplexity ↓", fontsize=10)
ax.set_ylabel("Avg zero-shot accuracy (%)", fontsize=10)
ax.set_title("Accuracy vs Perplexity — all models\n★=V0 baseline  dashes=EGPT→Mixed within family", fontsize=10)
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_all5_ppl_vs_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_all5_ppl_vs_acc")
