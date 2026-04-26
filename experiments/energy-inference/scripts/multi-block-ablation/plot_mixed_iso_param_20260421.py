"""
Focused iso-family comparison: Mixed vs pure architectures.

Group 1 (non-recurrent 12×1, ~143-162M): V0 GPT, V1 EGPT, V10 Mixed
Group 2 (recurrent 6×2, ~110-118M):       V2 EGPT, V11 Mixed

FLOPs/token (forward pass, s=4096, s-dependent attention included):
  V0  321 MFLOPs  V1  359  V2  359  V10  285  V11  314
"""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

RESULT_FILES = {
    "V0 GPT\n162M":       BASE / "v0_gpt_baseline_d768/unsharded/harness_results_2026-04-19T09-04-34.326698.json",
    "V1 EGPT\n12×1 143M": BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_2026-04-19T15-07-35.156466.json",
    "V10 Mixed\n12×1 144M": BASE / "v10_mixed_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T03-56-05.097549.json",
    "V2 EGPT\n6×2 110M":  BASE / "v2_6x2_d768/unsharded/harness_results_2026-04-19T09-24-43.813924.json",
    "V11 Mixed\n6×2 118M": BASE / "v11_mixed_6x2_d768_lr2e3/unsharded/harness_results_2026-04-21T03-36-11.074064.json",
    "V12 GPT\n6×2 120M":  BASE / "v12_gpt_6x2_d768_lr2e3/unsharded/harness_results_2026-04-21T11-24-43.182905.json",
}

PARAMS = {"V0 GPT\n162M": 162, "V1 EGPT\n12×1 143M": 143, "V10 Mixed\n12×1 144M": 144,
          "V2 EGPT\n6×2 110M": 110, "V11 Mixed\n6×2 118M": 118, "V12 GPT\n6×2 120M": 120}
FLOPS  = {"V0 GPT\n162M": 321, "V1 EGPT\n12×1 143M": 359, "V10 Mixed\n12×1 144M": 285,
          "V2 EGPT\n6×2 110M": 359, "V11 Mixed\n6×2 118M": 314, "V12 GPT\n6×2 120M": 321}

TASK_KEYS = {
    "arc_challenge": "acc_norm,none", "arc_easy": "acc,none", "boolq": "acc,none",
    "copa": "acc,none", "hellaswag": "acc_norm,none", "openbookqa": "acc_norm,none",
    "piqa": "acc_norm,none", "sciq": "acc,none", "winogrande": "acc,none", "mmlu": "acc,none",
}
TASK_LABELS = {
    "arc_challenge": "ARC-C", "arc_easy": "ARC-E", "boolq": "BoolQ", "copa": "COPA",
    "hellaswag": "HellaSwag", "openbookqa": "ObQA", "piqa": "PIQA", "sciq": "SciQ",
    "winogrande": "Wino.", "mmlu": "MMLU",
}

COLORS = {
    "V0 GPT\n162M":        "#1f77b4",
    "V1 EGPT\n12×1 143M":  "#ff7f0e",
    "V10 Mixed\n12×1 144M":"#9467bd",
    "V2 EGPT\n6×2 110M":   "#2ca02c",
    "V11 Mixed\n6×2 118M": "#8c564b",
    "V12 GPT\n6×2 120M":   "#d62728",
}
GROUP_COLORS = ["#1f77b4", "#ff7f0e", "#9467bd", "#2ca02c", "#8c564b", "#d62728"]

def load(path):
    with open(path) as f:
        data = json.load(f)
    accs = {}
    for task, key in TASK_KEYS.items():
        if task in data["results"]:
            v = data["results"][task].get(key, data["results"][task].get("acc,none"))
            if v is not None:
                accs[task] = v
    avg = sum(accs.values()) / len(accs) if accs else 0
    ppl = data["results"].get("wikitext", {}).get("word_perplexity,none")
    gsm = data["results"].get("gsm8k", {})
    gsm_score = gsm.get("exact_match,flexible-extract", gsm.get("exact_match,none", 0))
    return accs, avg, ppl, gsm_score

results = {name: load(path) for name, path in RESULT_FILES.items()}

names = list(results.keys())
group1 = ["V0 GPT\n162M", "V1 EGPT\n12×1 143M", "V10 Mixed\n12×1 144M"]
group2 = ["V2 EGPT\n6×2 110M", "V11 Mixed\n6×2 118M", "V12 GPT\n6×2 120M"]

# ── Fig 1: Grouped bar chart (avg accuracy + PPL), two groups ─────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, group, title in [(ax1, group1, "Non-recurrent (12 block applications)"),
                          (ax2, group2, "Recurrent 6×2 (12 block applications)")]:
    avgs = [results[n][1] * 100 for n in group]
    colors = [COLORS[n] for n in group]
    x = np.arange(len(group))
    bars = ax.bar(x, avgs, color=colors, width=0.5, edgecolor="white", linewidth=1.2)
    ax.set_xticks(x)
    ax.set_xticklabels([n.replace("\n", "\n") for n in group], fontsize=9)
    ax.set_ylabel("Avg zero-shot accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(42, 51)
    for bar, v, n in zip(bars, avgs, group):
        p = PARAMS[n]
        f = FLOPS[n]
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.1,
                f"{v:.1f}%\n{p}M/{f}M", ha="center", va="bottom", fontsize=7.5)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("Avg Zero-Shot Accuracy: Mixed vs Pure Architectures (iso-family)", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_iso_avg_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_iso_avg_acc")

# ── Fig 2: PPL grouped bar ────────────────────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for ax, group, title in [(ax1, group1, "Non-recurrent 12×1"), (ax2, group2, "Recurrent 6×2")]:
    ppls = [results[n][2] for n in group]
    colors = [COLORS[n] for n in group]
    x = np.arange(len(group))
    bars = ax.bar(x, ppls, color=colors, width=0.5, edgecolor="white")
    ax.set_xticks(x)
    ax.set_xticklabels(group, fontsize=9)
    ax.set_ylabel("WikiText Word Perplexity ↓")
    ax.set_title(title, fontsize=10)
    for bar, v in zip(bars, ppls):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.4, f"{v:.1f}", ha="center", va="bottom", fontsize=9)
    ax.grid(axis="y", alpha=0.3)

fig.suptitle("WikiText Perplexity: Mixed vs Pure Architectures (iso-family)", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_iso_ppl.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_iso_ppl")

# ── Fig 3: FLOPs/token vs accuracy scatter ────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))
for name, (accs, avg, ppl, gsm) in results.items():
    ax.scatter(FLOPS[name], avg * 100, color=COLORS[name], s=PARAMS[name] * 2,
               zorder=5, edgecolors="white", linewidth=0.8)
    label = name.replace("\n", " ")
    ax.annotate(label, (FLOPS[name], avg * 100), textcoords="offset points",
                xytext=(7, 2), fontsize=8.5, color=COLORS[name])

# draw group hulls
ax.set_xlabel("FLOPs/token (MFLOPs, forward pass, s=4096)", fontsize=10)
ax.set_ylabel("Avg zero-shot accuracy (%)", fontsize=10)
ax.set_title("Accuracy vs Compute: Mixed heads are strictly more efficient\nthan pure EGPT at same architecture family", fontsize=10)
ax.set_xlim(250, 400)
ax.set_ylim(43.5, 49)
ax.grid(alpha=0.3)

# Annotate efficiency arrows
ax.annotate("", xy=(FLOPS["V10 Mixed\n12×1 144M"], results["V10 Mixed\n12×1 144M"][1]*100),
            xytext=(FLOPS["V1 EGPT\n12×1 143M"], results["V1 EGPT\n12×1 143M"][1]*100),
            arrowprops=dict(arrowstyle="->", color="purple", lw=1.5))
ax.annotate("", xy=(FLOPS["V11 Mixed\n6×2 118M"], results["V11 Mixed\n6×2 118M"][1]*100),
            xytext=(FLOPS["V2 EGPT\n6×2 110M"], results["V2 EGPT\n6×2 110M"][1]*100),
            arrowprops=dict(arrowstyle="->", color="saddlebrown", lw=1.5))

fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_flops_vs_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_flops_vs_acc")

# ── Fig 4: Per-task detail for group 1 (12x1 family) ─────────────────────────
tasks_ordered = ["arc_challenge","arc_easy","boolq","copa","hellaswag",
                 "openbookqa","piqa","sciq","winogrande","mmlu"]
task_labels = [TASK_LABELS[t] for t in tasks_ordered]

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
for ax, group, title in [(axes[0], group1, "12×1 family (~143–162M)"),
                          (axes[1], group2, "6×2 recurrent family (~110–120M)")]:
    x = np.arange(len(tasks_ordered))
    n = len(group)
    width = 0.22
    offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width
    for i, name in enumerate(group):
        accs, _, _, _ = results[name]
        vals = [accs.get(t, 0) * 100 for t in tasks_ordered]
        label = name.replace("\n", " ")
        ax.bar(x + offsets[i], vals, width=width, label=label, color=COLORS[name], alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, rotation=25, ha="right", fontsize=8.5)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    ax.set_ylim(20, 90)

fig.suptitle("Per-task accuracy: Mixed vs pure architectures", fontsize=11)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"mixed_per_task.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved mixed_per_task")

# ── Summary table ─────────────────────────────────────────────────────────────
print("\n=== Summary ===")
print(f"{'Model':<24} {'Params':>6} {'FLOPs/tok':>10} {'Avg Acc':>8} {'PPL':>7} {'GSM8K':>7}")
for name in names:
    accs, avg, ppl, gsm = results[name]
    label = name.replace("\n", " ")
    print(f"{label:<24} {PARAMS[name]:>5}M {FLOPS[name]:>8}M  {avg*100:>7.1f}%  {ppl:>6.2f}  {gsm*100:>6.2f}%")

print("\n=== Group 1 delta (V10 vs V0, V1) ===")
k_v10 = "V10 Mixed\n12×1 144M"
for ref in ["V0 GPT\n162M", "V1 EGPT\n12×1 143M"]:
    ref_avg, ref_ppl = results[ref][1], results[ref][2]
    v10_avg, v10_ppl = results[k_v10][1], results[k_v10][2]
    print(f"V10 vs {ref.replace(chr(10),' ')}: Δacc={(v10_avg-ref_avg)*100:+.2f}pp  ΔPPL={v10_ppl-ref_ppl:+.2f}  ΔFLOPs={FLOPS[k_v10]-FLOPS[ref]:+d}M")

print("\n=== Group 2 delta (V11 vs V2) ===")
k_v11, k_v2 = "V11 Mixed\n6×2 118M", "V2 EGPT\n6×2 110M"
v11_avg, v11_ppl = results[k_v11][1], results[k_v11][2]
v2_avg,  v2_ppl  = results[k_v2][1],  results[k_v2][2]
print(f"V11 vs V2: Δacc={(v11_avg-v2_avg)*100:+.2f}pp  ΔPPL={v11_ppl-v2_ppl:+.2f}  ΔFLOPs={FLOPS[k_v11]-FLOPS[k_v2]:+d}M")
