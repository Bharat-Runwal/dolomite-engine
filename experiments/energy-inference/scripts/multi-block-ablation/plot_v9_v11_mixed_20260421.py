"""Plot V9 (GPT 354M), V10 (Mixed 12x1 144M), V11 (Mixed 6x2 118M) results vs V0/V1 baselines."""

import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np

BASE = Path(__file__).resolve().parents[2] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

# ── load results ──────────────────────────────────────────────────────────────
RESULT_FILES = {
    "V0 GPT 143M":      BASE / "v0_gpt_baseline_d768/unsharded/harness_results_2026-04-19T09-04-34.326698.json",
    "V1 EGPT 143M":     BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_2026-04-19T15-07-35.156466.json",
    "V1 400M EGPT":     BASE / "v1_400m_d1024_lr7e4/unsharded/harness_results_2026-04-20T04-28-32.509768.json",
    "V9 GPT 354M":      BASE / "v9_gpt_baseline_d1024_lr1e3/unsharded/harness_results_2026-04-21T02-13-24.774032.json",
    "V10 Mixed 12×1":   BASE / "v10_mixed_12x1_d768_lr2e3/unsharded/harness_results_2026-04-21T03-56-05.097549.json",
    "V11 Mixed 6×2":    BASE / "v11_mixed_6x2_d768_lr2e3/unsharded/harness_results_2026-04-21T03-36-11.074064.json",
}

TASK_KEYS = {
    "arc_challenge": "acc_norm,none",
    "arc_easy":      "acc,none",
    "boolq":         "acc,none",
    "copa":          "acc,none",
    "hellaswag":     "acc_norm,none",
    "openbookqa":    "acc_norm,none",
    "piqa":          "acc_norm,none",
    "sciq":          "acc,none",
    "winogrande":    "acc,none",
    "mmlu":          "acc,none",
}
TASK_LABELS = {
    "arc_challenge": "ARC-C", "arc_easy": "ARC-E", "boolq": "BoolQ",
    "copa": "COPA", "hellaswag": "HellaSwag", "openbookqa": "ObQA",
    "piqa": "PIQA", "sciq": "SciQ", "winogrande": "Wino.", "mmlu": "MMLU",
}

COLORS = {
    "V0 GPT 143M":    "#1f77b4",
    "V1 EGPT 143M":   "#ff7f0e",
    "V1 400M EGPT":   "#d62728",
    "V9 GPT 354M":    "#2ca02c",
    "V10 Mixed 12×1": "#9467bd",
    "V11 Mixed 6×2":  "#8c564b",
}
LINESTYLES = {
    "V0 GPT 143M":    "-",
    "V1 EGPT 143M":   "--",
    "V1 400M EGPT":   "-",
    "V9 GPT 354M":    "--",
    "V10 Mixed 12×1": "-.",
    "V11 Mixed 6×2":  ":",
}

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
    return accs, avg, ppl

results = {}
for name, path in RESULT_FILES.items():
    results[name] = load(path)

# ── Fig 1: avg accuracy bar chart ─────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
names = list(results.keys())
avgs = [results[n][1] * 100 for n in names]
colors = [COLORS[n] for n in names]
bars = ax.bar(range(len(names)), avgs, color=colors, width=0.6, edgecolor="white")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Avg zero-shot accuracy (%)")
ax.set_title("V9–V11 vs Baselines: Average Accuracy (10 tasks)")
ax.set_ylim(42, 52)
for bar, v in zip(bars, avgs):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.1, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax.axhline(avgs[0], color=COLORS["V0 GPT 143M"], linestyle="--", linewidth=0.8, alpha=0.5, label="V0 baseline")
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"v9_v11_avg_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved v9_v11_avg_acc")

# ── Fig 2: per-task breakdown (radar-style bar) ───────────────────────────────
tasks = list(TASK_KEYS.keys())
task_labels = [TASK_LABELS[t] for t in tasks]
x = np.arange(len(tasks))
width = 0.13
n = len(results)
offsets = np.linspace(-(n-1)/2, (n-1)/2, n) * width

fig, ax = plt.subplots(figsize=(12, 5))
for i, (name, (accs, avg, ppl)) in enumerate(results.items()):
    vals = [accs.get(t, 0) * 100 for t in tasks]
    ax.bar(x + offsets[i], vals, width=width, label=name, color=COLORS[name], alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels(task_labels, fontsize=9)
ax.set_ylabel("Accuracy (%)")
ax.set_title("Per-task breakdown: V9–V11 vs Baselines")
ax.legend(fontsize=7, loc="upper left", ncol=2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"v9_v11_per_task.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved v9_v11_per_task")

# ── Fig 3: PPL comparison ─────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 4))
ppls = [results[n][2] for n in names]
bars = ax.bar(range(len(names)), ppls, color=colors, width=0.6, edgecolor="white")
ax.set_xticks(range(len(names)))
ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
ax.set_ylabel("Wikitext Word Perplexity ↓")
ax.set_title("V9–V11 vs Baselines: WikiText Perplexity")
for bar, v in zip(bars, ppls):
    ax.text(bar.get_x() + bar.get_width()/2, v + 0.3, f"{v:.1f}", ha="center", va="bottom", fontsize=8)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"v9_v11_wikitext_ppl.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved v9_v11_wikitext_ppl")

# ── Fig 4: params vs avg accuracy scatter ─────────────────────────────────────
PARAMS = {
    "V0 GPT 143M":    143,
    "V1 EGPT 143M":   143,
    "V1 400M EGPT":   354,
    "V9 GPT 354M":    354,
    "V10 Mixed 12×1": 144,
    "V11 Mixed 6×2":  118,
}
fig, ax = plt.subplots(figsize=(6, 5))
for name, (accs, avg, ppl) in results.items():
    ax.scatter(PARAMS[name], avg * 100, color=COLORS[name], s=100, zorder=5)
    ax.annotate(name, (PARAMS[name], avg * 100), textcoords="offset points",
                xytext=(5, 3), fontsize=8, color=COLORS[name])
ax.set_xlabel("Parameters (M)")
ax.set_ylabel("Avg zero-shot accuracy (%)")
ax.set_title("Accuracy vs Model Size")
ax.grid(alpha=0.3)
fig.tight_layout()
for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"v9_v11_params_vs_acc.{ext}", dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved v9_v11_params_vs_acc")

print("\nSummary:")
print(f"{'Model':<22} {'Params':>7} {'Avg Acc':>8} {'PPL':>7}")
for name, (accs, avg, ppl) in results.items():
    print(f"{name:<22} {PARAMS[name]:>7}M {avg*100:>7.1f}% {ppl:>7.2f}" if ppl else f"{name:<22} {PARAMS[name]:>7}M {avg*100:>7.1f}%   N/A")
