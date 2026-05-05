"""
Plot cosreg performance comparison: loss + benchmarks for V0/V1/V39/V48 (176M) and V9/V1-400M/V40/V50 (400M).
"""
import json
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

_HERE = str(Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
from make_tables import AVG_TASKS_10, TASK_DISPLAY  # canonical metrics

PLOTS_DIR = Path(__file__).parent.parent.parent / "results/multi-block-ablation/plots"
PLOTS_DIR.mkdir(exist_ok=True)

BASE = Path(__file__).parent.parent.parent / "results/multi-block-ablation"

def load_metrics(json_path):
    """Load harness results and extract per-task accuracy using canonical metrics."""
    if not json_path.exists():
        return {}
    with open(json_path) as f:
        d = json.load(f)
    results = d.get("results", {})
    out = {}
    for task, metric in AVG_TASKS_10:
        v = results.get(task)
        if v is not None and metric in v:
            out[task] = v[metric]
    return out

# ---------- data ----------
models_176m = [
    {
        "label": "V0\nGPT",
        "loss": 3.121,
        "color": "#4C72B0",
        "hatch": "",
        "metrics": load_metrics(BASE / "v0_gpt_baseline_d768/unsharded/harness_results_2026-04-19T08-15-02.012321.json"),
    },
    {
        "label": "V1\nEGPT",
        "loss": 3.271,
        "color": "#DD8452",
        "hatch": "",
        "metrics": load_metrics(BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_2026-04-19T09-26-11.882459.json"),
    },
    {
        "label": "V39\nact-cosreg\n(λ=0.01)",
        "loss": 3.273,
        "color": "#DD8452",
        "hatch": "///",
        "metrics": load_metrics(BASE / "v39_egpt_cosreg_12x1_d768_lr2e3/unsharded/harness_results_2026-04-24T08-13-05.481033.json"),
    },
    {
        "label": "V48\nwt-cosreg\n(λ=0.01)",
        "loss": 3.287,
        "color": "#DD8452",
        "hatch": "xxx",
        "metrics": {},  # pending
    },
]

models_400m = [
    {
        "label": "V9\nGPT",
        "loss": 2.931,
        "color": "#4C72B0",
        "hatch": "",
        "metrics": load_metrics(BASE / "v9_gpt_baseline_d1024_lr1e3/unsharded/harness_results_2026-04-21T02-13-24.774032.json"),
    },
    {
        "label": "V1\nEGPT",
        "loss": 3.117,
        "color": "#DD8452",
        "hatch": "",
        "metrics": load_metrics(BASE / "v1_400m_d1024_lr7e4/unsharded/harness_results_2026-04-20T04-28-32.509768.json"),
    },
    {
        "label": "V40\nact-cosreg\n(λ=0.01)*",
        "loss": 3.225,  # last known at step 15900 (incomplete)
        "color": "#DD8452",
        "hatch": "///",
        "metrics": {},  # pending
        "incomplete": True,
    },
    {
        "label": "V50\nwt-cosreg\n(λ=0.01)*",
        "loss": 3.259,  # at step 15000 (running)
        "color": "#DD8452",
        "hatch": "xxx",
        "metrics": {},  # pending
        "incomplete": True,
    },
]

BENCH_TASKS = [t for t, _ in AVG_TASKS_10]

def avg_bench(m):
    """Mean over the canonical 10-task set; require all tasks present."""
    vals = [m[k] for k in BENCH_TASKS if k in m]
    if len(vals) != len(BENCH_TASKS):
        return None
    return sum(vals) / len(vals)

# ---------- figure ----------
fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
fig.suptitle("Cosine Similarity Regularizer: Performance Impact (λ=0.01, 30k steps)", fontsize=12, fontweight="bold")

# --- Panel 1: loss 176M ---
ax = axes[0]
ax.set_title("Training Loss — 176M", fontsize=10)
labels = [m["label"] for m in models_176m]
losses = [m["loss"] for m in models_176m]
colors = [m["color"] for m in models_176m]
hatches = [m["hatch"] for m in models_176m]
x = np.arange(len(labels))
bars = ax.bar(x, losses, color=colors, hatch=hatches, edgecolor="white", width=0.55)
for bar, loss in zip(bars, losses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{loss:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("Train loss (lower = better)")
ax.set_ylim(3.05, 3.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel 2: loss 400M ---
ax = axes[1]
ax.set_title("Training Loss — 400M  (* = incomplete)", fontsize=10)
labels_4 = [m["label"] for m in models_400m]
losses_4 = [m["loss"] for m in models_400m]
colors_4 = [m["color"] for m in models_400m]
hatches_4 = [m["hatch"] for m in models_400m]
incomplete_4 = [m.get("incomplete", False) for m in models_400m]
x = np.arange(len(labels_4))
bars = ax.bar(x, losses_4, color=colors_4, hatch=hatches_4, edgecolor="white", width=0.55, alpha=1.0)
for i, (bar, loss, inc) in enumerate(zip(bars, losses_4, incomplete_4)):
    if inc:
        bar.set_alpha(0.55)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{loss:.3f}",
            ha="center", va="bottom", fontsize=8, alpha=0.6 if inc else 1.0)
ax.set_xticks(x)
ax.set_xticklabels(labels_4, fontsize=8)
ax.set_ylabel("Train loss (lower = better)")
ax.set_ylim(2.85, 3.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# --- Panel 3: benchmark avg zero-shot (available models only) ---
ax = axes[2]
ax.set_title("Avg Zero-Shot Accuracy (7 tasks)", fontsize=10)
bench_models = [m for m in (models_176m + models_400m) if avg_bench(m["metrics"]) is not None]
labels_b = [m["label"].replace("\n", " ") for m in bench_models]
accs_b = [avg_bench(m["metrics"]) for m in bench_models]
colors_b = [m["color"] for m in bench_models]
hatches_b = [m["hatch"] for m in bench_models]
# add scale suffix
scale_suffix = ["176M"] * 4 + ["400M"] * 2
labels_b2 = [
    f"V0-GPT 176M", "V1-EGPT 176M", "V39-actreg 176M",
    "V9-GPT 400M", "V1-EGPT 400M"
]
x = np.arange(len(bench_models))
bars = ax.bar(x, accs_b, color=colors_b, hatch=hatches_b, edgecolor="white", width=0.6)
for bar, acc in zip(bars, accs_b):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002, f"{acc:.3f}",
            ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(labels_b2, fontsize=7, rotation=25, ha="right")
ax.set_ylabel("Avg accuracy (higher = better)")
ax.set_ylim(0.42, 0.56)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend patches
gpt_patch = mpatches.Patch(color="#4C72B0", label="GPT baseline")
egpt_patch = mpatches.Patch(color="#DD8452", label="EGPT")
act_patch = mpatches.Patch(color="#DD8452", hatch="///", label="EGPT + act-cosreg")
wt_patch = mpatches.Patch(color="#DD8452", hatch="xxx", label="EGPT + wt-cosreg")
fig.legend(handles=[gpt_patch, egpt_patch, act_patch, wt_patch],
           loc="lower center", ncol=4, fontsize=8, frameon=False,
           bbox_to_anchor=(0.5, -0.02))

plt.tight_layout(rect=[0, 0.06, 1, 1])

for ext in ["png", "pdf"]:
    fig.savefig(PLOTS_DIR / f"cosreg_performance_comparison.{ext}", dpi=150, bbox_inches="tight")

print(f"Saved to {PLOTS_DIR}/cosreg_performance_comparison.{{png,pdf}}")
