"""
Grouped bar charts for model-set comparisons.
Generates:
  - new_variants_bar.{pdf,png}  — V0/V1/V10/V15/V16 per-task + summary
  - mixed_iso_bar.{pdf,png}     — V0/V1/V10/V2/V11/V12 per-task + summary
"""
import json, glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
PLOTS_DIR = BASE / "plots"

BENCH = [
    ("arc_challenge", "acc_norm,none", "ARC-C"),
    ("arc_easy",      "acc_norm,none", "ARC-E"),
    ("boolq",         "acc,none",      "BoolQ"),
    ("copa",          "acc,none",      "COPA"),
    ("hellaswag",     "acc_norm,none", "HellaS"),
    ("mmlu",          "acc,none",      "MMLU"),
    ("openbookqa",    "acc_norm,none", "OBQA"),
    ("piqa",          "acc_norm,none", "PIQA"),
    ("sciq",          "acc_norm,none", "SciQ"),
    ("winogrande",    "acc,none",      "Wino"),
]
TASK_NAMES = [b[2] for b in BENCH]


def load(subdir):
    files = sorted(glob.glob(str(BASE / subdir / "unsharded" / "harness_results_*.json")))
    if not files:
        return None, None
    d = json.load(open(files[-1]))["results"]
    ppl = d.get("wikitext", {}).get("word_perplexity,none")
    scores = [d.get(t, {}).get(m) for t, m, _ in BENCH]
    return scores, ppl


def grouped_bar_chart(models, filename, title, show_ppl=True, show_avg=True):
    """
    models: list of (label, color, scores_list, ppl)
    scores_list: 10 floats aligned to BENCH
    """
    n_models = len(models)
    n_tasks = len(TASK_NAMES)

    # ── layout: task bars + optional PPL/Avg summary panel ───────────────────
    if show_ppl or show_avg:
        fig, (ax_tasks, ax_sum) = plt.subplots(
            1, 2, figsize=(14, 4.5),
            gridspec_kw={"width_ratios": [n_tasks, 2 if (show_ppl and show_avg) else 1]},
        )
    else:
        fig, ax_tasks = plt.subplots(figsize=(12, 4.5))
        ax_sum = None

    # ── per-task grouped bars ─────────────────────────────────────────────────
    width = 0.75 / n_models
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width
    x = np.arange(n_tasks)

    for (lbl, color, scores, _), off in zip(models, offsets):
        vals = [s if s is not None else 0 for s in scores]
        bars = ax_tasks.bar(x + off, vals, width=width * 0.92,
                            color=color, alpha=0.85, label=lbl)

    ax_tasks.set_xticks(x)
    ax_tasks.set_xticklabels(TASK_NAMES, fontsize=9)
    ax_tasks.set_ylabel("Accuracy")
    ax_tasks.set_ylim(0, 0.85)
    ax_tasks.set_title(f"{title} — per task")
    ax_tasks.legend(fontsize=8, loc="upper left", ncol=n_models, framealpha=0.8)
    ax_tasks.grid(axis="y", alpha=0.3)
    ax_tasks.axhline(0.25, color="#aaa", ls=":", lw=0.8, label="chance")

    # ── summary panel (PPL + avg acc) ─────────────────────────────────────────
    if ax_sum is not None:
        sum_metrics = []
        if show_avg:
            avgs = [np.mean([s for s in m[2] if s is not None]) for m in models]
            sum_metrics.append(("Avg Acc", avgs, 0.35, 0.60))
        if show_ppl:
            # normalise PPL so it fits on same axis: scale to ~0-1 range
            ppls = [m[3] for m in models]
            ppl_min, ppl_max = min(ppls), max(ppls)

        # two sub-axes stacked
        gs = ax_sum.get_subplotspec().subgridspec(2 if (show_avg and show_ppl) else 1, 1, hspace=0.5)
        fig.delaxes(ax_sum)
        sub_axes = [fig.add_subplot(sp) for sp in gs]

        if show_avg:
            axs = sub_axes[0]
            avgs = [np.mean([s for s in m[2] if s is not None]) for m in models]
            bars = axs.bar(range(n_models), avgs,
                           color=[m[1] for m in models], alpha=0.85, width=0.6)
            for bar, v in zip(bars, avgs):
                axs.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                         f"{v:.3f}", ha="center", va="bottom", fontsize=7.5)
            axs.set_xticks(range(n_models))
            axs.set_xticklabels([m[0] for m in models], fontsize=7.5, rotation=20, ha="right")
            axs.set_ylabel("Avg Acc ↑", fontsize=8)
            axs.set_ylim(min(avgs) - 0.02, max(avgs) + 0.025)
            axs.grid(axis="y", alpha=0.3)

        if show_ppl:
            axp = sub_axes[-1]
            ppls = [m[3] for m in models]
            bars = axp.bar(range(n_models), ppls,
                           color=[m[1] for m in models], alpha=0.85, width=0.6)
            for bar, v in zip(bars, ppls):
                axp.text(bar.get_x() + bar.get_width() / 2, v + 0.3,
                         f"{v:.1f}", ha="center", va="bottom", fontsize=7.5)
            axp.set_xticks(range(n_models))
            axp.set_xticklabels([m[0] for m in models], fontsize=7.5, rotation=20, ha="right")
            axp.set_ylabel("PPL ↓", fontsize=8)
            axp.set_ylim(0, max(ppls) * 1.18)
            axp.grid(axis="y", alpha=0.3)

    fig.suptitle(title, fontsize=11, y=1.01)
    fig.tight_layout()
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS_DIR / f"{filename}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {filename}")


# ── Output-rule ablation: V0, V1, V10, V15, V16 ───────────────────────────────
new_vars_models = [
    ("V0 GPT",    "#2196F3", *load("v0_gpt_baseline_d768")),
    ("V1 EGPT",   "#FF9800", *load("v1_12x1_d768_lr2e3")),
    ("V10 Mixed", "#9C27B0", *load("v10_mixed_12x1_d768_lr2e3")),
    ("V15 EGrad", "#F44336", *load("v15_energy_grad_mixed_12x1_d768_lr2e3")),
    ("V16 EDesc", "#4CAF50", *load("v16_mixed_energy_descent_12x1_d768_lr2e3")),
]
grouped_bar_chart(
    new_vars_models,
    "new_variants_bar",
    "Output-rule ablation: V0 / V1 / V10 Mixed / V15 EGrad / V16 EDesc",
)

# ── Iso-family mixed heads: V0, V1, V10, V2, V11, V12 ─────────────────────────
mixed_iso_models = [
    ("V0 GPT 12×1",  "#2196F3", *load("v0_gpt_baseline_d768")),
    ("V1 EGPT 12×1", "#FF9800", *load("v1_12x1_d768_lr2e3")),
    ("V10 Mix 12×1", "#9C27B0", *load("v10_mixed_12x1_d768_lr2e3")),
    ("V2 EGPT 6×2",  "#FFB74D", *load("v2_6x2_d768")),
    ("V11 Mix 6×2",  "#CE93D8", *load("v11_mixed_6x2_d768_lr2e3")),
    ("V12 GPT 6×2",  "#90CAF9", *load("v12_gpt_6x2_d768_lr2e3")),
]
grouped_bar_chart(
    mixed_iso_models,
    "mixed_iso_bar",
    "Iso-family comparison: non-recurrent 12×1 vs recurrent 6×2",
)
