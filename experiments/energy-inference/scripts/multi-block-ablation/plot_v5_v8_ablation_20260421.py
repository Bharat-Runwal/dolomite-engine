"""
Plots for V5–V8 Helmholtz decomposition ablation study.
Compares V0 GPT, V1 EGPT, V5 (attn-only energy), V6/V7/V8 (Helmholtz variants).
Saves PNG+PDF to results/multi-block-ablation/plots/.
"""

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_sys_dir = str(Path(__file__).resolve().parent)
if _sys_dir not in sys.path:
    sys.path.insert(0, _sys_dir)
from make_tables import AVG_TASKS_10, TASK_DISPLAY

RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "V0 GPT\n(162M)":    "v0_gpt_baseline_d768",
    "V1 EGPT\n(143M)":   "v1_12x1_d768_lr2e3",
    "V5 AttnOnly\n(143M)": "v5_12x1_d768_attn_only_energy_lr2e3",
    "V6 HelmF\n(140M)":  "v6_12x1_d768_helmholtz_factored_lr2e3",
    "V7 HelmD\n(140M)":  "v7_12x1_d768_helmholtz_dual_lr2e3",
    "V8 HelmDR\n(140M)": "v8_12x1_d768_helmholtz_dual_reversed_lr2e3",
}

ZERO_SHOT_TASKS = [t for t, _ in AVG_TASKS_10]
TASK_LABELS = [TASK_DISPLAY[t][0] for t in ZERO_SHOT_TASKS]

METRIC_MAP = {t: TASK_DISPLAY[t][1] for t in ZERO_SHOT_TASKS}
METRIC_MAP["wikitext"] = "word_perplexity,none"

# Colors: blue for GPT, orange for standard EGPT, green for V5, red shades for V6-V8
COLORS = {
    "V0 GPT\n(162M)":      "#4C72B0",
    "V1 EGPT\n(143M)":     "#DD8452",
    "V5 AttnOnly\n(143M)": "#55A868",
    "V6 HelmF\n(140M)":    "#C44E52",
    "V7 HelmD\n(140M)":    "#8172B2",
    "V8 HelmDR\n(140M)":   "#937860",
}


def load_results(model_dir: str) -> dict | None:
    unsharded = RESULTS_DIR / model_dir / "unsharded"
    cands = sorted(unsharded.glob("harness_results*.json"))
    if not cands:
        return None
    with open(cands[-1]) as f:
        return json.load(f)


def get_metric(results: dict, task: str) -> float | None:
    mkey = METRIC_MAP[task]
    v = results.get("results", {}).get(task, {}).get(mkey)
    return float(v) if v is not None else None


def savefig(fig, stem: str):
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"{stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def main():
    all_results = {label: load_results(d) for label, d in MODELS.items()}
    available = {l: r for l, r in all_results.items() if r is not None}
    missing = [l for l, r in all_results.items() if r is None]
    if missing:
        print(f"Missing: {missing}")

    labels = list(available.keys())
    colors = [COLORS[l] for l in labels]

    # ── 1. Average zero-shot accuracy bar chart ─────────────────────────────
    avg_acc = []
    for r in available.values():
        scores = [get_metric(r, t) for t in ZERO_SHOT_TASKS]
        scores = [s for s in scores if s is not None]
        avg_acc.append(np.mean(scores) if scores else None)

    fig, ax = plt.subplots(figsize=(9, 4))
    x = np.arange(len(labels))
    bars = ax.bar(x, [v * 100 for v in avg_acc], color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, avg_acc):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.15,
                f"{val * 100:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("Avg Zero-Shot Accuracy (%)", fontsize=10)
    ax.set_title("V5–V8 Ablation: Average Zero-Shot Accuracy (9 tasks)", fontsize=11, fontweight="bold")
    ax.set_ylim(40, 53)
    ax.axhline(avg_acc[0] * 100, color=COLORS["V0 GPT\n(162M)"], linestyle="--", linewidth=1.2, alpha=0.6, label="V0 GPT baseline")
    ax.legend(fontsize=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(fig, "v5_v8_avg_acc")

    # ── 2. WikiText PPL bar chart ────────────────────────────────────────────
    ppl_vals = [get_metric(r, "wikitext") for r in available.values()]
    fig, ax = plt.subplots(figsize=(9, 4))
    bars = ax.bar(x, ppl_vals, color=colors, edgecolor="white", linewidth=0.5)
    for bar, val in zip(bars, ppl_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                f"{val:.1f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("WikiText Word Perplexity (lower → better)", fontsize=10)
    ax.set_title("V5–V8 Ablation: WikiText Perplexity", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(fig, "v5_v8_wikitext_ppl")

    # ── 3. Per-task breakdown for V0/V1/V5 (focused comparison) ─────────────
    focus_labels = ["V0 GPT\n(162M)", "V1 EGPT\n(143M)", "V5 AttnOnly\n(143M)"]
    focus_results = [available[l] for l in focus_labels if l in available]
    focus_colors = [COLORS[l] for l in focus_labels if l in available]
    n_models = len(focus_results)
    n_tasks = len(ZERO_SHOT_TASKS)
    data = np.full((n_models, n_tasks), np.nan)
    for i, r in enumerate(focus_results):
        for j, t in enumerate(ZERO_SHOT_TASKS):
            v = get_metric(r, t)
            if v is not None:
                data[i, j] = v * 100

    fig, ax = plt.subplots(figsize=(11, 4.5))
    xt = np.arange(n_tasks)
    width = 0.25
    clean_labels = ["V0 GPT (162M)", "V1 EGPT (143M)", "V5 AttnOnly (143M)"]
    for i, (label, color) in enumerate(zip(clean_labels, focus_colors)):
        offset = (i - 1) * width
        vals = data[i]
        mask = ~np.isnan(vals)
        ax.bar(xt[mask] + offset, vals[mask], width, label=label, color=color,
               edgecolor="white", linewidth=0.3, alpha=0.9)
    ax.set_xticks(xt)
    ax.set_xticklabels(TASK_LABELS, fontsize=10)
    ax.set_ylabel("Accuracy (%)", fontsize=10)
    ax.set_title("Per-Task Comparison: V0 GPT vs V1 EGPT vs V5 AttnOnly", fontsize=11, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    savefig(fig, "v5_per_task")

    # ── 4. Full V0–V8 per-task heatmap ───────────────────────────────────────
    all_labels_clean = [l.replace("\n", " ") for l in labels]
    data_all = np.full((len(labels), n_tasks), np.nan)
    for i, r in enumerate(available.values()):
        for j, t in enumerate(ZERO_SHOT_TASKS):
            v = get_metric(r, t)
            if v is not None:
                data_all[i, j] = v * 100

    fig, ax = plt.subplots(figsize=(11, 4))
    im = ax.imshow(data_all, aspect="auto", cmap="RdYlGn", vmin=20, vmax=80)
    ax.set_xticks(range(n_tasks))
    ax.set_xticklabels(TASK_LABELS, fontsize=9)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(all_labels_clean, fontsize=9)
    for i in range(len(labels)):
        for j in range(n_tasks):
            if not np.isnan(data_all[i, j]):
                ax.text(j, i, f"{data_all[i, j]:.1f}", ha="center", va="center",
                        fontsize=7.5, color="black" if 30 < data_all[i, j] < 70 else "white")
    plt.colorbar(im, ax=ax, label="Accuracy (%)")
    ax.set_title("Per-Task Accuracy Heatmap: V0–V8 Variants", fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "v5_v8_heatmap")

    # ── Summary table ────────────────────────────────────────────────────────
    print("\n=== V5-V8 Ablation Summary ===")
    header = f"{'Model':<22} {'PPL':>6} {'Avg%':>6} " + "  ".join(f"{t[:5]:>5}" for t in TASK_LABELS)
    print(header)
    print("-" * len(header))
    for label, r, avg, ppl_v in zip(labels, available.values(), avg_acc, ppl_vals):
        row = f"{label.replace(chr(10), ' '):<22} {ppl_v or 0:>6.1f} {avg * 100:>5.1f}  "
        for t in ZERO_SHOT_TASKS:
            v = get_metric(r, t)
            row += f"  {v * 100:>5.1f}" if v else "    --"
        print(row)


if __name__ == "__main__":
    main()
