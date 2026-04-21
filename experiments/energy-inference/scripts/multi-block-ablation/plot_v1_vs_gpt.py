"""
Direct comparison: V1 EGPT lr=2e-3 (143M) vs V0 GPT (162M, 30k steps).

Produces:
  - A focused grouped bar chart (per-task + summary metrics)
  - A printed comparison table
  - Both PNG and PDF

Saved to: results/multi-block-ablation/plots/v1_vs_gpt.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "V0 GPT (162M)": "v0_gpt_baseline_d768",
    "V1 lr=2e-3 (143M)": "v1_12x1_d768_lr2e3",
}

TASKS = [
    ("ARC-C",     "arc_challenge", "acc_norm,none"),
    ("ARC-E",     "arc_easy",      "acc_norm,none"),
    ("HellaSwag", "hellaswag",     "acc_norm,none"),
    ("WinoGrande","winogrande",    "acc,none"),
    ("BoolQ",     "boolq",         "acc,none"),
    ("PIQA",      "piqa",          "acc_norm,none"),
    ("COPA",      "copa",          "acc,none"),
    ("OBQA",      "openbookqa",    "acc_norm,none"),
    ("SCIQ",      "sciq",          "acc,none"),
    ("MMLU",      "mmlu",          "acc,none"),
    ("GSM8K",     "gsm8k",         "exact_match,strict-match"),
    ("GSM8K-CoT", "gsm8k_cot",     "exact_match,strict-match"),
    ("WikiPPL↓",  "wikitext",      "word_perplexity,none"),
]
ZERO_SHOT = [t for t in TASKS if t[0] not in ("MMLU", "GSM8K", "GSM8K-CoT", "WikiPPL↓")]


def load(model_dir):
    unsharded = RESULTS_DIR / model_dir / "unsharded"
    candidates = sorted(unsharded.glob("harness_results*.json"))
    assert candidates, f"No results in {unsharded}"
    with open(candidates[-1]) as f:
        return json.load(f)["results"]


def get(results, task, metric):
    return results.get(task, {}).get(metric)


def main():
    data = {label: load(d) for label, d in MODELS.items()}
    labels = list(data.keys())
    colors = ["#4878CF", "#D65F5F"]  # blue for GPT, red for EGPT

    # ── grouped bar chart ────────────────────────────────────────────────────
    task_labels = [t[0] for t in TASKS]
    vals = np.array([
        [get(data[lbl], t[1], t[2]) or 0.0 for t in TASKS]
        for lbl in labels
    ])  # [2, n_tasks]

    n_tasks = len(TASKS)
    x = np.arange(n_tasks)
    width = 0.35

    fig, ax = plt.subplots(figsize=(15, 5))
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        offset = (i - 0.5) * width
        bars = ax.bar(x + offset, vals[i], width, label=lbl, color=col, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals[i]):
            if v > 0:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.3f}" if v < 10 else f"{v:.1f}",
                        ha="center", va="bottom", fontsize=6.5, rotation=45)

    # Mark WikiPPL separately — it's on a different scale
    ppl_idx = task_labels.index("WikiPPL↓")
    ax.axvline(ppl_idx - 0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.text(ppl_idx, max(vals[:, ppl_idx]) * 1.05, "PPL scale\n(lower=better)",
            ha="center", fontsize=7.5, color="gray")

    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=9)
    ax.set_ylabel("Score", fontsize=10)
    ax.set_title("V1 EGPT lr=2e-3 (143M) vs V0 GPT (162M) — 30k steps, 7.86B tokens",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Add avg zero-shot annotation
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        zs_scores = [get(data[lbl], t[1], t[2]) for t in ZERO_SHOT if get(data[lbl], t[1], t[2]) is not None]
        avg = np.mean(zs_scores)
        ax.annotate(f"{lbl}\nAvg 9-task: {avg:.4f}",
                    xy=(0.02 + i * 0.52, 0.96), xycoords="axes fraction",
                    fontsize=9, color=col, fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.7, edgecolor=col))

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"v1_vs_gpt.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)

    # ── summary table ────────────────────────────────────────────────────────
    print("\n=== V1 lr=2e-3 vs V0 GPT ===")
    col_w = 20
    header = f"{'Metric':<18} {'V0 GPT (162M)':>{col_w}} {'V1 lr=2e-3 (143M)':>{col_w}} {'Δ (EGPT − GPT)':>{col_w}}"
    print(header)
    print("─" * len(header))
    for label, task, metric in TASKS:
        gpt_v = get(data[labels[0]], task, metric)
        egpt_v = get(data[labels[1]], task, metric)
        if gpt_v is None or egpt_v is None:
            continue
        delta = egpt_v - gpt_v
        sign = "+" if delta >= 0 else ""
        better = "✓" if (delta > 0 and label != "WikiPPL↓") or (delta < 0 and label == "WikiPPL↓") else " "
        fmt = ".2f" if label == "WikiPPL↓" else ".4f"
        print(f"{label:<18} {gpt_v:>{col_w}{fmt}} {egpt_v:>{col_w}{fmt}} {sign}{delta:>{col_w-1}{fmt}} {better}")

    zs_gpt  = np.mean([get(data[labels[0]], t[1], t[2]) for t in ZERO_SHOT])
    zs_egpt = np.mean([get(data[labels[1]], t[1], t[2]) for t in ZERO_SHOT])
    print(f"\n{'Avg zero-shot':<18} {zs_gpt:>{col_w}.4f} {zs_egpt:>{col_w}.4f} {zs_egpt-zs_gpt:>+{col_w}.4f}")


if __name__ == "__main__":
    main()
