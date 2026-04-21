"""
Direct comparison: V1 EGPT lr=2e-3 (143M) vs V0 GPT (162M, 30k steps).
Layout: single row — wide accuracy subplot + narrow PPL subplot on the right.

Saved to: results/multi-block-ablation/plots/v1_vs_gpt.{png,pdf}
"""

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "V0 GPT (162M)": "v0_gpt_baseline_d768",
    "V1 lr=2e-3 (143M)": "v1_12x1_d768_lr2e3",
}

ACC_TASKS = [
    ("ARC-C",      "arc_challenge", "acc_norm,none"),
    ("ARC-E",      "arc_easy",      "acc_norm,none"),
    ("HellaSwag",  "hellaswag",     "acc_norm,none"),
    ("WinoGrande", "winogrande",    "acc,none"),
    ("BoolQ",      "boolq",         "acc,none"),
    ("PIQA",       "piqa",          "acc_norm,none"),
    ("COPA",       "copa",          "acc,none"),
    ("OBQA",       "openbookqa",    "acc_norm,none"),
    ("SCIQ",       "sciq",          "acc,none"),
    ("MMLU",       "mmlu",          "acc,none"),
    ("GSM8K",      "gsm8k",         "exact_match,strict-match"),
    ("GSM8K-CoT",  "gsm8k_cot",     "exact_match,strict-match"),
]
PPL_TASK = ("WikiPPL↓", "wikitext", "word_perplexity,none")
ZERO_SHOT = [t for t in ACC_TASKS if t[0] not in ("MMLU", "GSM8K", "GSM8K-CoT")]


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
    colors = ["#4878CF", "#D65F5F"]

    n_acc = len(ACC_TASKS)
    # width_ratios: acc gets 12 units, ppl gets 1 unit (~1/13 ≈ original 1-bar width)
    fig = plt.figure(figsize=(16, 5))
    gs = gridspec.GridSpec(1, 2, width_ratios=[n_acc, 1.4], wspace=0.06)
    ax_acc = fig.add_subplot(gs[0])
    ax_ppl = fig.add_subplot(gs[1])

    # ── accuracy panel ────────────────────────────────────────────────────────
    acc_vals = np.array([
        [get(data[lbl], t[1], t[2]) or 0.0 for t in ACC_TASKS]
        for lbl in labels
    ])

    x = np.arange(n_acc)
    width = 0.35
    for i, (lbl, col) in enumerate(zip(labels, colors)):
        offset = (i - 0.5) * width
        bars = ax_acc.bar(x + offset, acc_vals[i], width, label=lbl,
                          color=col, alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, acc_vals[i]):
            if v > 0:
                ax_acc.text(bar.get_x() + bar.get_width() / 2,
                            bar.get_height() + 0.004,
                            f"{v:.3f}",
                            ha="center", va="bottom", fontsize=6.5, rotation=45)

    ax_acc.axvline(ACC_TASKS.index(next(t for t in ACC_TASKS if t[0] == "MMLU")) - 0.5,
                   color="gray", linestyle=":", linewidth=0.8, alpha=0.7)

    for i, (lbl, col) in enumerate(zip(labels, colors)):
        zs = [get(data[lbl], t[1], t[2]) for t in ZERO_SHOT
              if get(data[lbl], t[1], t[2]) is not None]
        ax_acc.annotate(f"{lbl}\nAvg 9-task: {np.mean(zs):.4f}",
                        xy=(0.01 + i * 0.50, 0.97), xycoords="axes fraction",
                        fontsize=8.5, color=col, fontweight="bold", va="top",
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                                  alpha=0.7, edgecolor=col))

    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([t[0] for t in ACC_TASKS], fontsize=9)
    ax_acc.set_ylabel("Accuracy", fontsize=10)
    ax_acc.set_ylim(0, min(acc_vals.max() * 1.28, 1.0))
    ax_acc.set_title("V1 EGPT lr=2e-3 (143M) vs V0 GPT (162M) — 30k steps, 7.86B tokens",
                     fontsize=11, fontweight="bold")
    ax_acc.legend(fontsize=9, loc="upper right")
    ax_acc.spines["top"].set_visible(False)
    ax_acc.spines["right"].set_visible(False)

    # ── PPL panel (narrow, same row) ──────────────────────────────────────────
    ppl_vals = [get(data[lbl], PPL_TASK[1], PPL_TASK[2]) or 0.0 for lbl in labels]
    bars_ppl = ax_ppl.bar([0, 1], ppl_vals, 0.55, color=colors, alpha=0.85,
                          edgecolor="white")
    for bar, v in zip(bars_ppl, ppl_vals):
        ax_ppl.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.2,
                    f"{v:.1f}", ha="center", va="bottom", fontsize=9,
                    fontweight="bold")

    ppl_min = min(ppl_vals) * 0.88
    ax_ppl.set_ylim(ppl_min, max(ppl_vals) * 1.12)
    ax_ppl.set_xticks([0, 1])
    ax_ppl.set_xticklabels(["GPT", "EGPT"], fontsize=9)
    ax_ppl.set_ylabel("Word Perplexity ↓", fontsize=9)
    ax_ppl.set_title("WikiPPL\n(lower = better)", fontsize=9, fontweight="bold")
    ax_ppl.spines["top"].set_visible(False)
    ax_ppl.spines["right"].set_visible(False)

    # vertical separator between the two panels
    ax_ppl.spines["left"].set_linestyle(":")
    ax_ppl.spines["left"].set_color("gray")
    ax_ppl.spines["left"].set_alpha(0.6)

    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"v1_vs_gpt.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n=== V1 lr=2e-3 vs V0 GPT ===")
    col_w = 20
    header = (f"{'Metric':<18} {'V0 GPT (162M)':>{col_w}} "
              f"{'V1 lr=2e-3 (143M)':>{col_w}} {'Δ (EGPT − GPT)':>{col_w}}")
    print(header)
    print("─" * len(header))
    for label, task, metric in ACC_TASKS + [PPL_TASK]:
        gpt_v  = get(data[labels[0]], task, metric)
        egpt_v = get(data[labels[1]], task, metric)
        if gpt_v is None or egpt_v is None:
            continue
        delta = egpt_v - gpt_v
        sign  = "+" if delta >= 0 else ""
        better = "✓" if (delta > 0 and label != "WikiPPL↓") or (delta < 0 and label == "WikiPPL↓") else " "
        fmt = ".2f" if label == "WikiPPL↓" else ".4f"
        print(f"{label:<18} {gpt_v:>{col_w}{fmt}} {egpt_v:>{col_w}{fmt}} "
              f"{sign}{delta:>{col_w-1}{fmt}} {better}")

    zs_gpt  = np.mean([get(data[labels[0]], t[1], t[2]) for t in ZERO_SHOT])
    zs_egpt = np.mean([get(data[labels[1]], t[1], t[2]) for t in ZERO_SHOT])
    print(f"\n{'Avg zero-shot':<18} {zs_gpt:>{col_w}.4f} {zs_egpt:>{col_w}.4f} "
          f"{zs_egpt - zs_gpt:>+{col_w}.4f}")


if __name__ == "__main__":
    main()
