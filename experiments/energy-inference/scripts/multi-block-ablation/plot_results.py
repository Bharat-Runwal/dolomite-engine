"""
Bar chart plots for multi-block ablation benchmark results.
Reads harness_results.json from each model's unsharded/ dir.
Saves PNGs to results/multi-block-ablation/plots/.
"""

import json
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

MODELS = {
    "V0 GPT\n(162M)": "v0_gpt_baseline_d768",
    "V1\n(143M)": "v1_12x1_d768",
    "V2\n(110M)": "v2_6x2_d768",
    "V3 Shared\n(163M)": "v3_shared_d1152",
    "V1 lr=2e-3\n(143M)": "v1_12x1_d768_lr2e3",
    "V4\n(349M†)": "v4_shared_wide_d1152",
    "V4 lr=2e-3\n(349M†)": "v4_shared_wide_d1152_lr2e3",
    "V1 400M\n(400M)": "v1_400m_d1024_lr7e4",
    "V5 AttnOnly\n(~143M)": "v5_12x1_d768_attn_only_energy_lr2e3",
    "V6 HelmF\n(~140M)": "v6_12x1_d768_helmholtz_factored_lr2e3",
    "V7 HelmD\n(~140M)": "v7_12x1_d768_helmholtz_dual_lr2e3",
    "V8 HelmDR\n(~140M)": "v8_12x1_d768_helmholtz_dual_reversed_lr2e3",
}

ZERO_SHOT_TASKS = [
    "arc_challenge", "arc_easy", "hellaswag", "winogrande",
    "boolq", "piqa", "copa", "openbookqa", "sciq",
]

METRIC_MAP = {
    "arc_challenge":  ("acc_norm,none", "acc_norm"),
    "arc_easy":       ("acc_norm,none", "acc_norm"),
    "hellaswag":      ("acc_norm,none", "acc_norm"),
    "winogrande":     ("acc,none",      "acc"),
    "boolq":          ("acc,none",      "acc"),
    "piqa":           ("acc_norm,none", "acc_norm"),
    "copa":           ("acc,none",      "acc"),
    "openbookqa":     ("acc_norm,none", "acc_norm"),
    "sciq":           ("acc,none",      "acc"),
    "mmlu":           ("acc,none",      "acc"),
    "gsm8k":          ("exact_match,strict-match", "exact_match"),
    "gsm8k_cot":      ("exact_match,strict-match", "exact_match"),
    "wikitext":       ("word_perplexity,none", "word_perplexity"),
}


def load_results(model_dir: str) -> dict | None:
    unsharded = RESULTS_DIR / model_dir / "unsharded"
    # prefer explicit harness_results.json, else pick latest timestamped file
    candidates = sorted(unsharded.glob("harness_results*.json"))
    if not candidates:
        return None
    with open(candidates[-1]) as f:
        return json.load(f)


def get_metric(results: dict, task: str) -> float | None:
    metric_key, _ = METRIC_MAP[task]
    results_section = results.get("results", {})
    vals = results_section.get(task)
    if vals is not None:
        v = vals.get(metric_key)
        if v is not None:
            return float(v)
    return None


def bar_chart(labels, values, title, ylabel, filename, lower_is_better=False, color="steelblue"):
    valid = [(l, v) for l, v in zip(labels, values) if v is not None]
    if not valid:
        print(f"  No data for {title}, skipping.")
        return
    lbls, vals = zip(*valid)
    x = np.arange(len(lbls))
    fig, ax = plt.subplots(figsize=(max(6, len(lbls) * 1.2), 4.5))
    bars = ax.bar(x, vals, color=color, edgecolor="white", linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels(lbls, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for bar, val in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01 * max(vals),
                f"{val:.2f}", ha="center", va="bottom", fontsize=8)
    if lower_is_better:
        ax.invert_yaxis()
        ax.set_title(title + "  (lower → better)", fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / filename.replace(".png", f".{ext}")
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def main():
    all_results = {label: load_results(d) for label, d in MODELS.items()}
    available = {l: r for l, r in all_results.items() if r is not None}
    missing = [l for l, r in all_results.items() if r is None]
    print(f"Models with results: {list(available.keys())}")
    if missing:
        print(f"Missing (eval not done yet): {missing}")

    labels = list(available.keys())

    # 1. Wikitext perplexity
    ppl = [get_metric(r, "wikitext") for r in available.values()]
    bar_chart(labels, ppl, "Wikitext Word Perplexity", "PPL", "wikitext_ppl.png",
              lower_is_better=True, color="salmon")

    # 2. Average zero-shot accuracy
    avg_acc = []
    for r in available.values():
        scores = [get_metric(r, t) for t in ZERO_SHOT_TASKS]
        scores = [s for s in scores if s is not None]
        avg_acc.append(np.mean(scores) if scores else None)
    bar_chart(labels, avg_acc, "Average Zero-Shot Accuracy (9 tasks)", "Accuracy",
              "avg_zeroshot_acc.png", color="steelblue")

    # 3. MMLU
    mmlu = [get_metric(r, "mmlu") for r in available.values()]
    bar_chart(labels, mmlu, "MMLU Accuracy", "Accuracy", "mmlu_acc.png", color="mediumseagreen")

    # 4. GSM8K
    gsm = [get_metric(r, "gsm8k") for r in available.values()]
    bar_chart(labels, gsm, "GSM8K Exact Match (greedy)", "Exact Match", "gsm8k_acc.png", color="mediumpurple")

    # 5. GSM8K CoT
    gsm_cot = [get_metric(r, "gsm8k_cot") for r in available.values()]
    bar_chart(labels, gsm_cot, "GSM8K CoT Exact Match", "Exact Match", "gsm8k_cot_acc.png", color="darkorchid")

    # 6. Per-task breakdown (grouped bar)
    task_labels = ["arc_c", "arc_e", "hella", "wino", "boolq", "piqa", "copa", "obqa", "sciq"]
    n_models = len(labels)
    n_tasks = len(ZERO_SHOT_TASKS)
    data = np.full((n_models, n_tasks), np.nan)
    for i, r in enumerate(available.values()):
        for j, t in enumerate(ZERO_SHOT_TASKS):
            v = get_metric(r, t)
            if v is not None:
                data[i, j] = v
    fig, ax = plt.subplots(figsize=(13, 5))
    x = np.arange(n_tasks)
    width = 0.8 / n_models
    colors = plt.cm.tab10(np.linspace(0, 0.9, n_models))
    for i, (label, color) in enumerate(zip(labels, colors)):
        offset = (i - n_models / 2 + 0.5) * width
        vals = data[i]
        mask = ~np.isnan(vals)
        ax.bar(x[mask] + offset, vals[mask], width, label=label, color=color, alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(task_labels, fontsize=9)
    ax.set_ylabel("Accuracy", fontsize=10)
    ax.set_title("Per-Task Zero-Shot Accuracy", fontsize=11, fontweight="bold")
    ax.legend(fontsize=8, loc="upper left", ncol=2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"per_task_breakdown.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)

    # Print summary table
    print("\n=== Summary Table ===")
    header = f"{'Model':<20} {'PPL':>7} {'Avg':>6} {'MMLU':>6} {'GSM8K':>7} {'GSM8K-CoT':>10}"
    print(header)
    print("-" * len(header))
    for label, ppl_v, avg_v, mmlu_v, gsm_v, gsm_cot_v in zip(labels, ppl, avg_acc, mmlu, gsm, gsm_cot):
        print(f"{label:<20} {ppl_v or 0:>7.2f} {avg_v or 0:>6.3f} {mmlu_v or 0:>6.3f} {gsm_v or 0:>7.4f} {gsm_cot_v or 0:>10.4f}")


if __name__ == "__main__":
    main()
