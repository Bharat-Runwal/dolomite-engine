#!/usr/bin/env python
"""Generate paper bar charts for ≤5-model comparisons.

Mirrors make_tables.py: same MODELS registry, same harness JSON loaders.
Each chart is a single figure with one subplot per metric and one bar per
model. Output: results/multi-block-ablation/plots/bar_<name>.{pdf,png}.

Usage:
  python make_barcharts.py            # all charts
  python make_barcharts.py egpt_gap   # one chart
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from make_tables import MODELS, get_ppl, get_avg, get_task  # noqa: E402

PLOTS_DIR = HERE.parent.parent / "results/multi-block-ablation/plots"

# Family → color, kept consistent across charts so model identity is visible at a glance.
FAMILY_COLORS = {
    "gpt":          "#1f77b4",   # blue — pure GPT
    "energy":       "#ff7f0e",   # orange — pure EGPT (V=K)
    "mixed":        "#7f7f7f",   # grey — mixed-head plain (V10/V11)
    "mixhead_dag":  "#9467bd",   # purple — MixH† (V15…V26)
    "egrad":        "#d62728",   # red — true EGrad
    "edesc":        "#2ca02c",   # green — true EDesc
    "full_egrad":   "#8c564b",   # brown — full EGrad
}

# ---------------------------------------------------------------------------
# Charts: ≤5 models per chart, one figure with N subplots in a row.
# Metric kinds: 'ppl', 'avg', or ('task', task_name)
# Direction: 'lower' or 'higher' — drives bar-label placement and best-arrow.
# ---------------------------------------------------------------------------
CHARTS = [
    {
        "name": "egpt_gap",
        "title": "Diagnosing the EGPT compute gap",
        "models": ["V0", "V1", "V1_400m", "V9"],
        "metrics": [
            ("PPL ↓",   "ppl",  "lower"),
            ("Avg ↑",   "avg",  "higher"),
            ("ARC-C",   ("task", "arc_challenge"), "higher"),
            ("HellaS",  ("task", "hellaswag"),     "higher"),
        ],
    },
    {
        "name": "scaling",
        "title": "400M-scale headline: GPT vs EGPT vs MixH† vs EGrad/EDesc",
        "models": ["V9", "V1_400m", "V19", "V31", "V32"],
        "metrics": [
            ("PPL ↓",   "ppl",  "lower"),
            ("Avg ↑",   "avg",  "higher"),
            ("ARC-C",   ("task", "arc_challenge"), "higher"),
            ("HellaS",  ("task", "hellaswag"),     "higher"),
            ("GSM8K",   ("task", "gsm8k"),         "higher"),
        ],
    },
    {
        "name": "scaling_v25v26",
        "title": "Recurrence-schedule ablation (162M unique params, 503 MFLOPs/tok)",
        "models": ["V9", "V23", "V24", "V25", "V26"],
        "metrics": [
            ("PPL ↓",   "ppl",  "lower"),
            ("Avg ↑",   "avg",  "higher"),
            ("ARC-C",   ("task", "arc_challenge"), "higher"),
            ("HellaS",  ("task", "hellaswag"),     "higher"),
            ("GSM8K",   ("task", "gsm8k"),         "higher"),
        ],
    },
    {
        "name": "cosreg_benchmarks",
        "title": "Cosine-similarity regulariser — 176M scale (30k steps)",
        "models": ["V0", "V1", "V39", "V48", "V52", "V53"],
        "metrics": [
            ("Avg ↑",   "avg",  "higher"),
            ("ARC-C",   ("task", "arc_challenge"), "higher"),
            ("BoolQ",   ("task", "boolq"),         "higher"),
            ("HellaS",  ("task", "hellaswag"),     "higher"),
            ("PIQA",    ("task", "piqa"),          "higher"),
        ],
    },
]


# ---------------------------------------------------------------------------
# Metric resolution + value formatting
# ---------------------------------------------------------------------------
def get_value(model_id: str, metric):
    """Return (numeric_value, display_string, kind) for one cell."""
    if metric == "ppl":
        v = get_ppl(model_id)
        return v, (f"{v:.2f}" if v is not None else "—"), "ppl"
    if metric == "avg":
        v = get_avg(model_id, n_tasks=10)
        return v, (f"{v*100:.1f}" if v is not None else "—"), "pct"
    if isinstance(metric, tuple) and metric[0] == "task":
        task = metric[1]
        v = get_task(model_id, task)
        if task == "gsm8k":
            return v, (f"{v*100:.2f}" if v is not None else "—"), "pct"
        return v, (f"{v*100:.1f}" if v is not None else "—"), "pct"
    raise ValueError(f"unknown metric: {metric!r}")


def best_index(values, direction):
    """Return the index of the best value (None values skipped)."""
    idxs = [i for i, v in enumerate(values) if v is not None]
    if not idxs:
        return None
    if direction == "lower":
        return min(idxs, key=lambda i: values[i])
    return max(idxs, key=lambda i: values[i])


# ---------------------------------------------------------------------------
# Chart renderer
# ---------------------------------------------------------------------------
def render_chart(spec: dict) -> Path:
    models = spec["models"]
    metrics = spec["metrics"]
    n = len(metrics)

    # Per-subplot figure width grows with model count.
    fig_w = max(2.6 * n, 8.0)
    fig, axes = plt.subplots(1, n, figsize=(fig_w, 3.4), squeeze=False)
    axes = axes[0]

    # Short labels (drop the architecture suffix to keep x-axis tight)
    short_labels = [
        MODELS[m]["label"]
            .replace("$\\times$", "×")
            .replace("$\\dagger$", "†")
        for m in models
    ]
    # Trim "12×1", "24×1", etc. — keep first 2 tokens (e.g. "V0 GPT")
    short_labels = [" ".join(l.split()[:2]) for l in short_labels]

    bar_colors = [FAMILY_COLORS.get(MODELS[m]["family"], "#888888") for m in models]

    missing = []
    for ax, (header, metric, direction) in zip(axes, metrics):
        values = []
        labels = []
        for m in models:
            v, s, _kind = get_value(m, metric)
            values.append(v)
            labels.append(s)
            if v is None:
                missing.append((spec["name"], m, header))

        x = list(range(len(models)))
        bars = ax.bar(x, [v if v is not None else 0 for v in values], color=bar_colors,
                      edgecolor="white", linewidth=0.5)
        # Mark missing values with hatched bar at zero
        for i, v in enumerate(values):
            if v is None:
                bars[i].set_hatch("///")
                bars[i].set_alpha(0.3)

        # Highlight best with a thicker edge
        bi = best_index(values, direction)
        if bi is not None:
            bars[bi].set_edgecolor("black")
            bars[bi].set_linewidth(1.6)

        # Value labels above each bar
        ymax = max([v for v in values if v is not None] or [1.0])
        ymin = 0.0
        for i, (v, lbl) in enumerate(zip(values, labels)):
            if v is None:
                ax.text(i, 0, "—", ha="center", va="bottom", fontsize=8, color="#666")
                continue
            ax.text(i, v + 0.012 * ymax, lbl, ha="center", va="bottom", fontsize=8)

        # Cosmetic
        ax.set_xticks(x)
        ax.set_xticklabels(short_labels, rotation=30, ha="right", fontsize=8.5)
        ax.set_title(header, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_axisbelow(True)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        ax.set_ylim(ymin, ymax * 1.18)

    fig.suptitle(spec.get("title", spec["name"]), fontsize=11, y=1.02)
    fig.tight_layout()

    PLOTS_DIR.mkdir(exist_ok=True, parents=True)
    out_pdf = PLOTS_DIR / f"bar_{spec['name']}.pdf"
    out_png = PLOTS_DIR / f"bar_{spec['name']}.png"
    for path in (out_pdf, out_png):
        fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    if missing:
        for name, mid, hdr in missing:
            print(f"  warn: {name}/{mid}/{hdr} missing")
    return out_pdf


def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", help="single chart name to generate")
    args = p.parse_args()

    targets = [c for c in CHARTS if (args.name is None or c["name"] == args.name)]
    if not targets:
        print(f"No chart matches name={args.name}.  Known: {[c['name'] for c in CHARTS]}",
              file=sys.stderr)
        sys.exit(1)

    for c in targets:
        out = render_chart(c)
        print(f"  wrote {out.relative_to(PLOTS_DIR.parent.parent.parent)}")


if __name__ == "__main__":
    main()
