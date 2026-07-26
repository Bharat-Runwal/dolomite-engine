"""Register-EGPT vs non-EGPT baselines (RecGPT, deep-GPT), same plain
web-only data mix (no math), at two matched scales (H1 d768, V73 d1280).
Tests whether the register-EGPT GSM8K/BBH signal survives against a
same-datamix, same-depth non-EGPT architecture. CLUSTER JOB ONLY.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PLOTS = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/plots")

GROUPS = ["H1 scale (d=768, ~125-162M)", "V73 scale (d=1280, ~279-284M, iso-param)"]
MODELS = ["EGPT baseline", "EGPT+register", "RecGPT", "Deep-GPT"]
COLORS = {"EGPT baseline": "#888888", "EGPT+register": "#2f6f9f",
          "RecGPT": "#e67e22", "Deep-GPT": "#27ae60"}

# [H1-scale, V73-scale] per model
BBH = {
    "EGPT baseline":   [0.2704, 0.2880],
    "EGPT+register":   [0.2709, 0.2656],
    "RecGPT":          [0.2511, 0.2874],
    "Deep-GPT":        [0.2706, None],
}
GSM8K = {
    "EGPT baseline":   [0.0167, 0.0182],
    "EGPT+register":   [0.0235, 0.0152],
    "RecGPT":          [0.0136, 0.0190],
    "Deep-GPT":        [0.0197, None],
}

fig, axes = plt.subplots(2, 2, figsize=(11, 8), sharex="col")
x = np.arange(len(MODELS))

for col, group in enumerate(GROUPS):
    for row, (data, ylabel, title) in enumerate([
        (BBH, "BBH exact-match", "BBH"), (GSM8K, "GSM8K-CoT flex-extract", "GSM8K-CoT")
    ]):
        ax = axes[row][col]
        vals = [data[m][col] for m in MODELS]
        colors = [COLORS[m] for m in MODELS]
        bars = ax.bar(x, [v if v is not None else 0 for v in vals], color=colors,
                       edgecolor="white", linewidth=0.5)
        for xi, v in zip(x, vals):
            if v is None:
                ax.text(xi, 0.01, "n/a\n(no d1280\ndeep-GPT)", ha="center", va="bottom",
                        fontsize=7, color="gray", style="italic")
            else:
                ax.text(xi, v + 0.003, f"{100*v:.2f}%", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(MODELS if row == 1 else [], fontsize=8, rotation=15, ha="right")
        if col == 0:
            ax.set_ylabel(ylabel, fontsize=9)
        if row == 0:
            ax.set_title(group, fontsize=10, fontweight="bold")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(0, max(v for v in vals if v is not None) * 1.25)

fig.suptitle(
    "Register-EGPT vs. non-EGPT baselines, same plain web-only data mix (no math)\n"
    "Caveat: none of these runs use the math-mixed corpus used elsewhere in this paper",
    fontsize=10.5, fontweight="bold")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(str(PLOTS / f"lk_register_vs_nonegpt.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: lk_register_vs_nonegpt.{pdf,png}")
