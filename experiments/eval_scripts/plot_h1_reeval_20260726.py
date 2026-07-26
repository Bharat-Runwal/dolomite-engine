"""H1 variants: BBH/GSM8K before (bug) vs after (fix) the register-decode bug,
with the h1-base non-register baseline as a reference line. CLUSTER JOB ONLY.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

PLOTS = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation/plots")

MODELS = ["h1-all-128", "h1-all-256", "h1-sel-128"]
BBH_BUG = [None, None, None]          # never run pre-fix for these
BBH_FIX = [0.2623, 0.1543, 0.2709]
GSM8K_BUG = [0.0053, 0.0000, 0.0243]
GSM8K_FIX = [0.0136, 0.0000, 0.0235]
BASE_BBH = 0.2704
BASE_GSM8K = 0.0167

COL_BUG = "#b0413e"
COL_FIX = "#2f6f9f"
COL_BASE = "#444444"

fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
x = np.arange(len(MODELS))
w = 0.32

ax = axes[0]
ax.bar(x, BBH_FIX, width=w, color=COL_FIX, label="post-fix (no\\_cache)", edgecolor="white")
ax.axhline(BASE_BBH, color=COL_BASE, ls="--", lw=1.5, label="h1-base (R=0)")
ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
ax.set_ylabel("BBH exact-match")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("BBH", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

ax = axes[1]
ax.bar(x - w/2, GSM8K_BUG, width=w, color=COL_BUG, label="pre-fix (bypass)", edgecolor="white")
ax.bar(x + w/2, GSM8K_FIX, width=w, color=COL_FIX, label="post-fix (no\\_cache)", edgecolor="white")
ax.axhline(BASE_GSM8K, color=COL_BASE, ls="--", lw=1.5, label="h1-base (R=0)")
ax.set_xticks(x); ax.set_xticklabels(MODELS, fontsize=9)
ax.set_ylabel("GSM8K-CoT flexible-extract")
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("GSM8K-CoT", fontsize=10)
ax.legend(fontsize=8)
ax.grid(axis="y", alpha=0.3)

fig.suptitle("H1 selective vs all-layer registers: decode-bug fix effect", fontsize=11, fontweight="bold")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(str(PLOTS / f"lk_h1_reeval_bug_vs_fix.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: lk_h1_reeval_bug_vs_fix.{pdf,png}")
