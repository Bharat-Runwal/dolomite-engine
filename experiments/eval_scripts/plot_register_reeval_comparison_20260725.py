"""Bug (bypass) vs fix (no_cache) comparison for all 17 register checkpoints.
Reads register_reeval_comparison_20260725.json (produced by
collect_register_reeval_results_20260725.py). CLUSTER JOB ONLY.
"""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
rows = json.load(open(PLOTS / "register_reeval_comparison_20260725.json"))

# Sort by fixed BBH descending (missing -> -1 so they sort last)
rows = sorted(rows, key=lambda r: r["new_bbh"] if r["new_bbh"] is not None else -1, reverse=True)
labels = [r["label"] for r in rows]
y = np.arange(len(rows))

COL_BUG = "#b0413e"    # muted red — pre-fix (buggy bypass mode)
COL_FIX = "#2f6f9f"    # confident blue — post-fix (no_cache)
RAND_BBH = 0.24        # BBH random baseline (per REGISTER_DECODE_BUG.md, ~0.23-0.25)

fig, axes = plt.subplots(1, 2, figsize=(12, 8), sharey=True)

# --- Panel 1: BBH ---
ax = axes[0]
bug = [r["old_bbh"] if r["old_bbh"] is not None else np.nan for r in rows]
fix = [r["new_bbh"] if r["new_bbh"] is not None else np.nan for r in rows]
h = 0.34
ax.barh(y + h/2, [b if not np.isnan(b) else 0 for b in bug], height=h, color=COL_BUG,
        alpha=0.85, label="pre-fix (bypass, buggy)", edgecolor="white", linewidth=0.5)
ax.barh(y - h/2, fix, height=h, color=COL_FIX, alpha=0.9, label="post-fix (no\\_cache)",
        edgecolor="white", linewidth=0.5)
# mark missing pre-fix data
for yi, b in zip(y, bug):
    if np.isnan(b):
        ax.text(0.003, yi + h/2, "n/a", va="center", ha="left", fontsize=7, color="gray", style="italic")
ax.axvline(RAND_BBH, color="black", ls="--", lw=1.3, label="BBH random baseline (~24%)")
ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
ax.set_xlabel("BBH exact-match (rescored macro-avg)")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("BBH: pre- vs post-decode-bug-fix", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
ax.grid(axis="x", alpha=0.3)
ax.invert_yaxis()

# --- Panel 2: GSM8K-CoT flex-extract ---
ax = axes[1]
bug = [r["old_gsm8k_cot"] if r["old_gsm8k_cot"] is not None else np.nan for r in rows]
fix = [r["new_gsm8k_cot"] if r["new_gsm8k_cot"] is not None else np.nan for r in rows]
ax.barh(y + h/2, bug, height=h, color=COL_BUG, alpha=0.85, edgecolor="white", linewidth=0.5)
ax.barh(y - h/2, fix, height=h, color=COL_FIX, alpha=0.9, edgecolor="white", linewidth=0.5)
ax.set_xlabel("GSM8K-CoT flexible-extract exact-match")
ax.xaxis.set_major_formatter(mticker.PercentFormatter(1.0))
ax.set_title("GSM8K-CoT: pre- vs post-decode-bug-fix", fontsize=10)
ax.grid(axis="x", alpha=0.3)

fig.suptitle(
    "Register-decode RoPE bug: BBH mostly recovers to baseline; GSM8K stays near the\n"
    "small-model competence ceiling regardless of registers or the fix",
    fontsize=10.5, fontweight="bold")
plt.tight_layout()
for ext in ("pdf", "png"):
    fig.savefig(str(PLOTS / f"register_reeval_bug_vs_fix.{ext}"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("Saved: register_reeval_bug_vs_fix.{pdf,png}")
