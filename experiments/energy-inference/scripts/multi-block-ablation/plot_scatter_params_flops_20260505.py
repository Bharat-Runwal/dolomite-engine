"""Scatter plot: WikiText PPL and 10-task avg accuracy vs Parameters and FLOPs.

2×2 grid:
  [0,0] PPL         vs Params (M)
  [0,1] PPL         vs MFLOPs/token
  [1,0] 10-task avg vs Params (M)
  [1,1] 10-task avg vs MFLOPs/token

Includes V73 (6GPT+1EGPT×6, d=1280, 279M, 535 MFLOPs) alongside the best
100–400M models. V73 result is loaded if available, otherwise skipped.

Date: 2026-05-05
"""
from __future__ import annotations
import json, glob, sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE  = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))
from make_tables import AVG_TASKS_10   # canonical 10-task list

BASE      = HERE.parents[1] / "results/multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# ── Model registry ────────────────────────────────────────────────────────────
# (label, subdir, params_M, mflops_per_token, family)
# MFLOPs formula: 2 * sum(layers * iters * (4*d^2 + 3*d*ffn)) / 1e6
MODELS: dict[str, tuple] = {
    # ── GPT baselines ──
    "V0":      ("GPT 12×1 d=768",          "v0_gpt_baseline_d768",              162.0, 296,  "gpt"),
    "V9":      ("GPT 24×1 d=1024",         "v9_gpt_baseline_d1024_lr1e3",       354.5, 503,  "gpt"),
    "V12":     ("GPT 6×2 d=768",           "v12_gpt_6x2_d768_lr2e3",            119.5, 296,  "gpt"),
    "V54":     ("GPT par 12×1 d=768",      "v54_parallel_gpt_12x1_d768_lr2e3",  155.0, 296,  "gpt"),

    # ── Plain EGPT (V=K, dual projection) ──
    "V1":      ("EGPT 12×1 d=768",         "v1_12x1_d768_lr2e3",                143.0, 333,  "egpt"),
    "V1_400m": ("EGPT 24×1 d=1024",        "v1_400m_d1024_lr7e4",               354.4, 755,  "egpt"),

    # ── Recurrent EGPT (deep iter) ──
    "V56":     ("EGPT 1×12 d=768",         "v56_egpt_1x12_d768_lr2e3",          143.0, 333,  "recurrent"),
    "V57":     ("EGPT 1×6 d=768",          "v57_egpt_1x6_d768_lr2e3",           143.0, 333,  "recurrent"),
    "V58":     ("EGPT 1×24 d=1024",        "v58_egpt_1x24_d1024_lr1e3",         216.0, 755,  "recurrent"),
    "V59":     ("EGPT 1×12 d=1024",        "v59_egpt_1x12_d1024_lr1e3",         216.0, 503,  "recurrent"),
    "V63":     ("EGPT 1×12 d=1408",        "v63_egpt_1x12_d1408_lr2e3",         293.0, 600,  "recurrent"),

    # ── EGrad/EDesc (true W_Q^T energy gradient) ──
    "V19":     ("EGrad 24×1 d=1024",       "v19_energy_grad_24x1_d1024_lr1e3",  342.0, 503,  "egrad"),
    "V20":     ("EDesc 24×1 d=1024",       "v20_energy_desc_24x1_d1024_lr1e3",  342.0, 503,  "edesc"),
    "V23":     ("EGrad 12×2 d=1024",       "v23_energy_grad_12x2_d1024_lr1e3",  228.0, 503,  "egrad"),
    "V24":     ("EDesc 12×2 d=1024",       "v24_energy_desc_12x2_d1024_lr1e3",  228.0, 503,  "edesc"),
    "V31":     ("EGrad 24×1 d=1024 v2",    "v31_egrad_attn_24x1_d1024_lr1e3",   342.0, 503,  "egrad"),
    "V32":     ("EDesc 24×1 d=1024 v2",    "v32_edesc_24x1_d1024_lr1e3",        342.0, 503,  "edesc"),
    "V35":     ("EGrad 12×2 d=1024 v2",    "v35_egrad_attn_12x2_d1024_lr1e3",   228.0, 503,  "egrad"),
    "V36":     ("EDesc 12×2 d=1024 v2",    "v36_edesc_12x2_d1024_lr1e3",        228.0, 503,  "edesc"),
    "V38":     ("Full EGrad 24×1 d=1024",  "v38_full_egrad_24x1_d1024_lr1e3",   342.0, 503,  "full_egrad"),

    # ── RMS+Rayleigh hybrids (★ star marker) ──
    "V73":     ("6GPT+EGPT×6 d=1280 ★RMS-Ray","v73_6gpt_1egpt6x_rmsray_d1280",   279.2, 535,  "rmsray"),
    "V76":     ("4GPT+EGPT×6+128reg d=1024 ★","v76_4gpt_1egpt6x_rmsray_d1024_reg128", 182.0, 335, "rmsray_reg"),
}

FAMILY_STYLE: dict[str, dict] = {
    "gpt":        dict(color="#2196F3", marker="o",  ms=8,  zorder=3,  label="GPT baseline"),
    "egpt":       dict(color="#FF9800", marker="s",  ms=8,  zorder=3,  label="EGPT (plain)"),
    "recurrent":  dict(color="#9C27B0", marker="^",  ms=8,  zorder=3,  label="EGPT recurrent"),
    "egrad":      dict(color="#F44336", marker="D",  ms=7,  zorder=3,  label="EGrad/EDesc"),
    "edesc":      dict(color="#4CAF50", marker="D",  ms=7,  zorder=3,  label="EGrad/EDesc"),
    "full_egrad": dict(color="#E91E63", marker="P",  ms=9,  zorder=4,  label="Full EGrad"),
    "rmsray":     dict(color="#FF5722", marker="*",  ms=18, zorder=5,  label="★ RMS+Rayleigh"),
    "rmsray_reg": dict(color="#E91E63", marker="*",  ms=18, zorder=5,  label="★ RMS+Ray+Registers"),
}

def load_results(subdir: str):
    unsharded = BASE / subdir / "unsharded"
    # Try all harness_results*.json files (timestamped and non-timestamped)
    files = sorted(unsharded.glob("harness_results*.json"))
    # Prefer non-generation files (nogen) over gsm8k-only files
    nogen = [f for f in files if "nogen" in f.name]
    main = [f for f in files if "nogen" not in f.name and "gsm" not in f.name]
    candidates = main or nogen or files
    if not candidates:
        return None, None
    r = json.loads(candidates[-1].read_text()).get("results", {})
    ppl = r.get("wikitext", {}).get("word_perplexity,none")
    accs = [r.get(t, {}).get(m) for t, m in AVG_TASKS_10]
    accs = [a for a in accs if a is not None]
    avg = sum(accs) / len(accs) if len(accs) >= 8 else None
    return ppl, avg


def make_scatter():
    rows = []
    for key, (label, subdir, params, mflops, family) in MODELS.items():
        ppl, avg = load_results(subdir)
        if ppl is None or avg is None:
            print(f"  [skip] {key}: no results yet ({subdir})")
            continue
        rows.append(dict(key=key, label=label, params=params, mflops=mflops,
                         family=family, ppl=ppl, avg=avg * 100))

    if not rows:
        print("No results loaded.")
        return

    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle("PPL & Avg-10 Accuracy vs Parameters and Compute", fontsize=14, y=1.01)

    x_axes = [("params", "Parameters (M)"), ("mflops", "MFLOPs / token")]
    y_axes = [("ppl",    "WikiText PPL ↓"),  ("avg",    "10-task Avg Acc (%) ↑")]

    seen_labels: set[str] = set()
    legend_handles: list = []

    for col, (xkey, xlabel) in enumerate(x_axes):
        for row, (ykey, ylabel) in enumerate(y_axes):
            ax = axes[row, col]
            for r in rows:
                fam = r["family"]
                if fam == "edesc":
                    fam = "egrad"   # same visual group
                st = FAMILY_STYLE.get(fam, FAMILY_STYLE["gpt"])
                lbl = st["label"]
                ax.scatter(
                    r[xkey], r[ykey],
                    color=st["color"], marker=st["marker"],
                    s=st["ms"] ** 2, zorder=st["zorder"],
                    edgecolors="white", linewidths=0.5,
                )
                # Label only V73, V9, V0, V31, V38 to avoid clutter
                if r["key"] in ("V73", "V76", "V9", "V0", "V31", "V38", "V63"):
                    offset = (8, -14) if r["key"] in ("V73", "V76") else (5, 5)
                    ax.annotate(
                        r["key"],
                        xy=(r[xkey], r[ykey]),
                        xytext=offset, textcoords="offset points",
                        fontsize=7.5, fontweight="bold" if r["key"] == "V73" else "normal",
                        color=FAMILY_STYLE.get(r["family"], FAMILY_STYLE["gpt"])["color"],
                    )
                if lbl not in seen_labels:
                    seen_labels.add(lbl)
                    legend_handles.append(Line2D(
                        [0], [0], marker=st["marker"], color="w",
                        markerfacecolor=st["color"], markersize=st["ms"],
                        label=lbl, markeredgecolor="white",
                    ))

            ax.set_xscale("log")
            ax.set_xlabel(xlabel, fontsize=10)
            ax.set_ylabel(ylabel, fontsize=10)
            if ykey == "ppl":
                ax.invert_yaxis()
            ax.grid(True, alpha=0.3, linestyle="--")
            ax.tick_params(labelsize=9)

    # Deduplicate legend handles
    seen, deduped = set(), []
    for h in legend_handles:
        if h.get_label() not in seen:
            seen.add(h.get_label())
            deduped.append(h)

    fig.legend(deduped, [h.get_label() for h in deduped],
               loc="lower center", ncol=4, fontsize=9,
               bbox_to_anchor=(0.5, -0.04), framealpha=0.9)

    fig.tight_layout(rect=[0, 0.04, 1, 1])

    for ext in ("png", "pdf"):
        path = PLOTS_DIR / f"scatter_params_flops_20260505.{ext}"
        fig.savefig(path, dpi=150, bbox_inches="tight")
        print(f"Saved: {path}")

    plt.close(fig)


if __name__ == "__main__":
    make_scatter()
