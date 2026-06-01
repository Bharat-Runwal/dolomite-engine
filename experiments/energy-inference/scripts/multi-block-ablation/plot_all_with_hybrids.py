"""Generate scatter plots including V73, R1-R3, U1-U4 hybrid models.

Saves PDFs to experiments/energy-inference/paper/figs/ and
nima/paper/figs/ (the Overleaf bundle).

Usage: python plot_all_with_hybrids.py
"""
import json
import glob
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
OUT_DIR = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/paper/figs")
NIMA_DIR = Path("/proj/dmfexp/nima/Code/energy/energy-GPT-neurips2026/nima/figs")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ['arc_challenge','arc_easy','boolq','copa','hellaswag','openbookqa','piqa','sciq','winogrande','mmlu']
METS  = ['acc_norm,none','acc_norm,none','acc,none','acc,none','acc_norm,none','acc_norm,none','acc_norm,none','acc,none','acc,none','acc,none']

def load(subdir, alt=None):
    if alt:
        p = BASE / subdir / alt
        if p.exists():
            return json.load(open(p))["results"]
    files = sorted(glob.glob(str(BASE / subdir / "unsharded" / "harness_results_*.json")))
    if not files: return None
    return json.load(open(files[-1]))["results"]

def metrics(r):
    if r is None: return None, None, None
    ppl = r.get("wikitext",{}).get("word_perplexity,none", None)
    vals = [r.get(t,{}).get(m,0) for t,m in zip(TASKS,METS)]
    avg = sum(vals)/len(vals)
    gsm = r.get("gsm8k",{}).get("exact_match,flexible-extract", None)
    return ppl, avg, gsm

# Model registry: (key, label, subdir, params_M, mflops, family, alt_json)
MODELS = [
    # GPT baselines
    ("V0",  "V0 GPT",           "v0_gpt_baseline_d768",                      162, 321, "gpt",     None),
    ("V9",  "V9 GPT",           "v9_gpt_baseline_d1024_lr1e3",               354, 503, "gpt",     None),
    ("V54", "V54 ParGPT",       "v54_parallel_gpt_12x1_d768_lr2e3",          144, 321, "gpt",     None),
    # Deep EGPT
    ("V1",  "V1 EGPT",          "v1_12x1_d768_lr2e3",                        176, 359, "egpt",    None),
    ("V1e", "V1 EGPT 400M",     "v1_400m_d1024_lr7e4",                       354, 755, "egpt",    None),
    # Mixed-head / EGrad
    ("V10", "V10 Mixed",        "v10_mixed_12x1_d768_lr2e3",                 144, 285, "mixed",   None),
    ("V15", "V15 MixH",         "v15_energy_grad_mixed_12x1_d768_lr2e3",     144, 285, "mixhead", None),
    ("V16", "V16 MixH",         "v16_mixed_energy_descent_12x1_d768_lr2e3",  144, 285, "mixhead", None),
    ("V19", "V19 MixH",         "v19_energy_grad_24x1_d1024_lr1e3",          342, 503, "mixhead", None),
    ("V20", "V20 MixH",         "v20_energy_desc_24x1_d1024_lr1e3",          342, 503, "mixhead", None),
    ("V23", "V23 MixH",         "v23_energy_grad_12x2_d1024_lr1e3",          228, 503, "mixhead", None),
    ("V24", "V24 MixH",         "v24_energy_desc_12x2_d1024_lr1e3",          228, 503, "mixhead", None),
    ("V31", "V31 EGrad",        "v31_egrad_attn_24x1_d1024_lr1e3",           342, 503, "egrad",   None),
    ("V32", "V32 EDesc",        "v32_edesc_24x1_d1024_lr1e3",                342, 503, "egrad",   None),
    ("V35", "V35 EGrad",        "v35_egrad_attn_12x2_d1024_lr1e3",           228, 503, "egrad",   None),
    ("V38", "V38 EGrad",        "v38_full_egrad_24x1_d1024_lr1e3",           342, 503, "egrad",   None),
    # Hybrid GPT+EGPT
    ("V41", "V41 Sandwich",     "v41_sandwich_2gpt8e2gpt_d768_lr2e3",        143, 359, "hybrid",  None),
    ("V73", "V73 6G+1E×6",      "v73_6gpt_1egpt6x_rmsray_d1280",             282, 511, "hybrid",  None),
    ("R1",  "R1 4G+1E×6",       "r1_4gpt_1egpt6x_rmsray_d1024",              166, 247, "hybrid",  "harness_final_36k.json"),
    ("R2",  "R2 6G+1E×6",       "r2_6gpt_1egpt6x_rmsray_d1280",              213, 398, "hybrid",  "harness_final_36k.json"),
    ("R3",  "R3 11G+1E×6",      "r3_11gpt_1egpt6x_rmsray_d1280",             393, 734, "hybrid",  "harness_final_36k.json"),
    ("U1",  "U1 2G+4E×3+1G",    "u1_2gpt_4egpt3x_rmsray_d1280",              277, 622, "hybrid",  None),
    ("U2",  "U2 2G+4Grec+1G",   "u2_2gpt_4gptrec3x_d1280",                   284, 668, "hybrid",  None),
    ("U3",  "U3 2G+4E×3+1G",    "u3_2gpt_4egpt3x_rmsnorm_d1280",             277, 622, "hybrid",  None),
]

COLORS = {
    "gpt":     "#2196F3",   # blue
    "egpt":    "#F44336",   # red
    "mixed":   "#9C27B0",   # purple
    "mixhead": "#9C27B0",
    "egrad":   "#E91E63",   # pink-red
    "hybrid":  "#FF9800",   # orange
}
MARKERS = {
    "gpt":     "o",
    "egpt":    "s",
    "mixed":   "^",
    "mixhead": "^",
    "egrad":   "D",
    "hybrid":  "*",
}

rows = []
for entry in MODELS:
    key, label, subdir, params, mflops, fam, alt = entry
    r = load(subdir, alt)
    ppl, avg, gsm = metrics(r)
    if ppl is None: continue
    rows.append(dict(key=key, label=label, params=params, mflops=mflops,
                     fam=fam, ppl=ppl, avg=avg, gsm=gsm))

def make_scatter(ax, xs, ys, rows, xlabel, ylabel, xlim=None, ylim=None,
                 annotate_keys=None, log_x=True):
    for row in rows:
        x, y = row[xs], row[ys]
        if x is None or y is None: continue
        fam = row["fam"]
        ax.scatter(x, y, color=COLORS[fam], marker=MARKERS[fam],
                   s=80 if fam == "hybrid" else 50,
                   zorder=3 if fam == "hybrid" else 2,
                   alpha=0.9)
        if annotate_keys and row["key"] in annotate_keys:
            ax.annotate(row["key"], (x, y), textcoords="offset points",
                        xytext=(5, 3), fontsize=7)
    if log_x: ax.set_xscale("log")
    ax.set_xlabel(xlabel, fontsize=11)
    ax.set_ylabel(ylabel, fontsize=11)
    if xlim: ax.set_xlim(xlim)
    if ylim: ax.set_ylim(ylim)
    ax.grid(True, alpha=0.3)

LEGEND = [
    mpatches.Patch(color=COLORS["gpt"],     label="GPT baseline"),
    mpatches.Patch(color=COLORS["egpt"],    label="Deep EGPT"),
    mpatches.Patch(color=COLORS["mixhead"], label="Mixed-head / EGrad"),
    mpatches.Patch(color=COLORS["hybrid"],  label="Hybrid GPT+EGPT (new)"),
]
KEY_LABELS = {"V0","V9","V1","V31","V19","V73","R3","U1","V41"}

# ---------------------------------------------------------------------------
# Figure 1: PPL vs params
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
make_scatter(axes[0], "params", "ppl", rows,
             "Parameters (M)", "WikiText PPL ↓", annotate_keys=KEY_LABELS)
axes[0].set_title("PPL vs. Parameters")
make_scatter(axes[1], "params", "avg", rows,
             "Parameters (M)", "Avg 10-task accuracy ↑", annotate_keys=KEY_LABELS)
axes[1].set_title("Accuracy vs. Parameters")
for ax in axes:
    ax.legend(handles=LEGEND, fontsize=8, loc="best")
plt.suptitle("All models: deep EGPT, mixed-head, and hybrid GPT+EGPT",
             fontsize=12, fontweight="bold")
plt.tight_layout()
for out in [OUT_DIR / "scatter_all_with_hybrids_params.pdf",
            NIMA_DIR / "scatter_all_with_hybrids_params.pdf"]:
    plt.savefig(out, bbox_inches="tight")
print(f"Saved PPL/Acc vs params")
plt.close()

# ---------------------------------------------------------------------------
# Figure 2: PPL vs FLOPs
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
make_scatter(axes[0], "mflops", "ppl", rows,
             "FLOPs/token (MFLOPs)", "WikiText PPL ↓", annotate_keys=KEY_LABELS)
axes[0].set_title("PPL vs. FLOPs/token")
make_scatter(axes[1], "mflops", "avg", rows,
             "FLOPs/token (MFLOPs)", "Avg 10-task accuracy ↑", annotate_keys=KEY_LABELS)
axes[1].set_title("Accuracy vs. FLOPs/token")
for ax in axes:
    ax.legend(handles=LEGEND, fontsize=8, loc="best")
plt.suptitle("All models: efficiency frontier (log FLOPs axis)",
             fontsize=12, fontweight="bold")
plt.tight_layout()
for out in [OUT_DIR / "scatter_all_with_hybrids_flops.pdf",
            NIMA_DIR / "scatter_all_with_hybrids_flops.pdf"]:
    plt.savefig(out, bbox_inches="tight")
print(f"Saved PPL/Acc vs FLOPs")
plt.close()

# ---------------------------------------------------------------------------
# Figure 3: GSM8K vs params (only models with GSM8K)
gsm_rows = [r for r in rows if r["gsm"] is not None]
fig, ax = plt.subplots(figsize=(7, 5))
make_scatter(ax, "params", "gsm", gsm_rows,
             "Parameters (M)", "GSM8K exact-match (flexible) ↑",
             annotate_keys={r["key"] for r in gsm_rows})
ax.set_title("GSM8K vs. Parameters — GPT leads; EGPT/hybrid lag")
ax.legend(handles=LEGEND, fontsize=8)
plt.tight_layout()
for out in [OUT_DIR / "scatter_gsm8k_all_with_hybrids.pdf",
            NIMA_DIR / "scatter_gsm8k_all_with_hybrids.pdf"]:
    plt.savefig(out, bbox_inches="tight")
print(f"Saved GSM8K vs params")
plt.close()

print(f"\nAll plots saved to:\n  {OUT_DIR}\n  {NIMA_DIR}")
