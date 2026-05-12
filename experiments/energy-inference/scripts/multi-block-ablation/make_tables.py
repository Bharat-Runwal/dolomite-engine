#!/usr/bin/env python
"""Generate LaTeX tables for paper_v2.tex from harness_*.json files.

Output: experiments/energy-inference/paper/tables/<name>.tex — each file is a
self-contained `\\begin{tabular}...\\end{tabular}` block intended to be \\input{}
into the paper inside a `\\begin{table}...\\end{table}` wrapper that holds
caption/label.

Highlighting:
  * Best per "comparable" column: \\textbf{...}
  * Second best per comparable column: \\underline{...}
  * For PPL columns (lower is better) the comparison is reversed.
  * Static columns (Params, MFLOPs, Type) are never highlighted.

Run:
  python make_tables.py            # generate all tables
  python make_tables.py egpt_gap   # generate just one table

The MODELS registry below extends the one in plot_v27_egrad_edesc_20260423.py.
The TABLES registry mirrors the structure of every benchmark table in paper_v2.tex.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
HERE  = Path(__file__).resolve().parent
BASE  = HERE.parent.parent / "results/multi-block-ablation"
TABLES_DIR = HERE.parent.parent / "paper/tables"

# ---------------------------------------------------------------------------
# Models registry: id -> {label, subdir, params_M, mflops_per_tok, family,
#                          type_label, recur (e.g. "12x1"), gen ("d768"|"d1024")}
#
# Numbers (params, MFLOPs) are display values used in the existing paper tables
# and should NOT be recomputed here — they describe the model and are correct.
# `family` is one of: gpt, energy, mixed, mixhead_dag, egrad, edesc, full_egrad.
# `mflops_fwd` is forward-pass MFLOPs/token at s=4096 (matches paper convention).
# ---------------------------------------------------------------------------
MODELS = {
    # Baselines
    "V0":       dict(label="V0 GPT 12$\\times$1",          subdir="v0_gpt_baseline_d768",                       params_M=162, mflops=321, family="gpt",         type="GPT"),
    "V9":       dict(label="V9 GPT 24$\\times$1",          subdir="v9_gpt_baseline_d1024_lr1e3",                params_M=354, mflops=503, family="gpt",         type="GPT"),
    "V12":      dict(label="V12 GPT 6$\\times$2",          subdir="v12_gpt_6x2_d768_lr2e3",                     params_M=120, mflops=321, family="gpt",         type="GPT"),

    # Plain EGPT (V=K, true gradient via $W_Q^\top$ output, dual proj)
    "V1":       dict(label="V1 EGPT 12$\\times$1",         subdir="v1_12x1_d768_lr2e3",                         params_M=143, mflops=359, family="energy",      type="EGPT"),
    "V1_400m":  dict(label="V1 EGPT 24$\\times$1 (400M)",  subdir="v1_400m_d1024_lr7e4",                        params_M=400, mflops=755, family="energy",      type="EGPT"),
    "V2":       dict(label="V2 EGPT 6$\\times$2",          subdir="v2_6x2_d768",                                params_M=110, mflops=359, family="energy",      type="EGPT"),

    # Structural ablations
    "V5":       dict(label="V5 AttnOnly",                  subdir="v5_12x1_d768_attn_only_energy_lr2e3",        params_M=143, mflops=331, family="energy",      type="AttnOnly"),
    "V6":       dict(label="V6 HelmF",                     subdir="v6_12x1_d768_helmholtz_factored_lr2e3",      params_M=140, mflops=359, family="energy",      type="HelmF"),
    "V7":       dict(label="V7 HelmD",                     subdir="v7_12x1_d768_helmholtz_dual_lr2e3",          params_M=140, mflops=359, family="energy",      type="HelmD"),
    "V8":       dict(label="V8 HelmDR",                    subdir="v8_12x1_d768_helmholtz_dual_reversed_lr2e3", params_M=140, mflops=359, family="energy",      type="HelmDR"),

    # Mixed-head (additive update, shared $W_O$)
    "V10":      dict(label="V10 Mixed 12$\\times$1",       subdir="v10_mixed_12x1_d768_lr2e3",                  params_M=144, mflops=285, family="mixed",       type="Mixed"),
    "V11":      dict(label="V11 Mixed 6$\\times$2",        subdir="v11_mixed_6x2_d768_lr2e3",                   params_M=118, mflops=314, family="mixed",       type="Mixed"),
    "V13":      dict(label="V13 Mixed 8:4",                subdir="v13_mixed_8e4g_12x1_d768_lr2e3",             params_M=144, mflops=285, family="mixed",       type="Mixed"),
    "V14":      dict(label="V14 Mixed 10:2",               subdir="v14_mixed_10e2g_12x1_d768_lr2e3",            params_M=144, mflops=285, family="mixed",       type="Mixed"),

    # MixHead "dag" (intended as EGrad/EDesc but trained as plain MixedHead)
    "V15":      dict(label="V15 MixH$^\\dagger$ 12$\\times$1",  subdir="v15_energy_grad_mixed_12x1_d768_lr2e3",       params_M=144, mflops=285, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V16":      dict(label="V16 MixH$^\\dagger$ 12$\\times$1",  subdir="v16_mixed_energy_descent_12x1_d768_lr2e3",    params_M=144, mflops=285, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V17":      dict(label="V17 MixH$^\\dagger$ 6$\\times$2",   subdir="v17_energy_grad_6x2_d768_lr2e3",              params_M=118, mflops=314, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V18":      dict(label="V18 MixH$^\\dagger$ 6$\\times$2",   subdir="v18_energy_desc_6x2_d768_lr2e3",              params_M=118, mflops=314, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V19":      dict(label="V19 MixH$^\\dagger$ 24$\\times$1",  subdir="v19_energy_grad_24x1_d1024_lr1e3",            params_M=342, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V20":      dict(label="V20 MixH$^\\dagger$ 24$\\times$1",  subdir="v20_energy_desc_24x1_d1024_lr1e3",            params_M=342, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V21":      dict(label="V21 MixH$^\\dagger$ 6$\\times$2",   subdir="v21_energy_grad_6x2_d1024_lr1e3",             params_M=162, mflops=252, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V22":      dict(label="V22 MixH$^\\dagger$ 6$\\times$2",   subdir="v22_energy_desc_6x2_d1024_lr1e3",             params_M=162, mflops=252, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V23":      dict(label="V23 MixH$^\\dagger$ 12$\\times$2",  subdir="v23_energy_grad_12x2_d1024_lr1e3",            params_M=228, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V24":      dict(label="V24 MixH$^\\dagger$ 12$\\times$2",  subdir="v24_energy_desc_12x2_d1024_lr1e3",            params_M=228, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V25":      dict(label="V25 MixH$^\\dagger$ 6$\\times$4",   subdir="v25_energy_grad_6x4_d1024_lr1e3",             params_M=162, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),
    "V26":      dict(label="V26 MixH$^\\dagger$ 6$\\times$ramp",subdir="v26_energy_grad_6x_ramp_d1024_lr1e3",         params_M=162, mflops=503, family="mixhead_dag", type="MixH$^\\dagger$"),

    # True EGrad/EDesc (W_Q^T output for energy heads)
    "V27":      dict(label="V27 EGrad 12$\\times$1",       subdir="v27_egrad_attn_12x1_d768_lr2e3",             params_M=141, mflops=285, family="egrad",      type="EGrad"),
    "V28":      dict(label="V28 EDesc 12$\\times$1",       subdir="v28_edesc_12x1_d768_lr2e3",                  params_M=141, mflops=285, family="edesc",      type="EDesc"),
    "V29":      dict(label="V29 EGrad 6$\\times$2",        subdir="v29_egrad_attn_6x2_d768_lr2e3",              params_M=115, mflops=314, family="egrad",      type="EGrad"),
    "V30":      dict(label="V30 EDesc 6$\\times$2",        subdir="v30_edesc_6x2_d768_lr2e3",                   params_M=115, mflops=314, family="edesc",      type="EDesc"),
    "V31":      dict(label="V31 EGrad 24$\\times$1",       subdir="v31_egrad_attn_24x1_d1024_lr1e3",            params_M=342, mflops=503, family="egrad",      type="EGrad"),
    "V32":      dict(label="V32 EDesc 24$\\times$1",       subdir="v32_edesc_24x1_d1024_lr1e3",                 params_M=342, mflops=503, family="edesc",      type="EDesc"),
    "V33":      dict(label="V33 EGrad 6$\\times$2",        subdir="v33_egrad_attn_6x2_d1024_lr1e3",             params_M=162, mflops=252, family="egrad",      type="EGrad"),
    "V34":      dict(label="V34 EDesc 6$\\times$2",        subdir="v34_edesc_6x2_d1024_lr1e3",                  params_M=162, mflops=252, family="edesc",      type="EDesc"),
    "V35":      dict(label="V35 EGrad 12$\\times$2",       subdir="v35_egrad_attn_12x2_d1024_lr1e3",            params_M=228, mflops=503, family="egrad",      type="EGrad"),
    "V36":      dict(label="V36 EDesc 12$\\times$2",       subdir="v36_edesc_12x2_d1024_lr1e3",                 params_M=228, mflops=503, family="edesc",      type="EDesc"),
    "V37":      dict(label="V37 Full EGrad 12$\\times$1",  subdir="v37_full_egrad_12x1_d768_lr2e3",             params_M=141, mflops=285, family="full_egrad", type="Full EGrad"),
    "V38":      dict(label="V38 Full EGrad 24$\\times$1",  subdir="v38_full_egrad_24x1_d1024_lr1e3",            params_M=342, mflops=503, family="full_egrad", type="Full EGrad"),

    # Cosreg / merger / sandwich
    "V39":      dict(label="V39 act-cosreg",               subdir="v39_egpt_cosreg_12x1_d768_lr2e3",            params_M=143, mflops=359, family="energy",      type="EGPT+cosreg"),
    "V40":      dict(label="V40 act-cosreg 400M",          subdir="v40_egpt_cosreg_24x1_d1024_lr7e4",           params_M=354, mflops=654, family="energy",      type="EGPT+cosreg"),
    "V41":      dict(label="V41 Sandwich",                 subdir="v41_sandwich_2gpt8e2gpt_d768_lr2e3",         params_M=143, mflops=359, family="mixed",       type="Sandwich"),
    "V48":      dict(label="V48 wt-cosreg",                subdir="v48_egpt_weight_cosreg_12x1_d768_lr2e3",     params_M=143, mflops=359, family="energy",      type="EGPT+wt-cosreg"),
    "V52":      dict(label="V52 act-cosreg $\\lambda{=}0.1$", subdir="v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3",  params_M=143, mflops=359, family="energy",      type="EGPT+act-cosreg"),
    "V53":      dict(label="V53 act-cosreg ramp$\\to$1.0",  subdir="v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3",   params_M=143, mflops=359, family="energy",      type="EGPT+act-cosreg"),

    # Misc shared
    # NOTE: V3/V4 use shared_backbone (1 unique block × 12 iterations).
    # params_M = UNIQUE parameter count (1 block); actual training used ~2× due to
    # FSDP-2 weight-tying bug (weights were duplicated 12×).  Use with caution in
    # params-axis scatter plots — the 174M/163M represents capacity, not memory.
    "V3":       dict(label="V3 Shared $d{=}1152$",        subdir="v3_shared_d1152",                            params_M=163, mflops=359, family="energy",      type="Shared"),
    "V4":       dict(label="V4 Shared wide$^\\dagger$",   subdir="v4_shared_wide_d1152_lr2e3",                 params_M=174, mflops=359, family="energy",      type="Shared"),

    # Hybrid GPT+EGPT (new series: V73, R1–R3, U1–U4)
    # FLOPs = 2 × effective_params_per_token ≈ 2 × (GPT_unique + EGPT_unique × iters)
    "V73":  dict(label="V73 6GPT+1E$\\times$6",           subdir="v73_6gpt_1egpt6x_rmsray_d1280",              params_M=282, mflops=511, family="hybrid",      type="Hybrid"),
    "V41":  dict(label="V41 Sandwich 2G+8E+2G",           subdir="v41_sandwich_2gpt8e2gpt_d768_lr2e3",         params_M=143, mflops=359, family="hybrid",      type="Hybrid"),
    "R1":   dict(label="R1 4GPT+1E$\\times$6",            subdir="r1_4gpt_1egpt6x_rmsray_d1024",               params_M=166, mflops=247, family="hybrid",      type="Hybrid",  alt_json="harness_final_36k.json"),
    "R2":   dict(label="R2 6GPT+1E$\\times$6",            subdir="r2_6gpt_1egpt6x_rmsray_d1280",               params_M=213, mflops=398, family="hybrid",      type="Hybrid",  alt_json="harness_final_36k.json"),
    "R3":   dict(label="R3 11GPT+1E$\\times$6",           subdir="r3_11gpt_1egpt6x_rmsray_d1280",              params_M=393, mflops=734, family="hybrid",      type="Hybrid",  alt_json="harness_final_36k.json"),
    "U1":   dict(label="U1 2G+4E$\\times$3+1G",           subdir="u1_2gpt_4egpt3x_rmsray_d1280",               params_M=277, mflops=622, family="hybrid",      type="Hybrid"),
    "U2":   dict(label="U2 2G+4GPTrec+1G",                subdir="u2_2gpt_4gptrec3x_d1280",                    params_M=284, mflops=668, family="hybrid",      type="Hybrid"),
    "U3":   dict(label="U3 2G+4E$\\times$3+1G (RMS)",     subdir="u3_2gpt_4egpt3x_rmsnorm_d1280",              params_M=277, mflops=622, family="hybrid",      type="Hybrid"),
    "U4":   dict(label="U4 2G+4E$\\times$3+1G $d{=}1024$",subdir="u4_2gpt_4egpt3x_rmsray_d1024",               params_M=214, mflops=483, family="hybrid",      type="Hybrid"),
}

# ---------------------------------------------------------------------------
# Avg-accuracy tasks (the canonical 10-task set used in plot_v27_egrad_edesc).
# Tuple: (task_name, harness_metric_key)
# ---------------------------------------------------------------------------
AVG_TASKS_10 = [
    ("arc_challenge", "acc_norm,none"),
    ("arc_easy",      "acc_norm,none"),
    ("boolq",         "acc,none"),
    ("copa",          "acc,none"),
    ("hellaswag",     "acc_norm,none"),
    ("mmlu",          "acc,none"),
    ("openbookqa",    "acc_norm,none"),
    ("piqa",          "acc_norm,none"),
    ("sciq",          "acc,none"),
    ("winogrande",    "acc,none"),
]
# Single canonical list. Importers rely on this name being stable.
AVG_TASKS = AVG_TASKS_10

# Per-task display defaults (for individual columns).
TASK_DISPLAY = {
    "arc_challenge": ("ARC-C", "acc_norm,none", "pct"),
    "arc_easy":      ("ARC-E", "acc_norm,none", "pct"),
    "boolq":         ("BoolQ", "acc,none",      "pct"),
    "copa":          ("COPA",  "acc,none",      "pct"),
    "hellaswag":     ("HellaS","acc_norm,none", "pct"),
    "mmlu":          ("MMLU",  "acc,none",      "pct"),
    "openbookqa":    ("ObQA",  "acc_norm,none", "pct"),
    "piqa":          ("PIQA",  "acc_norm,none", "pct"),
    "sciq":          ("SciQ",  "acc,none",      "pct"),
    "winogrande":    ("Wino.", "acc,none",      "pct"),
    "gsm8k":         ("GSM8K", "exact_match,flexible-extract", "pct_pct"),
}

# ---------------------------------------------------------------------------
# Loading & metric extraction
# ---------------------------------------------------------------------------
_results_cache: dict[str, dict] = {}

def load_results(model_id: str) -> dict | None:
    """Return the harness `results` dict for model_id, or None if missing."""
    if model_id in _results_cache:
        return _results_cache[model_id]
    m = MODELS[model_id]
    sub = m["subdir"]
    # Some models (R1-R3) store results as harness_final_36k.json at the subdir root
    alt = m.get("alt_json")
    if alt:
        alt_path = BASE / sub / alt
        if alt_path.exists():
            res = json.load(open(alt_path))["results"]
            _results_cache[model_id] = res
            return res
    files = sorted(glob.glob(str(BASE / sub / "unsharded" / "harness_results_*.json")))
    if not files:
        _results_cache[model_id] = None
        return None
    res = json.load(open(files[-1]))["results"]
    _results_cache[model_id] = res
    return res

def get_ppl(model_id: str) -> float | None:
    r = load_results(model_id)
    if not r:
        return None
    return r.get("wikitext", {}).get("word_perplexity,none")

def get_avg(model_id: str, n_tasks: int = 10) -> float | None:
    """Return mean accuracy (0..1) over the canonical 10-task set.
    n_tasks is kept for back-compat but always resolved against AVG_TASKS_10."""
    r = load_results(model_id)
    if not r:
        return None
    vals = []
    for t, m in AVG_TASKS_10:
        v = r.get(t, {}).get(m)
        if v is None:
            return None
        vals.append(v)
    return sum(vals) / len(vals)

def get_task(model_id: str, task: str, metric: str | None = None) -> float | None:
    """Return raw 0..1 accuracy for one task; metric defaults to TASK_DISPLAY default."""
    r = load_results(model_id)
    if not r:
        return None
    if metric is None:
        metric = TASK_DISPLAY[task][1]
    return r.get(task, {}).get(metric)

# ---------------------------------------------------------------------------
# Cell formatting
# ---------------------------------------------------------------------------
def fmt_cell(value, fmt: str) -> str:
    """Format a single cell value (without highlighting)."""
    if value is None:
        return "---"
    if fmt == "ppl":
        return f"{value:.2f}"
    if fmt == "pct":           # raw 0..1 -> "47.4"
        return f"{value*100:.1f}"
    if fmt == "pct_pct":       # raw 0..1 -> "2.20\\%"
        return f"{value*100:.2f}\\%"
    if fmt == "avg_pct":       # raw 0..1 -> "47.4\\%"
        return f"{value*100:.1f}\\%"
    if fmt == "params":        # int M -> "144M"
        return f"{int(round(value))}M"
    if fmt == "int":
        return f"{int(round(value))}"
    if fmt == "str":
        return str(value)
    raise ValueError(f"unknown fmt: {fmt}")

def highlight(strings: list[str], values: list, lower_is_better: bool, eligible: list[bool]) -> list[str]:
    """Wrap best in \\textbf{} and 2nd best in \\underline{} among `eligible` rows.

    Ties: only the first occurrence is highlighted. Ties for 2nd between best
    duplicates are not separately marked.
    """
    idxs = [i for i, e in enumerate(eligible) if e and values[i] is not None]
    if not idxs:
        return strings
    sorted_idxs = sorted(idxs, key=lambda i: (values[i] if not lower_is_better else -values[i]), reverse=True)
    if not sorted_idxs:
        return strings
    out = list(strings)
    best = sorted_idxs[0]
    out[best] = f"\\textbf{{{out[best]}}}"
    if len(sorted_idxs) > 1:
        second = sorted_idxs[1]
        # Avoid underlining a tied-best; if best & second have equal values, skip 2nd.
        if values[second] != values[best]:
            out[second] = f"\\underline{{{out[second]}}}"
    return out

# ---------------------------------------------------------------------------
# Column resolvers
# ---------------------------------------------------------------------------
def resolve_value_and_fmt(model_id: str, col: dict):
    """Return (numeric_value_for_ranking, formatted_string, lower_is_better, comparable).

    `col` is one of:
      {"kind": "label"}
      {"kind": "static", "key": "params_M"|"mflops"|"type"}
      {"kind": "ppl"}                     # wikitext word_perplexity
      {"kind": "avg", "n": 7|10}          # avg accuracy
      {"kind": "task", "task": ...,       # individual task
                       "metric": ...,
                       "fmt": "pct"|"pct_pct"}
      {"kind": "literal", "value": "...", "comparable": False}
    """
    info = MODELS[model_id]
    k = col["kind"]
    if k == "label":
        return None, info["label"], False, False
    if k == "static":
        key = col["key"]
        v = info.get(key)
        fmt = col.get("fmt", {"params_M": "params", "mflops": "int", "type": "str"}[key])
        return None, fmt_cell(v, fmt), False, False
    if k == "ppl":
        v = get_ppl(model_id)
        return v, fmt_cell(v, "ppl"), True, True
    if k == "avg":
        v = get_avg(model_id, n_tasks=col.get("n", 10))
        return v, fmt_cell(v, "avg_pct"), False, True
    if k == "task":
        task = col["task"]
        metric = col.get("metric") or TASK_DISPLAY[task][1]
        fmt = col.get("fmt") or TASK_DISPLAY[task][2]
        v = get_task(model_id, task, metric)
        return v, fmt_cell(v, fmt), False, True
    if k == "literal":
        return None, col["value"], False, False
    raise ValueError(f"unknown col kind: {k}")

# ---------------------------------------------------------------------------
# Table renderer
# ---------------------------------------------------------------------------
def render_table(spec: dict) -> str:
    """Render one table to LaTeX tabular content (no \\caption / \\label / table env)."""
    cols = spec["columns"]
    rows = spec["rows"]                 # list of model_ids OR dicts {sep: ...}

    # ColSpec for tabular alignment: 'l' for label, 'r' for numeric/static
    align = "".join(c.get("align", "r") for c in cols)
    align = "l" + align[1:] if cols and cols[0]["kind"] == "label" else align

    # Header row
    header_cells = [c.get("header", "") for c in cols]
    header = " & ".join(f"\\textbf{{{h}}}" if h and not h.startswith("\\") else h for h in header_cells)

    # Build cell matrix: one list per column (across rows), so we can highlight column-wise
    n_cols = len(cols)
    model_rows = [r for r in rows if isinstance(r, str)]
    n_data = len(model_rows)
    raw_values: list[list] = [[None] * n_data for _ in cols]
    raw_strings: list[list[str]] = [["---"] * n_data for _ in cols]
    eligible: list[list[bool]] = [[False] * n_data for _ in cols]

    for i, mid in enumerate(model_rows):
        for j, col in enumerate(cols):
            v, s, lib, comp = resolve_value_and_fmt(mid, col)
            raw_values[j][i] = v
            raw_strings[j][i] = s
            eligible[j][i] = comp and (v is not None)

    # Highlight comparable columns (use the col-level `lower_is_better` flag).
    for j, col in enumerate(cols):
        if not any(eligible[j]):
            continue
        lib = (col["kind"] == "ppl")  # only PPL is lower-is-better at present
        raw_strings[j] = highlight(raw_strings[j], raw_values[j], lib, eligible[j])

    # Optionally restrict highlighting to subgroups (if "highlight_groups" specified).
    # Format: list of (start, end) row indices (inclusive). When provided,
    # highlight independently within each group.
    groups = spec.get("highlight_groups")
    if groups:
        for j, col in enumerate(cols):
            if not any(eligible[j]):
                continue
            lib = (col["kind"] == "ppl")
            new = list(raw_strings[j])
            # First strip any existing highlighting (we redo per-group)
            for i in range(n_data):
                v = raw_values[j][i]
                if v is None or not eligible[j][i]:
                    continue
                # re-format raw cell from the value, ignoring earlier wrapping
                _, s, _, _ = resolve_value_and_fmt(model_rows[i], col)
                new[i] = s
            for (g0, g1) in groups:
                idxs = list(range(g0, g1 + 1))
                grp_eligible = [eligible[j][i] for i in idxs]
                grp_values   = [raw_values[j][i] for i in idxs]
                grp_strings  = [new[i] for i in idxs]
                hi = highlight(grp_strings, grp_values, lib, grp_eligible)
                for k, idx in enumerate(idxs):
                    new[idx] = hi[k]
            raw_strings[j] = new

    # Assemble row strings, inserting \midrule separators specified by `rows`.
    body_lines = []
    data_idx = 0
    for r in rows:
        if isinstance(r, dict) and r.get("sep") == "midrule":
            body_lines.append("\\midrule")
        else:
            cells = [raw_strings[j][data_idx] for j in range(n_cols)]
            body_lines.append(" & ".join(cells) + " \\\\")
            data_idx += 1

    # Generated-by header comment
    out = []
    out.append("% Auto-generated by experiments/energy-inference/scripts/multi-block-ablation/make_tables.py")
    out.append(f"% Table: {spec['name']}")
    out.append("% DO NOT edit by hand — re-run the generator instead.")
    out.append("\\begin{tabular}{" + align + "}")
    out.append("\\toprule")
    out.append(header + " \\\\")
    out.append("\\midrule")
    out.extend(body_lines)
    out.append("\\bottomrule")
    out.append("\\end{tabular}")
    return "\n".join(out) + "\n"

# ---------------------------------------------------------------------------
# Common column definitions
# ---------------------------------------------------------------------------
COL_LABEL   = dict(kind="label", header="Model", align="l")
COL_TYPE    = dict(kind="static", key="type", fmt="str", header="Type", align="l")
COL_PARAMS  = dict(kind="static", key="params_M", header="Params", align="r")
COL_MFLOPS  = dict(kind="static", key="mflops", header="MFLOPs/tok", align="r")
COL_PPL     = dict(kind="ppl", header="PPL$\\downarrow$", align="r")
COL_AVG10   = dict(kind="avg", n=10, header="Avg$\\uparrow$", align="r")

def col_task(task, header=None, fmt=None, metric=None, align="r"):
    return dict(kind="task", task=task, metric=metric, fmt=fmt,
                header=header or TASK_DISPLAY[task][0], align=align)

# ---------------------------------------------------------------------------
# Tables: each entry mirrors one paper table.
# ---------------------------------------------------------------------------
TABLES = [
    # ---- main text -----------------------------------------------------------------
    {
        "name": "egpt_gap",
        "columns": [COL_LABEL, COL_PARAMS, COL_MFLOPS, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"), col_task("hellaswag")],
        "rows": [
            "V0", "V1",
            {"sep": "midrule"},
            "V1_400m", "V9",
        ],
    },
    {
        "name": "mixed_results",
        "columns": [COL_LABEL,
                    dict(kind="literal", value="0:12", header="E:G", align="c"),
                    COL_PARAMS, COL_MFLOPS, COL_PPL, COL_AVG10,
                    col_task("gsm8k", header="GSM8K")],
        # E:G column will be set per-row via 'literals' (see below);
        # for simplicity, render via post-processing:
        "rows": [
            "V0", "V1", "V10",
            {"sep": "midrule"},
            "V11", "V12",
        ],
        "literals": {  # row_index (0..n_data-1) -> {col_index: text}
            "E:G": {"V0": "0:12", "V1": "12:0", "V10": "6:6",
                    "V11": "6:6", "V12": "0:12"},
        },
    },
    {
        "name": "output_rule",
        "columns": [COL_LABEL, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("hellaswag"),
                    col_task("sciq"),
                    col_task("gsm8k", header="GSM8K")],
        "rows": [
            "V0", "V1", "V10",
            {"sep": "midrule"},
            "V15", "V16",
            {"sep": "midrule"},
            "V27", "V28",
        ],
    },
    {
        "name": "scaling",
        "columns": [COL_LABEL, COL_TYPE, COL_PARAMS, COL_MFLOPS, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("hellaswag"),
                    col_task("gsm8k", header="GSM8K")],
        "rows": [
            "V9", "V1_400m",
            {"sep": "midrule"},
            "V19", "V31", "V32",
        ],
    },
    {
        "name": "scaling_v25v26",
        "columns": [COL_LABEL, COL_TYPE, COL_PARAMS, COL_MFLOPS, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("hellaswag"),
                    col_task("gsm8k", header="GSM8K")],
        "rows": [
            "V9",
            {"sep": "midrule"},
            "V23", "V24",
            {"sep": "midrule"},
            "V25", "V26",
        ],
    },
    # ---- appendix: full 400M ----
    {
        "name": "scaling_full",
        "columns": [COL_LABEL, COL_TYPE, COL_PARAMS, COL_MFLOPS, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("hellaswag"),
                    col_task("gsm8k", header="GSM8K")],
        "rows": [
            "V9", "V1_400m",
            {"sep": "midrule"},
            "V19", "V20", "V31", "V32",
            {"sep": "midrule"},
            "V21", "V22", "V33", "V34",
            {"sep": "midrule"},
            "V23", "V24", "V35", "V36",
        ],
    },
    # ---- appendix: structural ablations ----
    {
        "name": "v5_v8",
        "columns": [COL_LABEL, COL_PARAMS, COL_PPL, COL_AVG10,
                    col_task("boolq"), col_task("copa")],
        "rows": [
            "V0", "V1",
            {"sep": "midrule"},
            "V5", "V6", "V7", "V8",
        ],
    },
    # ---- appendix: full small-scale benchmarks ----
    {
        "name": "all_small",
        "columns": [COL_LABEL, COL_PPL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("arc_easy"),
                    col_task("hellaswag"),
                    col_task("winogrande"),
                    col_task("boolq"),
                    col_task("piqa"),
                    col_task("copa"),
                    col_task("openbookqa"),
                    col_task("sciq"),
                    col_task("gsm8k", header="GSM8K")],
        "rows": [
            # Sorted roughly by average accuracy (will be re-sorted live by the sort_by hook below).
            "V0", "V1", "V2", "V5", "V6", "V7", "V8",
            "V10", "V11", "V12", "V13", "V14", "V15", "V16",
        ],
        "sort_by_avg": True,
    },
    # ---- cosreg benchmarks: use the canonical 10-task avg + TASK_DISPLAY default
    # metrics (acc_norm where applicable) so V0/V1 numbers match the headline tables.
    {
        "name": "cosreg_benchmarks",
        "columns": [COL_LABEL, COL_AVG10,
                    col_task("arc_challenge"),
                    col_task("arc_easy"),
                    col_task("boolq"),
                    col_task("hellaswag"),
                    col_task("mmlu"),
                    col_task("piqa"),
                    col_task("winogrande")],
        "rows": [
            "V0", "V1", "V39", "V48", "V52", "V53",
        ],
    },
]

# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def render_one(spec: dict, sort_by_avg: bool = False) -> str:
    if spec.get("sort_by_avg"):
        # Replace `rows` (list of model ids only, no separators) with sorted list
        ids = [r for r in spec["rows"] if isinstance(r, str)]
        ids.sort(key=lambda mid: (get_avg(mid) or 0.0), reverse=True)
        spec = {**spec, "rows": ids, "sort_by_avg": False}
    return render_table(spec)

def main():
    p = argparse.ArgumentParser()
    p.add_argument("name", nargs="?", help="single table name to generate")
    p.add_argument("--quiet", "-q", action="store_true")
    args = p.parse_args()

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    targets = [t for t in TABLES if (args.name is None or t["name"] == args.name)]
    if not targets:
        print(f"No table matches name={args.name}.  Known: {[t['name'] for t in TABLES]}",
              file=sys.stderr)
        sys.exit(1)

    for t in targets:
        out = render_one(t)
        path = TABLES_DIR / f"{t['name']}.tex"
        path.write_text(out)
        if not args.quiet:
            print(f"  wrote {path.relative_to(TABLES_DIR.parent.parent.parent)}  ({path.stat().st_size} bytes)")

if __name__ == "__main__":
    main()
