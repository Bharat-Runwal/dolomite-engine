#!/usr/bin/env python3
"""Build a comprehensive CSV of all benchmark results for the EGPT ablation study.

Covers: V0–V78, R1–R3, U1–U4, B1–B5, C1–C4, H1–H6, reg_* series.
Outputs:
  - experiments/analysis/all_results.csv  (canonical)
  - ../energy-GPT-neurips2026/nima/data/all_results.csv  (Overleaf bundle)

Usage: python build_results_csv.py
"""

import csv
import glob
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
BASE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
BMOE = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/results")
OUT_DIR = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/analysis")
NIMA_DIR = Path("/proj/dmfexp/nima/Code/energy/energy-GPT-neurips2026/nima/data")
NIMA_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Model metadata: (subdir, result_base, label, model_type, family,
#                  params_M, mflops, d, n_layers_unique, max_iters, notes)
# model_type: GPT | EGPT | MixH | EGrad | Hybrid | Recurrent-EGPT | Recurrent-GPT
#             Sandwich | MoE | Register-EGPT | Register-GPT | Register-Hybrid
# family groups for plotting
# ---------------------------------------------------------------------------
MODELS = [
    # ── GPT baselines ──────────────────────────────────────────────────────
    ("v0_gpt_baseline_d768",                BASE, "V0 GPT 12×1",            "GPT",            "gpt",       162, 321,  768, 12, 1,  ""),
    ("v9_gpt_baseline_d1024_lr1e3",         BASE, "V9 GPT 24×1",            "GPT",            "gpt",       354, 503, 1024, 24, 1,  ""),
    ("v12_gpt_6x2_d768_lr2e3",              BASE, "V12 GPT 6×2",            "Recurrent-GPT",  "gpt",       120, 321,  768,  6, 2,  ""),
    ("v54_parallel_gpt_12x1_d768_lr2e3",    BASE, "V54 ParallelGPT 12×1",   "GPT",            "gpt",       144, 321,  768, 12, 1,  "parallel layout"),
    # ── Deep EGPT (N distinct blocks × 1) ─────────────────────────────────
    ("v1_12x1_d768_lr2e3",                  BASE, "V1 EGPT 12×1",           "EGPT",           "egpt",      176, 359,  768, 12, 1,  "lr=2e-3"),
    ("v1_400m_d1024_lr7e4",                 BASE, "V1-400M EGPT 24×1",      "EGPT",           "egpt",      354, 755, 1024, 24, 1,  ""),
    ("v2_6x2_d768",                         BASE, "V2 EGPT 6×2",            "EGPT",           "egpt",      110, 359,  768,  6, 2,  "lr=3e-4"),
    ("v5_12x1_d768_attn_only_energy_lr2e3", BASE, "V5 AttnOnly 12×1",       "EGPT",           "egpt",      143, 331,  768, 12, 1,  "proj_mlp removed"),
    ("v6_12x1_d768_helmholtz_factored_lr2e3",BASE,"V6 HelmF 12×1",          "EGPT",           "egpt",      140, 359,  768, 12, 1,  "Helmholtz factored"),
    ("v7_12x1_d768_helmholtz_dual_lr2e3",   BASE, "V7 HelmD 12×1",          "EGPT",           "egpt",      140, 359,  768, 12, 1,  "Helmholtz dual"),
    ("v8_12x1_d768_helmholtz_dual_reversed_lr2e3",BASE,"V8 HelmDR 12×1",    "EGPT",           "egpt",      140, 359,  768, 12, 1,  "Helmholtz reversed"),
    ("v37_full_egrad_12x1_d768_lr2e3",      BASE, "V37 FullEGrad 12×1",     "EGrad",          "egrad",     141, 285,  768, 12, 1,  "EGrad attn+MLP"),
    ("v38_full_egrad_24x1_d1024_lr1e3",     BASE, "V38 FullEGrad 24×1",     "EGrad",          "egrad",     342, 503, 1024, 24, 1,  ""),
    # ── Mixed-head / EGrad ────────────────────────────────────────────────
    ("v10_mixed_12x1_d768_lr2e3",           BASE, "V10 Mixed 12×1",         "MixH",           "mixhead",   144, 285,  768, 12, 1,  "6E+6G heads"),
    ("v11_mixed_6x2_d768_lr2e3",            BASE, "V11 Mixed 6×2",          "MixH",           "mixhead",   118, 314,  768,  6, 2,  ""),
    ("v13_mixed_8e4g_12x1_d768_lr2e3",      BASE, "V13 Mixed 8E+4G 12×1",   "MixH",           "mixhead",   144, 285,  768, 12, 1,  "8E+4G heads"),
    ("v14_mixed_10e2g_12x1_d768_lr2e3",     BASE, "V14 Mixed 10E+2G 12×1",  "MixH",           "mixhead",   144, 285,  768, 12, 1,  ""),
    ("v15_energy_grad_mixed_12x1_d768_lr2e3",BASE,"V15 EGrad-MixH 12×1",   "MixH",           "mixhead",   144, 285,  768, 12, 1,  "true EGrad output"),
    ("v16_mixed_energy_descent_12x1_d768_lr2e3",BASE,"V16 EDesc-MixH 12×1","MixH",           "mixhead",   144, 285,  768, 12, 1,  "EDesc output"),
    ("v17_energy_grad_6x2_d768_lr2e3",      BASE, "V17 EGrad-MixH 6×2",    "MixH",           "mixhead",   118, 314,  768,  6, 2,  ""),
    ("v18_energy_desc_6x2_d768_lr2e3",      BASE, "V18 EDesc-MixH 6×2",    "MixH",           "mixhead",   118, 314,  768,  6, 2,  ""),
    ("v19_energy_grad_24x1_d1024_lr1e3",    BASE, "V19 EGrad-MixH 24×1",   "MixH",           "mixhead",   342, 503, 1024, 24, 1,  ""),
    ("v20_energy_desc_24x1_d1024_lr1e3",    BASE, "V20 EDesc-MixH 24×1",   "MixH",           "mixhead",   342, 503, 1024, 24, 1,  ""),
    ("v21_energy_grad_6x2_d1024_lr1e3",     BASE, "V21 EGrad-MixH 6×2 d1024","MixH",         "mixhead",   162, 252, 1024,  6, 2,  ""),
    ("v22_energy_desc_6x2_d1024_lr1e3",     BASE, "V22 EDesc-MixH 6×2 d1024","MixH",         "mixhead",   162, 252, 1024,  6, 2,  ""),
    ("v23_energy_grad_12x2_d1024_lr1e3",    BASE, "V23 EGrad-MixH 12×2",   "MixH",           "mixhead",   228, 503, 1024, 12, 2,  ""),
    ("v24_energy_desc_12x2_d1024_lr1e3",    BASE, "V24 EDesc-MixH 12×2",   "MixH",           "mixhead",   228, 503, 1024, 12, 2,  ""),
    ("v25_energy_grad_6x4_d1024_lr1e3",     BASE, "V25 EGrad-MixH 6×4",    "MixH",           "mixhead",   162, 503, 1024,  6, 4,  "uniform schedule"),
    ("v26_energy_grad_6x_ramp_d1024_lr1e3", BASE, "V26 EGrad-MixH 6×ramp", "MixH",           "mixhead",   162, 503, 1024,  6, 9,  "ramp schedule"),
    ("v27_egrad_attn_12x1_d768_lr2e3",      BASE, "V27 EGrad-Attn 12×1",   "EGrad",          "egrad",     141, 285,  768, 12, 1,  "attn-only EGrad"),
    ("v28_edesc_12x1_d768_lr2e3",           BASE, "V28 EDesc 12×1",         "EGrad",          "egrad",     141, 285,  768, 12, 1,  ""),
    ("v29_egrad_attn_6x2_d768_lr2e3",       BASE, "V29 EGrad-Attn 6×2",    "EGrad",          "egrad",     118, 314,  768,  6, 2,  ""),
    ("v30_edesc_6x2_d768_lr2e3",            BASE, "V30 EDesc 6×2",          "EGrad",          "egrad",     118, 314,  768,  6, 2,  ""),
    ("v31_egrad_attn_24x1_d1024_lr1e3",     BASE, "V31 EGrad-Attn 24×1",   "EGrad",          "egrad",     342, 503, 1024, 24, 1,  ""),
    ("v32_edesc_24x1_d1024_lr1e3",          BASE, "V32 EDesc 24×1",         "EGrad",          "egrad",     342, 503, 1024, 24, 1,  ""),
    ("v33_egrad_attn_6x2_d1024_lr1e3",      BASE, "V33 EGrad-Attn 6×2 d1024","EGrad",        "egrad",     162, 252, 1024,  6, 2,  ""),
    ("v34_edesc_6x2_d1024_lr1e3",           BASE, "V34 EDesc 6×2 d1024",   "EGrad",          "egrad",     162, 252, 1024,  6, 2,  ""),
    ("v35_egrad_attn_12x2_d1024_lr1e3",     BASE, "V35 EGrad-Attn 12×2",   "EGrad",          "egrad",     228, 503, 1024, 12, 2,  ""),
    ("v36_edesc_12x2_d1024_lr1e3",          BASE, "V36 EDesc 12×2",         "EGrad",          "egrad",     228, 503, 1024, 12, 2,  ""),
    # ── Cosreg / alignment ────────────────────────────────────────────────
    ("v39_egpt_cosreg_12x1_d768_lr2e3",     BASE, "V39 ActCosreg λ=0.01",  "EGPT",           "cosreg",    176, 359,  768, 12, 1,  "act cosreg"),
    ("v40_egpt_cosreg_24x1_d1024_lr7e4",    BASE, "V40 ActCosreg 400M",    "EGPT",           "cosreg",    354, 654, 1024, 24, 1,  "400M act cosreg"),
    ("v41_sandwich_2gpt8e2gpt_d768_lr2e3",  BASE, "V41 Sandwich 2G+8E+2G", "Sandwich",       "hybrid",    143, 359,  768, 12, 1,  ""),
    ("v48_egpt_weight_cosreg_12x1_d768_lr2e3",BASE,"V48 WtCosreg λ=0.01",  "EGPT",           "cosreg",    143, 359,  768, 12, 1,  "weight cosreg"),
    ("v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3",BASE,"V52 ActCosreg λ=0.1",  "EGPT",           "cosreg",    143, 359,  768, 12, 1,  ""),
    ("v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3",BASE,"V53 ActCosreg ramp",  "EGPT",           "cosreg",    143, 359,  768, 12, 1,  "cosine ramp to 1.0"),
    # ── Recurrent EGPT (1 block × T) ──────────────────────────────────────
    ("v56_egpt_1x12_d768_lr2e3",            BASE, "V56 Recurrent 1×12",    "Recurrent-EGPT", "recurrent", 143, 359,  768,  1,12, "RMSNorm"),
    ("v57_egpt_1x6_d768_lr2e3",             BASE, "V57 Recurrent 1×6",     "Recurrent-EGPT", "recurrent", 143, 242,  768,  1, 6,  ""),
    ("v58_egpt_1x24_d1024_lr1e3",           BASE, "V58 Recurrent 1×24",    "Recurrent-EGPT", "recurrent", 113, 755, 1024,  1,24, ""),
    ("v59_egpt_1x12_d1024_lr1e3",           BASE, "V59 Recurrent 1×12 d1024","Recurrent-EGPT","recurrent",113, 503, 1024,  1,12, ""),
    ("v63_egpt_1x12_d1408_lr2e3",           BASE, "V63 Recurrent 1×12 d1408","Recurrent-EGPT","recurrent",285, 685, 1408,  1,12, "wider"),
    ("v65_egpt_1x12_d768_layernorm_lr2e3",  BASE, "V65 Recurrent LN",      "Recurrent-EGPT", "recurrent", 143, 359,  768,  1,12, "LayerNorm"),
    ("v66_egpt_1x12_d768_rmsnorm_reileigh_lr2e3",BASE,"V66 Recurrent Rayleigh","Recurrent-EGPT","recurrent",143,359, 768,  1,12, "RMSNorm+Rayleigh"),
    # ── Hybrid: N GPT + 1 recurrent EGPT ─────────────────────────────────
    ("v73_6gpt_1egpt6x_rmsray_d1280",       BASE, "V73 6GPT+1E×6 d1280",   "Hybrid",         "hybrid",    282, 511, 1280,  7, 6,  "Rayleigh"),
    ("v76_4gpt_1egpt6x_rmsray_d1024_reg128",BASE,"V76 4GPT+1E×6+R128",    "Register-Hybrid","register",  185, 500, 1024,  5, 6,  "128 registers"),
    ("r1_4gpt_1egpt6x_rmsray_d1024",        BASE, "R1 4GPT+1E×6 d1024",    "Hybrid",         "hybrid",    166, 247, 1024,  5, 6,  "36k steps"),
    ("r2_6gpt_1egpt6x_rmsray_d1280",        BASE, "R2 6GPT+1E×6 d1280",    "Hybrid",         "hybrid",    213, 398, 1280,  7, 6,  "36k steps"),
    ("r3_11gpt_1egpt6x_rmsray_d1280",       BASE, "R3 11GPT+1E×6 d1280",   "Hybrid",         "hybrid",    393, 734, 1280, 12, 6,  "36k steps"),
    # ── U-series: GPT + multiple recurrent EGPT blocks ────────────────────
    ("u1_2gpt_4egpt3x_rmsray_d1280",        BASE, "U1 2G+4E×3+1G d1280",   "Hybrid",         "hybrid",    277, 622, 1280,  7, 3,  "Rayleigh"),
    ("u2_2gpt_4gptrec3x_d1280",             BASE, "U2 2G+4GPTrec×3+1G",    "Recurrent-GPT",  "hybrid",    284, 668, 1280,  7, 3,  "GPT control"),
    ("u3_2gpt_4egpt3x_rmsnorm_d1280",       BASE, "U3 2G+4E×3+1G RMS",     "Hybrid",         "hybrid",    277, 622, 1280,  7, 3,  "no Rayleigh"),
    ("u4_2gpt_4egpt3x_rmsray_d1024",        BASE, "U4 2G+4E×3+1G d1024",   "Hybrid",         "hybrid",    214, 483, 1024,  7, 3,  "d=1024"),
    # ── 160M hybrid series ────────────────────────────────────────────────
    ("h1_6gpt_1egpt6x_d768",                BASE, "H1 6GPT+1E×6 d768",     "Hybrid",         "hybrid_160",126, 165,  768,  7, 6,  "V73-small"),
    ("h2_6gpt_2egpt3x_d768",                BASE, "H2 6GPT+2E×3 d768",     "Hybrid",         "hybrid_160",133, 193,  768,  8, 3,  ""),
    ("h3_6gpt_4egpt_d768",                  BASE, "H3 6GPT+4EGPT d768",    "Hybrid",         "hybrid_160",143, 252,  768, 10, 1,  "deep"),
    ("h4_4gpt_1egpt8x_d768",                BASE, "H4 4GPT+1E×8 d768",     "Hybrid",         "hybrid_160",119, 165,  768,  5, 8,  "long recurrence"),
    ("h5_6gpt_1egpt1x_d768",                BASE, "H5 6GPT+1E×1 d768",     "Hybrid",         "hybrid_160",120, 165,  768,  7, 1,  "minimal EGPT"),
    ("h6_4gpt_4egpt_4gpt_d768",             BASE, "H6 4GPT+4E+4GPT d768",  "Sandwich",       "hybrid_160",162, 321,  768, 12, 1,  "balanced sandwich"),
    # ── Register experiments ───────────────────────────────────────────────
    ("reg_v0_gpt_12x1_d768_r128",           BASE, "reg-V0-GPT+R128",        "Register-GPT",   "register",  162, 321,  768, 12, 1,  "GPT+128 regs"),
    ("reg_v1_egpt_12x1_d768_r128",          BASE, "reg-V1-EGPT+R128",       "Register-EGPT",  "register",  176, 359,  768, 12, 1,  "EGPT+128 regs"),
    ("reg_v1_egpt_12x1_d768_r16",           BASE, "reg-V1-EGPT+R16",        "Register-EGPT",  "register",  176, 359,  768, 12, 1,  "EGPT+16 regs"),
    ("reg_v56_1x12_d768_r128",              BASE, "reg-V56-Rec+R128",       "Register-EGPT",  "register",  143, 359,  768,  1,12, "Recurrent+128 regs"),
    ("reg_v73_6gpt_1egpt6x_d1280_r128",     BASE, "reg-V73+R128",           "Register-Hybrid","register",  282, 511, 1280,  7, 6,  "V73+128 regs"),
    ("reg_v41_sandwich_2g8e2g_d768_r128",   BASE, "reg-V41-Sandwich+R128",  "Register-Hybrid","register",  143, 359,  768, 12, 1,  "Sandwich+128 regs"),
    ("reg_h1_6gpt_1egpt6x_d768_r128",       BASE, "reg-H1+R128",            "Register-Hybrid","register",  126, 165,  768,  7, 6,  "H1+128 regs"),
    # ── BoltzmannMoE ─────────────────────────────────────────────────────
    ("b1_boltz_moe_16x1024_d768_lr2e3",     BASE, "B1 BoltzMoE no-reg",     "MoE",            "moe",       407, 285,  768, 12, 1,  "16 iso-param experts"),
    ("b2_boltz_moe_repulsion_16x1024_d768_lr2e3",BASE,"B2 BoltzMoE rep0.01","MoE",            "moe",       407, 285,  768, 12, 1,  "repulsion λ=0.01"),
    ("b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3",BASE,"B3 BoltzMoE drop+WD","MoE",           "moe",       407, 285,  768, 12, 1,  "dropout+WD=0.3"),
    ("b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3",BASE,"B4 BoltzMoE rep0.1","MoE",      "moe",       407, 285,  768, 12, 1,  "repulsion λ=0.1"),
    ("b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3",BASE,"B5 BoltzMoE rep+drop","MoE","moe",      407, 285,  768, 12, 1,  "rep0.1+drop+WD"),
    ("c1_topk_energy_moe_4x2048_top2_d768", BMOE, "C1 TopK 4×2048 top2",   "MoE",            "moe",       256, 285,  768, 12, 1,  "proper MoE top-2"),
    ("c2_surrogate_boltz_16x1024_d768",     BMOE, "C2 SurrogateBoltz",      "MoE",            "moe",       407, 285,  768, 12, 1,  "KL distil during train"),
    ("c3_attn_moe_2x_d768",                 BMOE, "C3 AttnMoE 2x",          "MoE",            "moe",       156, 359,  768, 12, 1,  "attn MoE alignment route"),
    ("c4_paired_unit_moe_2x_d768",          BMOE, "C4 PairedUnit 2x",       "MoE",            "moe",       250, 359,  768, 12, 1,  "paired attn+FFN units"),
]

# ---------------------------------------------------------------------------
# Benchmark tasks and metrics
# ---------------------------------------------------------------------------
BENCH_TASKS = [
    ("arc_challenge", "arc_c", "acc_norm,none"),
    ("arc_easy",      "arc_e", "acc_norm,none"),
    ("boolq",         "boolq", "acc,none"),
    ("copa",          "copa",  "acc,none"),
    ("hellaswag",     "hella", "acc_norm,none"),
    ("openbookqa",    "obqa",  "acc_norm,none"),
    ("piqa",          "piqa",  "acc_norm,none"),
    ("sciq",          "sciq",  "acc,none"),
    ("winogrande",    "wino",  "acc,none"),
    ("mmlu",          "mmlu",  "acc,none"),
    ("gsm8k",         "gsm8k", "exact_match,flexible-extract"),
]
AVG_TASKS = [t for t,_,_ in BENCH_TASKS if t != "gsm8k"]
AVG_METS  = [m for _,_,m in BENCH_TASKS if _ != "gsm8k"]

def load_results(subdir, base):
    # Try standard harness results
    files = sorted(glob.glob(str(base / subdir / "unsharded" / "harness_results*.json")))
    if files:
        return json.load(open(files[-1]))["results"]
    # Try harness_final_36k.json (R-series)
    alt = base / subdir / "harness_final_36k.json"
    if alt.exists():
        return json.load(open(alt))["results"]
    return None

# ---------------------------------------------------------------------------
header = (
    ["model_id", "label", "model_type", "family",
     "params_M", "mflops_tok", "hidden_size", "n_layers_unique", "max_iters",
     "notes", "ppl", "avg10"]
    + [col for _,col,_ in BENCH_TASKS]
)

rows = []
for (subdir, base, label, mtype, family, params, mflops, d, nlayers, maxiters, notes) in MODELS:
    r = load_results(subdir, base)
    if r is None:
        continue  # skip models with no eval

    ppl = r.get("wikitext",{}).get("word_perplexity,none", None)
    bench_vals = {}
    for task, col, met in BENCH_TASKS:
        bench_vals[col] = r.get(task, {}).get(met, None)

    # 10-task average (excluding gsm8k)
    avg_vals = [r.get(t,{}).get(m, None) for t,m in zip(AVG_TASKS, AVG_METS)]
    avg_vals = [v for v in avg_vals if v is not None]
    avg10 = sum(avg_vals)/len(avg_vals) if avg_vals else None

    row = {
        "model_id": subdir,
        "label": label,
        "model_type": mtype,
        "family": family,
        "params_M": params,
        "mflops_tok": mflops,
        "hidden_size": d,
        "n_layers_unique": nlayers,
        "max_iters": maxiters,
        "notes": notes,
        "ppl": round(ppl, 4) if ppl else None,
        "avg10": round(avg10 * 100, 2) if avg10 else None,
    }
    for task, col, met in BENCH_TASKS:
        v = bench_vals[col]
        row[col] = round(v * 100, 3) if v is not None else None
    rows.append(row)

# Write CSV
for out_path in [OUT_DIR / "all_results.csv", NIMA_DIR / "all_results.csv"]:
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=header)
        w.writeheader()
        w.writerows(rows)
    print(f"Written {len(rows)} rows → {out_path}")

# Quick summary
print(f"\nTotal models with eval: {len(rows)}")
print(f"model_type distribution:")
from collections import Counter
for k,v in sorted(Counter(r["model_type"] for r in rows).items()):
    print(f"  {k:<22}: {v}")
