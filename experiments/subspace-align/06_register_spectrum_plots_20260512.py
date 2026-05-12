"""LM-head spectrum alignment for register vs non-register models.

Hypothesis: if register tokens provide explicit scratch space (L⊥ working memory),
EGPT content positions should rely LESS on the null space of W_U → the bimodal
bottom-k concentration should DECREASE as n_registers increases.

Comparisons (matched architecture, only n_registers differs):
  Group A: reg_v73 — 6GPT+1EGPT×6, d=1280, n_reg ∈ {0, 128, 256}
  Group B: reg_v1_400m — 24-block EGPT, d=1024, n_reg ∈ {0, 128, 256}

Same plots as 05_lk_spectrum_plots_20260512.py:
  Fig A: 5-bin bar chart
  Fig B: cumulative alignment curves

Outputs: plots/lk_register_5bin.{pdf,png}, lk_register_cumulative.{pdf,png}

CLUSTER JOB ONLY — never run on login node.
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/energy-inference/results/multi-block-ablation"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM

# ── Model registry ───────────────────────────────────────────────────────────
# (path, n_reg, group, display_label)
MODELS = {
    # Group A: 6GPT+1EGPT×6, d=1280
    "v73_r0":   (BASE / "v73_6gpt_1egpt6x_rmsray_d1280/unsharded",         0,   "A", "V73 (R=0)"),
    "v73_r128": (BASE / "reg_v73_6gpt_1egpt6x_d1280_r128/unsharded",      128,  "A", "V73+R128"),
    "v73_r256": (BASE / "reg_v73_6gpt_1egpt6x_d1280_r256/unsharded",      256,  "A", "V73+R256"),
    # Group B: 400M EGPT, d=1024
    "v1_r0":    (BASE / "v1_400m_d1024_lr7e4/unsharded",                    0,   "B", "V1-400M (R=0)"),
    "v1_r128":  (BASE / "reg_v1_400m_d1024_r128/unsharded",               128,  "B", "V1-400M+R128"),
    "v1_r256":  (BASE / "reg_v1_400m_d1024_r256/unsharded",               256,  "B", "V1-400M+R256"),
}

# Colour ramps: darker = more registers
COLORS_A = {0: "#e74c3c", 128: "#c0392b", 256: "#922b21"}  # reds
COLORS_B = {0: "#2980b9", 128: "#1a5276", 256: "#154360"}  # blues
OP_LABELS = {"J": r"$J$", "PiJ": r"$\Pi J$", "WOV": r"$W_O W_V$"}


# ── Weight utilities (same as script 04/05) ──────────────────────────────────

def get_lm_head(model) -> torch.Tensor:
    for attr in ("lm_head", "embed_out"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach().float().cpu()
    t = getattr(model, "transformer", None) or getattr(model, "model", None)
    if t is not None:
        wte = getattr(t, "wte", None)
        if wte is not None:
            return wte.weight.detach().float().cpu()
    raise RuntimeError("Cannot find LM head")


def full_svd_via_gram(W: torch.Tensor, device="cuda"):
    W = W.to(device)
    WtW = W.T @ W
    eigenvals, V = torch.linalg.eigh(WtW)
    S = eigenvals.clamp(min=0).sqrt().flip(0).cpu().numpy()
    Vh = V.T.flip(0).cpu().numpy()  # row 0 = top SV
    return S, Vh


def get_write_ops(model, d: int) -> dict[str, np.ndarray]:
    from collections import defaultdict
    layer_iters = model.config.layer_iterations

    def get_blocks(m):
        for attr in ("transformer", "model"):
            t = getattr(m, attr, None)
            if t is not None and hasattr(t, "h"):
                return t.h
            if t is not None and hasattr(t, "transformer"):
                return t.transformer.h
        raise AttributeError

    blocks = get_blocks(model)
    egpt_idxs = [i for i, it in enumerate(layer_iters) if it > 1]
    if not egpt_idxs:
        egpt_idxs = list(range(len(blocks)))

    ops_acc: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in egpt_idxs:
        blk = blocks[i]
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(getattr(attn, "c_attn", None), "weight"):
            w = attn.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                ops_acc["J"].append(J.reshape(-1).numpy())
                for pname in ("proj_attn", "proj"):
                    p = getattr(blk, pname, None)
                    if p is not None and hasattr(p, "weight"):
                        ops_acc["PiJ"].append(
                            (p.weight.detach().float().cpu() @ J).reshape(-1).numpy())
                        break
            continue
        seq = getattr(blk, "sequence_mixer", None)
        if seq is not None and hasattr(getattr(seq, "c_attn", None), "weight"):
            w = seq.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                ops_acc["J"].append(J.reshape(-1).numpy())
            if w.shape[0] >= 3 * d:
                cp = getattr(seq, "c_proj", None)
                if cp is not None and hasattr(cp, "weight"):
                    ops_acc["WOV"].append(
                        (cp.weight.detach().float().cpu() @ w[2*d:]).reshape(-1).numpy())

    return {k: np.mean(v, axis=0) for k, v in ops_acc.items() if v}


def bin_energy(W_flat, Vh, bin_edges):
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    W_sq = float((W ** 2).sum())
    fracs = []
    for s, e in bin_edges:
        Lk = torch.tensor(Vh[s:e].T, dtype=torch.float32)
        proj = Lk @ (Lk.T @ W)
        fracs.append(float((proj ** 2).sum()) / W_sq)
    return fracs


def cumul_align(W_flat, Vh, k_vals, top=True):
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    Wn = float(W.norm())
    aligns = []
    for k in k_vals:
        rows = Vh[:k] if top else Vh[-k:]
        Lk = torch.tensor(rows.T, dtype=torch.float32)
        proj = Lk @ (Lk.T @ W)
        aligns.append(float(proj.norm()) / Wn)
    return aligns


# ── Load and analyze ──────────────────────────────────────────────────────────

def analyze_one(mid, path, device):
    print(f"  Loading {mid}...")
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    d = m.config.hidden_size
    W_U = get_lm_head(m).to(device)
    S, Vh = full_svd_via_gram(W_U, device)
    ops = get_write_ops(m, d)
    del m, W_U; torch.cuda.empty_cache()
    print(f"    d={d}  ops={list(ops.keys())}  n_SV={len(S)}")
    return d, S, Vh, ops


# ── Figure A: 5-bin bar chart ─────────────────────────────────────────────────

def plot_5bin_register(all_data):
    # Separate into groups by dimension
    groups = {}
    for mid, (d, S, Vh, ops) in all_data.items():
        _, n_reg, grp, label = MODELS[mid]
        groups.setdefault(grp, []).append((mid, n_reg, d, Vh, ops, label))

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 2, figsize=(13, 4.5 * n_groups), squeeze=False)
    n_bins = 5

    for row_idx, (grp_key, members) in enumerate(sorted(groups.items())):
        d = members[0][2]
        bin_w = d // n_bins
        bin_edges = [(i * bin_w, min((i+1)*bin_w, d)) for i in range(n_bins)]
        bin_pct_labels = [
            f"Top\n0–{int(100*bin_w/d)}%",
            f"{int(100*bin_w/d)}–{int(200*bin_w/d)}%",
            f"Mid\n{int(200*bin_w/d)}–{int(300*bin_w/d)}%",
            f"{int(300*bin_w/d)}–{int(400*bin_w/d)}%",
            f"Bot\n{int(400*bin_w/d)}–100%",
        ]
        random_frac = 1.0 / n_bins
        x = np.arange(n_bins)
        color_map = COLORS_A if grp_key == "A" else COLORS_B

        for col_idx, op_key in enumerate(["PiJ", "J"]):
            ax = axes[row_idx][col_idx]
            n_models = sum(1 for _, nr, _, _, ops, _ in members if op_key in ops)
            width = min(0.8 / max(n_models, 1), 0.25)
            i_bar = 0

            for mid, n_reg, dv, Vh, ops, label in sorted(members, key=lambda x: x[1]):
                if op_key not in ops:
                    continue
                fracs = bin_energy(ops[op_key], Vh, bin_edges)
                color = color_map.get(n_reg, "#888888")
                ax.bar(x + i_bar * width - (n_models - 1) * width / 2,
                       fracs, width=width, color=color, alpha=0.8,
                       label=f"{label} (R={n_reg})", edgecolor="white", lw=0.5)
                i_bar += 1

            ax.axhline(random_frac, color="black", ls="--", lw=1.5,
                       label=f"Random ({100*random_frac:.0f}%)")
            ax.set_xticks(x)
            ax.set_xticklabels(bin_pct_labels, fontsize=8)
            ax.set_ylabel("Fraction of energy")
            op_label = {"PiJ": r"$\Pi J$ (write op)", "J": r"$J = W_Q^\top W_K$ (kernel)"}
            title_grp = "6GPT+1EGPT×6, d=1280" if grp_key == "A" else "EGPT 400M, d=1024"
            ax.set_title(f"{title_grp} — {op_label.get(op_key, op_key)}")
            ax.legend(fontsize=7)
            ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
            ax.grid(axis="y", alpha=0.3)

    fig.suptitle("Register tokens vs LM-head spectral alignment: does R reduce null-space use?",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(PLOTS / f"lk_register_5bin.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_register_5bin.{pdf,png}")


# ── Figure B: Cumulative curves ───────────────────────────────────────────────

def plot_cumulative_register(all_data):
    groups = {}
    for mid, (d, S, Vh, ops) in all_data.items():
        _, n_reg, grp, label = MODELS[mid]
        groups.setdefault(grp, []).append((mid, n_reg, d, Vh, ops, label))

    n_groups = len(groups)
    fig, axes = plt.subplots(n_groups, 2, figsize=(13, 5 * n_groups), squeeze=False)

    for row_idx, (grp_key, members) in enumerate(sorted(groups.items())):
        d = members[0][2]
        k_vals = sorted(set([16, 32, 64, 128, 192, 256, 320, 384, 512,
                              int(d*0.6), int(d*0.7), int(d*0.8), int(d*0.9), d]))
        k_vals = [k for k in k_vals if 0 < k <= d]
        k_arr = np.array(k_vals)
        rand_base = np.sqrt(k_arr / d)
        color_map = COLORS_A if grp_key == "A" else COLORS_B

        for col_idx, direction in enumerate(["top", "bot"]):
            ax = axes[row_idx][col_idx]
            ax.plot(k_arr, rand_base, "k--", lw=1.5, label=r"Random $\sqrt{k/d}$")

            for mid, n_reg, dv, Vh, ops, label in sorted(members, key=lambda x: x[1]):
                op_key = "PiJ" if "PiJ" in ops else ("J" if "J" in ops else None)
                if op_key is None:
                    continue
                W_flat = ops[op_key]
                aligns = cumul_align(W_flat, Vh, k_vals, top=(direction == "top"))
                color = color_map.get(n_reg, "#888888")
                ls = "-" if n_reg == 0 else ("--" if n_reg == 128 else ":")
                ax.plot(k_arr, aligns, color=color, ls=ls, lw=2,
                        label=f"{label} (R={n_reg}) {op_key}")

            ax.set_xlabel(r"$k$")
            ax.set_ylabel(r"align$(W, L_k)$")
            dir_label = "top-$k$ (dominant vocab)" if direction == "top" else "bottom-$k$ (near-null)"
            grp_label = "6GPT+1EGPT×6, d=1280" if grp_key == "A" else "EGPT 400M, d=1024"
            ax.set_title(f"{grp_label} — {dir_label}")
            ax.legend(fontsize=7)
            ax.grid(alpha=0.3)
            ax.set_xlim(0, d)
            ax.set_ylim(0, 1.05)

    fig.suptitle(r"Cumulative LM-head alignment: effect of register count $R$",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(PLOTS / f"lk_register_cumulative.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_register_cumulative.{pdf,png}")


# ── Entry point ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_data = {}
    for mid, (path, n_reg, grp, label) in MODELS.items():
        if not path.exists():
            print(f"SKIP {mid}: {path}")
            continue
        d, S, Vh, ops = analyze_one(mid, path, args.device)
        all_data[mid] = (d, S, Vh, ops)

    if not all_data:
        print("No models loaded."); return

    print("\nGenerating 5-bin chart...")
    plot_5bin_register(all_data)
    print("Generating cumulative curves...")
    plot_cumulative_register(all_data)

    # Save bin data
    records = {}
    for mid, (d, S, Vh, ops) in all_data.items():
        _, n_reg, grp, label = MODELS[mid]
        n_bins = 5; bin_w = d // n_bins
        bin_edges = [(i*bin_w, min((i+1)*bin_w, d)) for i in range(n_bins)]
        records[mid] = {"d": d, "n_reg": n_reg, "group": grp, "label": label,
                        "random_frac": 1.0/n_bins, "ops": {}}
        for op_key, W_flat in ops.items():
            fracs = bin_energy(W_flat, Vh, bin_edges)
            records[mid]["ops"][op_key] = fracs
            print(f"  {label} {op_key}: bins={[round(f,3) for f in fracs]}")

    (PLOTS / "lk_register_5bin_data.json").write_text(json.dumps(records, indent=2))
    print(f"\nSaved: {PLOTS}/lk_register_5bin_data.json")
    print("=== done ===")


if __name__ == "__main__":
    main()
