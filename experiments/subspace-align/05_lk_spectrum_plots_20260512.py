"""LM-head spectrum alignment plots (post-DL, arXiv/camera-ready).

Two complementary figures:

Fig A — 5-bin bar chart (headline):
    Divide W_U's d right singular vectors into 5 equal-width bins
    (top-20%, 20-40%, 40-60%, 60-80%, bottom-20%).
    For each write operator, show the fraction of Frobenius energy
    in each bin.  Random baseline: 20% per bin (dashed horizontal line).
    Bimodal pattern = high bars at both extremes, dip in the middle.

Fig B — Cumulative alignment curve (supporting):
    Two panels: sweep top-k (k from 0 to d) and bottom-k.
    y-axis: align(W, L_k) = frac of energy in top/bottom-k SVs.
    Dashed line: random baseline sqrt(k/d).
    The bimodal pattern shows as EGPT rising steeply at BOTH ends.

Models: matched EGPT (410m) + RecGPT (410m) + EGPT-Dual (V71).
Operators: J=W_Q^T W_K (scoring kernel) and Pi@J / W_O W_V (write ops).

Usage (cluster job only — never run on login node):
    python 05_lk_spectrum_plots_20260512.py
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

MODELS = {
    "410m_egpt":     (BASE / "410m_hybrid_s8e4/unsharded",   True,  "EGPT"),
    "410m_recgpt":   (BASE / "410m_recgpt_s8e4/unsharded",   False, "RecGPT"),
    "v71_egpt_dual": (BASE / "v71_hybrid_8gpt_4egpt_rmsray_d1280/unsharded", True, "EGPT-Dual"),
}

# Colours per model-operator pair
STYLES = {
    ("410m_egpt",     "J"):    dict(color="#e74c3c", ls="-",  marker="o", lw=2.0),
    ("410m_egpt",     "PiJ"):  dict(color="#c0392b", ls="--", marker="s", lw=2.0),
    ("410m_recgpt",   "J"):    dict(color="#2980b9", ls="-",  marker="o", lw=2.0),
    ("410m_recgpt",   "WOV"):  dict(color="#1a5276", ls="--", marker="s", lw=2.0),
    ("v71_egpt_dual", "J"):    dict(color="#e67e22", ls="-",  marker="^", lw=1.5),
    ("v71_egpt_dual", "PiJ"):  dict(color="#d35400", ls="--", marker="v", lw=1.5),
}

OP_LABELS = {"J": r"$J = W_Q^\top W_K$", "PiJ": r"$\Pi J$",
             "WOV": r"$W_O W_V$", "PiaJ": r"$\Pi_a J$"}


# ── Weight extraction ────────────────────────────────────────────────────────

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
    """Full right-singular basis via W^T W eigendecomposition on GPU.
    Returns Vh: [d, d], row 0 = most dominant SV, row -1 = least dominant."""
    W = W.to(device)
    WtW = W.T @ W                          # [d, d]
    eigenvals, V = torch.linalg.eigh(WtW)  # eigenvals ascending; V: [d, d]
    S = eigenvals.clamp(min=0).sqrt()
    # Flip: row 0 = largest SV
    S = S.flip(0).cpu().numpy()
    Vh = V.T.flip(0).cpu().numpy()  # [d, d]
    return S, Vh


def get_write_ops(model, d: int, device="cuda") -> dict[str, np.ndarray]:
    """Extract flat write operator vectors for all recurrent EGPT blocks."""
    from collections import defaultdict
    layer_iters = model.config.layer_iterations
    ops_acc: dict[str, list[np.ndarray]] = defaultdict(list)

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

    for i in egpt_idxs:
        blk = blocks[i]
        # EGPT attention block
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(getattr(attn, "c_attn", None), "weight"):
            w = attn.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                ops_acc["J"].append(J.reshape(-1).numpy())
                for pname in ("proj_attn", "proj"):
                    p = getattr(blk, pname, None)
                    if p is not None and hasattr(p, "weight"):
                        Pi = p.weight.detach().float().cpu()
                        ops_acc["PiJ"].append((Pi @ J).reshape(-1).numpy())
                        break
            continue
        # RecGPT / GPT block
        seq = getattr(blk, "sequence_mixer", None)
        if seq is not None and hasattr(getattr(seq, "c_attn", None), "weight"):
            w = seq.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                ops_acc["J"].append(J.reshape(-1).numpy())
            if w.shape[0] >= 3 * d:
                cproj = getattr(seq, "c_proj", None)
                if cproj is not None and hasattr(cproj, "weight"):
                    WO = cproj.weight.detach().float().cpu()
                    ops_acc["WOV"].append((WO @ w[2*d:]).reshape(-1).numpy())

    # Average across blocks
    return {k: np.mean(v, axis=0) for k, v in ops_acc.items() if v}


def bin_energy(W_flat: np.ndarray, Vh: np.ndarray,
               bin_edges: list[tuple[int, int]]) -> list[float]:
    """Fraction of ||W||_F^2 in each spectral bin defined by (start, end) SV indices."""
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    W_norm_sq = float((W ** 2).sum())
    fracs = []
    for start, end in bin_edges:
        Lk = torch.tensor(Vh[start:end].T, dtype=torch.float32)  # [d, width]
        proj = Lk @ (Lk.T @ W)
        fracs.append(float((proj ** 2).sum()) / W_norm_sq)
    return fracs


def cumul_align(W_flat: np.ndarray, Vh: np.ndarray,
                k_vals: list[int], top: bool) -> list[float]:
    """align(W, L_k) for each k in k_vals, from top or bottom."""
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    W_norm = float(W.norm())
    aligns = []
    for k in k_vals:
        rows = Vh[:k] if top else Vh[-k:]
        Lk = torch.tensor(rows.T, dtype=torch.float32)
        proj = Lk @ (Lk.T @ W)
        aligns.append(float(proj.norm()) / W_norm)
    return aligns


# ── Main analysis ────────────────────────────────────────────────────────────

def load_and_analyze(model_id: str, path: Path, device: str):
    print(f"\n=== {model_id} ===")
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    d = m.config.hidden_size

    W_U = get_lm_head(m).to(device)
    S, Vh = full_svd_via_gram(W_U, device)
    ops = get_write_ops(m, d, device)
    del m, W_U
    torch.cuda.empty_cache()
    print(f"  d={d}  ops={list(ops.keys())}  SV range=[{S[0]:.2f},{S[-1]:.4f}]")
    return d, S, Vh, ops


# ── Figure A: 5-bin bar chart ────────────────────────────────────────────────

def plot_5bin(all_data: dict, d: int = 1280):
    n_bins = 5
    bin_w = d // n_bins  # 256 for d=1280
    bin_edges = [(i * bin_w, (i + 1) * bin_w) for i in range(n_bins)]
    bin_labels = [f"Top\n{int(100*s/d)}-{int(100*e/d)}%" for s, e in bin_edges]
    bin_labels[-1] = f"Bot\n{int(100*(d-bin_w)/d)}-100%"

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    random_frac = 1.0 / n_bins

    for ax_idx, (group_label, op_keys, ax) in enumerate([
        ("J = W_Q^T W_K  (scoring kernel)", ["J"], axes[0]),
        ("Write operators (Pi J and W_O W_V)", ["PiJ", "WOV"], axes[1]),
    ]):
        x = np.arange(n_bins)
        width = 0.8 / max(len(all_data) * len(op_keys), 1)
        i_bar = 0
        for mid, (d_m, S, Vh, ops) in all_data.items():
            _, _, label = MODELS[mid]
            # Only use the first model's Vh for bin definitions (all d=1280 matched)
            for op_key in op_keys:
                if op_key not in ops:
                    continue
                fracs = bin_energy(ops[op_key], Vh, bin_edges)
                style = STYLES.get((mid, op_key), {})
                color = style.get("color", "gray")
                lbl = f"{label} {OP_LABELS.get(op_key, op_key)}"
                ax.bar(x + i_bar * width - 0.4 + width / 2, fracs,
                       width=width, color=color, alpha=0.75, label=lbl,
                       edgecolor="white", linewidth=0.5)
                i_bar += 1

        ax.axhline(random_frac, color="black", ls="--", lw=1.5,
                   label=f"Random ({100*random_frac:.0f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(bin_labels, fontsize=8)
        ax.set_ylabel("Fraction of operator energy")
        ax.set_title(group_label)
        ax.legend(fontsize=7, loc="upper center")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        r"Spectral energy distribution across LM-head $\mathbf{W}_U$ singular-vector spectrum",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fpath = PLOTS / f"lk_5bin_spectrum_{ext}"
        fig.savefig(str(fpath).replace("_pdf", f".{ext}").replace("_png", f".{ext}"),
                    dpi=150, bbox_inches="tight")
    fig.savefig(str(PLOTS / "lk_5bin_spectrum.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(str(PLOTS / "lk_5bin_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_5bin_spectrum.{pdf,png}")


# ── Figure B: Cumulative alignment curves ───────────────────────────────────

def plot_cumulative(all_data: dict):
    k_vals = [16, 32, 64, 128, 192, 256, 320, 384, 512, 640, 768, 896, 1024,
              1088, 1152, 1216, 1280]
    k_arr = np.array(k_vals)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax_top, ax_bot = axes

    for mid, (d, S, Vh, ops) in all_data.items():
        rand = np.sqrt(k_arr / d)
        _, _, label = MODELS[mid]

        for op_key, W_flat in ops.items():
            style = STYLES.get((mid, op_key), dict(color="gray", ls="-", lw=1.5))
            lbl = f"{label} {OP_LABELS.get(op_key, op_key)}"
            k_use = [k for k in k_vals if k <= d]
            ka = np.array(k_use)

            top_align = cumul_align(W_flat, Vh, k_use, top=True)
            bot_align = cumul_align(W_flat, Vh, k_use, top=False)

            ax_top.plot(ka, top_align, label=lbl, **style)
            ax_bot.plot(ka, bot_align, label=lbl, **style)

    # Random baseline (use d=1280 for the dashed line)
    d_ref = 1280
    k_ref = np.array([k for k in k_vals if k <= d_ref])
    rand_ref = np.sqrt(k_ref / d_ref)
    for ax in (ax_top, ax_bot):
        ax.plot(k_ref, rand_ref, "k--", lw=1.5, label=r"Random $\sqrt{k/d}$")
        ax.set_xlabel("$k$ (number of singular vectors)")
        ax.set_ylabel(r"align$(W, L_k)$")
        ax.legend(fontsize=7, loc="upper left")
        ax.grid(alpha=0.3)
        ax.set_xlim(0, max(k_vals))
        ax.set_ylim(0, 1.05)

    ax_top.set_title(r"$\mathrm{align}(W,\, \mathrm{top}\text{-}k)$ — dominant vocab")
    ax_bot.set_title(r"$\mathrm{align}(W,\, \mathrm{bot}\text{-}k)$ — near-null space")

    fig.suptitle(
        r"Cumulative LM-head alignment: top-$k$ and bottom-$k$ right singular subspaces",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS / "lk_cumulative_spectrum.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(str(PLOTS / "lk_cumulative_spectrum.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_cumulative_spectrum.{pdf,png}")


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_data = {}
    for mid, (path, is_egpt, label) in MODELS.items():
        if not path.exists():
            print(f"SKIP {mid}: {path}")
            continue
        d, S, Vh, ops = load_and_analyze(mid, path, args.device)
        all_data[mid] = (d, S, Vh, ops)

    if not all_data:
        print("No models found.")
        return

    print("\nGenerating 5-bin bar chart...")
    plot_5bin(all_data)

    print("Generating cumulative alignment curves...")
    plot_cumulative(all_data)

    # Save per-bin data for reference
    records = {}
    for mid, (d, S, Vh, ops) in all_data.items():
        n_bins = 5
        bin_w = d // n_bins
        bin_edges = [(i * bin_w, (i + 1) * bin_w) for i in range(n_bins)]
        records[mid] = {
            "d": d, "ops": {},
            "bin_edges": bin_edges,
            "random_frac": 1.0 / n_bins,
        }
        for op_key, W_flat in ops.items():
            records[mid]["ops"][op_key] = bin_energy(W_flat, Vh, bin_edges)
    (PLOTS / "lk_5bin_data.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {PLOTS}/lk_5bin_data.json")
    print("\n=== done ===")


if __name__ == "__main__":
    main()
