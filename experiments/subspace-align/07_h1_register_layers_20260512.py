"""H1 register variant analysis: EGPT vs GPT layer spectral alignment.

H1 = 6GPT + 1EGPT×6, d=768. Three register variants:
  h1_base    : no registers (R=0)
  reg_h1_r128: registers on ALL 7 layers (R=128, crashed GSM8K)
  reg_h1_r256: registers on ALL 7 layers (R=256)
  h1_sel_r128: selective — registers ONLY on EGPT block 6 (R=128, good GSM8K)

Plot layout (2 rows × N_models columns, one figure per op type):
  Top row:    EGPT block (block 6, energy_attention) — Pi@J alignment
  Bottom row: GPT blocks (0-5, softmax_attention) — W_O@W_V alignment
              (averaged across the 6 GPT blocks)

This directly shows whether:
  (a) EGPT-layer null-space concentration changes with different register schemes
  (b) GPT layers are affected when registers touch them (reg_h1) vs not (sel)

CLUSTER JOB ONLY.
"""
from __future__ import annotations

import json, sys, math
from pathlib import Path
from collections import defaultdict

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
    "h1_base":      (BASE / "h1_6gpt_1egpt6x_d768/unsharded",             0,   "H1 base\n(R=0)"),
    "h1_all_r128":  (BASE / "reg_h1_6gpt_1egpt6x_d768_r128/unsharded",   128,  "H1 all-layers\n(R=128)"),
    "h1_all_r256":  (BASE / "reg_h1_6gpt_1egpt6x_d768_r256/unsharded",   256,  "H1 all-layers\n(R=256)"),
    "h1_sel_r128":  (BASE / "h1_sel_reg_128_d768/unsharded",              128,  "H1 sel-EGPT\n(R=128)"),
}

N_BINS = 5
RANDOM_FRAC = 1.0 / N_BINS

# Colour: red shades for EGPT-layer ops, blue shades for GPT-layer ops
COLORS = {
    "h1_base":     "#e74c3c",
    "h1_all_r128": "#c0392b",
    "h1_all_r256": "#922b21",
    "h1_sel_r128": "#f39c12",   # orange to distinguish selective
}


# ── Utilities (same pattern as scripts 04-06) ────────────────────────────────

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
    eigenvals, V = torch.linalg.eigh(W.T @ W)
    S = eigenvals.clamp(min=0).sqrt().flip(0).cpu().numpy()
    Vh = V.T.flip(0).cpu().numpy()
    return S, Vh


def bin_energy(W_flat, Vh, bin_edges) -> list[float]:
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    W_sq = float((W ** 2).sum())
    return [float(((torch.tensor(Vh[s:e].T) @ (torch.tensor(Vh[s:e].T).T @ W)) ** 2).sum()) / W_sq
            for s, e in bin_edges]


def get_blocks(model):
    for attr in ("transformer", "model"):
        t = getattr(model, attr, None)
        if t is not None and hasattr(t, "h"):
            return t.h
        if t is not None and hasattr(t, "transformer"):
            return t.transformer.h
    raise AttributeError


def extract_layer_ops(model, d: int) -> dict:
    """Return {'egpt': {op: vec}, 'gpt': {op: vec}} separated by block type."""
    blocks = get_blocks(model)
    mixers = model.config.sequence_mixer_blocks

    egpt_ops: dict[str, list] = defaultdict(list)
    gpt_ops: dict[str, list] = defaultdict(list)

    for i, blk in enumerate(blocks):
        mixer_type = mixers[i].sequence_mixer_type if i < len(mixers) else "?"
        is_egpt = "energy" in mixer_type

        # EGPT block: extract J and Pi@J
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(getattr(attn, "c_attn", None), "weight"):
            w = attn.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                egpt_ops["J"].append(J.reshape(-1).numpy())
                for pname in ("proj_attn", "proj"):
                    p = getattr(blk, pname, None)
                    if p is not None and hasattr(p, "weight"):
                        egpt_ops["PiJ"].append(
                            (p.weight.detach().float().cpu() @ J).reshape(-1).numpy())
                        break
            continue

        # GPT block: extract J and W_O W_V
        seq = getattr(blk, "sequence_mixer", None)
        if seq is not None and hasattr(getattr(seq, "c_attn", None), "weight"):
            w = seq.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2 * d:
                J = w[:d].T @ w[d:2*d]
                gpt_ops["J"].append(J.reshape(-1).numpy())
            if w.shape[0] >= 3 * d:
                cp = getattr(seq, "c_proj", None)
                if cp is not None and hasattr(cp, "weight"):
                    gpt_ops["WOV"].append(
                        (cp.weight.detach().float().cpu() @ w[2*d:]).reshape(-1).numpy())

    # Average across blocks within each type
    return {
        "egpt": {k: np.mean(v, axis=0) for k, v in egpt_ops.items() if v},
        "gpt":  {k: np.mean(v, axis=0) for k, v in gpt_ops.items()  if v},
    }


# ── Plot: 2-row 5-bin chart ──────────────────────────────────────────────────

def plot_h1_layer_bins(all_data: dict, d: int = 768):
    bin_w = d // N_BINS
    bin_edges = [(i * bin_w, min((i + 1) * bin_w, d)) for i in range(N_BINS)]
    x_labels = [
        f"Top\n0–{int(100*bin_w/d)}%",
        f"{int(100*bin_w/d)}–{int(200*bin_w/d)}%",
        "Mid",
        f"{int(300*bin_w/d)}–{int(400*bin_w/d)}%",
        f"Bot\n{int(400*bin_w/d)}–100%",
    ]
    x = np.arange(N_BINS)
    n_models = len(all_data)
    bar_w = min(0.8 / n_models, 0.18)
    offsets = np.linspace(-(n_models - 1) * bar_w / 2, (n_models - 1) * bar_w / 2, n_models)

    # 2 rows: EGPT (top) and GPT (bottom); 2 columns: PiJ/WOV and J
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    row_config = [
        ("egpt", "PiJ",  r"EGPT block — $\Pi J$ (write op)",       "Top row: EGPT layer"),
        ("egpt", "J",    r"EGPT block — $J = W_Q^\top W_K$ (kernel)", ""),
        ("gpt",  "WOV",  r"GPT blocks (avg) — $W_O W_V$ (write op)", "Bot row: GPT layers"),
        ("gpt",  "J",    r"GPT blocks (avg) — $J$ (kernel)",          ""),
    ]

    for (row, col), (layer_key, op_key, subtitle, row_label) in zip(
        [(0,0),(0,1),(1,0),(1,1)], row_config
    ):
        ax = axes[row][col]
        for i_m, (mid, (d_m, Vh, layer_ops, label)) in enumerate(all_data.items()):
            ops = layer_ops.get(layer_key, {})
            if op_key not in ops:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes, color="gray")
                continue
            fracs = bin_energy(ops[op_key], Vh, bin_edges)
            color = COLORS.get(mid, "gray")
            ax.bar(x + offsets[i_m], fracs, width=bar_w, color=color,
                   alpha=0.8, label=label.replace("\n", " "),
                   edgecolor="white", linewidth=0.5)

        ax.axhline(RANDOM_FRAC, color="black", ls="--", lw=1.5,
                   label=f"Random ({100*RANDOM_FRAC:.0f}%)")
        ax.set_xticks(x)
        ax.set_xticklabels(x_labels, fontsize=8)
        ax.set_title(subtitle, fontsize=9)
        ax.set_ylabel("Energy fraction")
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))
        ax.legend(fontsize=7, loc="upper center")
        ax.grid(axis="y", alpha=0.3)

        if row_label:
            ax.set_ylabel(f"{row_label}\nEnergy fraction", fontsize=8)

    fig.suptitle(
        "H1 register variants: EGPT vs GPT layer spectral alignment\n"
        r"(6GPT + 1EGPT$\times$6, $d{=}768$; all-layer vs selective register application)",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(PLOTS / f"lk_h1_layers_5bin.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_h1_layers_5bin.{pdf,png}")


def plot_h1_cumulative(all_data: dict, d: int = 768):
    k_vals = [8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 614, 691, 768]
    k_vals = [k for k in k_vals if k <= d]
    k_arr = np.array(k_vals)
    rand_base = np.sqrt(k_arr / d)

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    row_config = [
        ("egpt", "PiJ",  r"EGPT block — $\Pi J$ write op"),
        ("egpt", "J",    r"EGPT block — $J$ kernel"),
        ("gpt",  "WOV",  r"GPT blocks — $W_O W_V$ write op"),
        ("gpt",  "J",    r"GPT blocks — $J$ kernel"),
    ]
    dir_pairs = [(True, "top"), (False, "bot")]

    for (row, col), (layer_key, op_key, subtitle) in zip(
        [(0,0),(0,1),(1,0),(1,1)], row_config
    ):
        ax = axes[row][col]
        ax.plot(k_arr, rand_base, "k--", lw=1.5, label=r"Random $\sqrt{k/d}$")

        for mid, (d_m, Vh, layer_ops, label) in all_data.items():
            ops = layer_ops.get(layer_key, {})
            if op_key not in ops:
                continue
            W_flat = ops[op_key]
            W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
            Wn = float(W.norm())
            color = COLORS.get(mid, "gray")
            ls_top, ls_bot = "-", ":"

            top_aligns, bot_aligns = [], []
            for k in k_vals:
                Lk_top = torch.tensor(Vh[:k].T, dtype=torch.float32)
                Lk_bot = torch.tensor(Vh[-k:].T, dtype=torch.float32)
                top_aligns.append(float((Lk_top @ (Lk_top.T @ W)).norm()) / Wn)
                bot_aligns.append(float((Lk_bot @ (Lk_bot.T @ W)).norm()) / Wn)

            lbl = label.replace("\n", " ")
            ax.plot(k_arr, top_aligns, color=color, ls=ls_top, lw=2, label=f"{lbl} top-k")
            ax.plot(k_arr, bot_aligns, color=color, ls=ls_bot, lw=1.5, label=f"{lbl} bot-k", alpha=0.7)

        ax.set_title(subtitle, fontsize=9)
        ax.set_xlabel(r"$k$")
        ax.set_ylabel(r"align$(W, L_k)$")
        ax.legend(fontsize=6, loc="upper left", ncol=2)
        ax.grid(alpha=0.3); ax.set_xlim(0, d); ax.set_ylim(0, 1.05)

        row_label = "EGPT layer" if row == 0 else "GPT layers"
        ax.set_ylabel(f"{row_label}\nalign$(W, L_k)$", fontsize=8)

    fig.suptitle(
        "H1 register variants: cumulative alignment (solid=top-k, dotted=bot-k)\n"
        r"6GPT + 1EGPT$\times$6, $d{=}768$",
        fontsize=11, fontweight="bold"
    )
    plt.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(str(PLOTS / f"lk_h1_layers_cumulative.{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_h1_layers_cumulative.{pdf,png}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    all_data = {}
    for mid, (path, n_reg, label) in MODELS.items():
        if not path.exists():
            print(f"SKIP {mid}: {path}")
            continue
        print(f"\n=== {label.replace(chr(10),' ')} ===")
        m = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
        ).to(args.device).eval()
        d = m.config.hidden_size
        W_U = get_lm_head(m).to(args.device)
        _, Vh = full_svd_via_gram(W_U, args.device)
        layer_ops = extract_layer_ops(m, d)
        del m, W_U; torch.cuda.empty_cache()

        print(f"  d={d}  egpt_ops={list(layer_ops['egpt'].keys())}  gpt_ops={list(layer_ops['gpt'].keys())}")
        for layer_key in ("egpt", "gpt"):
            for op_key, W_flat in layer_ops[layer_key].items():
                bin_w = d // N_BINS
                be = [(i*bin_w, min((i+1)*bin_w, d)) for i in range(N_BINS)]
                fracs = bin_energy(W_flat, Vh, be)
                print(f"  {layer_key} {op_key}: {[round(f,3) for f in fracs]}")
        all_data[mid] = (d, Vh, layer_ops, label)

    print("\nPlotting 5-bin chart...")
    plot_h1_layer_bins(all_data)
    print("Plotting cumulative curves...")
    plot_h1_cumulative(all_data)

    # Save data
    records = {}
    for mid, (d, Vh, layer_ops, label) in all_data.items():
        _, n_reg, _ = MODELS[mid]
        bin_w = d // N_BINS
        be = [(i*bin_w, min((i+1)*bin_w, d)) for i in range(N_BINS)]
        records[mid] = {"d": d, "n_reg": n_reg, "label": label.replace("\n"," "),
                        "random_frac": RANDOM_FRAC, "layers": {}}
        for lk in ("egpt", "gpt"):
            records[mid]["layers"][lk] = {
                op: bin_energy(wf, Vh, be)
                for op, wf in layer_ops.get(lk, {}).items()
            }
    (PLOTS / "lk_h1_layers_data.json").write_text(json.dumps(records, indent=2))
    print(f"Saved: {PLOTS}/lk_h1_layers_data.json")
    print("=== done ===")


if __name__ == "__main__":
    main()
