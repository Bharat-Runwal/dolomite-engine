"""Dedicated bar plots for update-rule ablation (parallel vs sequential, energy vs standard).

Four models, all d=768, 1.31B tokens, 12 total iterations:
  v1  EGPT 12x1   — parallel energy descent (12 distinct EGPT blocks × 1 iter)
  v54 ParGPT 12x1 — parallel standard      (12 distinct ParGPT blocks × 1 iter)
  v12 SeqGPT 6x2  — sequential standard    (6 sequential GPT blocks × 2 iter)
  v56 EGPT 1x12   — recurrent energy       (1 EGPT block × 12 iter, shared weights)

Two figures:
  Fig A: 5-bin bar chart (top 20% / 20-40% / mid / 60-80% / bot 20%)
         One panel for PiJ/WOV (write op), one for J (scoring kernel)
         Highlights that update rule barely matters; recurrence amplifies bottom

  Fig B: Simple 3-bar summary chart (top-32, random, bot-32 excess)
         Cleaner visual isolating the key message

CLUSTER JOB ONLY.
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
    "v1_egpt_12x1":  (BASE/"v1_12x1_d768_lr2e3/unsharded",
                      "EGPT 12×1\n(par. energy)"),
    "v54_par_gpt":   (BASE/"v54_parallel_gpt_12x1_d768_lr2e3/unsharded",
                      "ParGPT 12×1\n(par. standard)"),
    "v12_seq_gpt":   (BASE/"v12_gpt_6x2_d768_lr2e3/unsharded",
                      "SeqGPT 6×2\n(seq. standard)"),
    "v56_egpt_1x12": (BASE/"v56_egpt_1x12_d768_lr2e3/unsharded",
                      "EGPT 1×12\n(recurrent energy)"),
}

# Highlight: recurrent EGPT in orange, others in palette
COLORS = {
    "v1_egpt_12x1":  "#e74c3c",   # red
    "v54_par_gpt":   "#2980b9",   # blue
    "v12_seq_gpt":   "#27ae60",   # green
    "v56_egpt_1x12": "#e67e22",   # orange (stands out)
}
N_BINS = 5


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
    raise RuntimeError("no LM head")


def full_svd(W: torch.Tensor, device="cuda"):
    W = W.to(device)
    ev, V = torch.linalg.eigh(W.T @ W)
    S = ev.clamp(min=0).sqrt().flip(0).cpu().numpy()
    Vh = V.T.flip(0).cpu().numpy()
    return S, Vh


def get_write_ops(model, d: int):
    from collections import defaultdict
    layer_iters = model.config.layer_iterations
    def get_blocks(m):
        for a in ("transformer","model"):
            t = getattr(m, a, None)
            if t is not None and hasattr(t, "h"): return t.h
            if t is not None and hasattr(t, "transformer"): return t.transformer.h
        raise AttributeError
    blocks = get_blocks(model)
    egpt_idxs = [i for i,it in enumerate(layer_iters) if it>1]
    if not egpt_idxs: egpt_idxs = list(range(len(blocks)))
    ops = defaultdict(list)
    for i in egpt_idxs:
        blk = blocks[i]
        attn = getattr(blk, "attn", None)
        if attn is not None and hasattr(getattr(attn,"c_attn",None),"weight"):
            w = attn.c_attn.weight.detach().float().cpu()
            if w.shape[0] >= 2*d:
                J = w[:d].T @ w[d:2*d]
                ops["J"].append(J.reshape(-1).numpy())
                for pn in ("proj_attn","proj"):
                    p = getattr(blk,pn,None)
                    if p is not None and hasattr(p,"weight"):
                        ops["PiJ"].append((p.weight.detach().float().cpu()@J).reshape(-1).numpy())
                        break
            continue
        seq = getattr(blk,"sequence_mixer",None)
        if seq is not None and hasattr(getattr(seq,"c_attn",None),"weight"):
            w = seq.c_attn.weight.detach().float().cpu()
            if w.shape[0]>=2*d:
                J = w[:d].T @ w[d:2*d]; ops["J"].append(J.reshape(-1).numpy())
            if w.shape[0]>=3*d:
                cp = getattr(seq,"c_proj",None)
                if cp and hasattr(cp,"weight"):
                    ops["WOV"].append((cp.weight.detach().float().cpu()@w[2*d:]).reshape(-1).numpy())
    return {k: np.mean(v,axis=0) for k,v in ops.items() if v}


def bin_energy(W_flat, Vh, bin_edges):
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d,-1), dtype=torch.float32)
    Wsq = float((W**2).sum())
    return [float(((torch.tensor(Vh[s:e].T)@(torch.tensor(Vh[s:e].T).T@W))**2).sum())/Wsq
            for s,e in bin_edges]


def align_k(W_flat, Vh, k, top=True):
    d = Vh.shape[1]
    W = torch.tensor(W_flat.reshape(d,-1), dtype=torch.float32)
    rows = Vh[:k] if top else Vh[-k:]
    Lk = torch.tensor(rows.T, dtype=torch.float32)
    return float((Lk@(Lk.T@W)).norm()) / float(W.norm())


def load_one(mid, path, device):
    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    d = m.config.hidden_size
    W_U = get_lm_head(m).to(device)
    S, Vh = full_svd(W_U, device)
    ops = get_write_ops(m, d)
    del m, W_U; torch.cuda.empty_cache()
    print(f"  {mid}: d={d}  ops={list(ops.keys())}")
    return d, S, Vh, ops


# ── Figure A: 5-bin bar chart ─────────────────────────────────────────────────
def plot_5bin(all_data):
    bin_labels = ["Top\n0-20%", "20-40%", "Mid\n40-60%", "60-80%", "Bot\n80-100%"]
    x = np.arange(N_BINS)
    rand = 1.0 / N_BINS
    n_models = len(all_data)
    bw = 0.8 / n_models

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel_ops = [("Write op ($\\Pi J$ or $W_OW_V$)", ["PiJ","WOV"]),
                 ("Scoring kernel $J = W_Q^\\top W_K$",   ["J"])]

    for ax, (title, op_keys) in zip(axes, panel_ops):
        for i_m, (mid, (d, S, Vh, ops)) in enumerate(all_data.items()):
            label = MODELS[mid][1].replace("\n", " ")
            bin_w = d // N_BINS
            be = [(j*bin_w, min((j+1)*bin_w, d)) for j in range(N_BINS)]
            op_key = next((k for k in op_keys if k in ops), None)
            if op_key is None: continue
            fracs = bin_energy(ops[op_key], Vh, be)
            offset = (i_m - (n_models-1)/2) * bw
            ax.bar(x + offset, fracs, width=bw, color=COLORS[mid],
                   alpha=0.85, label=label, edgecolor="white", lw=0.5)

        ax.axhline(rand, color="black", ls="--", lw=1.5, label=f"Random (20%)")
        ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=9)
        ax.set_ylabel("Fraction of operator energy")
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0))

    fig.suptitle(
        "Update-rule ablation: parallel vs sequential, energy vs standard\n"
        r"All $d{=}768$, 1.31B tokens, 12 total iterations",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS/"lk_update_rule_5bin_v2.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(str(PLOTS/"lk_update_rule_5bin_v2.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_update_rule_5bin_v2.{pdf,png}")


# ── Figure B: top-32 / bot-32 excess summary ──────────────────────────────────
def plot_excess_summary(all_data):
    """Grouped bar: for each model show top-32 excess and bot-32 excess side by side."""
    k = 32
    mids = list(all_data.keys())
    labels = [MODELS[m][1] for m in mids]
    colors = [COLORS[m] for m in mids]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    panel_ops = [("Write op ($\\Pi J$ / $W_OW_V$)", ["PiJ","WOV"]),
                 ("Scoring kernel $J$",               ["J"])]

    for ax, (title, op_keys) in zip(axes, panel_ops):
        x = np.arange(len(mids))
        top_exc, bot_exc = [], []
        rand_top = rand_bot = None

        for mid, (d, S, Vh, ops) in all_data.items():
            rand = math.sqrt(k / d)
            rand_top = rand_bot = rand
            op_key = next((k_ for k_ in op_keys if k_ in ops), None)
            if op_key:
                t = align_k(ops[op_key], Vh, k, top=True)
                b = align_k(ops[op_key], Vh, k, top=False)
                top_exc.append(t - rand)
                bot_exc.append(b - rand)
            else:
                top_exc.append(0); bot_exc.append(0)

        bw = 0.35
        bars_top = ax.bar(x - bw/2, top_exc, width=bw, color=colors,
                          alpha=0.85, edgecolor="white", lw=0.5,
                          label="top-32 excess")
        bars_bot = ax.bar(x + bw/2, bot_exc, width=bw, color=colors,
                          alpha=0.50, edgecolor="gray", lw=0.5,
                          hatch="//", label="bot-32 excess")

        # Add value labels
        for bar in list(bars_top) + list(bars_bot):
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.003,
                    f"{h:+.3f}", ha="center", va="bottom", fontsize=7)

        ax.axhline(0, color="black", lw=1.0)
        ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
        ax.set_ylabel(r"Excess over $\sqrt{k/d}$")
        ax.set_title(f"{title}\n(solid = top-32, hatched = bot-32)", fontsize=9)
        ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)
        ax.set_ylim(-0.15, 0.25)

    fig.suptitle(
        r"Top-32 vs bot-32 excess at $k{=}32$, all $d{=}768$, 1.31B tokens",
        fontsize=11, fontweight="bold")
    plt.tight_layout()
    fig.savefig(str(PLOTS/"lk_update_rule_excess.pdf"), dpi=150, bbox_inches="tight")
    fig.savefig(str(PLOTS/"lk_update_rule_excess.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved: lk_update_rule_excess.{pdf,png}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = p.parse_args()

    all_data = {}
    for mid, (path, label) in MODELS.items():
        if not path.exists():
            print(f"SKIP {mid}: {path}"); continue
        d, S, Vh, ops = load_one(mid, path, args.device)
        all_data[mid] = (d, S, Vh, ops)

    plot_5bin(all_data)
    plot_excess_summary(all_data)
    print("=== done ===")


if __name__ == "__main__":
    main()
