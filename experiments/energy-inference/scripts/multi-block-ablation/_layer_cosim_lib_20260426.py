"""Shared helpers for layer-cosim analyses (20260426).

Centralises the activation capture + plotting used by both
analyze_cosreg_layer_cosim_20260426.py and analyze_layer_cosim_400m_20260426.py
so each driver script is just a model registry + main() call.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
RESULTS_DIR = Path(__file__).resolve().parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: E402,F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer  # noqa: E402

SEQ_LEN = 512
PREFILL = (
    "The tower is part of a complex of buildings that includes the Palace of Westminster, "
    "Westminster Abbey, and St Margaret's Church. The tower was completed in 1859 and contains "
    "the famous bell known as Big Ben. The clock faces are 23 feet in diameter and illuminated "
    "at night. The tower stands 315 feet tall at the north end of the Palace of Westminster."
)


# ---------------------------------------------------------------------------
# Activation capture
# ---------------------------------------------------------------------------
def get_blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    raise AttributeError(f"Cannot find transformer blocks on {type(model)}")


def is_egpt_v1(model) -> bool:
    blk = get_blocks(model)[0]
    return hasattr(blk, "attn") and hasattr(blk, "ffwd")


def capture(model, input_ids):
    blocks = get_blocks(model)
    nl = len(blocks)
    legacy = is_egpt_v1(model)
    buf = {k: [None] * nl for k in ["h_in", "h_out", "attn", "ffwd"]}
    handles = []
    for i, block in enumerate(blocks):
        def pre_h(mod, inp, i=i):
            buf["h_in"][i] = inp[0].detach().float().squeeze(0)
        def post_h(mod, inp, out, i=i):
            buf["h_out"][i] = out.detach().float().squeeze(0)
        handles += [block.register_forward_pre_hook(pre_h),
                    block.register_forward_hook(post_h)]
        attn_src, ffwd_src = (block.attn, block.ffwd) if legacy else (block.sequence_mixer, block.mlp_block)
        def post_attn(mod, inp, out, i=i):
            buf["attn"][i] = out.detach().float().squeeze(0)
        def post_ffwd(mod, inp, out, i=i):
            buf["ffwd"][i] = out.detach().float().squeeze(0)
        handles += [attn_src.register_forward_hook(post_attn),
                    ffwd_src.register_forward_hook(post_ffwd)]
    with torch.no_grad():
        model(input_ids[:, :SEQ_LEN])
    for h in handles:
        h.remove()
    return {
        "h_out":  [buf["h_out"][i] for i in range(nl)],
        "attn":   [buf["attn"][i]  for i in range(nl)],
        "ffwd":   [buf["ffwd"][i]  for i in range(nl)],
        "update": [buf["h_out"][i] - buf["h_in"][i] for i in range(nl)],
        "nl": nl,
    }


def cosine_pair(X, Y):
    return (F.normalize(X.float(), dim=-1) * F.normalize(Y.float(), dim=-1)).sum(-1).mean().item()


def cosine_matrix(acts):
    n = len(acts)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            M[i, j] = cosine_pair(acts[i], acts[j])
    return M


def consecutive_sims(acts):
    return [cosine_pair(acts[i], acts[i + 1]) for i in range(len(acts) - 1)]


def off_diag_mean(M):
    n = M.shape[0]
    return M[~np.eye(n, dtype=bool)].mean()


def load_and_analyse(label, ckpt_path, input_ids):
    if not Path(ckpt_path).exists():
        print(f"  Missing checkpoint: {ckpt_path}")
        return None, 0
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path), dtype=torch.float32, trust_remote_code=True, device_map="cpu"
    )
    model.eval()
    acts = capture(model, input_ids)
    nl = acts["nl"]
    print(f"  {label}: {nl} blocks, legacy={is_egpt_v1(model)}")
    result = {}
    for comp in ["attn", "ffwd", "update"]:
        M = cosine_matrix(acts[comp])
        c = consecutive_sims(acts[comp])
        result[comp] = {"cosine_matrix": M, "consecutive": c, "off_diag": off_diag_mean(M)}
    del model
    return result, nl


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def _save(fig, stem):
    for ext in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.{{pdf,png}}")


def plot_heatmaps(results, nl, stem, title_suffix):
    names = list(results.keys())
    n = len(names)
    rows = [("attn", "attn_out  (pre-residual attention)"),
            ("ffwd", "ffwd_out  (pre-residual FFN)"),
            ("update", "block update  (Δh = h_out − h_in)")]
    fig, axes = plt.subplots(len(rows), n, figsize=(4.5 * n, 4.0 * len(rows)), squeeze=False)
    for r, (comp, comp_label) in enumerate(rows):
        for c, mname in enumerate(names):
            ax = axes[r][c]
            M = results[mname][comp]["cosine_matrix"]
            od = results[mname][comp]["off_diag"]
            im = ax.imshow(M, vmin=-0.2, vmax=1.0, cmap="RdYlGn", aspect="auto")
            ax.set_title(f"{mname}\n{comp_label}", fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=8)
            ax.set_ylabel("Layer", fontsize=8)
            ticks = list(range(0, nl, max(1, nl // 6)))
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            ax.text(0.97, 0.03, f"mean={od:.3f}", transform=ax.transAxes,
                    ha="right", va="bottom", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8))
    fig.suptitle(f"Inter-layer cosine similarity — {title_suffix}",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, stem)


def plot_consecutive(results, nl, stem, title_suffix, colors, markers):
    names = list(results.keys())
    x = np.arange(nl - 1)
    pair_labels = [f"{i}→{i+1}" for i in range(nl - 1)]
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    comps = [("attn",   "attn_out (pre-residual attention)"),
             ("ffwd",   "ffwd_out (pre-residual FFN)"),
             ("update", "block update (Δh)")]
    for ax, (comp, ctitle) in zip(axes, comps):
        for mname in names:
            vals = results[mname][comp]["consecutive"]
            ax.plot(x[:len(vals)], vals,
                    color=colors.get(mname, "#888"),
                    marker=markers.get(mname, "o"),
                    markersize=5, linewidth=1.8, label=mname)
        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        step = max(1, (nl - 1) // 12)
        idx = list(range(0, nl - 1, step))
        ax.set_xticks([x[i] for i in idx])
        ax.set_xticklabels([pair_labels[i] for i in idx], fontsize=8, rotation=45)
        ax.set_xlabel("Layer pair", fontsize=10)
        ax.set_ylabel("Mean cosine similarity", fontsize=10)
        ax.set_title(ctitle, fontsize=11, fontweight="bold")
        ax.legend(fontsize=8)
    fig.suptitle(f"Consecutive-layer cosine similarity — {title_suffix}",
                 fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, stem)


def plot_summary_bars(results, stem, title, colors):
    """Two-panel: off-diagonal mean (left) and consecutive mean (right) for attn_out."""
    names = list(results.keys())
    fig, axes = plt.subplots(1, 2, figsize=(max(8.0, 1.1 * len(names) * 2), 4.5))
    for ax, key, ylabel in zip(
        axes,
        ["off_diag_mean", "consec_mean"],
        ["Mean off-diagonal cosim (attn_out)",
         "Mean consecutive cosim (attn_out)"],
    ):
        if key == "off_diag_mean":
            vals = [results[m]["attn"]["off_diag"] for m in names]
        else:
            vals = [float(np.mean(results[m]["attn"]["consecutive"])) for m in names]
        bars = ax.bar(range(len(names)), vals,
                      color=[colors.get(m, "#888") for m in names],
                      edgecolor="white", alpha=0.9)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.003,
                    f"{v:.3f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    plt.tight_layout()
    _save(fig, stem)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def run(models: dict, colors: dict, markers: dict, tag: str, title_suffix: str,
        summary_title: str):
    """`models`: {label: ckpt_path}. tag drives output filename suffix."""
    first_label = next(iter(models))
    first_ckpt = models[first_label]
    tokenizer = AutoTokenizer.from_pretrained(str(first_ckpt), trust_remote_code=True)
    input_ids = tokenizer(PREFILL, return_tensors="pt")["input_ids"][:, :SEQ_LEN]
    print(f"Prefill: {input_ids.shape[1]} tokens")

    results = {}
    nl = 0
    for label, ckpt in models.items():
        print(f"\nLoading {label}  ←  {ckpt}")
        res, n = load_and_analyse(label, ckpt, input_ids)
        if res is not None:
            results[label] = res
            nl = max(nl, n)

    if not results:
        print("No models loaded — abort.")
        return

    plot_heatmaps(results, nl, f"layer_sim_cosine_{tag}_full", title_suffix)
    plot_consecutive(results, nl, f"consecutive_sim_{tag}_full",
                     title_suffix, colors, markers)
    plot_summary_bars(results, f"layer_sim_summary_{tag}_full",
                      summary_title, colors)
    print("\nDone — plots in", PLOTS_DIR)
