"""Projected-contribution decomposition: proj_attn vs proj_mlp vs scale_ff.

Answers WHY V1's same-input full-block delta cos-sim (0.536) is so much higher
than raw attn (0.198) and raw FFN (~0.001).

For dual_unconstrained EGPT blocks:
    Δ_i = -proj_attn_i(a_i) - proj_mlp_i(f_i)

where a_i = attn(ln(h)), f_i = ffn(ln(h)).

We capture FOUR intermediate quantities per block and compare across adjacent blocks:
  1. attn_raw  — raw a_i before projection
  2. proj_attn — proj_attn_i(a_i)
  3. ffn_raw   — raw f_i before projection  (scaled by scale_ff first for single-proj models)
  4. proj_ffn  — proj_mlp_i(f_i)

The key question: does proj amplify or suppress cross-block alignment?
"""

import sys, os, warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, message="Mean of empty slice")
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
sys.path.insert(0, REPO)
import lm_engine.hf_models  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE  = Path(f"{REPO}/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"

EVAL_TEXT = """
The transformer architecture has become the dominant paradigm in natural language processing.
Self-attention mechanisms allow models to relate different positions in a sequence to compute
representations. The key insight is that language understanding requires tracking long-range
dependencies between words and phrases. Energy-based models offer an alternative perspective,
where the attention computation corresponds to gradient descent on an energy function. This
connection between energy minimization and attention is not merely an analogy but a precise
mathematical relationship: the output of energy attention equals the gradient of the energy
with respect to the hidden state. Recurrent computation in transformer blocks may thus
implement an iterative optimization process, converging toward a low-energy state that captures
the semantic content of the input. Layer similarity then measures whether different blocks are
performing redundant optimization steps or distinct transformations.
""".strip()

N_TOKENS = 256

MODELS = {
    "V0 GPT\n12×1": {
        "path": BASE / "v0_gpt_baseline_d768" / "unsharded",
        "color": "#4477AA", "n_layers": 12, "arch": "GPT", "linestyle": "-",
    },
    "V1 EGPT\n12×1": {
        "path": BASE / "v1_12x1_d768_lr2e3" / "unsharded",
        "color": "#EE6677", "n_layers": 12, "arch": "EGPT", "linestyle": "-",
    },
    "V39 EGPT\n+act-cosreg": {
        "path": BASE / "v39_egpt_cosreg_12x1_d768_lr2e3" / "unsharded",
        "color": "#AA3377", "n_layers": 12, "arch": "EGPT+cosreg", "linestyle": "--",
    },
    "V41 Sandwich\n2G+8E+2G": {
        "path": BASE / "v41_sandwich_2gpt8e2gpt_d768_lr2e3" / "unsharded",
        "color": "#228833", "n_layers": 12, "arch": "Sandwich", "linestyle": "-.",
    },
    "V42 Naive\n+5k FT": {
        "path": BASE / "v42_egpt_6layer_naive_merge_finetune_d768" / "unsharded",
        "color": "#EE8800", "n_layers": 6, "arch": "Merged+FT", "linestyle": "-",
    },
    "V43 Procrustes\n+5k FT": {
        "path": BASE / "v43_egpt_6layer_procrustes_merge_finetune_d768" / "unsharded",
        "color": "#0077BB", "n_layers": 6, "arch": "Merged+FT", "linestyle": "-",
    },
}


def load_model(info):
    path = info["path"]
    if not Path(path).exists():
        print(f"  Missing: {path}")
        return None
    print(f"  Loading from {path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(
            str(path), torch_dtype=torch.float32, device_map="cpu"
        )
        model.eval()
        return model
    except Exception as e:
        print(f"  Error: {e}")
        return None


def get_input_tokens(tokenizer, text, n_tokens):
    ids = tokenizer.encode(text)[:n_tokens]
    return torch.tensor([ids])


def get_embedding(model, input_ids):
    transformer = model.transformer
    with torch.no_grad():
        h = transformer.wte(input_ids).float()
        if hasattr(transformer, "wpe"):
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            h = h + transformer.wpe(pos).float()
    return h


def get_rope_cos_sin(model, input_ids, h_base):
    transformer = model.transformer
    if not hasattr(transformer, "position_embedding_type"):
        return None
    if transformer.position_embedding_type != "rope":
        return None
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        return transformer._get_rope_cos_sin(
            key_length=input_ids.shape[1],
            position_ids=position_ids,
            dtype=h_base.dtype,
        )


def _make_capture_hook(buf):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        buf.append(o.detach().float().squeeze(0))
    return hook


def same_input_extended(model, h_base, rope_cos_sin=None):
    """Feed h_base independently to each block; return 6-tuple of lists.

    Returns:
        full_deltas   : list[(T,d)]  Δ_i = Block_i(h_base) - h_base
        attn_raws     : list[(T,d)]  raw attn sub-output a_i (pre-proj)
        ffn_raws      : list[(T,d)]  raw FFN sub-output f_i (pre-proj; None for GPT)
        proj_attn_outs: list[(T,d)]  proj_attn(a_i) for dual_unconstrained; else None
        proj_ffn_outs : list[(T,d)]  proj_mlp(f_i)  for dual_unconstrained; else None
        scale_ffs     : list[float]  scale_ff value per block (1.0 if frozen; None for GPT)
    """
    blocks = list(model.transformer.h)
    full_deltas    = []
    attn_raws      = []
    ffn_raws       = []
    proj_attn_outs = []
    proj_ffn_outs  = []
    scale_ffs      = []

    for blk in blocks:
        proj_type = getattr(blk, "proj_type", None)

        # scale_ff
        if hasattr(blk, "scale_ff"):
            scale_ffs.append(blk.scale_ff.item())
        else:
            scale_ffs.append(None)

        ab, fb, pab, pfb = [], [], [], []
        hooks = []

        for attr in ("attn", "sequence_mixer"):
            if hasattr(blk, attr):
                hooks.append(getattr(blk, attr).register_forward_hook(_make_capture_hook(ab)))
                break
        for attr in ("ffwd", "mlp", "feed_forward", "ff"):
            if hasattr(blk, attr):
                hooks.append(getattr(blk, attr).register_forward_hook(_make_capture_hook(fb)))
                break

        if proj_type == "dual_unconstrained":
            if hasattr(blk, "proj_attn"):
                hooks.append(blk.proj_attn.register_forward_hook(_make_capture_hook(pab)))
            if hasattr(blk, "proj_mlp"):
                hooks.append(blk.proj_mlp.register_forward_hook(_make_capture_hook(pfb)))

        with torch.no_grad():
            out = blk(h_base, rope_cos_sin=rope_cos_sin)
        h_out = out[0] if isinstance(out, (tuple, list)) else out

        for hk in hooks:
            hk.remove()

        full_deltas.append((h_out.float() - h_base).squeeze(0))
        attn_raws.append(ab[0] if ab else None)
        ffn_raws.append(fb[0] if fb else None)
        proj_attn_outs.append(pab[0] if pab else None)
        proj_ffn_outs.append(pfb[0] if pfb else None)

    return full_deltas, attn_raws, ffn_raws, proj_attn_outs, proj_ffn_outs, scale_ffs


def cos_sim_matrix(act_list):
    valid = [a for a in act_list if a is not None]
    if not valid:
        return None
    n = len(valid)
    vecs = [a.reshape(-1) for a in valid]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cs = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            mat[i, j] = cs
            mat[j, i] = cs
    return mat


def succ_sim(act_list):
    """Position-aligned successive cos-sim; NaN for None pairs."""
    result = []
    for i in range(len(act_list) - 1):
        a, b = act_list[i], act_list[i + 1]
        if a is not None and b is not None:
            cs = F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()
            result.append(cs)
        else:
            result.append(float("nan"))
    return result


def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


# ─── Plot: 5-panel heatmap per model ────────────────────────────────────────

def plot_five_heatmaps(full_mat, attn_raw_mat, proj_attn_mat,
                       ffn_raw_mat, proj_ffn_mat, label, stem):
    """5-panel: full / attn_raw / proj_attn / ffn_raw / proj_ffn."""
    panels = [
        (full_mat,      "Full block Δ"),
        (attn_raw_mat,  "Attn raw (pre-proj)"),
        (proj_attn_mat, "proj_attn(attn)"),
        (ffn_raw_mat,   "FFN raw (pre-proj)"),
        (proj_ffn_mat,  "proj_mlp(FFN)"),
    ]
    n_panels = sum(1 for m, _ in panels if m is not None)
    fig, axes = plt.subplots(1, n_panels, figsize=(n_panels * 4.5, 5))
    if n_panels == 1:
        axes = [axes]

    ax_idx = 0
    for mat, name in panels:
        if mat is None:
            continue
        ax = axes[ax_idx]; ax_idx += 1
        n = mat.shape[0]
        off_vals = [mat[i, j] for i in range(n) for j in range(n) if i != j]
        vmin, vmax = min(off_vals), max(off_vals)
        im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        ax.set_xticks(range(n)); ax.set_xticklabels([str(i) for i in range(n)], fontsize=max(5, 8 - n//4))
        ax.set_yticks(range(n)); ax.set_yticklabels([str(i) for i in range(n)], fontsize=max(5, 8 - n//4))
        ax.set_title(name, fontsize=9, fontweight="bold")
        vcenter = (vmin + vmax) / 2
        for i in range(n):
            for j in range(n):
                val = mat[i, j]
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=max(4, 6 - n // 4),
                        color="white" if val < vcenter else "black")

    fig.suptitle(f"Proj decomposition cosine sim — {label.replace(chr(10),' ')}",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, stem)


# ─── Plot: successive line plot with all 5 curves ───────────────────────────

def plot_successive_proj(all_results):
    """Multi-panel successive cos-sim: full / attn_raw / proj_attn / ffn_raw / proj_ffn."""
    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(5 * n_models, 5), sharey=False)
    if n_models == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, all_results.items()):
        info  = MODELS.get(label, {})
        color = info.get("color", "#888888")
        ls    = info.get("linestyle", "-")
        x     = np.arange(len(data["full_succ"]))

        ax.plot(x, data["full_succ"],     color=color, lw=2.5, marker="o", ms=6, ls=ls,
                label="Full block Δ")
        ax.plot(x, data["attn_raw_succ"], color=color, lw=1.8, marker="s", ms=4, ls=ls, alpha=0.7,
                label="Attn raw")
        if not all(np.isnan(v) for v in data["proj_attn_succ"]):
            ax.plot(x, data["proj_attn_succ"], color=color, lw=1.8, marker="^", ms=4, ls="--", alpha=0.85,
                    label="proj_attn(attn)")
        ax.plot(x, data["ffn_raw_succ"],  color="gray", lw=1.4, marker="s", ms=3, ls=ls, alpha=0.5,
                label="FFN raw")
        if not all(np.isnan(v) for v in data["proj_ffn_succ"]):
            ax.plot(x, data["proj_ffn_succ"],  color="gray", lw=1.4, marker="^", ms=3, ls="--", alpha=0.65,
                    label="proj_mlp(FFN)")

        ax.axhline(0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Layer pair (i, i+1)", fontsize=10)
        ax.set_ylabel("Same-input cos-sim", fontsize=10)
        ax.set_title(label.replace("\n", " "), fontsize=10, fontweight="bold")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Successive same-input delta cos-sim: raw vs projected contributions",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "proj_decomp_successive")


# ─── Plot: scale_ff per layer ────────────────────────────────────────────────

def plot_scale_ff(all_results):
    """Bar chart of scale_ff values per layer for each model."""
    has_any = any(
        any(v is not None and v != 1.0 for v in data["scale_ffs"])
        for data in all_results.values()
    )
    if not has_any:
        # All models have scale_ff frozen at 1.0 — skip the plot
        print("  scale_ff = 1.0 for all blocks; skipping scale_ff plot.")
        return

    n_models = len(all_results)
    fig, axes = plt.subplots(1, n_models, figsize=(4 * n_models, 4))
    if n_models == 1:
        axes = [axes]

    for ax, (label, data) in zip(axes, all_results.items()):
        info  = MODELS.get(label, {})
        color = info.get("color", "#888888")
        sff   = [v if v is not None else 0.0 for v in data["scale_ffs"]]
        ax.bar(range(len(sff)), sff, color=color, alpha=0.8, edgecolor="white")
        ax.axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.5)
        ax.set_xlabel("Block index", fontsize=9)
        ax.set_ylabel("scale_ff", fontsize=9)
        ax.set_title(label.replace("\n", " "), fontsize=9, fontweight="bold")
        ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("scale_ff per layer (1.0 = frozen default for dual_unconstrained)",
                 fontsize=10, fontweight="bold")
    plt.tight_layout()
    savefig(fig, "scale_ff_per_layer")


# ─── Plot: summary bar with all 5 metrics ───────────────────────────────────

def plot_summary_bar(all_results):
    """Grouped bar: full / attn_raw / proj_attn / ffn_raw / proj_ffn (off-diag mean)."""
    labels_list = []
    full_v, attn_v, pa_v, ffn_v, pf_v = [], [], [], [], []
    colors_list = []

    for label, data in all_results.items():
        n     = data["n_layers"]
        mask  = ~np.eye(n, dtype=bool)
        full_v.append(float(data["full_mat"][mask].mean()))

        for mat_key, lst in [("attn_raw_mat", attn_v), ("proj_attn_mat", pa_v),
                              ("ffn_raw_mat", ffn_v),   ("proj_ffn_mat", pf_v)]:
            m = data.get(mat_key)
            if m is not None:
                mm = ~np.eye(m.shape[0], dtype=bool)
                lst.append(float(m[mm].mean()))
            else:
                lst.append(float("nan"))

        labels_list.append(label.replace("\n", " "))
        colors_list.append(MODELS.get(label, {}).get("color", "#888888"))

    x     = np.arange(len(labels_list))
    width = 0.16
    fig, ax = plt.subplots(figsize=(max(9, len(labels_list) * 1.8), 5))

    offsets = [-2, -1, 0, 1, 2]
    vals    = [full_v, attn_v, pa_v, ffn_v, pf_v]
    names   = ["Full Δ", "Attn raw", "proj_attn", "FFN raw", "proj_mlp"]
    alphas  = [1.0, 0.75, 0.90, 0.5, 0.65]

    for off, vs, name, alpha in zip(offsets, vals, names, alphas):
        bars = ax.bar(x + off * width, vs, width,
                      label=name, color=colors_list, edgecolor="white", alpha=alpha)
        for i, (bar, v) in enumerate(zip(bars, vs)):
            if not np.isnan(v):
                ax.text(bar.get_x() + bar.get_width() / 2, v + 0.003,
                        f"{v:.2f}", ha="center", va="bottom", fontsize=7, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(labels_list, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg same-input cos-sim (off-diagonal)", fontsize=11)
    ax.set_title("Same-input delta: raw vs projected contributions",
                 fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", ls="--", lw=0.8)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, "proj_decomp_bar")


# ─── Main ────────────────────────────────────────────────────────────────────

def analyze_model(label, info, input_ids, tokenizer):
    print(f"\n=== {label.replace(chr(10), ' ')} ===")
    model = load_model(info)
    if model is None:
        return None

    h_base = get_embedding(model, input_ids)
    rope   = get_rope_cos_sin(model, input_ids, h_base)

    full_deltas, attn_raws, ffn_raws, proj_attn_outs, proj_ffn_outs, scale_ffs = \
        same_input_extended(model, h_base, rope)

    n = len(full_deltas)
    n_proj = sum(p is not None for p in proj_attn_outs)
    print(f"  {n} blocks, {sum(a is not None for a in attn_raws)} attn, "
          f"{sum(f is not None for f in ffn_raws)} ffn, {n_proj} proj_attn")
    if any(s is not None for s in scale_ffs):
        scale_str = [f"{s:.2f}" if s is not None else "—" for s in scale_ffs]
        print(f"  scale_ff: {scale_str}")

    full_mat      = cos_sim_matrix(full_deltas)
    attn_raw_mat  = cos_sim_matrix([a for a in attn_raws      if a is not None]) if any(a is not None for a in attn_raws)      else None
    proj_attn_mat = cos_sim_matrix([a for a in proj_attn_outs if a is not None]) if any(a is not None for a in proj_attn_outs) else None
    ffn_raw_mat   = cos_sim_matrix([a for a in ffn_raws       if a is not None]) if any(a is not None for a in ffn_raws)       else None
    proj_ffn_mat  = cos_sim_matrix([a for a in proj_ffn_outs  if a is not None]) if any(a is not None for a in proj_ffn_outs)  else None

    full_succ      = [full_mat[i, i+1] for i in range(n - 1)]
    attn_raw_succ  = succ_sim(attn_raws)
    proj_attn_succ = succ_sim(proj_attn_outs)
    ffn_raw_succ   = succ_sim(ffn_raws)
    proj_ffn_succ  = succ_sim(proj_ffn_outs)

    print(f"  Full Δ succ     (mean={np.nanmean(full_succ):.4f}): {[f'{v:.3f}' for v in full_succ]}")
    print(f"  Attn raw succ   (mean={np.nanmean(attn_raw_succ):.4f}): {[f'{v:.3f}' for v in attn_raw_succ]}")
    print(f"  proj_attn succ  (mean={np.nanmean(proj_attn_succ):.4f}): {[f'{v:.3f}' for v in proj_attn_succ]}")
    print(f"  FFN raw succ    (mean={np.nanmean(ffn_raw_succ):.4f}):  {[f'{v:.3f}' for v in ffn_raw_succ]}")
    print(f"  proj_ffn succ   (mean={np.nanmean(proj_ffn_succ):.4f}):  {[f'{v:.3f}' for v in proj_ffn_succ]}")

    del model
    return {
        "n_layers":      n,
        "full_mat":      full_mat,
        "attn_raw_mat":  attn_raw_mat,
        "proj_attn_mat": proj_attn_mat,
        "ffn_raw_mat":   ffn_raw_mat,
        "proj_ffn_mat":  proj_ffn_mat,
        "full_succ":      full_succ,
        "attn_raw_succ":  attn_raw_succ,
        "proj_attn_succ": proj_attn_succ,
        "ffn_raw_succ":   ffn_raw_succ,
        "proj_ffn_succ":  proj_ffn_succ,
        "scale_ffs":      scale_ffs,
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    input_ids = get_input_tokens(tokenizer, EVAL_TEXT, N_TOKENS)
    print(f"Using {input_ids.shape[1]} tokens.")

    all_results = {}
    for label, info in MODELS.items():
        r = analyze_model(label, info, input_ids, tokenizer)
        if r is not None:
            all_results[label] = r

    # 1. Per-model 5-panel heatmaps
    for label, data in all_results.items():
        stem = ("proj_5panel_" +
                label.replace("\n","_").replace(" ","").replace("/","")
                     .replace("(","").replace(")","").replace("+","p").replace("×","x"))
        plot_five_heatmaps(
            data["full_mat"], data["attn_raw_mat"], data["proj_attn_mat"],
            data["ffn_raw_mat"], data["proj_ffn_mat"], label, stem,
        )

    # 2. Successive line plots per model (all 5 curves)
    plot_successive_proj(all_results)

    # 3. scale_ff per layer
    plot_scale_ff(all_results)

    # 4. Summary bar chart
    plot_summary_bar(all_results)

    # 5. Summary table
    print("\n\n=== SUMMARY TABLE ===")
    hdr = f"{'Model':<32} {'N':>3} {'Full Δ':>8} {'Attn raw':>10} {'proj_attn':>10} {'FFN raw':>9} {'proj_FFN':>9}"
    print(hdr)
    print("-" * len(hdr))
    for label, data in all_results.items():
        def avg(lst):
            vals = [v for v in lst if not np.isnan(v)]
            return np.mean(vals) if vals else float("nan")
        print(f"{label.replace(chr(10),' '):<32} {data['n_layers']:>3} "
              f"{avg(data['full_succ']):>8.4f} "
              f"{avg(data['attn_raw_succ']):>10.4f} "
              f"{avg(data['proj_attn_succ']):>10.4f} "
              f"{avg(data['ffn_raw_succ']):>9.4f} "
              f"{avg(data['proj_ffn_succ']):>9.4f}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
