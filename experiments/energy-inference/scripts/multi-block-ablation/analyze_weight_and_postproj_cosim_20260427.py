"""analyze_weight_and_postproj_cosim_20260427.py

Two complementary analyses of consecutive-layer similarity:

  A) Weight-space cosim
     For each consecutive block pair, compute cosine similarity of individual
     weight matrices:
       W_Q, W_K      — from attn.c_attn (EGPT) / sequence_mixer.c_attn (GPT)
       W1, W2        — from ffwd.W1/W2 (EGPT) / mlp_block.c_fc split (GPT)
       proj_attn     — from blk.proj_attn.weight (EGPT only)
       proj_mlp      — from blk.proj_mlp.weight (EGPT only)
       W_V, W_O      — from GPT c_attn[2d:] / c_proj
     Hypothesis: proj_attn and proj_mlp are the "exploration/exploitation"
     matrices and may be LESS similar across layers than W_Q/W_K/W1/W2,
     which could reflect diverse transformation strategies per block despite
     similar attention patterns.

  B) Post-projection activation cosim
     Hook into proj_attn and proj_mlp outputs to capture the projected
     attention and FFN contributions to the residual stream, separately.
     Pre-proj cosim (attn_out, ffwd_out) was already captured in
     _layer_cosim_lib_20260426.py; this adds the POST-proj view.
     Also captures the proj_attn and proj_mlp contributions for the
     parallel GPT model (V55) where attn and FFN streams are cleanly additive.

Outputs (in results/multi-block-ablation/plots/):
  weight_cosim_consec_all.{pdf,png}        — per-type consecutive lines, all models
  weight_cosim_summary_bar.{pdf,png}       — mean consec cosim per weight type per model
  weight_cosim_heatmap_{tag}.{pdf,png}     — full i×j weight cosim heatmaps
  postproj_cosim_consec_all.{pdf,png}      — post-proj consecutive lines
  postproj_cosim_summary_bar.{pdf,png}     — summary: pre vs post proj
  layer_cosim_stats_weight.json            — all stats
"""

from __future__ import annotations

import json
import sys
import warnings
warnings.filterwarnings("ignore")
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

HERE = Path(__file__).resolve().parent
REPO = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401

RESULTS_DIR = REPO / "experiments/energy-inference/results/multi-block-ablation"
PLOTS_DIR   = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

SEQ_LEN = 512
PREFILL = (
    "The tower is part of a complex of buildings that includes the Palace of Westminster, "
    "Westminster Abbey, and St Margaret's Church. The tower was completed in 1859 and contains "
    "the famous bell known as Big Ben. The clock faces are 23 feet in diameter and illuminated "
    "at night. The tower stands 315 feet tall at the north end of the Palace of Westminster."
)

# ── Model registry ────────────────────────────────────────────────────────────
MODELS_176M = {
    "V0 GPT":                 RESULTS_DIR / "v0_gpt_baseline_d768"                           / "unsharded",
    "V1 EGPT":                RESULTS_DIR / "v1_12x1_d768_lr2e3"                             / "unsharded",
    "V39 act-cosreg\nλ=0.01": RESULTS_DIR / "v39_egpt_cosreg_12x1_d768_lr2e3"                / "unsharded",
    "V52 act-cosreg\nλ=0.1":  RESULTS_DIR / "v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3"         / "unsharded",
    "V53 act-cosreg\nramp→1": RESULTS_DIR / "v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3"        / "unsharded",
    "V48 wt-cosreg":          RESULTS_DIR / "v48_egpt_weight_cosreg_12x1_d768_lr2e3"         / "unsharded",
}
MODELS_400M = {
    "V9 GPT 400M":            RESULTS_DIR / "v9_gpt_baseline_d1024_lr1e3"                    / "unsharded",
    "V1 EGPT 400M":           RESULTS_DIR / "v1_400m_d1024_lr7e4"                            / "unsharded",
    "V31 EGrad":              RESULTS_DIR / "v31_egrad_attn_24x1_d1024_lr1e3"                / "unsharded",
    "V32 EDesc":              RESULTS_DIR / "v32_edesc_24x1_d1024_lr1e3"                     / "unsharded",
    "V40 act-cosreg\n400M":   RESULTS_DIR / "v40_egpt_cosreg_24x1_d1024_lr7e4"               / "unsharded",
    "V55 ParGPT\n400M":       RESULTS_DIR / "v55_parallel_gpt_24x1_d1024_lr1p5e3"            / "unsharded",
}

COLORS_176M = {
    "V0 GPT":                 "#4878CF",
    "V1 EGPT":                "#D65F5F",
    "V39 act-cosreg\nλ=0.01": "#F4A582",
    "V52 act-cosreg\nλ=0.1":  "#B2182B",
    "V53 act-cosreg\nramp→1": "#762A83",
    "V48 wt-cosreg":          "#2CA02C",
}
COLORS_400M = {
    "V9 GPT 400M":           "#4878CF",
    "V1 EGPT 400M":          "#D65F5F",
    "V31 EGrad":             "#2CA02C",
    "V32 EDesc":             "#FF7F0E",
    "V40 act-cosreg\n400M":  "#8C564B",
    "V55 ParGPT\n400M":      "#17BECF",
}
MARKERS = {n: m for n, m in zip(
    list(MODELS_176M) + list(MODELS_400M),
    ["s","o","^","D","P","X","s","o","^","D","P","*"]
)}


# ── Utilities ─────────────────────────────────────────────────────────────────

def flat_cosim(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a.float().reshape(1, -1), b.float().reshape(1, -1)
    return F.cosine_similarity(a, b).item()


def weight_cosim_matrix(weights_list):
    """Full n×n cosim matrix for a list of weight tensors (one per layer)."""
    n = len(weights_list)
    M = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if weights_list[i] is not None and weights_list[j] is not None:
                M[i, j] = flat_cosim(weights_list[i], weights_list[j])
            else:
                M[i, j] = float("nan")
    return M


def consec(lst):
    return [flat_cosim(lst[i], lst[i+1])
            for i in range(len(lst)-1)
            if lst[i] is not None and lst[i+1] is not None]


def nanmean(arr):
    vals = [v for v in arr if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def off_diag_mean(M: np.ndarray) -> float:
    n = M.shape[0]
    mask = ~np.eye(n, dtype=bool) & ~np.isnan(M)
    return float(M[mask].mean()) if mask.any() else float("nan")


def _save(fig, stem):
    for ext in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {stem}.{{pdf,png}}")


# ── Weight extraction ─────────────────────────────────────────────────────────

def extract_weights(model):
    """
    Return dict: weight_type → list[tensor | None], one entry per block.
    Types: W_Q, W_K, W_V (GPT only), W_O / c_proj (attn output),
           W1, W2 (FFN), proj_attn, proj_mlp.
    Each tensor is the raw weight matrix (flattened for cosim).
    """
    blocks = model.transformer.h
    nl = len(blocks)
    # W_down = FFN down projection (GPT c_proj); separate from W_O (attn output proj)
    wts = {k: [] for k in ["W_Q","W_K","W_V","W_O","W1","W2","W_down","proj_attn","proj_mlp"]}

    def _app(key, val):
        wts[key].append(val)

    for blk in blocks:
        # ---- Attention weights ----
        # Resolve the attention module regardless of attribute name
        attn_mod = (getattr(blk, "attn", None) or getattr(blk, "sequence_mixer", None))
        if attn_mod is None:
            for k in ["W_Q","W_K","W_V","W_O"]: _app(k, None)
        elif hasattr(attn_mod, "c_attn"):
            # Pure EGPT (EnergyAttention_QK): c_attn = [W_Q; W_K]
            ca = attn_mod.c_attn.weight
            d = ca.shape[0] // 2
            _app("W_Q", ca[:d].detach()); _app("W_K", ca[d:].detach())
            _app("W_V", None); _app("W_O", None)
        elif hasattr(attn_mod, "c_attn_energy"):
            # EGrad/EDesc MixedHead: c_attn_energy=[W_Q_e;W_K_e], c_attn_gpt=[W_Q_g;W_K_g;W_V_g]
            cae = attn_mod.c_attn_energy.weight   # (2*d_e, d)
            d_e = cae.shape[0] // 2
            # Use energy heads as the primary W_Q/W_K (they implement true gradient)
            _app("W_Q", cae[:d_e].detach()); _app("W_K", cae[d_e:].detach())
            cag = attn_mod.c_attn_gpt.weight      # (3*d_g, d)
            d_g = cag.shape[0] // 3
            _app("W_V", cag[2*d_g:].detach())    # GPT-head V
            _app("W_O", attn_mod.W_O_gpt.weight.detach() if hasattr(attn_mod, "W_O_gpt") else None)
        elif hasattr(attn_mod, "c_attn") or True:
            # Plain GPT sequence_mixer: c_attn = [W_Q; W_K; W_V]
            if hasattr(attn_mod, "c_attn"):
                ca = attn_mod.c_attn.weight
                d = ca.shape[0] // 3
                _app("W_Q", ca[:d].detach()); _app("W_K", ca[d:2*d].detach())
                _app("W_V", ca[2*d:].detach())
            else:
                _app("W_Q", None); _app("W_K", None); _app("W_V", None)
            _app("W_O", attn_mod.c_proj.weight.detach() if hasattr(attn_mod, "c_proj") else None)

        # ---- FFN weights ----
        ffn = getattr(blk, "ffwd", None) or getattr(blk, "mlp_block", None)
        if ffn is not None and hasattr(ffn, "W1"):
            # EGPT Energy_MLP: W1 and W2 are weight-tied up/down projections
            _app("W1", ffn.W1.weight.detach())
            _app("W2", ffn.W2.weight.detach())
            _app("W_down", None)
        elif ffn is not None and hasattr(ffn, "c_fc"):
            # GPT SwiGLU: c_fc = [gate; up] stacked; c_proj = down
            cf = ffn.c_fc.weight
            half = cf.shape[0] // 2
            _app("W1", cf[:half].detach())   # gate
            _app("W2", cf[half:].detach())   # up (value)
            _app("W_down", ffn.c_proj.weight.detach() if hasattr(ffn, "c_proj") else None)
        else:
            _app("W1", None); _app("W2", None); _app("W_down", None)

        # ---- Projection weights (EGPT only) ----
        _app("proj_attn", blk.proj_attn.weight.detach() if hasattr(blk, "proj_attn") else None)
        _app("proj_mlp",  blk.proj_mlp.weight.detach()  if hasattr(blk, "proj_mlp")  else None)

    # Ensure all lists are length nl
    for k in wts:
        while len(wts[k]) < nl:
            wts[k].append(None)

    return wts


# ── Activation post-proj capture ──────────────────────────────────────────────

def capture_postproj(model, input_ids):
    """
    Run a forward pass and collect:
      pre_attn:   attn sub-output BEFORE proj_attn  (pre-proj)
      post_attn:  output of proj_attn                (post-proj attn contribution)
      pre_ffn:    ffn sub-output BEFORE proj_mlp     (pre-proj)
      post_ffn:   output of proj_mlp                 (post-proj FFN contribution)
      delta:      h_out - h_in                        (full block update)
    For parallel GPT (proj_attn/proj_mlp both present) this cleanly separates
    the two additive contributions to the residual stream.
    """
    blocks = model.transformer.h
    nl = len(blocks)
    buf = {k: [None]*nl for k in
           ["pre_attn","post_attn","pre_ffn","post_ffn","delta"]}
    handles = []

    for i, blk in enumerate(blocks):
        def make_h_hooks(i=i):
            def pre_h(mod, inp):
                buf["delta"][i] = inp[0].detach().float().squeeze(0)  # store h_in
            def post_h(mod, inp, out):
                h_out = out[0] if isinstance(out, tuple) else out
                if buf["delta"][i] is not None:
                    buf["delta"][i] = (h_out.detach().float().squeeze(0)
                                       - buf["delta"][i])
            return pre_h, post_h

        pre_h, post_h = make_h_hooks()
        handles += [blk.register_forward_pre_hook(pre_h),
                    blk.register_forward_hook(post_h)]

        # Pre-proj attn
        attn_src = getattr(blk, "attn", None) or getattr(blk, "sequence_mixer", None)
        if attn_src:
            def make_pre_attn(i=i):
                def fn(m, inp, out):
                    buf["pre_attn"][i] = (out[0] if isinstance(out,tuple) else out).detach().float().squeeze(0)
                return fn
            handles.append(attn_src.register_forward_hook(make_pre_attn()))

        # Post-proj attn
        if hasattr(blk, "proj_attn"):
            def make_post_attn(i=i):
                def fn(m, inp, out):
                    buf["post_attn"][i] = (out[0] if isinstance(out,tuple) else out).detach().float().squeeze(0)
                return fn
            handles.append(blk.proj_attn.register_forward_hook(make_post_attn()))
        elif attn_src and hasattr(attn_src, "c_proj"):
            def make_post_attn_gpt(i=i):
                def fn(m, inp, out):
                    buf["post_attn"][i] = (out[0] if isinstance(out,tuple) else out).detach().float().squeeze(0)
                return fn
            handles.append(attn_src.c_proj.register_forward_hook(make_post_attn_gpt()))

        # Pre-proj FFN
        ffn_src = getattr(blk, "ffwd", None) or getattr(blk, "mlp_block", None)
        if ffn_src:
            def make_pre_ffn(i=i):
                def fn(m, inp, out):
                    buf["pre_ffn"][i] = (out[0] if isinstance(out,tuple) else out).detach().float().squeeze(0)
                return fn
            handles.append(ffn_src.register_forward_hook(make_pre_ffn()))

        # Post-proj FFN
        if hasattr(blk, "proj_mlp"):
            def make_post_ffn(i=i):
                def fn(m, inp, out):
                    buf["post_ffn"][i] = (out[0] if isinstance(out,tuple) else out).detach().float().squeeze(0)
                return fn
            handles.append(blk.proj_mlp.register_forward_hook(make_post_ffn()))

    with torch.no_grad():
        model(input_ids[:, :SEQ_LEN])
    for h in handles:
        h.remove()

    acts = {}
    for k in buf:
        acts[k] = buf[k]
    return acts


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyse_model(label, ckpt_path, input_ids):
    if not Path(ckpt_path).exists():
        print(f"  SKIP {label} — checkpoint not found: {ckpt_path}")
        return None

    print(f"  Loading {label} ...")
    model = AutoModelForCausalLM.from_pretrained(
        str(ckpt_path), dtype=torch.float32, trust_remote_code=True, device_map="cpu"
    )
    model.eval()
    nl = len(model.transformer.h)
    print(f"    {nl} blocks")

    # A) Weight cosim
    wts = extract_weights(model)
    wt_result = {}
    for wtype, wlist in wts.items():
        valid = [w for w in wlist if w is not None]
        if len(valid) < 2:
            continue
        M = weight_cosim_matrix(wlist)
        c = consec(wlist)
        wt_result[wtype] = {
            "cosine_matrix": M,
            "consecutive":   c,
            "off_diag":      off_diag_mean(M),
            "consec_mean":   nanmean(c),
        }

    # B) Post-proj activation cosim
    acts = capture_postproj(model, input_ids)
    act_result = {}
    for atype, alist in acts.items():
        valid = [a for a in alist if a is not None]
        if len(valid) < 2:
            continue
        flat = [a.reshape(-1) if a is not None else None for a in alist]
        c = consec(flat)
        act_result[atype] = {
            "consecutive": c,
            "consec_mean": nanmean(c),
        }

    del model
    return {"weights": wt_result, "activations": act_result, "nl": nl}


# ── Plotting helpers ──────────────────────────────────────────────────────────

def plot_weight_consec_lines(all_results, colors, stem, title):
    """
    One subplot per weight type, one line per model.
    Layout: 2 rows × N_types/2 cols.
    """
    # Gather weight types present in any model
    wtypes = []
    for mname, res in all_results.items():
        for wt in res["weights"]:
            if wt not in wtypes:
                wtypes.append(wt)

    WTYPE_LABELS = {
        "W_Q": "W_Q  (query)",
        "W_K": "W_K  (key)",
        "W_V": "W_V  (value, GPT only)",
        "W_O": "W_O  (attn output, GPT)",
        "W1":  "W1  (FFN gate/up)",
        "W2":  "W2  (FFN value/up)",
        "proj_attn": "proj_attn  (attn→residual)",
        "proj_mlp":  "proj_mlp  (FFN→residual)",
    }
    n = len(wtypes)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5 * ncols, 4 * nrows), squeeze=False)

    for idx, wtype in enumerate(wtypes):
        ax = axes[idx // ncols][idx % ncols]
        for mname, res in all_results.items():
            if wtype not in res["weights"]:
                continue
            c = res["weights"][wtype]["consecutive"]
            x = np.arange(len(c))
            ax.plot(x, c,
                    color=colors.get(mname, "#888"),
                    marker=MARKERS.get(mname, "o"),
                    markersize=4, linewidth=1.7,
                    label=mname.replace("\n", " "))
        ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(WTYPE_LABELS.get(wtype, wtype), fontsize=9, fontweight="bold")
        ax.set_xlabel("Layer pair i→i+1", fontsize=8)
        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.legend(fontsize=6, ncol=1)

    # Hide unused axes
    for idx in range(len(wtypes), nrows * ncols):
        axes[idx // ncols][idx % ncols].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, stem)


def plot_weight_summary_bar(all_results, colors, stem, title):
    """
    Bar chart: x = model, groups = weight type.
    Shows mean consecutive cosim per (model, weight_type).
    This is the KEY figure to answer the projection-diversity hypothesis.
    """
    mnames = list(all_results.keys())
    # Collect all weight types
    all_wtypes = []
    for res in all_results.values():
        for wt in res["weights"]:
            if wt not in all_wtypes:
                all_wtypes.append(wt)

    WTYPE_SHORT = {
        "W_Q": "W_Q", "W_K": "W_K", "W_V": "W_V", "W_O": "W_O",
        "W1": "W1", "W2": "W2", "proj_attn": "proj_a", "proj_mlp": "proj_m",
    }
    # Color each weight type distinctly
    wtype_colors = {
        "W_Q": "#1f77b4", "W_K": "#aec7e8",
        "W_V": "#ffbb78", "W_O": "#ff7f0e",
        "W1":  "#2ca02c", "W2":  "#98df8a",
        "proj_attn": "#d62728", "proj_mlp": "#ff9896",
    }

    n_models = len(mnames)
    n_types = len(all_wtypes)
    width = 0.8 / n_types
    x = np.arange(n_models)

    fig, ax = plt.subplots(figsize=(max(10, 1.5 * n_models), 5))
    for ti, wtype in enumerate(all_wtypes):
        vals = []
        for mname in mnames:
            v = all_results[mname]["weights"].get(wtype, {}).get("consec_mean", float("nan"))
            vals.append(v)
        offset = (ti - n_types / 2 + 0.5) * width
        bars = ax.bar(x + offset, vals, width=width * 0.9,
                      color=wtype_colors.get(wtype, "#888"),
                      label=WTYPE_SHORT.get(wtype, wtype),
                      alpha=0.85, edgecolor="white")
        for bar, v in zip(bars, vals):
            if not np.isnan(v) and v > 0.01:
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005,
                        f"{v:.2f}", ha="center", va="bottom",
                        fontsize=6, rotation=90)

    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("\n", " ") for m in mnames],
                       rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("Mean consecutive weight cosim", fontsize=10)
    ax.legend(title="Weight type", fontsize=8, ncol=4,
              bbox_to_anchor=(1, 1), loc="upper left")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.4)
    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, stem)


def plot_postproj_consec_lines(all_results, colors, stem, title):
    """
    Compare pre- vs post-proj consecutive activation cosim.
    Panels: pre_attn, post_attn, pre_ffn, post_ffn, delta.
    """
    ATYPES = [
        ("pre_attn",  "attn_out (pre-proj)"),
        ("post_attn", "proj_attn(attn)  (post-proj)"),
        ("pre_ffn",   "ffn_out  (pre-proj)"),
        ("post_ffn",  "proj_mlp(ffn)  (post-proj)"),
        ("delta",     "full block Δh"),
    ]
    available = [at for at in ATYPES
                 if any(at[0] in res["activations"] for res in all_results.values())]
    n = len(available)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)

    for ax, (atype, alabel) in zip(axes[0], available):
        for mname, res in all_results.items():
            if atype not in res["activations"]:
                continue
            c = res["activations"][atype]["consecutive"]
            x = np.arange(len(c))
            ax.plot(x, c,
                    color=colors.get(mname, "#888"),
                    marker=MARKERS.get(mname, "o"),
                    markersize=4, linewidth=1.7,
                    label=mname.replace("\n", " "))
        ax.axhline(0, color="gray", lw=0.6, ls="--", alpha=0.5)
        ax.set_title(alabel, fontsize=9, fontweight="bold")
        ax.set_xlabel("Layer pair i→i+1", fontsize=8)
        ax.set_ylabel("Cosine similarity", fontsize=8)
        ax.legend(fontsize=6)

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.01)
    plt.tight_layout()
    _save(fig, stem)


def plot_postproj_summary_bar(all_results, colors, stem, title):
    """
    Bar chart showing pre- vs post-proj mean consecutive cosim side-by-side.
    Highlights the amplification (or dampening) by the proj matrices.
    """
    mnames = list(all_results.keys())
    pairs = [
        ("pre_attn",  "post_attn", "Attn: pre vs post proj", "#aec7e8", "#1f77b4"),
        ("pre_ffn",   "post_ffn",  "FFN: pre vs post proj",  "#98df8a", "#2ca02c"),
    ]
    fig, axes = plt.subplots(1, len(pairs), figsize=(7 * len(pairs), 5))
    if len(pairs) == 1:
        axes = [axes]

    for ax, (pre_k, post_k, ptitle, cpre, cpost) in zip(axes, pairs):
        x = np.arange(len(mnames))
        pre_vals  = [all_results[m]["activations"].get(pre_k,  {}).get("consec_mean", float("nan")) for m in mnames]
        post_vals = [all_results[m]["activations"].get(post_k, {}).get("consec_mean", float("nan")) for m in mnames]
        w = 0.35
        ax.bar(x - w/2, pre_vals,  width=w, color=cpre,  label="pre-proj",  alpha=0.85, edgecolor="white")
        ax.bar(x + w/2, post_vals, width=w, color=cpost, label="post-proj", alpha=0.85, edgecolor="white")
        for xi, (vpre, vpost) in enumerate(zip(pre_vals, post_vals)):
            for xoff, v in [(-w/2, vpre), (w/2, vpost)]:
                if not np.isnan(v):
                    ax.text(xi + xoff, v + 0.005, f"{v:.3f}",
                            ha="center", va="bottom", fontsize=7)
        ax.set_xticks(x)
        ax.set_xticklabels([m.replace("\n", " ") for m in mnames],
                           rotation=20, ha="right", fontsize=8)
        ax.set_ylabel("Mean consecutive cosim", fontsize=10)
        ax.set_title(ptitle, fontsize=10, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(title, fontsize=11, fontweight="bold")
    plt.tight_layout()
    _save(fig, stem)


# ── Main ──────────────────────────────────────────────────────────────────────

def run_group(models_dict, colors, tag, title_base):
    first_ckpt = next(p for p in models_dict.values() if Path(p).exists())
    tok = AutoTokenizer.from_pretrained(str(first_ckpt), trust_remote_code=True)
    input_ids = tok(PREFILL, return_tensors="pt")["input_ids"][:, :SEQ_LEN]
    print(f"Prefill: {input_ids.shape[1]} tokens")

    all_results = {}
    for label, ckpt in models_dict.items():
        print(f"\n--- {label} ---")
        res = analyse_model(label, ckpt, input_ids)
        if res is not None:
            all_results[label] = res

    if not all_results:
        print(f"No models loaded for {tag}. Skipping.")
        return {}

    print(f"\nPlotting {tag} ...")
    plot_weight_consec_lines(
        all_results, colors,
        f"weight_cosim_consec_{tag}",
        f"Per-weight-type consecutive cosim — {title_base}"
    )
    plot_weight_summary_bar(
        all_results, colors,
        f"weight_cosim_summary_{tag}",
        f"Mean consecutive weight cosim by type — {title_base}\n"
        r"Hypothesis: proj_attn / proj_mlp show lower cosim = more diverse strategies"
    )
    plot_postproj_consec_lines(
        all_results, colors,
        f"postproj_cosim_consec_{tag}",
        f"Pre- vs post-projection activation cosim — {title_base}"
    )
    plot_postproj_summary_bar(
        all_results, colors,
        f"postproj_cosim_summary_{tag}",
        f"Pre vs post-proj mean cosim (attn & FFN) — {title_base}"
    )

    # Save stats JSON
    stats = {}
    for mname, res in all_results.items():
        stats[mname] = {
            "weights": {
                wt: {
                    "consec_mean": res["weights"][wt]["consec_mean"],
                    "off_diag":    res["weights"][wt]["off_diag"],
                    "consecutive": [round(v, 6) for v in res["weights"][wt]["consecutive"]],
                }
                for wt in res["weights"]
            },
            "activations": {
                at: {
                    "consec_mean": res["activations"][at]["consec_mean"],
                    "consecutive": [round(v, 6) for v in res["activations"][at]["consecutive"]],
                }
                for at in res["activations"]
            },
        }
    out_json = PLOTS_DIR / f"layer_cosim_stats_weight_{tag}.json"
    with open(out_json, "w") as f:
        json.dump(stats, f, indent=2)
    print(f"  saved {out_json.name}")
    return all_results


if __name__ == "__main__":
    print("=== 176M cosreg sweep ===")
    run_group(MODELS_176M, COLORS_176M, "176m",
              "176M cosreg sweep (V0/V1/V39/V52/V53/V48)")

    print("\n=== 400M family ===")
    run_group(MODELS_400M, COLORS_400M, "400m",
              "400M family (V9/V1/V31/V32/V40/V55)")

    print("\nAll done.")
