"""analyze_consec_bars_20260425d.py

Generates consecutive-layer avg cosine similarity bar plots — the "consecutive avg"
counterpart to the off-diagonal avg bars in paper_v2.

For each model, loads once and computes:
  1. Weight-space merger cosim (full / attn / ffn)
  2. Activation residual-stream cosim (full / attn / ffn)
  3. Same-input delta cosim (full / attn / ffn)
  4. Proj-decomp cosim (full Δ / attn_raw / proj_attn / ffn_raw / proj_ffn)

Each bar includes a 'V41 Sandwich (EGPT only)' entry using only the 7
EGPT-EGPT consecutive pairs (layers 2-3, 3-4, ..., 8-9 in the 12-layer model).

Output PDFs:
  merger_consec_avg_cosim_bar.pdf
  merger_consec_avg_cosim_attn_ffn_bar.pdf
  act_consec_avg_cosim_attn_ffn_bar.pdf
  same_delta_consec_breakdown_bar.pdf
  proj_decomp_consec_bar.pdf
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
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

# V41: layers 0,1=GPT; layers 2-9=EGPT; layers 10,11=GPT
# EGPT-EGPT consecutive pairs = indices 2..8 in the length-11 successive array
V41_EGPT_SLICE = slice(2, 9)

MODELS = {
    "V0 GPT\n12×1": {
        "path": BASE / "v0_gpt_baseline_d768" / "unsharded",
        "color": "#4477AA", "n_layers": 12, "arch": "GPT",
    },
    "V1 EGPT\n12×1": {
        "path": BASE / "v1_12x1_d768_lr2e3" / "unsharded",
        "color": "#EE6677", "n_layers": 12, "arch": "EGPT",
    },
    "V39 EGPT\n+cosreg": {
        "path": BASE / "v39_egpt_cosreg_12x1_d768_lr2e3" / "unsharded",
        "color": "#AA3377", "n_layers": 12, "arch": "EGPT+cosreg",
    },
    "V41 Sandwich\n2G+8E+2G": {
        "path": BASE / "v41_sandwich_2gpt8e2gpt_d768_lr2e3" / "unsharded",
        "color": "#228833", "n_layers": 12, "arch": "Sandwich",
        "egpt_only": True,
    },
    "V42 Naive\nmerge (init)": {
        "path": BASE / "v42_merged_naive_init",
        "color": "#CCBB44", "n_layers": 6, "arch": "Merged",
    },
    "V42 Naive\n+5k FT": {
        "path": BASE / "v42_egpt_6layer_naive_merge_finetune_d768" / "unsharded",
        "color": "#EE8800", "n_layers": 6, "arch": "Merged+FT",
    },
    "V43 Procrustes\nmerge (init)": {
        "path": BASE / "v43_merged_procrustes_init",
        "color": "#66CCEE", "n_layers": 6, "arch": "Merged",
    },
    "V43 Procrustes\n+5k FT": {
        "path": BASE / "v43_egpt_6layer_procrustes_merge_finetune_d768" / "unsharded",
        "color": "#0077BB", "n_layers": 6, "arch": "Merged+FT",
    },
}

V41_EGPT_COLOR = "#009944"


# ── Utilities ─────────────────────────────────────────────────────────────────

def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


def flat_cosim(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def succ_cosim(lst):
    """Consecutive cosim for a list of tensors; NaN where either is None."""
    result = []
    for i in range(len(lst) - 1):
        a, b = lst[i], lst[i + 1]
        if a is not None and b is not None:
            result.append(flat_cosim(a, b))
        else:
            result.append(float("nan"))
    return result


def nanmean(arr):
    vals = [v for v in arr if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ── Weight-space analysis ──────────────────────────────────────────────────────

def _weight_vec(block, key_words=None):
    return {k: p.data.float().reshape(-1) for k, p in block.named_parameters()
            if key_words is None or any(kw in k for kw in key_words)}


def _dict_cosim(d1, d2):
    common = sorted(set(d1) & set(d2))
    if not common:
        return float("nan")
    v1 = torch.cat([d1[k] for k in common])
    v2 = torch.cat([d2[k] for k in common])
    return F.cosine_similarity(v1.unsqueeze(0), v2.unsqueeze(0)).item()


def merger_consec(blocks):
    """Weight-space consecutive cosim: full / attn / ffn."""
    n = len(blocks)
    ATTN_KW = ("attn", "c_attn", "W_Q", "sequence_mixer")
    FFN_KW  = ("ffwd", "mlp", "feed_forward", "ff")
    full_vecs = [_weight_vec(b) for b in blocks]
    attn_vecs = [_weight_vec(b, ATTN_KW) for b in blocks]
    ffn_vecs  = [_weight_vec(b, FFN_KW)  for b in blocks]

    full_succ = [_dict_cosim(full_vecs[i], full_vecs[i+1]) for i in range(n-1)]
    attn_succ = [_dict_cosim(attn_vecs[i], attn_vecs[i+1]) if attn_vecs[i] and attn_vecs[i+1]
                 else float("nan") for i in range(n-1)]
    ffn_succ  = [_dict_cosim(ffn_vecs[i],  ffn_vecs[i+1])  if ffn_vecs[i]  and ffn_vecs[i+1]
                 else float("nan") for i in range(n-1)]
    return full_succ, attn_succ, ffn_succ


# ── Activation-space analysis ─────────────────────────────────────────────────

def get_embedding(model, input_ids):
    transformer = model.transformer
    with torch.no_grad():
        h = transformer.wte(input_ids).float()
        if hasattr(transformer, "wpe"):
            pos = torch.arange(input_ids.shape[1]).unsqueeze(0)
            h = h + transformer.wpe(pos).float()
    return h


def get_rope(model, input_ids, h):
    t = model.transformer
    if not hasattr(t, "position_embedding_type") or t.position_embedding_type != "rope":
        return None
    pos_ids = torch.arange(input_ids.shape[1]).unsqueeze(0)
    with torch.no_grad():
        return t._get_rope_cos_sin(key_length=input_ids.shape[1],
                                   position_ids=pos_ids, dtype=h.dtype)


def _make_hook(buf):
    def hook(mod, inp, out):
        o = out[0] if isinstance(out, (tuple, list)) else out
        buf.append(o.detach().float().squeeze(0))
    return hook


def act_consec(model, input_ids):
    """Sequential forward pass: consecutive cosim for residual stream / attn / ffn."""
    blocks = list(model.transformer.h)
    blk_outs, attn_outs, ffn_outs = [], [], []
    hooks = []

    for blk in blocks:
        hooks.append(blk.register_forward_hook(_make_hook(blk_outs)))
        for name in ("attn", "sequence_mixer"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(attn_outs)))
                break
        for name in ("ffwd", "mlp", "feed_forward", "ff"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(ffn_outs)))
                break

    with torch.no_grad():
        model(input_ids)
    for h in hooks:
        h.remove()

    n = len(blk_outs)
    # Pad attn/ffn lists to n if shorter (V41 GPT blocks may not have all)
    while len(attn_outs) < n:
        attn_outs.append(None)
    while len(ffn_outs) < n:
        ffn_outs.append(None)

    full_succ = succ_cosim(blk_outs[:n])
    attn_succ = succ_cosim(attn_outs[:n])
    ffn_succ  = succ_cosim(ffn_outs[:n])
    return full_succ, attn_succ, ffn_succ


# ── Same-input delta analysis ─────────────────────────────────────────────────

def same_delta_consec(model, h_base, rope):
    """Same-input delta + attn/ffn sub-outputs: consecutive cosim."""
    blocks = list(model.transformer.h)
    n = len(blocks)
    full_deltas, attn_outs, ffn_outs = [], [], []

    for blk in blocks:
        ab, fb = [], []
        hooks = []
        for name in ("attn", "sequence_mixer"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(ab)))
                break
        for name in ("ffwd", "mlp", "feed_forward", "ff"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(fb)))
                break

        with torch.no_grad():
            out = blk(h_base, rope_cos_sin=rope)
        h_out = out[0] if isinstance(out, (tuple, list)) else out
        for hk in hooks:
            hk.remove()

        full_deltas.append((h_out.float() - h_base).squeeze(0))
        attn_outs.append(ab[0] if ab else None)
        ffn_outs.append(fb[0] if fb else None)

    rand_model = AutoModelForCausalLM.from_config(model.config)
    rand_model.eval()
    rand_rope = get_rope(rand_model, torch.zeros(1, h_base.shape[1], dtype=torch.long), h_base)
    rand_deltas = []
    for blk in rand_model.transformer.h:
        with torch.no_grad():
            out = blk(h_base, rope_cos_sin=rand_rope)
        h_out = out[0] if isinstance(out, (tuple, list)) else out
        rand_deltas.append((h_out.float() - h_base).squeeze(0))
    del rand_model

    full_succ = succ_cosim(full_deltas)
    attn_succ = succ_cosim(attn_outs)
    ffn_succ  = succ_cosim(ffn_outs)
    rand_succ = succ_cosim(rand_deltas)
    return full_succ, attn_succ, ffn_succ, rand_succ


# ── Proj-decomp analysis ──────────────────────────────────────────────────────

def proj_decomp_consec(model, h_base, rope):
    """For dual_unconstrained EGPT blocks: full Δ / attn_raw / proj_attn / ffn_raw / proj_ffn consecutive cosim."""
    blocks = list(model.transformer.h)
    n = len(blocks)
    full_deltas, attn_raws, proj_attn_outs, ffn_raws, proj_ffn_outs = [], [], [], [], []

    for blk in blocks:
        proj_type = getattr(blk, "proj_type", None)
        ab, pab, fb, pfb = [], [], [], []
        hooks = []
        for name in ("attn", "sequence_mixer"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(ab)))
                break
        for name in ("ffwd", "mlp", "feed_forward", "ff"):
            if hasattr(blk, name):
                hooks.append(getattr(blk, name).register_forward_hook(_make_hook(fb)))
                break
        if proj_type == "dual_unconstrained":
            if hasattr(blk, "proj_attn"):
                hooks.append(blk.proj_attn.register_forward_hook(_make_hook(pab)))
            if hasattr(blk, "proj_mlp"):
                hooks.append(blk.proj_mlp.register_forward_hook(_make_hook(pfb)))

        with torch.no_grad():
            out = blk(h_base, rope_cos_sin=rope)
        h_out = out[0] if isinstance(out, (tuple, list)) else out
        for hk in hooks:
            hk.remove()

        full_deltas.append((h_out.float() - h_base).squeeze(0))
        attn_raws.append(ab[0] if ab else None)
        proj_attn_outs.append(pab[0] if pab else None)
        ffn_raws.append(fb[0] if fb else None)
        proj_ffn_outs.append(pfb[0] if pfb else None)

    return (succ_cosim(full_deltas), succ_cosim(attn_raws), succ_cosim(proj_attn_outs),
            succ_cosim(ffn_raws), succ_cosim(proj_ffn_outs))


# ── Per-model analysis ────────────────────────────────────────────────────────

def analyze_model(label, info, input_ids):
    path = info["path"]
    if not Path(path).exists():
        print(f"  Missing: {path}")
        return None
    print(f"  Loading {path}")
    try:
        model = AutoModelForCausalLM.from_pretrained(str(path), torch_dtype=torch.float32, device_map="cpu")
        model.eval()
    except Exception as e:
        print(f"  Error: {e}")
        return None

    blocks = list(model.transformer.h)
    n = len(blocks)
    print(f"  {n} blocks")

    # 1. Weight merger
    merger_full, merger_attn, merger_ffn = merger_consec(blocks)

    # 2. Activation
    act_full, act_attn, act_ffn = act_consec(model, input_ids)

    # 3. Same-input delta
    h_base = get_embedding(model, input_ids)
    rope   = get_rope(model, input_ids, h_base)
    sd_full, sd_attn, sd_ffn, sd_rand = same_delta_consec(model, h_base, rope)

    # 4. Proj decomp
    pd_full, pd_attn_raw, pd_proj_attn, pd_ffn_raw, pd_proj_ffn = proj_decomp_consec(model, h_base, rope)

    del model

    result = {
        "n_layers": n,
        "merger_full": merger_full, "merger_attn": merger_attn, "merger_ffn": merger_ffn,
        "act_full": act_full, "act_attn": act_attn, "act_ffn": act_ffn,
        "sd_full": sd_full, "sd_attn": sd_attn, "sd_ffn": sd_ffn, "sd_rand": sd_rand,
        "pd_full": pd_full, "pd_attn_raw": pd_attn_raw, "pd_proj_attn": pd_proj_attn,
        "pd_ffn_raw": pd_ffn_raw, "pd_proj_ffn": pd_proj_ffn,
    }

    for key in result:
        if isinstance(result[key], list):
            print(f"  {key}: mean={nanmean(result[key]):.4f}")

    return result


# ── Plotting helpers ──────────────────────────────────────────────────────────

def _make_bar_entries(all_results, key, include_egpt_only=True):
    """Build (labels, values, colors) for a bar chart of mean consecutive cosim for `key`."""
    labels, vals, colors = [], [], []
    for label, data in all_results.items():
        if data is None:
            continue
        arr = data.get(key, [])
        labels.append(label.replace("\n", " "))
        vals.append(nanmean(arr))
        colors.append(MODELS[label]["color"])

        # Add EGPT-only entry immediately after V41
        if include_egpt_only and MODELS[label].get("egpt_only"):
            egpt_arr = arr[V41_EGPT_SLICE]
            labels.append("V41 Sandwich\n(EGPT only)".replace("\n", " "))
            vals.append(nanmean(egpt_arr))
            colors.append(V41_EGPT_COLOR)

    return labels, vals, colors


def plot_single_bar(all_results, key, stem, ylabel, title):
    labels, vals, colors = _make_bar_entries(all_results, key)
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, vals, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, vals):
        if not np.isnan(val):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, stem)


def plot_grouped_bar(all_results, keys, labels_legend, alphas, stem, ylabel, title):
    """Grouped bar chart with one group of bars per model."""
    model_labels = []
    all_vals = [[] for _ in keys]
    all_colors = []

    for label, data in all_results.items():
        if data is None:
            continue
        model_labels.append(label.replace("\n", " "))
        all_colors.append(MODELS[label]["color"])
        for i, key in enumerate(keys):
            arr = data.get(key, [])
            all_vals[i].append(nanmean(arr))

        # Add EGPT-only entry after V41
        if MODELS[label].get("egpt_only"):
            model_labels.append("V41 Sandwich\n(EGPT only)".replace("\n", " "))
            all_colors.append(V41_EGPT_COLOR)
            for i, key in enumerate(keys):
                arr = data.get(key, [])[V41_EGPT_SLICE]
                all_vals[i].append(nanmean(arr))

    n_models = len(model_labels)
    n_groups = len(keys)
    width = 0.8 / n_groups
    offsets = np.linspace(-(n_groups-1)/2, (n_groups-1)/2, n_groups) * width

    fig, ax = plt.subplots(figsize=(max(8, n_models * 1.4), 5))
    x = np.arange(n_models)

    for i, (key, lbl, alpha, offset) in enumerate(zip(keys, labels_legend, alphas, offsets)):
        bars = ax.bar(x + offset, all_vals[i], width,
                      label=lbl, color=all_colors, edgecolor="white", alpha=alpha)
        for bar, val in zip(bars, all_vals[i]):
            if not np.isnan(val):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
                        f"{val:.2f}", ha="center", va="bottom", fontsize=7, alpha=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(model_labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel(ylabel, fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    plt.tight_layout()
    savefig(fig, stem)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    input_ids = torch.tensor([tokenizer.encode(EVAL_TEXT)[:N_TOKENS]])
    print(f"Using {input_ids.shape[1]} tokens.")

    all_results = {}
    for label, info in MODELS.items():
        print(f"\n=== {label.replace(chr(10), ' ')} ===")
        r = analyze_model(label, info, input_ids)
        all_results[label] = r

    # 1. Merger weight: single bar (consecutive mean)
    plot_single_bar(all_results, "merger_full",
                    "merger_consec_avg_cosim_bar",
                    "Mean consecutive weight cosine similarity",
                    "Consecutive-Layer Weight Cosine Similarity per Model")

    # 2. Merger weight: grouped bar (full / attn / ffn)
    plot_grouped_bar(all_results,
                     ["merger_full", "merger_attn", "merger_ffn"],
                     ["Full block", "Attn only", "FFN only"],
                     [0.9, 0.65, 0.40],
                     "merger_consec_avg_cosim_attn_ffn_bar",
                     "Mean consecutive weight cosine similarity",
                     "Consecutive-Layer Weight Similarity: Full vs Attn vs FFN")

    # 3. Activations: grouped bar (residual / attn / ffn)
    plot_grouped_bar(all_results,
                     ["act_full", "act_attn", "act_ffn"],
                     ["Full block output", "Attn sub-output", "FFN sub-output"],
                     [0.9, 0.65, 0.40],
                     "act_consec_avg_cosim_attn_ffn_bar",
                     "Mean consecutive activation cosine similarity",
                     "Consecutive-Layer Activation Similarity: Full vs Attn vs FFN")

    # 4. Same-input delta: grouped bar (full / attn / ffn / random)
    plot_grouped_bar(all_results,
                     ["sd_full", "sd_attn", "sd_ffn", "sd_rand"],
                     ["Full block Δ", "Attn sub-out Δ", "FFN sub-out Δ", "Random init Δ"],
                     [0.95, 0.65, 0.40, 0.50],
                     "same_delta_consec_breakdown_bar",
                     "Mean consecutive same-input delta cosine similarity",
                     "Consecutive-Layer Same-Input Delta Breakdown")

    # 5. Proj-decomp: grouped bar (5 measures)
    plot_grouped_bar(all_results,
                     ["pd_full", "pd_attn_raw", "pd_proj_attn", "pd_ffn_raw", "pd_proj_ffn"],
                     ["Full Δ", "Attn raw", "proj_attn", "FFN raw", "proj_mlp"],
                     [1.0, 0.75, 0.90, 0.5, 0.65],
                     "proj_decomp_consec_bar",
                     "Mean consecutive same-input delta cosine similarity",
                     "Consecutive-Layer Proj Decomposition")

    # Summary
    print("\n=== SUMMARY ===")
    print(f"{'Model':<35} {'m_full':>8} {'act_full':>8} {'sd_full':>8} {'pd_full':>8}")
    for label, data in all_results.items():
        if data is None:
            continue
        print(f"{label.replace(chr(10),' '):<35} "
              f"{nanmean(data['merger_full']):>8.4f} "
              f"{nanmean(data['act_full']):>8.4f} "
              f"{nanmean(data['sd_full']):>8.4f} "
              f"{nanmean(data['pd_full']):>8.4f}")
        if MODELS[label].get("egpt_only"):
            print(f"  {'(EGPT only)':<33} "
                  f"{nanmean(data['merger_full'][V41_EGPT_SLICE]):>8.4f} "
                  f"{nanmean(data['act_full'][V41_EGPT_SLICE]):>8.4f} "
                  f"{nanmean(data['sd_full'][V41_EGPT_SLICE]):>8.4f} "
                  f"{nanmean(data['pd_full'][V41_EGPT_SLICE]):>8.4f}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
