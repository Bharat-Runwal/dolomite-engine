"""Activation-space layer similarity analysis for V39-V43 experiments.

Feeds real text through each model, captures the output of every transformer block,
and computes cosine similarity between those activation vectors across layers.

This measures functional similarity (what each layer actually computes) rather than
weight-space similarity (parameter overlap). Key advantages:
- Works naturally across mixed architectures (V41 Sandwich: GPT+EGPT blocks)
- Directly captures what V39 cosreg actually regularized (attn output activations)
- Tells us: do adjacent layers do similar things to the same input?

Captures per block:
- Full block output: h_out = block(h_in) — residual stream after block
- Attn sub-output: attn(h_in)  (before residual add)
- FFN sub-output:  ffn(h_in)   (before residual add)

Cosine similarity is computed over the (batch×seq, d_model) activation matrix.
"""

import sys, os, json
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from contextlib import contextmanager

REPO = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO)
import lm_engine.hf_models  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE = Path(f"{REPO}/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"

# Fixed evaluation text — same for all models
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

N_TOKENS = 256  # tokens to use for activation capture

# ── Model catalog ────────────────────────────────────────────────────────────
MODELS = {
    "V0 GPT\n12×1 d768": {
        "path": BASE / "v0_gpt_baseline_d768" / "unsharded",
        "color": "#4477AA", "n_layers": 12, "arch": "GPT", "marker": "s",
    },
    "V1 EGPT\n12×1 d768": {
        "path": BASE / "v1_12x1_d768_lr2e3" / "unsharded",
        "color": "#EE6677", "n_layers": 12, "arch": "EGPT", "marker": "o",
    },
    "V39 EGPT\n+cosreg": {
        "path": BASE / "v39_egpt_cosreg_12x1_d768_lr2e3" / "unsharded",
        "color": "#AA3377", "n_layers": 12, "arch": "EGPT+cosreg", "marker": "o",
    },
    "V41 Sandwich\n2G+8E+2G": {
        "path": BASE / "v41_sandwich_2gpt8e2gpt_d768_lr2e3" / "unsharded",
        "color": "#228833", "n_layers": 12, "arch": "Sandwich", "marker": "^",
    },
    "V42 Naive\nmerge (init)": {
        "path": BASE / "v42_merged_naive_init",
        "color": "#CCBB44", "n_layers": 6, "arch": "Merged", "marker": "o",
    },
    "V42 Naive\n+5k FT": {
        "path": BASE / "v42_egpt_6layer_naive_merge_finetune_d768" / "unsharded",
        "color": "#EE8800", "n_layers": 6, "arch": "Merged+FT", "marker": "o",
    },
    "V43 Procrustes\nmerge (init)": {
        "path": BASE / "v43_merged_procrustes_init",
        "color": "#66CCEE", "n_layers": 6, "arch": "Merged", "marker": "D",
    },
    "V43 Procrustes\n+5k FT": {
        "path": BASE / "v43_egpt_6layer_procrustes_merge_finetune_d768" / "unsharded",
        "color": "#0077BB", "n_layers": 6, "arch": "Merged+FT", "marker": "D",
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
    return torch.tensor([ids])  # (1, T)


def get_embedding(model, input_ids):
    """Return the post-embedding hidden state (after wte + wpe, before any blocks)."""
    transformer = model.transformer
    with torch.no_grad():
        h = transformer.wte(input_ids).float()
        if hasattr(transformer, "wpe"):
            pos = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            h = h + transformer.wpe(pos).float()
    return h  # (1, T, d)


def get_rope_cos_sin(model, input_ids, h_base):
    """Compute RoPE cos/sin for the given sequence length, or None if not using RoPE."""
    transformer = model.transformer
    if not hasattr(transformer, "position_embedding_type"):
        return None
    if transformer.position_embedding_type != "rope":
        return None
    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
    with torch.no_grad():
        rope_cos_sin = transformer._get_rope_cos_sin(
            key_length=input_ids.shape[1],
            position_ids=position_ids,
            dtype=h_base.dtype,
        )
    return rope_cos_sin


def same_input_deltas(model, h_base, rope_cos_sin=None):
    """Feed the SAME h_base to every block independently, return Δ_i = Block_i(h) - h.

    This is the user-intended analysis: treat each block as an independent operator
    on a shared embedding space and compare their actions on identical input.
    Unlike sequential deltas, this is unconfounded by input distribution shift.
    h_base: (1, T, d) float tensor (post-embedding, pre-blocks).
    """
    blocks = list(model.transformer.h)
    deltas = []
    with torch.no_grad():
        for blk in blocks:
            out = blk(h_base, rope_cos_sin=rope_cos_sin)
            h_out = out[0] if isinstance(out, (tuple, list)) else out
            deltas.append((h_out.float() - h_base).squeeze(0))  # (T, d)
    return deltas


def capture_activations(model, input_ids):
    """Run sequential forward pass capturing block I/O and sub-module outputs.

    Returns:
        block_ins:  list of (T, d) — residual stream entering each block
        block_outs: list of (T, d) — residual stream after each block
        attn_outs:  list of (T, d) — attn sub-module output
        ffn_outs:   list of (T, d) — ffn sub-module output

    Sequential delta Δ_i = block_outs[i] - block_ins[i] (each block sees its own evolved input).
    For same-input operator comparison use same_input_deltas() instead.
    """
    block_ins  = []
    block_outs = []
    attn_outs  = []
    ffn_outs   = []
    hooks = []

    transformer = model.transformer if hasattr(model, "transformer") else None
    if transformer is None:
        print("  Cannot find model.transformer")
        return [], [], [], []

    blocks = list(transformer.h)

    def make_block_hooks(in_store, out_store):
        def pre_hook(module, inp):
            h = inp[0] if isinstance(inp, (tuple, list)) else inp
            in_store.append(h.detach().float().squeeze(0))
        def post_hook(module, inp, out):
            h = out[0] if isinstance(out, (tuple, list)) else out
            out_store.append(h.detach().float().squeeze(0))
        return pre_hook, post_hook

    for blk in blocks:
        pre, post = make_block_hooks(block_ins, block_outs)
        hooks.append(blk.register_forward_pre_hook(pre))
        hooks.append(blk.register_forward_hook(post))

    for blk in blocks:
        attn_mod = None
        for name in ["attn", "sequence_mixer"]:
            if hasattr(blk, name):
                attn_mod = getattr(blk, name); break
        if attn_mod is not None:
            def make_attn_hook(store):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    store.append(h.detach().float().squeeze(0))
                return hook
            hooks.append(attn_mod.register_forward_hook(make_attn_hook(attn_outs)))

    for blk in blocks:
        ffn_mod = None
        for name in ["ffwd", "mlp", "feed_forward", "ff"]:
            if hasattr(blk, name):
                ffn_mod = getattr(blk, name); break
        if ffn_mod is not None:
            def make_ffn_hook(store):
                def hook(module, inp, out):
                    h = out[0] if isinstance(out, (tuple, list)) else out
                    store.append(h.detach().float().squeeze(0))
                return hook
            hooks.append(ffn_mod.register_forward_hook(make_ffn_hook(ffn_outs)))

    with torch.no_grad():
        model(input_ids)

    for h in hooks:
        h.remove()

    return block_ins, block_outs, attn_outs, ffn_outs


def act_cos_sim_matrix(act_list):
    """(N, N) cosine similarity matrix over list of (T, d) activation tensors.

    Flatten T×d → vector, then cosine sim. This captures overall activation pattern sim.
    """
    if not act_list:
        return None
    n = len(act_list)
    vecs = [a.reshape(-1) for a in act_list]
    mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i, n):
            cs = F.cosine_similarity(vecs[i].unsqueeze(0), vecs[j].unsqueeze(0)).item()
            mat[i, j] = cs
            mat[j, i] = cs
    return mat


def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


def plot_heatmap(mat, title, stem, labels=None):
    n = mat.shape[0]
    labels = labels or [str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(max(5, n * 0.55 + 1), max(4, n * 0.55)))
    # off-diagonal range for colormap
    off_vals = [mat[i, j] for i in range(n) for j in range(n) if i != j]
    vmin, vmax = min(off_vals), max(off_vals)
    vcenter = (vmin + vmax) / 2
    im = ax.imshow(mat, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set_xticks(range(n)); ax.set_xticklabels(labels, fontsize=8)
    ax.set_yticks(range(n)); ax.set_yticklabels(labels, fontsize=8)
    ax.set_title(title, fontsize=11, fontweight="bold")
    for i in range(n):
        for j in range(n):
            val = mat[i, j]
            color = "white" if val < vcenter else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=max(5, 7 - n // 4), color=color)
    savefig(fig, stem)


def plot_successive_line(results, stem, title, key="successive"):
    fig, ax = plt.subplots(figsize=(8, 4))
    for label, data in results.items():
        cs = data.get(key)
        if cs is None: continue
        x = np.arange(len(cs))
        info = MODELS.get(label, {})
        color = info.get("color")
        marker = info.get("marker", "o")
        ax.plot(x, cs, marker=marker, label=label.replace("\n", " "),
                color=color, linewidth=2, markersize=5)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Layer pair (i, i+1)", fontsize=11)
    ax.set_ylabel("Cosine similarity (activations)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, stem)


def plot_avg_sim_bar(results, stem, title):
    labels, avgs, colors = [], [], []
    for label, data in results.items():
        mat = data.get("matrix")
        if mat is None: continue
        n = mat.shape[0]
        mask = ~np.eye(n, dtype=bool)
        avgs.append(float(mat[mask].mean()))
        labels.append(label.replace("\n", " "))
        colors.append(MODELS[label]["color"] if label in MODELS else "#888888")

    fig, ax = plt.subplots(figsize=(max(6, len(labels) * 1.2), 5))
    x = np.arange(len(labels))
    bars = ax.bar(x, avgs, color=colors, edgecolor="white", linewidth=0.8)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg activation cosine sim (off-diagonal)", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    for bar, val in zip(bars, avgs):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                f"{val:.3f}", ha="center", va="bottom", fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, stem)


def plot_avg_sim_bar_attn_ffn(results, stem, title):
    labels = []
    full_avgs, attn_avgs, ffn_avgs = [], [], []
    colors = []
    for label, data in results.items():
        if data.get("matrix") is None: continue
        n = data["matrix"].shape[0]
        mask = ~np.eye(n, dtype=bool)
        full_avgs.append(float(data["matrix"][mask].mean()))
        attn_avgs.append(float(data["attn_matrix"][mask].mean()) if data.get("attn_matrix") is not None else float("nan"))
        ffn_avgs.append(float(data["ffn_matrix"][mask].mean()) if data.get("ffn_matrix") is not None else float("nan"))
        labels.append(label.replace("\n", " "))
        colors.append(MODELS[label]["color"] if label in MODELS else "#888888")

    x = np.arange(len(labels))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(labels) * 1.4), 5))
    ax.bar(x - width, full_avgs, width, label="Full block output", color=colors, edgecolor="white", alpha=0.9)
    ax.bar(x,         attn_avgs, width, label="Attn sub-output",   color=colors, edgecolor="white", alpha=0.6)
    ax.bar(x + width, ffn_avgs,  width, label="FFN sub-output",    color=colors, edgecolor="white", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg activation cosine sim", fontsize=11)
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, stem)


def analyze_model(label, info, tokenizer, input_ids):
    print(f"\n=== {label.replace(chr(10), ' ')} ===")
    model = load_model(info)
    if model is None:
        return None

    block_ins, block_outs, attn_outs, ffn_outs = capture_activations(model, input_ids)
    n = len(block_outs)
    if n == 0:
        print("  No block outputs captured!")
        del model
        return None

    print(f"  {n} blocks, {len(attn_outs)} attn, {len(ffn_outs)} ffn sub-outputs")

    # Residual-stream cosine similarity (biased by shared passthrough)
    mat      = act_cos_sim_matrix(block_outs)
    attn_mat = act_cos_sim_matrix(attn_outs) if len(attn_outs) == n else None
    ffn_mat  = act_cos_sim_matrix(ffn_outs)  if len(ffn_outs) == n else None

    # Sequential delta: Δ_i = h_out_i - h_in_i (each block sees its own evolved input)
    seq_deltas = [block_outs[i] - block_ins[i] for i in range(n)]
    ratio = np.mean([seq_deltas[i].norm().item() / block_ins[i].norm().item() for i in range(n)])
    print(f"  Mean ||Δ||/||h_in|| = {ratio:.4f}")
    seq_delta_mat = act_cos_sim_matrix(seq_deltas)

    # Same-input delta: feed same base embedding h to every block independently
    # Δ_i = Block_i(h) - h for the SAME h → unconfounded operator comparison
    h_base = get_embedding(model, input_ids)
    rope_cos_sin = get_rope_cos_sin(model, input_ids, h_base)
    same_deltas = same_input_deltas(model, h_base, rope_cos_sin)
    same_delta_mat = act_cos_sim_matrix(same_deltas)

    # Random-init baseline (same-input): same architecture, random weights, same h
    rand_model = AutoModelForCausalLM.from_config(model.config)
    rand_model.eval()
    rand_rope = get_rope_cos_sin(rand_model, input_ids, h_base)
    rand_same_deltas = same_input_deltas(rand_model, h_base, rand_rope)
    rand_same_delta_mat = act_cos_sim_matrix(rand_same_deltas)
    # Also sequential random for residual baseline
    rand_ins, rand_outs, _, _ = capture_activations(rand_model, input_ids)
    rand_mat = act_cos_sim_matrix(rand_outs)
    rand_seq_deltas = [rand_outs[i] - rand_ins[i] for i in range(len(rand_outs))]
    rand_seq_delta_mat = act_cos_sim_matrix(rand_seq_deltas)
    del rand_model

    successive          = [mat[i, i+1]               for i in range(n - 1)]
    seq_delta_succ      = [seq_delta_mat[i, i+1]      for i in range(n - 1)]
    same_delta_succ     = [same_delta_mat[i, i+1]     for i in range(n - 1)]
    rand_succ           = [rand_mat[i, i+1]            for i in range(n - 1)]
    rand_seq_delta_succ = [rand_seq_delta_mat[i, i+1] for i in range(n - 1)]
    rand_same_delta_succ= [rand_same_delta_mat[i, i+1]for i in range(n - 1)]

    print(f"  Residual-stream successive     (mean={np.mean(successive):.4f}): {[f'{v:.3f}' for v in successive]}")
    print(f"  Seq-delta successive           (mean={np.mean(seq_delta_succ):.4f}): {[f'{v:.3f}' for v in seq_delta_succ]}")
    print(f"  Same-input delta successive    (mean={np.mean(same_delta_succ):.4f}): {[f'{v:.3f}' for v in same_delta_succ]}")
    print(f"  Rand same-input delta succ     (mean={np.mean(rand_same_delta_succ):.4f}): {[f'{v:.3f}' for v in rand_same_delta_succ]}")

    del model
    return {
        "matrix":               mat,
        "seq_delta_matrix":     seq_delta_mat,
        "same_delta_matrix":    same_delta_mat,
        "rand_matrix":          rand_mat,
        "rand_seq_delta_matrix":rand_seq_delta_mat,
        "rand_same_delta_matrix":rand_same_delta_mat,
        "attn_matrix":          attn_mat,
        "ffn_matrix":           ffn_mat,
        "successive":           successive,
        "seq_delta_succ":       seq_delta_succ,
        "same_delta_succ":      same_delta_succ,
        "rand_succ":            rand_succ,
        "rand_seq_delta_succ":  rand_seq_delta_succ,
        "rand_same_delta_succ": rand_same_delta_succ,
        "delta_ratio":          ratio,
        "n_layers":             n,
    }


def main():
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_PATH)
    input_ids = get_input_tokens(tokenizer, EVAL_TEXT, N_TOKENS)
    print(f"Using {input_ids.shape[1]} tokens for activation capture.")

    all_results = {}
    for label, info in MODELS.items():
        r = analyze_model(label, info, tokenizer, input_ids)
        if r is not None:
            all_results[label] = r

    # 1. Per-model heatmaps — residual stream, sequential delta, same-input delta
    for label, data in all_results.items():
        stem = label.replace("\n", "_").replace(" ", "").replace("/", "").replace("(", "").replace(")", "").replace("+", "p")
        n = data["n_layers"]
        ll = [str(i) for i in range(n)]
        plot_heatmap(data["matrix"],
                     f"Residual-Stream Cosine Sim — {label.replace(chr(10), ' ')}",
                     f"act_heatmap_{stem}", labels=ll)
        plot_heatmap(data["same_delta_matrix"],
                     f"Same-Input Delta Cosine Sim — {label.replace(chr(10), ' ')}",
                     f"same_delta_heatmap_{stem}", labels=ll)
        if data["attn_matrix"] is not None:
            plot_heatmap(data["attn_matrix"],
                         f"Attn Output Cosine Sim — {label.replace(chr(10), ' ')}",
                         f"act_attn_heatmap_{stem}", labels=ll)

    # 2. Successive line plots for 12-layer and 6-layer models
    models_12 = {k: v for k, v in all_results.items() if v["n_layers"] == 12}
    models_6  = {k: v for k, v in all_results.items() if v["n_layers"] == 6}

    if models_12:
        plot_successive_line(models_12, "act_successive_cosim_12layer",
                             "Residual-Stream Cosine Sim, Consecutive Layers (12-layer)")
        plot_successive_line(models_12, "same_delta_successive_cosim_12layer",
                             "Same-Input Delta Cosine Sim, Consecutive Layers (12-layer)",
                             key="same_delta_succ")
    if models_6:
        plot_successive_line(models_6, "act_successive_cosim_6layer",
                             "Residual-Stream Cosine Sim, Consecutive Layers (6-layer merged)")
        plot_successive_line(models_6, "same_delta_successive_cosim_6layer",
                             "Same-Input Delta Cosine Sim, Consecutive Layers (6-layer merged)",
                             key="same_delta_succ")

    # 3. Three-panel overlay: residual / same-input delta / random same-input delta
    if models_12:
        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        for label, data in models_12.items():
            info = MODELS.get(label, {})
            color = info.get("color")
            marker = info.get("marker", "o")
            x = np.arange(len(data["successive"]))
            axes[0].plot(x, data["successive"],          color=color, linewidth=2, marker=marker, markersize=4,
                         label=label.replace("\n", " "))
            axes[0].plot(x, data["rand_succ"],           color=color, linewidth=1.2, linestyle=":", marker=marker, markersize=3)
            axes[1].plot(x, data["seq_delta_succ"],      color=color, linewidth=2, marker=marker, markersize=4,
                         label=label.replace("\n", " "))
            axes[1].plot(x, data["rand_seq_delta_succ"], color=color, linewidth=1.2, linestyle=":", marker=marker, markersize=3)
            axes[2].plot(x, data["same_delta_succ"],     color=color, linewidth=2, marker=marker, markersize=4,
                         label=label.replace("\n", " "))
            axes[2].plot(x, data["rand_same_delta_succ"],color=color, linewidth=1.2, linestyle=":", marker=marker, markersize=3)
        titles = [
            "Residual stream\n(biased by passthrough)",
            "Sequential delta\n(each layer sees evolved input)",
            "Same-input delta ← primary\n(all layers see same h)",
        ]
        for ax, title in zip(axes, titles):
            ax.set_xlabel("Layer pair (i, i+1)", fontsize=9)
            ax.set_ylabel("Cosine similarity", fontsize=9)
            ax.set_title(title, fontsize=9, fontweight="bold")
            ax.legend(fontsize=7, ncol=1)
            ax.grid(True, alpha=0.3)
            ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
        fig.suptitle("Three measures of layer similarity — solid=trained, dotted=random init (12-layer)",
                     fontsize=11, fontweight="bold")
        plt.tight_layout()
        savefig(fig, "act_three_measures_12layer")

    # 4. Bar chart: all three measures, all models
    labels_list = []
    res_avgs, same_delta_avgs, rand_same_delta_avgs, colors_list = [], [], [], []
    for label, data in all_results.items():
        n = data["n_layers"]
        mask = ~np.eye(n, dtype=bool)
        res_avgs.append(float(data["matrix"][mask].mean()))
        same_delta_avgs.append(float(data["same_delta_matrix"][mask].mean()))
        rand_same_delta_avgs.append(float(data["rand_same_delta_matrix"][mask].mean()))
        labels_list.append(label.replace("\n", " "))
        colors_list.append(MODELS[label]["color"] if label in MODELS else "#888888")

    x = np.arange(len(labels_list))
    width = 0.27
    fig, ax = plt.subplots(figsize=(max(7, len(labels_list) * 1.4), 5))
    ax.bar(x - width, res_avgs,            width, label="Residual stream",          color=colors_list, edgecolor="white", alpha=0.9)
    ax.bar(x,         same_delta_avgs,     width, label="Same-input Δ (trained)",   color=colors_list, edgecolor="white", alpha=0.6)
    ax.bar(x + width, rand_same_delta_avgs,width, label="Same-input Δ (random init)",color="#888888",   edgecolor="white", alpha=0.4)
    ax.set_xticks(x); ax.set_xticklabels(labels_list, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg cosine similarity (off-diagonal)", fontsize=11)
    ax.set_title("Layer similarity: residual vs same-input Δ vs random baseline", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, "act_three_measures_bar")

    # 5. Combined all-models successive line plot (normalised x-axis)
    fig, ax = plt.subplots(figsize=(9, 4))
    for label, data in all_results.items():
        cs = data["successive"]
        n = len(cs)
        x = np.linspace(0, 1, n)
        info = MODELS.get(label, {})
        color = info.get("color")
        marker = info.get("marker", "o")
        ls = "--" if data["n_layers"] == 6 else "-"
        ax.plot(x, cs, marker=marker, markersize=4, label=label.replace("\n", " "),
                color=color, linewidth=2, linestyle=ls)
    ax.axhline(0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
    ax.set_xlabel("Normalized layer position", fontsize=11)
    ax.set_ylabel("Residual-stream cosine sim (i, i+1)", fontsize=11)
    ax.set_title("Consecutive-Layer Activation Similarity — All Models", fontsize=12, fontweight="bold")
    ax.legend(fontsize=8, loc="upper right", ncol=2)
    ax.grid(True, alpha=0.3)
    savefig(fig, "act_successive_cosim_all")

    fig, ax = plt.subplots(figsize=(max(6, len(all_results) * 1.2), 5))
    labels_bar = [l.replace("\n", " ") for l in all_results]
    avgs_bar = [float(np.mean(d["successive"])) for d in all_results.values()]
    colors_bar = [MODELS.get(l, {}).get("color", "#888888") for l in all_results]
    bars = ax.bar(np.arange(len(labels_bar)), avgs_bar, color=colors_bar, edgecolor="white")
    ax.set_xticks(np.arange(len(labels_bar)))
    ax.set_xticklabels(labels_bar, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Mean residual-stream cosine sim", fontsize=11)
    ax.set_title("Mean Activation Similarity per Model", fontsize=12, fontweight="bold")
    for bar, val in zip(bars, avgs_bar):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, f"{val:.3f}",
                ha="center", va="bottom", fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, "act_avg_cosim_bar")

    # Attn vs FFN breakdown bar
    labels_bar2, attn_avgs, ffn_avgs, colors_bar2 = [], [], [], []
    for label, data in all_results.items():
        n = data["n_layers"]
        mask = ~np.eye(n, dtype=bool)
        labels_bar2.append(label.replace("\n", " "))
        attn_avgs.append(float(data["attn_matrix"][mask].mean()) if data.get("attn_matrix") is not None else float("nan"))
        ffn_avgs.append(float(data["ffn_matrix"][mask].mean()) if data.get("ffn_matrix") is not None else float("nan"))
        colors_bar2.append(MODELS.get(label, {}).get("color", "#888888"))
    x2 = np.arange(len(labels_bar2))
    width = 0.35
    fig, ax = plt.subplots(figsize=(max(7, len(labels_bar2) * 1.4), 5))
    ax.bar(x2 - width/2, attn_avgs, width, label="Attn",  color=colors_bar2, edgecolor="white", alpha=0.85)
    ax.bar(x2 + width/2, ffn_avgs,  width, label="FFN",   color=colors_bar2, edgecolor="white", alpha=0.45)
    ax.set_xticks(x2); ax.set_xticklabels(labels_bar2, fontsize=9, rotation=30, ha="right")
    ax.set_ylabel("Avg activation cosine sim", fontsize=11)
    ax.set_title("Activation Similarity: Attn vs FFN sub-outputs", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(True, axis="y", alpha=0.3)
    savefig(fig, "act_avg_cosim_attn_ffn_bar")

    # 6. Summary table
    print("\n\n=== SUMMARY TABLE ===")
    print(f"{'Model':<35} {'N':>3} {'Res.Succ':>10} {'SeqΔ.Succ':>10} {'SameΔ.Succ':>11} {'Rand.SameΔ':>11} {'||Δ||/||h||':>12}")
    print("-" * 110)
    for label, data in all_results.items():
        print(f"{label.replace(chr(10),' '):<35} {data['n_layers']:>3} "
              f"{np.mean(data['successive']):>10.4f} "
              f"{np.mean(data['seq_delta_succ']):>10.4f} "
              f"{np.mean(data['same_delta_succ']):>11.4f} "
              f"{np.mean(data['rand_same_delta_succ']):>11.4f} "
              f"{data['delta_ratio']:>12.4f}")

    print(f"\nAll plots saved to {PLOTS}")


if __name__ == "__main__":
    main()
