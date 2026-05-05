"""
Layer similarity analysis extended to EGrad and EDesc models.

Two scale groups:
  Small (d=768, 12 blocks): V0 GPT, V1 EGPT, V15 EGrad, V16 EDesc
  400M  (d=1024, 24 blocks): V9 GPT, V19 EGrad, V20 EDesc

For each group produces:
  - layer_sim_cosine_attn_{small,400m}.{pdf,png}   — L×L cosine heatmaps (attn_out + ffwd_out)
  - consecutive_sim_{small,400m}.{pdf,png}          — consecutive-layer cosine line plots
  - layer_sim_summary_egrad_edesc.{pdf,png}         — bar summary comparing off-diag means
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).parents[4]
RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401 — registers energy model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

# ── model registry ────────────────────────────────────────────────────────────

SMALL_MODELS = {
    "V0 GPT":   (RESULTS_DIR / "v0_gpt_baseline_d768" / "unsharded",             "gpt"),
    "V1 EGPT":  (RESULTS_DIR / "v1_12x1_d768_lr2e3"   / "unsharded",             "egpt"),
    "V15 EGrad":(RESULTS_DIR / "v15_energy_grad_mixed_12x1_d768_lr2e3" / "unsharded", "gpt"),
    "V16 EDesc":(RESULTS_DIR / "v16_mixed_energy_descent_12x1_d768_lr2e3" / "unsharded","gpt"),
}
SMALL_RANDOM_CKPT = RESULTS_DIR / "v1_12x1_d768_lr2e3" / "unsharded"
SMALL_RANDOM_LABEL = "V1 EGPT\nrandom"

LARGE_MODELS = {
    "V9 GPT":   (RESULTS_DIR / "v9_gpt_baseline_d1024_lr1e3" / "unsharded",      "gpt"),
    "V19 EGrad":(RESULTS_DIR / "v19_energy_grad_24x1_d1024_lr1e3" / "unsharded", "gpt"),
    "V20 EDesc":(RESULTS_DIR / "v20_energy_desc_24x1_d1024_lr1e3" / "unsharded", "gpt"),
}

# ── colors / markers ──────────────────────────────────────────────────────────

COLORS = {
    "V0 GPT":       "#4878CF",
    "V1 EGPT":      "#D65F5F",
    "V15 EGrad":    "#2CA02C",
    "V16 EDesc":    "#FF7F0E",
    "V9 GPT":       "#4878CF",
    "V19 EGrad":    "#2CA02C",
    "V20 EDesc":    "#FF7F0E",
    "V1 EGPT\nrandom": "#888888",
}
MARKERS = {
    "V0 GPT": "s", "V1 EGPT": "o", "V15 EGrad": "D", "V16 EDesc": "^",
    "V9 GPT": "s", "V19 EGrad": "D", "V20 EDesc": "^",
    "V1 EGPT\nrandom": "x",
}

SEQ_LEN = 512

PREFILL = (
    "The tower is part of a complex of buildings that includes the Palace of Westminster, "
    "Westminster Abbey, and St Margaret's Church. The tower was completed in 1859 and contains "
    "the famous bell known as Big Ben. The clock faces are 23 feet in diameter and illuminated "
    "at night. The tower stands 315 feet tall at the north end of the Palace of Westminster. "
    "The clock mechanism was designed by Edmund Beckett Denison and George Airy. The tower has "
    "been a Grade I listed building since 1970 and is part of a UNESCO World Heritage Site. "
    "The clock became internationally famous for its reliability and accuracy, with the BBC "
    "having broadcast the chimes since 1924. The tower's official name was changed to the "
    "Elizabeth Tower in 2012, in honour of Queen Elizabeth II's Diamond Jubilee. The pendulum "
    "inside the clock weighs 660 pounds. The clock is regulated by a stack of coins placed on "
    "the pendulum: adding a coin minutely increases the pendulum's effective length, causing "
    "the clock to gain two-fifths of a second per day. The original tower was destroyed in the "
    "fire of 1834. A competition was held for the design of the new palace. Charles Barry won. "
    "The first bell was cast in 1856 but cracked beyond repair. A second bell was cast in 1858 "
    "and has been in use ever since, though it too cracked in 1859. The crack was rotated a "
    "quarter turn away from the hammer and the bell has been in use continuously ever since."
)


# ── model / block utilities ───────────────────────────────────────────────────

def get_blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    raise AttributeError(f"Cannot find transformer blocks on {type(model)}")


def is_egpt_v1(model) -> bool:
    """Return True only for original V1 EGPT which has block.attn / block.ffwd."""
    blk = get_blocks(model)[0]
    return hasattr(blk, "attn") and hasattr(blk, "ffwd")


# ── activation capture ────────────────────────────────────────────────────────

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

        if legacy:
            attn_src, ffwd_src = block.attn, block.ffwd
        else:
            attn_src, ffwd_src = block.sequence_mixer, block.mlp_block

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


# ── similarity metrics ────────────────────────────────────────────────────────

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


# ── load and analyse a model ──────────────────────────────────────────────────

def load_and_analyse(label, ckpt_path, input_ids, is_random=False, random_ckpt=None):
    if is_random:
        cfg = AutoConfig.from_pretrained(str(random_ckpt), trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
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
        od = off_diag_mean(M)
        print(f"    {comp}: off-diag cosine={od:.4f}, consec_max={max(c):.4f} @ pair {np.argmax(c)}")
        result[comp] = {"cosine_matrix": M, "consecutive": c, "off_diag": od}
    del model
    return result, nl


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_heatmaps(results_dict, nl, tag, title_suffix):
    """Cosine heatmaps for attn_out and ffwd_out side-by-side, one column per model."""
    model_names = list(results_dict.keys())
    n = len(model_names)
    comps = [("attn", "attn_out  (pre-residual attention)"),
             ("ffwd", "ffwd_out  (pre-residual FFN)")]

    fig, axes = plt.subplots(len(comps), n, figsize=(4.5 * n, 4.0 * len(comps)), squeeze=False)

    for row, (comp, comp_label) in enumerate(comps):
        for col, mname in enumerate(model_names):
            ax = axes[row][col]
            M = results_dict[mname][comp]["cosine_matrix"]
            od = results_dict[mname][comp]["off_diag"]
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

    fig.suptitle(f"Inter-layer cosine similarity — {title_suffix}\n"
                 "attn_out = pre-residual attention output;  ffwd_out = pre-residual FFN output",
                 fontsize=10, fontweight="bold", y=1.01)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"layer_sim_cosine_{tag}.{ext}", dpi=150, bbox_inches="tight")
        print(f"  Saved layer_sim_cosine_{tag}.{ext}")
    plt.close(fig)


def plot_consecutive(results_dict, nl, tag, title_suffix):
    """Consecutive-layer cosine similarity line plots (attn and ffwd)."""
    model_names = list(results_dict.keys())
    x = np.arange(nl - 1)
    pair_labels = [f"{i}→{i+1}" for i in range(nl - 1)]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, comp, ctitle in zip(
        axes,
        ["attn", "ffwd"],
        ["attn_out  (pre-residual attention)", "ffwd_out  (pre-residual FFN)"],
    ):
        for mname in model_names:
            vals = results_dict[mname][comp]["consecutive"]
            ax.plot(x, vals, color=COLORS[mname], marker=MARKERS[mname],
                    markersize=5, linewidth=1.8, label=mname)
            peak = int(np.argmax(np.abs(vals)))
            ax.annotate(f"{vals[peak]:.3f}", xy=(x[peak], vals[peak]),
                        xytext=(0, 8), textcoords="offset points",
                        ha="center", fontsize=7.5, color=COLORS[mname])

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        step = max(1, (nl - 1) // 12)
        show_idx = list(range(0, nl - 1, step))
        ax.set_xticks([x[i] for i in show_idx])
        ax.set_xticklabels([pair_labels[i] for i in show_idx], fontsize=8, rotation=45)
        ax.set_xlabel("Layer pair", fontsize=10)
        ax.set_ylabel("Mean cosine similarity", fontsize=10)
        ax.set_title(ctitle, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        f"Consecutive-layer cosine similarity — {title_suffix}\n"
        "cos_sim(layer[i], layer[i+1])  |  higher = adjacent layers produce more similar outputs",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"consecutive_sim_{tag}.{ext}", dpi=150, bbox_inches="tight")
        print(f"  Saved consecutive_sim_{tag}.{ext}")
    plt.close(fig)


def plot_summary(small_results, large_results):
    """Bar chart: off-diagonal mean cosine similarity for attn_out across all models."""
    all_models = list(small_results.keys()) + list(large_results.keys())
    attn_vals = [small_results[m]["attn"]["off_diag"] for m in small_results] + \
                [large_results[m]["attn"]["off_diag"] for m in large_results]
    colors = [COLORS[m] for m in all_models]

    fig, ax = plt.subplots(figsize=(10, 5))
    bars = ax.bar(range(len(all_models)), attn_vals, color=colors, edgecolor="white", alpha=0.88)
    for bar, v in zip(bars, attn_vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
                f"{v:.3f}", ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax.axvline(len(small_results) - 0.5, color="gray", linewidth=1, linestyle="--", alpha=0.7)
    ax.text(len(small_results) / 2 - 0.5, max(attn_vals) * 0.98, "d=768 (12×1)",
            ha="center", fontsize=9, color="gray")
    ax.text(len(small_results) + len(large_results) / 2 - 0.5, max(attn_vals) * 0.98,
            "d=1024 (24×1)", ha="center", fontsize=9, color="gray")
    ax.set_xticks(range(len(all_models)))
    ax.set_xticklabels(all_models, fontsize=10)
    ax.set_ylabel("Mean off-diagonal cosine similarity (attn_out)", fontsize=10)
    ax.set_title("Inter-layer attention similarity: EGrad and EDesc vs baselines\n"
                 "Higher = more similar gradient signal across layers (shared energy structure)",
                 fontsize=11, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    for ext in ["pdf", "png"]:
        fig.savefig(PLOTS_DIR / f"layer_sim_summary_egrad_edesc.{ext}", dpi=150, bbox_inches="tight")
        print(f"  Saved layer_sim_summary_egrad_edesc.{ext}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    tokenizer = AutoTokenizer.from_pretrained(
        str(SMALL_MODELS["V0 GPT"][0]), trust_remote_code=True
    )
    input_ids = tokenizer(PREFILL, return_tensors="pt")["input_ids"][:, :SEQ_LEN]
    print(f"Prefill: {input_ids.shape[1]} tokens")

    # ── small scale (d=768, 12 blocks) ────────────────────────────────────────
    print("\n=== Small scale (d=768) ===")
    small_results = {}
    for label, (ckpt, _) in SMALL_MODELS.items():
        print(f"\nLoading {label}")
        res, nl_small = load_and_analyse(label, ckpt, input_ids)
        small_results[label] = res

    # random-init baseline for small models
    print(f"\nBuilding random-init V1 EGPT")
    rand_res, _ = load_and_analyse(
        SMALL_RANDOM_LABEL, None, input_ids, is_random=True, random_ckpt=SMALL_RANDOM_CKPT
    )
    small_results[SMALL_RANDOM_LABEL] = rand_res

    # ── 400M scale (d=1024, 24 blocks) ───────────────────────────────────────
    print("\n=== 400M scale (d=1024) ===")
    large_results = {}
    for label, (ckpt, _) in LARGE_MODELS.items():
        print(f"\nLoading {label}")
        res, nl_large = load_and_analyse(label, ckpt, input_ids)
        large_results[label] = res

    # ── summary table ─────────────────────────────────────────────────────────
    print("\n=== Summary: off-diagonal mean cosine similarity (attn_out) ===")
    for group, results in [("Small (d=768)", small_results), ("400M (d=1024)", large_results)]:
        print(f"\n{group}:")
        for m, r in results.items():
            print(f"  {m:<20} attn={r['attn']['off_diag']:.4f}  "
                  f"ffwd={r['ffwd']['off_diag']:.4f}  "
                  f"update={r['update']['off_diag']:.4f}")

    # ── plots: small scale ────────────────────────────────────────────────────
    # Show only trained models in heatmaps (exclude random from heatmaps to keep it compact)
    small_trained = {k: v for k, v in small_results.items() if "random" not in k}
    plot_heatmaps(small_trained, nl_small, "small", "d=768, 12×1  (V0 GPT, V1 EGPT, V15 EGrad, V16 EDesc)")
    # Include random in consecutive plots
    plot_consecutive(small_results, nl_small, "small",
                     "d=768, 12×1  (V0 GPT, V1 EGPT, V15 EGrad, V16 EDesc)")

    # ── plots: 400M scale ─────────────────────────────────────────────────────
    plot_heatmaps(large_results, nl_large, "400m", "d=1024, 24×1  (V9 GPT, V19 EGrad, V20 EDesc)")
    plot_consecutive(large_results, nl_large, "400m",
                     "d=1024, 24×1  (V9 GPT, V19 EGrad, V20 EDesc)")

    # ── summary bar chart (no random) ─────────────────────────────────────────
    plot_summary(small_trained, large_results)

    print("\nDone. All plots saved to:", PLOTS_DIR)


if __name__ == "__main__":
    main()
