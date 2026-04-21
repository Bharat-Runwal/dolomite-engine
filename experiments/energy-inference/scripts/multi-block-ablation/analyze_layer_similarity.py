"""
Layer similarity analysis: V1 EGPT lr=2e-3 vs V0 GPT vs V1 EGPT (random init).

Interpretation framing:
  - attn_out / ffwd_out  = pre-proj activations = the "gradient" signal of the
    implicit energy (what EGPT uses as grad_E before projecting).
  - update (Δ residual)  = the full step applied = proj(grad_E); the proj is
    the "policy" (learning rate / exploration-exploitation strategy).
  - h_out                = residual stream state after the step.

If all EGPT layers truly optimize the same energy:
  → pre-proj CKA should be HIGH across layers (same gradient landscape)
  → update CKA could be lower (different policy weights per block)
  → Both should exceed the random-init baseline.

Three models compared:
  1. V1 EGPT lr=2e-3 (trained, 30k steps)
  2. V0 GPT baseline  (trained, 30k steps)
  3. V1 EGPT random   (same config, untrained — establishes baseline)

Four components × two metrics (CKA, cosine sim) × three models.

Saved:
  results/multi-block-ablation/plots/layer_sim_cka.{png,pdf}
  results/multi-block-ablation/plots/layer_sim_cosine.{png,pdf}
  results/multi-block-ablation/plots/layer_sim_summary.{png,pdf}
"""

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch

REPO = Path(__file__).parents[4]
RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401 — registers energy model with HF AutoModel
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TRAINED_MODELS = {
    "V1 EGPT\nlr=2e-3": RESULTS_DIR / "v1_12x1_d768_lr2e3" / "unsharded",
    "V0 GPT\nbaseline":  RESULTS_DIR / "v0_gpt_baseline_d768" / "unsharded",
}
# Random init uses V1 EGPT config but no trained weights
RANDOM_CKPT = RESULTS_DIR / "v1_12x1_d768_lr2e3" / "unsharded"

SEQ_LEN = 512

PREFILL_TEXT = """\
The tower is part of a complex of buildings that includes the Palace of Westminster, \
Westminster Abbey, and St Margaret's Church. The tower was completed in 1859 and contains \
the famous bell known as Big Ben, which is actually the name of the largest bell inside \
the tower. The clock faces are 23 feet (7.0 m) in diameter and are illuminated at night. \
The tower stands 315 feet (96 m) tall and is located at the north end of the Palace of \
Westminster. Robert Stephenson, the famous railway engineer, was one of the engineers \
involved in the design of the tower. The clock mechanism was designed by Edmund Beckett \
Denison and George Airy. The tower has been a Grade I listed building since 1970 and is \
part of a UNESCO World Heritage Site. The clock became internationally famous for its \
reliability and accuracy, with the BBC having broadcast the chimes since 1924. \
The tower's official name was changed to the Elizabeth Tower in 2012, in honour of \
Queen Elizabeth II's Diamond Jubilee. The tower has four clock faces, each of which \
is about 23 feet (7 m) in diameter. The minute hands on each clock face are about \
14 feet (4.3 m) long. Each numeral on the clock face is 2 feet (0.61 m) long. \
The pendulum inside the clock weighs 660 pounds (300 kg). The clock is regulated by \
a stack of coins placed on the pendulum: adding a coin has the effect of minutely \
increasing the pendulum's effective length, causing the clock to gain two-fifths of a \
second over the course of a day. The original tower, along with the rest of the palace, \
was destroyed in the fire of 1834. A competition was held for the design of the new \
palace, with Gothic or Elizabethan style stipulated. Charles Barry won the competition. \
The first bell was cast in 1856 but cracked beyond repair a short time later. \
A second bell was cast in 1858 and has been in use ever since, though it too cracked \
in 1859. The crack was rotated a quarter turn away from the hammer, and the bell has \
been in use continuously ever since. The tower itself has never had an official name, \
though it has always been referred to as the Clock Tower or St. Stephen's Tower \
until the renaming in 2012. The tower leans slightly to the northwest, at around \
0.26 degrees, and this lean is increasing very slowly over time due to tunnelling \
and underground construction in the area. Visitors can climb the 334 steps to the \
belfry where the bells are housed, though access is restricted to UK residents only. \
The tower was used as a prison during the 19th century, with both Members of Parliament \
and ordinary citizens being detained there for various offences, including contempt of \
Parliament. The last person to be imprisoned in the tower was Charles Bradlaugh in 1880.\
"""

COMP_NAMES = [
    "h_out\n(residual stream)",
    "attn_out\n(pre-proj = grad_E attn)",
    "ffwd_out\n(pre-proj = grad_E ffwd)",
    "update\n(proj policy step)",
]
COMP_KEYS = ["h_out", "attn", "ffwd", "update"]


# ── similarity metrics ────────────────────────────────────────────────────────

def linear_cka(X: torch.Tensor, Y: torch.Tensor) -> float:
    X = X - X.mean(0, keepdim=True)
    Y = Y - Y.mean(0, keepdim=True)
    XXT = X @ X.T
    YYT = Y @ Y.T
    num = (XXT * YYT).sum()
    denom = XXT.norm("fro") * YYT.norm("fro")
    return (num / (denom + 1e-12)).item()


def mean_cosine_sim(X: torch.Tensor, Y: torch.Tensor) -> float:
    X_n = torch.nn.functional.normalize(X.float(), dim=-1)
    Y_n = torch.nn.functional.normalize(Y.float(), dim=-1)
    return (X_n * Y_n).sum(-1).mean().item()


def sim_matrix(acts: list, fn) -> np.ndarray:
    L = len(acts)
    M = np.zeros((L, L))
    for i in range(L):
        for j in range(L):
            M[i, j] = fn(acts[i], acts[j])
    return M


# ── model utilities ───────────────────────────────────────────────────────────

def get_blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer.h
    raise AttributeError(f"Cannot find transformer blocks on {type(model)}")


def detect_type(model) -> str:
    block = get_blocks(model)[0]
    return "egpt" if (hasattr(block, "attn") and hasattr(block, "ffwd")) else "gpt"


# ── hook-based activation capture ────────────────────────────────────────────

def capture_activations(model, input_ids: torch.Tensor) -> dict:
    blocks = get_blocks(model)
    mtype = detect_type(model)
    nl = len(blocks)
    buf = {k: [None] * nl for k in ["h_in", "h_out", "attn", "ffwd"]}
    handles = []

    for i, block in enumerate(blocks):
        def pre(mod, inp, i=i):
            buf["h_in"][i] = inp[0].detach().float().squeeze(0)
        def post(mod, inp, out, i=i):
            buf["h_out"][i] = out.detach().float().squeeze(0)
        handles += [
            block.register_forward_pre_hook(pre),
            block.register_forward_hook(post),
        ]
        if mtype == "egpt":
            def ha(mod, inp, out, i=i):
                buf["attn"][i] = out.detach().float().squeeze(0)
            def hf(mod, inp, out, i=i):
                buf["ffwd"][i] = out.detach().float().squeeze(0)
            handles += [
                block.attn.register_forward_hook(ha),
                block.ffwd.register_forward_hook(hf),
            ]
        else:
            def ha(mod, inp, out, i=i):
                buf["attn"][i] = out.detach().float().squeeze(0)
            def hf(mod, inp, out, i=i):
                buf["ffwd"][i] = out.detach().float().squeeze(0)
            handles += [
                block.sequence_mixer.register_forward_hook(ha),
                block.mlp_block.register_forward_hook(hf),
            ]

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
        "mtype": mtype,
    }


# ── per-model analysis ────────────────────────────────────────────────────────

def analyze(label: str, model, input_ids: torch.Tensor) -> dict:
    print(f"\n  {label}  ({detect_type(model)}, {len(get_blocks(model))} blocks)")
    acts = capture_activations(model, input_ids)
    nl = acts["nl"]

    result = {}
    for key, name in zip(COMP_KEYS, COMP_NAMES):
        act_list = acts[key]
        if any(a is None for a in act_list):
            print(f"    SKIP {key}: missing activations")
            continue
        cka_mat    = sim_matrix(act_list, linear_cka)
        cosine_mat = sim_matrix(act_list, mean_cosine_sim)
        off_mask = ~np.eye(nl, dtype=bool)
        result[key] = {"cka": cka_mat, "cosine": cosine_mat, "nl": nl}
        print(f"    {key:<10}  CKA off-diag={cka_mat[off_mask].mean():.3f}"
              f"  cosine off-diag={cosine_mat[off_mask].mean():.3f}")
    return result


# ── plotting ──────────────────────────────────────────────────────────────────

MODEL_COLORS = {
    "V1 EGPT\nlr=2e-3": "#D65F5F",
    "V0 GPT\nbaseline":  "#4878CF",
    "V1 EGPT\nrandom":   "#888888",
}
MODEL_SHORT = {
    "V1 EGPT\nlr=2e-3": "V1 EGPT",
    "V0 GPT\nbaseline":  "V0 GPT",
    "V1 EGPT\nrandom":   "Random",
}


def plot_heatmaps(all_results: dict, metric: str, fname_stem: str):
    """4 rows (components) × N cols (models) heatmap grid."""
    model_names = list(all_results.keys())
    n_models = len(model_names)
    cmap = "RdYlGn" if metric == "cka" else "coolwarm"
    vmin, vmax = 0.0, 1.0

    fig, axes = plt.subplots(len(COMP_KEYS), n_models,
                              figsize=(5 * n_models, 4.2 * len(COMP_KEYS)),
                              squeeze=False)

    for row, (key, cname) in enumerate(zip(COMP_KEYS, COMP_NAMES)):
        for col, mname in enumerate(model_names):
            ax = axes[row][col]
            if key not in all_results[mname]:
                ax.set_visible(False)
                continue
            M = all_results[mname][key][metric]
            nl = all_results[mname][key]["nl"]
            im = ax.imshow(M, vmin=vmin, vmax=vmax, cmap=cmap, aspect="auto")
            short = MODEL_SHORT[mname]
            ax.set_title(f"{short}\n{cname}", fontsize=8.5, fontweight="bold")
            ax.set_xlabel("Layer", fontsize=8)
            ax.set_ylabel("Layer", fontsize=8)
            ticks = list(range(0, nl, 2)) if nl > 8 else list(range(nl))
            ax.set_xticks(ticks); ax.set_xticklabels(ticks, fontsize=7)
            ax.set_yticks(ticks); ax.set_yticklabels(ticks, fontsize=7)
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
            off = M[~np.eye(nl, dtype=bool)].mean()
            ax.text(0.97, 0.03, f"mean={off:.3f}",
                    transform=ax.transAxes, ha="right", va="bottom",
                    fontsize=8, bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.75))

    metric_label = "CKA" if metric == "cka" else "Mean Cosine Similarity"
    fig.suptitle(
        f"Layer Representation Similarity ({metric_label})\n"
        f"512-token WikiText prefill  |  pre-proj = gradient signal  |  update = policy step",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"{fname_stem}.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def plot_summary_bars(all_results: dict):
    """Bar chart: off-diagonal mean CKA for each component × model."""
    model_names = list(all_results.keys())
    n_models = len(model_names)
    n_comps = len(COMP_KEYS)
    x = np.arange(n_comps)
    width = 0.22
    offsets = np.linspace(-(n_models - 1) / 2, (n_models - 1) / 2, n_models) * width

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax_idx, metric in enumerate(["cka", "cosine"]):
        ax = axes[ax_idx]
        for mi, mname in enumerate(model_names):
            vals = []
            for key in COMP_KEYS:
                if key not in all_results[mname]:
                    vals.append(0.0)
                    continue
                M = all_results[mname][key][metric]
                nl = all_results[mname][key]["nl"]
                vals.append(M[~np.eye(nl, dtype=bool)].mean())
            col = MODEL_COLORS[mname]
            bars = ax.bar(x + offsets[mi], vals, width * 0.9,
                          label=MODEL_SHORT[mname], color=col, alpha=0.85, edgecolor="white")
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                        f"{v:.3f}", ha="center", va="bottom", fontsize=7.5, rotation=60)

        ax.set_xticks(x)
        short_names = [c.split("\n")[0] for c in COMP_NAMES]
        ax.set_xticklabels(short_names, fontsize=9)
        metric_label = "Linear CKA" if metric == "cka" else "Mean Cosine Sim"
        ax.set_ylabel(f"Mean off-diagonal {metric_label}", fontsize=10)
        ax.set_title(
            f"Inter-layer similarity ({metric_label})\n"
            "(higher = layers produce more similar representations)",
            fontsize=10, fontweight="bold"
        )
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        if metric == "cosine":
            ax.axhline(0, color="gray", linewidth=0.8, linestyle="--")

    # Add annotation explaining components
    fig.text(0.5, -0.04,
             "attn_out / ffwd_out = pre-proj activations = gradient signal of the energy  |  "
             "update = proj(grad) = policy step applied to residual",
             ha="center", fontsize=8.5, style="italic", color="#444444")

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"layer_sim_summary.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    tokenizer = AutoTokenizer.from_pretrained(str(RANDOM_CKPT), trust_remote_code=True)
    input_ids = tokenizer(PREFILL_TEXT, return_tensors="pt")["input_ids"]
    actual_len = min(input_ids.shape[1], SEQ_LEN)
    print(f"Prefill: {input_ids.shape[1]} tokens, using {actual_len}")
    input_ids = input_ids[:, :actual_len]

    all_results = {}

    print("\n" + "=" * 60)
    for label, ckpt in TRAINED_MODELS.items():
        print(f"Loading {label.strip()} from {ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), dtype=torch.float32, trust_remote_code=True, device_map="cpu"
        )
        model.eval()
        all_results[label] = analyze(label, model, input_ids)
        del model

    # Random init: same config as V1 EGPT, no trained weights
    print("\n" + "=" * 60)
    print("Building random-init V1 EGPT (same config, untrained)...")
    config = AutoConfig.from_pretrained(str(RANDOM_CKPT), trust_remote_code=True)
    rand_model = AutoModelForCausalLM.from_config(config)
    rand_model.eval()
    rand_label = "V1 EGPT\nrandom"
    all_results[rand_label] = analyze(rand_label, rand_model, input_ids)
    del rand_model

    # ── summary table ──────────────────────────────────────────────────────────
    print("\n=== Off-diagonal mean similarity ===")
    nl = next(v["nl"] for res in all_results.values() for v in res.values())
    header = f"{'Component':<12} {'Metric':>8}"
    for m in all_results:
        header += f"  {MODEL_SHORT[m]:>12}"
    print(header)
    print("─" * len(header))
    for key in COMP_KEYS:
        for metric in ["cka", "cosine"]:
            row = f"{key:<12} {metric.upper():>8}"
            for mname in all_results:
                if key not in all_results[mname]:
                    row += f"  {'N/A':>12}"
                else:
                    M = all_results[mname][key][metric]
                    off = M[~np.eye(nl, dtype=bool)].mean()
                    row += f"  {off:>12.4f}"
            print(row)

    # ── plots ──────────────────────────────────────────────────────────────────
    plot_heatmaps(all_results, "cka",    "layer_sim_cka")
    plot_heatmaps(all_results, "cosine", "layer_sim_cosine")
    plot_summary_bars(all_results)


if __name__ == "__main__":
    main()
