"""
Two analyses:
1. Consecutive-layer cosine similarity: cos_sim(attn[i], attn[i+1]) and
   cos_sim(ffwd[i], ffwd[i+1]) for i=0..10, for V1 EGPT, V0 GPT, random.
   Answers: are nearby layers more similar? Do later layers cluster?

2. ffwd_out sanity check: print mean/std/norm of ffwd_out at each layer
   and a sample cosine similarity matrix to verify activations are non-trivial.

Saved: results/multi-block-ablation/plots/consecutive_layer_sim.{png,pdf}
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
import lm_engine.hf_models  # noqa: F401
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

TRAINED_MODELS = {
    "V1 EGPT lr=2e-3": RESULTS_DIR / "v1_12x1_d768_lr2e3"  / "unsharded",
    "V0 GPT baseline":  RESULTS_DIR / "v0_gpt_baseline_d768" / "unsharded",
}
RANDOM_CKPT = RESULTS_DIR / "v1_12x1_d768_lr2e3" / "unsharded"

MODEL_COLORS = {
    "V1 EGPT lr=2e-3": "#D65F5F",
    "V0 GPT baseline":  "#4878CF",
    "V1 EGPT random":   "#888888",
}
MODEL_MARKERS = {
    "V1 EGPT lr=2e-3": "o",
    "V0 GPT baseline":  "s",
    "V1 EGPT random":   "^",
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


def get_blocks(model):
    if hasattr(model, "transformer"):
        return model.transformer.h
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer.h
    raise AttributeError(f"Cannot find blocks on {type(model)}")


def detect_type(model):
    b = get_blocks(model)[0]
    return "egpt" if (hasattr(b, "attn") and hasattr(b, "ffwd")) else "gpt"


def capture(model, input_ids):
    blocks = get_blocks(model)
    mtype  = detect_type(model)
    nl     = len(blocks)
    buf    = {k: [None] * nl for k in ["attn", "ffwd"]}
    handles = []

    for i, block in enumerate(blocks):
        if mtype == "egpt":
            handles += [
                block.attn.register_forward_hook(
                    lambda m, inp, out, i=i: buf["attn"].__setitem__(i, out.detach().float().squeeze(0))),
                block.ffwd.register_forward_hook(
                    lambda m, inp, out, i=i: buf["ffwd"].__setitem__(i, out.detach().float().squeeze(0))),
            ]
        else:
            handles += [
                block.sequence_mixer.register_forward_hook(
                    lambda m, inp, out, i=i: buf["attn"].__setitem__(i, out.detach().float().squeeze(0))),
                block.mlp_block.register_forward_hook(
                    lambda m, inp, out, i=i: buf["ffwd"].__setitem__(i, out.detach().float().squeeze(0))),
            ]

    with torch.no_grad():
        model(input_ids[:, :SEQ_LEN])
    for h in handles:
        h.remove()

    return buf, nl, mtype


def cosine_sim_pair(X, Y):
    """Mean cosine similarity between two (T, d) tensors."""
    return (F.normalize(X, dim=-1) * F.normalize(Y, dim=-1)).sum(-1).mean().item()


def consecutive_sims(buf, nl, key):
    """cos_sim(buf[key][i], buf[key][i+1]) for i in 0..nl-2."""
    return [cosine_sim_pair(buf[key][i], buf[key][i+1]) for i in range(nl - 1)]


def ffwd_sanity(label, buf, nl):
    """Print statistics to verify ffwd_out activations are non-trivial."""
    print(f"\n  [{label}] ffwd_out sanity check:")
    print(f"  {'Layer':>5}  {'mean':>8}  {'std':>8}  {'l2norm':>8}  {'cos_sim(i,i+1)':>15}")
    for i in range(nl):
        x = buf["ffwd"][i]  # (T, d)
        mn = x.mean().item()
        sd = x.std().item()
        nm = x.norm(dim=-1).mean().item()  # mean token norm
        cs = cosine_sim_pair(x, buf["ffwd"][i + 1]) if i < nl - 1 else float("nan")
        print(f"  {i:>5}  {mn:>8.4f}  {sd:>8.4f}  {nm:>8.4f}  {cs:>15.4f}")


def main():
    tokenizer = AutoTokenizer.from_pretrained(str(RANDOM_CKPT), trust_remote_code=True)
    input_ids = tokenizer(PREFILL, return_tensors="pt")["input_ids"]
    input_ids = input_ids[:, :SEQ_LEN]
    print(f"Using {input_ids.shape[1]} tokens")

    all_sims = {}  # label -> {"attn": [...], "ffwd": [...]}

    # ── trained models ────────────────────────────────────────────────────────
    for label, ckpt in TRAINED_MODELS.items():
        print(f"\nLoading {label} from {ckpt}")
        model = AutoModelForCausalLM.from_pretrained(
            str(ckpt), dtype=torch.float32, trust_remote_code=True, device_map="cpu"
        )
        model.eval()
        buf, nl, mtype = capture(model, input_ids)
        print(f"  type={mtype}, {nl} blocks")
        ffwd_sanity(label, buf, nl)
        all_sims[label] = {
            "attn": consecutive_sims(buf, nl, "attn"),
            "ffwd": consecutive_sims(buf, nl, "ffwd"),
            "nl": nl,
        }
        del model

    # ── random init ───────────────────────────────────────────────────────────
    rand_label = "V1 EGPT random"
    print(f"\nBuilding random-init V1 EGPT...")
    config = AutoConfig.from_pretrained(str(RANDOM_CKPT), trust_remote_code=True)
    model  = AutoModelForCausalLM.from_config(config)
    model.eval()
    buf, nl, mtype = capture(model, input_ids)
    ffwd_sanity(rand_label, buf, nl)
    all_sims[rand_label] = {
        "attn": consecutive_sims(buf, nl, "attn"),
        "ffwd": consecutive_sims(buf, nl, "ffwd"),
        "nl": nl,
    }
    del model

    # ── print table ───────────────────────────────────────────────────────────
    nl = all_sims[list(all_sims.keys())[0]]["nl"]
    pairs = [f"{i}→{i+1}" for i in range(nl - 1)]
    print(f"\n{'Pair':>6}", end="")
    for lbl in all_sims:
        print(f"  {lbl[:15]:>15}(attn)  {lbl[:15]:>15}(ffwd)", end="")
    print()
    print("─" * (6 + len(all_sims) * 36))
    for j, pair in enumerate(pairs):
        print(f"{pair:>6}", end="")
        for lbl in all_sims:
            a = all_sims[lbl]["attn"][j]
            f = all_sims[lbl]["ffwd"][j]
            print(f"  {a:>21.4f}  {f:>21.4f}", end="")
        print()

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)
    x = np.arange(nl - 1)
    pair_labels = pairs

    for ax, comp, title in zip(
        axes,
        ["attn", "ffwd"],
        ["attn_out  (pre-proj gradient signal)", "ffwd_out  (pre-proj FFN signal)"],
    ):
        for label, sims in all_sims.items():
            vals = sims[comp]
            ax.plot(x, vals,
                    color=MODEL_COLORS[label],
                    marker=MODEL_MARKERS[label],
                    markersize=6, linewidth=1.8,
                    label=label)
            # highlight max
            peak = int(np.argmax(vals))
            ax.annotate(f"{vals[peak]:.3f}",
                        xy=(x[peak], vals[peak]),
                        xytext=(0, 7), textcoords="offset points",
                        ha="center", fontsize=7.5,
                        color=MODEL_COLORS[label])

        ax.axhline(0, color="gray", linewidth=0.7, linestyle="--", alpha=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(pair_labels, fontsize=8, rotation=45)
        ax.set_xlabel("Layer pair", fontsize=10)
        ax.set_ylabel("Mean cosine similarity", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.legend(fontsize=9)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    fig.suptitle(
        "Consecutive-layer cosine similarity: cos_sim(layer[i], layer[i+1])\n"
        "Higher = adjacent layers produce more similar pre-projection outputs",
        fontsize=11, fontweight="bold", y=1.01,
    )
    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"consecutive_layer_sim.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Saved {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
