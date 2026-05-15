"""
BoltzmannMoE Expert Specialization Analysis
============================================
For each finished/checkpointed model, runs inference on benchmark samples
and captures per-layer Boltzmann routing weights.  Produces:

  expert_routing_heatmap_<model>.pdf   -- mean routing weight per expert per layer,
                                          split by task category
  expert_pca_<model>.pdf               -- PCA of per-sample routing vectors,
                                          coloured by category
  expert_correlation_<model>.pdf       -- expert × category correlation matrix
  expert_load_<model>.pdf              -- expert load (fraction as argmax) by category
  expert_specialization_summary.txt    -- text report

Usage (interactive node with 1 GPU):
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  cd /proj/dmfexp/nima/Code/dolomite-engine
  python experiments/energy-inference/scripts/multi-block-ablation/\\
      analyze_boltz_expert_specialization_20260429.py [--model b1] [--n_per_cat 80]
"""

import argparse, json, re
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import torch
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import lm_engine.hf_models  # noqa: registers 'energy' model type with HF AutoModel
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO = Path(__file__).parents[4]
RESULTS = REPO / "experiments/boltzmann-moe/results"
PLOTS   = RESULTS / "plots"
PLOTS.mkdir(exist_ok=True)

MODELS = {
    "b1": RESULTS / "b1_boltz_moe_16x1024_d768_lr2e3/unsharded",
    "b5": RESULTS / "b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3/unsharded",
    # checkpoints (sharded) — use latest global_step dir as a proxy via config only
    # (full inference requires unsharded; skip if not available)
}

N_LAYERS  = 12
N_EXPERTS = 16
MAX_LEN   = 256   # tokens per sample

# ── Dataset builders ────────────────────────────────────────────────────────

MMLU_GROUPS = {
    "STEM":       ["abstract_algebra","college_mathematics","high_school_mathematics",
                   "high_school_physics","high_school_chemistry","college_physics",
                   "computer_security","high_school_computer_science"],
    "Humanities": ["high_school_history","world_history","philosophy",
                   "international_law","jurisprudence"],
    "Social":     ["high_school_psychology","sociology","economics","political_science",
                   "high_school_government_and_politics"],
    "Medical":    ["clinical_knowledge","medical_genetics","anatomy",
                   "college_medicine","professional_medicine"],
    "Logic":      ["formal_logic","logical_fallacies"],
}

def load_mmlu(group_name, subjects, n_per_subj=10):
    from datasets import load_dataset
    texts = []
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        except Exception:
            continue
        for ex in list(ds)[:n_per_subj]:
            choices = "\n".join(f"{c}. {ex['choices'][i]}"
                                for i, c in enumerate("ABCD"))
            texts.append(f"Question: {ex['question']}\n{choices}\nAnswer:")
    return texts


def load_boolq(n=80):
    from datasets import load_dataset
    ds = load_dataset("boolq", split="validation", trust_remote_code=True)
    return [f"Passage: {ex['passage'][:300]}\nQuestion: {ex['question']}\nTrue or False:"
            for ex in list(ds)[:n]]


def load_copa(n=80):
    from datasets import load_dataset
    ds = load_dataset("super_glue", "copa", split="validation", trust_remote_code=True)
    texts = []
    for ex in list(ds)[:n]:
        premise = ex["premise"].rstrip(".")
        texts.append(f"Premise: {premise}. "
                     f"A: {ex['choice1']}  B: {ex['choice2']}  "
                     f"{'Because' if ex['question']=='cause' else 'So'}:")
    return texts


def load_gsm8k(n=80):
    from datasets import load_dataset
    ds = load_dataset("gsm8k", "main", split="test", trust_remote_code=True)
    return [f"Problem: {ex['question']}\nSolution:" for ex in list(ds)[:n]]


def build_categories(n_per_cat):
    cats = {}
    for grp, subjects in MMLU_GROUPS.items():
        texts = load_mmlu(grp, subjects, n_per_subj=max(4, n_per_cat // len(subjects)))
        if texts:
            cats[f"MMLU-{grp}"] = texts[:n_per_cat]
    cats["BoolQ"]  = load_boolq(n_per_cat)
    cats["COPA"]   = load_copa(n_per_cat)
    cats["GSM8k"]  = load_gsm8k(n_per_cat)
    return cats


# ── Routing weight capture ───────────────────────────────────────────────────

def attach_routing_hooks(model):
    """Monkey-patch BoltzmannMoE forward to expose routing weights.

    Stores last routing weights in module._captured_routing (B,T,n_experts).
    Returns list of hooked modules in layer order.
    """
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP
    import torch.nn.functional as F
    import math

    hooked = []
    for name, module in model.named_modules():
        if not isinstance(module, BoltzmannMoE_Energy_MLP):
            continue
        module._captured_routing = None

        orig_forward = module.forward  # already-bound method — captures self correctly

        def make_patched(m, orig):
            def patched_forward(x):
                leading = x.shape[:-1]
                with torch.no_grad():
                    W1x = m.W1(x).view(*leading, m.n_experts, m.expert_I)
                    W2_e = m.W2.weight.view(m.n_experts, m.expert_I, m.hidden_size)
                    phi = F.gelu(W1x)
                    term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
                    E = torch.einsum("...h,...eh->...e", x, term1)
                    p = F.softmax(E / m.temperature, dim=-1)
                m._captured_routing = p.detach().float().cpu()
                return orig(x)  # call original bound method normally
            return patched_forward

        module.forward = make_patched(module, orig_forward)
        hooked.append((name, module))

    return hooked


def collect_routing(model, tokenizer, texts, device, batch_size=4):
    """Run texts through model; return array (n_samples, n_layers, n_experts).

    Each entry is the mean routing weight (averaged over sequence tokens) for
    that sample × layer × expert.
    """
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP

    # Get ordered list of BoltzmannMoE modules (layer 0..11)
    moe_modules = [(n, m) for n, m in model.named_modules()
                   if isinstance(m, BoltzmannMoE_Energy_MLP)]
    moe_modules.sort(key=lambda x: int(re.search(r'\.(\d+)\.', x[0]).group(1))
                     if re.search(r'\.(\d+)\.', x[0]) else 0)
    n_layers = len(moe_modules)

    all_routing = []  # list of (n_layers, n_experts)

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        enc = tokenizer(batch_texts, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            model(**enc)

        # collect routing: average over (batch, seq_tokens) per sample
        # _captured_routing shape: (B, T, n_experts) or (B*T, n_experts) padding-free
        batch_size_actual = len(batch_texts)
        for b in range(batch_size_actual):
            sample_routing = []
            for _, m in moe_modules:
                if m._captured_routing is None:
                    sample_routing.append(np.full(N_EXPERTS, 1.0/N_EXPERTS))
                    continue
                p = m._captured_routing  # (B, T, n_experts)
                if p.dim() == 2:  # padding-free (B*T, n_experts)
                    p_b = p.numpy()
                else:
                    p_b = p[b].numpy()  # (T, n_experts)
                # mean over tokens (non-padding)
                if enc.attention_mask is not None and p.dim() == 3:
                    mask = enc.attention_mask[b].cpu().numpy().astype(bool)
                    p_b = p_b[mask]
                sample_routing.append(p_b.mean(axis=0))  # (n_experts,)
            all_routing.append(np.stack(sample_routing))  # (n_layers, n_experts)

    return np.array(all_routing)  # (n_samples, n_layers, n_experts)


# ── Plotting ─────────────────────────────────────────────────────────────────

COLORS_CAT = {
    "MMLU-STEM":       "#e74c3c",
    "MMLU-Humanities": "#e67e22",
    "MMLU-Social":     "#f1c40f",
    "MMLU-Medical":    "#2ecc71",
    "MMLU-Logic":      "#1abc9c",
    "BoolQ":           "#3498db",
    "COPA":            "#9b59b6",
    "GSM8k":           "#e91e63",
}


def plot_heatmap(cat_routing, model_tag):
    """Mean routing weight per expert per layer, split by category."""
    cats = list(cat_routing.keys())
    n_cats = len(cats)
    fig, axes = plt.subplots(1, n_cats, figsize=(3*n_cats, 4), sharey=True)
    if n_cats == 1:
        axes = [axes]

    for ax, cat in zip(axes, cats):
        mat = cat_routing[cat].mean(axis=0)  # (n_layers, n_experts)
        im = ax.imshow(mat, aspect="auto", vmin=0, vmax=1.0/N_EXPERTS*3,
                       cmap="YlOrRd", origin="upper")
        ax.set_title(cat.replace("MMLU-",""), fontsize=8, fontweight="bold")
        ax.set_xlabel("Expert", fontsize=7)
        ax.tick_params(labelsize=6)
    axes[0].set_ylabel("Layer", fontsize=8)
    axes[0].set_yticks(range(N_LAYERS))

    fig.suptitle(f"{model_tag} — mean routing weight (layer × expert)", fontweight="bold")
    plt.colorbar(im, ax=axes[-1], shrink=0.8, label="mean p_i")
    plt.tight_layout()
    _save(fig, f"expert_routing_heatmap_{model_tag}")


def plot_dominant_expert_distribution(cat_routing, model_tag):
    """For each category, which experts are dominant (argmax) and their load."""
    cats = list(cat_routing.keys())
    n_cats = len(cats)
    fig, axes = plt.subplots(2, n_cats, figsize=(3*n_cats, 6), sharey="row")
    if n_cats == 1:
        axes = [[axes[0]], [axes[1]]]

    for col, cat in enumerate(cats):
        data = cat_routing[cat]  # (n_samples, n_layers, n_experts)
        # Average over layers → (n_samples, n_experts)
        mean_over_layers = data.mean(axis=1)

        # Top panel: load = fraction of samples where expert i is argmax
        argmax = mean_over_layers.argmax(axis=-1)  # (n_samples,)
        load = np.bincount(argmax, minlength=N_EXPERTS) / len(argmax)
        axes[0][col].bar(range(N_EXPERTS), load,
                         color=COLORS_CAT.get(cat, "steelblue"), alpha=0.8)
        axes[0][col].axhline(1/N_EXPERTS, color="gray", ls="--", lw=0.8)
        axes[0][col].set_title(cat.replace("MMLU-",""), fontsize=8)
        axes[0][col].set_ylim(0, max(load.max()*1.2, 2/N_EXPERTS))

        # Bottom panel: mean routing weight
        axes[1][col].bar(range(N_EXPERTS), mean_over_layers.mean(axis=0),
                         color=COLORS_CAT.get(cat, "steelblue"), alpha=0.8)
        axes[1][col].axhline(1/N_EXPERTS, color="gray", ls="--", lw=0.8)

    axes[0][0].set_ylabel("Argmax fraction", fontsize=8)
    axes[1][0].set_ylabel("Mean routing weight", fontsize=8)
    for col in range(n_cats):
        for row in range(2):
            axes[row][col].set_xlabel("Expert", fontsize=7)
            axes[row][col].tick_params(labelsize=6)

    fig.suptitle(f"{model_tag} — expert load by category", fontweight="bold")
    plt.tight_layout()
    _save(fig, f"expert_load_{model_tag}")


def plot_pca(cat_routing, model_tag):
    """PCA of per-sample routing vectors (n_experts × n_layers flattened)."""
    all_vecs, all_labels = [], []
    for cat, data in cat_routing.items():
        flat = data.reshape(len(data), -1)  # (n_samples, n_layers*n_experts)
        all_vecs.append(flat)
        all_labels.extend([cat] * len(data))

    X = np.vstack(all_vecs)
    X_norm = normalize(X, norm="l2")
    pca = PCA(n_components=2)
    coords = pca.fit_transform(X_norm)

    fig, ax = plt.subplots(figsize=(7, 5))
    seen = set()
    for i, label in enumerate(all_labels):
        color = COLORS_CAT.get(label, "black")
        kw = dict(label=label) if label not in seen else {}
        ax.scatter(coords[i, 0], coords[i, 1], c=color, alpha=0.5, s=15, **kw)
        seen.add(label)

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    ax.legend(fontsize=7, markerscale=1.5, loc="best")
    ax.set_title(f"{model_tag} — PCA of routing vectors by category", fontweight="bold")
    ax.grid(alpha=0.2)
    plt.tight_layout()
    _save(fig, f"expert_pca_{model_tag}")


def plot_correlation(cat_routing, model_tag):
    """Correlation between category mean-routing vectors."""
    cats = list(cat_routing.keys())
    # mean routing vector per category, averaged over samples and layers
    cat_vecs = np.stack([cat_routing[c].mean(axis=(0,1)) for c in cats])  # (n_cats, n_experts)

    # category × category cosine similarity
    from sklearn.metrics.pairwise import cosine_similarity
    sim = cosine_similarity(cat_vecs)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Left: category-category similarity
    im0 = axes[0].imshow(sim, vmin=0, vmax=1, cmap="RdYlGn")
    axes[0].set_xticks(range(len(cats))); axes[0].set_xticklabels(cats, rotation=45, ha="right", fontsize=7)
    axes[0].set_yticks(range(len(cats))); axes[0].set_yticklabels(cats, fontsize=7)
    axes[0].set_title("Category routing similarity")
    plt.colorbar(im0, ax=axes[0], shrink=0.8)
    for i in range(len(cats)):
        for j in range(len(cats)):
            axes[0].text(j, i, f"{sim[i,j]:.2f}", ha="center", va="center", fontsize=6)

    # Right: category × expert heatmap (which experts does each category prefer)
    im1 = axes[1].imshow(cat_vecs, aspect="auto", cmap="YlOrRd")
    axes[1].set_yticks(range(len(cats))); axes[1].set_yticklabels(cats, fontsize=7)
    axes[1].set_xlabel("Expert")
    axes[1].set_title("Mean routing weight per category × expert")
    plt.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f"{model_tag} — expert specialization by category", fontweight="bold")
    plt.tight_layout()
    _save(fig, f"expert_correlation_{model_tag}")


def _save(fig, name):
    for ext in (".pdf", ".png"):
        p = PLOTS / (name + ext)
        fig.savefig(p, bbox_inches="tight", dpi=150 if ext == ".png" else None)
    plt.close(fig)
    print(f"  saved {name}.pdf")


def print_summary(cat_routing, model_tag, out_file):
    lines = [f"\n{'='*60}", f"Expert Specialization Summary — {model_tag}", f"{'='*60}"]
    for cat, data in cat_routing.items():
        flat = data.mean(axis=1)  # (n_samples, n_experts)
        argmax = flat.argmax(axis=-1)
        n_active = len(np.unique(argmax))
        top_expert = int(np.bincount(argmax, minlength=N_EXPERTS).argmax())
        top_load = np.bincount(argmax, minlength=N_EXPERTS).max() / len(argmax)
        entropy = -np.sum(flat.mean(0) * np.log(flat.mean(0)+1e-8))
        lines.append(f"\n{cat} (n={len(data)})")
        lines.append(f"  Active experts (argmax): {n_active}/{N_EXPERTS}")
        lines.append(f"  Top expert: #{top_expert} ({top_load*100:.1f}% of samples)")
        lines.append(f"  Routing entropy: {entropy:.3f} / {np.log(N_EXPERTS):.3f}")

    text = "\n".join(lines)
    print(text)
    with open(out_file, "a") as f:
        f.write(text + "\n")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="b1", choices=list(MODELS.keys()),
                        help="Which unsharded model to analyse")
    parser.add_argument("--n_per_cat", type=int, default=60,
                        help="Max samples per category")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    model_path = MODELS[args.model]
    if not model_path.exists():
        print(f"Model not found: {model_path}")
        return

    print(f"Loading model from {model_path} ...")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16,
        device_map=args.device, trust_remote_code=True,
    )
    model.eval()
    print(f"  params: {sum(p.numel() for p in model.parameters())/1e6:.0f}M")

    print("Attaching routing hooks ...")
    attach_routing_hooks(model)

    print(f"Building categories (n_per_cat={args.n_per_cat}) ...")
    categories = build_categories(args.n_per_cat)
    for cat, texts in categories.items():
        print(f"  {cat}: {len(texts)} samples")

    print("Running inference and collecting routing ...")
    cat_routing = {}
    for cat, texts in categories.items():
        print(f"  {cat} ...", end=" ", flush=True)
        routing = collect_routing(model, tokenizer, texts, args.device, batch_size=4)
        cat_routing[cat] = routing
        print(f"shape={routing.shape}")

    tag = args.model
    summary_file = PLOTS / "expert_specialization_summary.txt"

    print("\nGenerating plots ...")
    plot_heatmap(cat_routing, tag)
    plot_dominant_expert_distribution(cat_routing, tag)
    plot_pca(cat_routing, tag)
    plot_correlation(cat_routing, tag)
    print_summary(cat_routing, tag, summary_file)

    print(f"\nDone. Plots in {PLOTS}")
    print(f"Summary appended to {summary_file}")


if __name__ == "__main__":
    main()
