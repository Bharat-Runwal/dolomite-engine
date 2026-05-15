"""
BoltzmannMoE Deep Expert Specialization Analysis
=================================================
Enhanced version of the routing analysis:
  - More samples per category (default 200)
  - Mean-centred routing vectors to reveal residual specialization
  - PC1/2, PC1/3, PC2/3 scatter + KDE contour plots
  - Category centroids ± std ellipses on PCA
  - Pairwise Mahalanobis separation score matrix
  - Individual mean ± std routing profiles per category
  - Saves raw routing arrays to disk for fast re-analysis

Usage:
  python analyze_boltz_expert_deep_20260429.py --model b1 [--n_per_cat 200]
  python analyze_boltz_expert_deep_20260429.py --model b1 --reuse  # skip inference
"""

import argparse, sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import scipy.stats as stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize
from sklearn.covariance import LedoitWolf

sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
import lm_engine.hf_models  # noqa
import torch
import re, types
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModelForCausalLM

REPO    = Path(__file__).resolve().parents[4]
RESULTS = REPO / "experiments/boltzmann-moe/results"
PLOTS   = RESULTS / "plots"
CACHE   = RESULTS / "plots/routing_cache"
PLOTS.mkdir(exist_ok=True)
CACHE.mkdir(exist_ok=True)

MODELS = {
    "b1": RESULTS / "b1_boltz_moe_16x1024_d768_lr2e3/unsharded",
    "b2": RESULTS / "b2_boltz_moe_repulsion_16x1024_d768_lr2e3/unsharded",
    "b3": RESULTS / "b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3/unsharded",
    "b4": RESULTS / "b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3/unsharded",
    "b5": RESULTS / "b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3/unsharded",
}

N_LAYERS  = 12
N_EXPERTS = 16
MAX_LEN   = 256

COLORS = {
    "MMLU-STEM":       "#c0392b",   # dark red
    "MMLU-Humanities": "#d35400",   # burnt orange (distinct from red)
    "MMLU-Social":     "#7f6000",   # olive/dark yellow (distinct from orange)
    "MMLU-Medical":    "#1a7a4a",   # dark green
    "MMLU-Logic":      "#5b2c6f",   # deep purple (distinct from COPA)
    "BoolQ":           "#1a5276",   # dark blue
    "COPA":            "#e91e63",   # magenta/hot pink (most distinct)
    "GSM8k":           "#b7950b",   # amber/gold
}
MARKERS = {k: s for k, s in zip(COLORS.keys(), ["o","s","D","^","v","P","*","X"])}

# ── Datasets ─────────────────────────────────────────────────────────────────

MMLU_GROUPS = {
    "MMLU-STEM":       ["abstract_algebra","college_mathematics","high_school_mathematics",
                        "high_school_physics","high_school_chemistry","college_physics",
                        "computer_security","high_school_computer_science","astronomy"],
    "MMLU-Humanities": ["high_school_history","world_history","philosophy",
                        "international_law","jurisprudence","moral_scenarios"],
    "MMLU-Social":     ["high_school_psychology","sociology","economics","political_science",
                        "high_school_government_and_politics","public_relations"],
    "MMLU-Medical":    ["clinical_knowledge","medical_genetics","anatomy",
                        "college_medicine","professional_medicine","virology"],
    "MMLU-Logic":      ["formal_logic","logical_fallacies"],
}

def load_mmlu(subjects, n_total):
    from datasets import load_dataset
    texts, per = [], max(4, n_total // len(subjects))
    for subj in subjects:
        try:
            ds = load_dataset("cais/mmlu", subj, split="test", trust_remote_code=True)
        except Exception:
            continue
        for ex in list(ds)[:per]:
            choices = "\n".join(f"{c}. {ex['choices'][i]}" for i,c in enumerate("ABCD"))
            texts.append(f"Question: {ex['question']}\n{choices}\nAnswer:")
    return texts[:n_total]

def load_boolq(n):
    from datasets import load_dataset
    ds = load_dataset("boolq", split="validation", trust_remote_code=True)
    return [f"Passage: {ex['passage'][:300]}\nQuestion: {ex['question']}\nTrue or False:"
            for ex in list(ds)[:n]]

def load_copa(n):
    from datasets import load_dataset
    ds = load_dataset("super_glue","copa",split="validation",trust_remote_code=True)
    texts = []
    for ex in list(ds)[:n]:
        texts.append(f"Premise: {ex['premise'].rstrip('.')}. "
                     f"A: {ex['choice1']}  B: {ex['choice2']}  "
                     f"{'Because' if ex['question']=='cause' else 'So'}:")
    return texts

def load_gsm8k(n):
    from datasets import load_dataset
    ds = load_dataset("gsm8k","main",split="test",trust_remote_code=True)
    return [f"Problem: {ex['question']}\nSolution:" for ex in list(ds)[:n]]

def build_categories(n):
    cats = {}
    for grp, subjects in MMLU_GROUPS.items():
        t = load_mmlu(subjects, n)
        if t: cats[grp] = t
    cats["BoolQ"]  = load_boolq(n)
    cats["COPA"]   = load_copa(n)
    cats["GSM8k"]  = load_gsm8k(n)
    return cats

# ── Routing capture ───────────────────────────────────────────────────────────

def attach_hooks(model):
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP
    hooked = []
    for name, m in model.named_modules():
        if not isinstance(m, BoltzmannMoE_Energy_MLP): continue
        m._cap_routing = None
        orig = m.forward
        def make_patch(mod, o):
            def patched(x):
                leading = x.shape[:-1]
                with torch.no_grad():
                    W1x = mod.W1(x).view(*leading, mod.n_experts, mod.expert_I)
                    W2_e = mod.W2.weight.view(mod.n_experts, mod.expert_I, mod.hidden_size)
                    phi = F.gelu(W1x)
                    term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
                    E = torch.einsum("...h,...eh->...e", x, term1)
                    p = F.softmax(E / mod.temperature, dim=-1)
                mod._cap_routing = p.detach().float().cpu()
                return o(x)
            return patched
        m.forward = make_patch(m, orig)
        hooked.append((name, m))
    return hooked

def collect_routing(model, tokenizer, texts, device, batch_size=4):
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP
    moe_mods = sorted(
        [(n, m) for n, m in model.named_modules() if isinstance(m, BoltzmannMoE_Energy_MLP)],
        key=lambda x: int(re.search(r'\.(\d+)\.', x[0]).group(1)) if re.search(r'\.(\d+)\.', x[0]) else 0
    )
    all_routing = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=True, max_length=MAX_LEN).to(device)
        with torch.no_grad():
            model(**enc)
        for b in range(len(batch)):
            sample = []
            for _, m in moe_mods:
                if m._cap_routing is None:
                    sample.append(np.full(N_EXPERTS, 1./N_EXPERTS))
                    continue
                p = m._cap_routing
                p_b = p[b] if p.dim()==3 else p
                if enc.attention_mask is not None and p.dim()==3:
                    mask = enc.attention_mask[b].cpu().numpy().astype(bool)
                    p_b = p_b.numpy()[mask]
                else:
                    p_b = p_b.numpy()
                sample.append(p_b.mean(axis=0))
            all_routing.append(np.stack(sample))
    return np.array(all_routing)  # (N, n_layers, n_experts)

# ── PCA helpers ───────────────────────────────────────────────────────────────

def build_pca_features(cat_routing, mean_centre=True):
    """Flatten routing vectors, optionally mean-centre, run PCA, return coords + pca."""
    cats = list(cat_routing.keys())
    all_vecs, labels = [], []
    for cat in cats:
        flat = cat_routing[cat].reshape(len(cat_routing[cat]), -1)
        all_vecs.append(flat); labels.extend([cat]*len(flat))
    X = np.vstack(all_vecs)
    if mean_centre:
        X = X - X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(8, X.shape[1], X.shape[0]-1))
    coords = pca.fit_transform(X)
    return coords, labels, pca, cats

def confidence_ellipse(ax, x, y, color, n_std=1.5, **kwargs):
    """Draw a covariance ellipse for points (x, y)."""
    if len(x) < 3: return
    cov = np.cov(x, y)
    vals, vecs = np.linalg.eigh(cov)
    order = vals.argsort()[::-1]
    vals, vecs = vals[order], vecs[:, order]
    angle = np.degrees(np.arctan2(*vecs[:, 0][::-1]))
    w, h = 2 * n_std * np.sqrt(vals)
    ell = Ellipse(xy=(np.mean(x), np.mean(y)), width=w, height=h, angle=angle,
                  facecolor="none", edgecolor=color, linewidth=2, **kwargs)
    ax.add_patch(ell)

def plot_pca_panel(ax, coords, labels, cats, pc_x, pc_y, pca,
                  mean_centre, show_kde=True, show_ellipse=True):
    """Single PCA scatter panel with optional KDE contours and ellipses."""
    from scipy.stats import gaussian_kde
    cx, cy = coords[:, pc_x], coords[:, pc_y]

    ax.set_facecolor("white")

    # KDE contours — outlines only, no fill (fill causes color mixing and grey background)
    if show_kde:
        for cat in cats:
            idx = [i for i,l in enumerate(labels) if l==cat]
            if len(idx) < 10: continue
            x_, y_ = cx[idx], cy[idx]
            color = COLORS.get(cat, "gray")
            try:
                kde = gaussian_kde(np.vstack([x_, y_]))
                xi = np.linspace(cx.min(), cx.max(), 80)
                yi = np.linspace(cy.min(), cy.max(), 80)
                Xi, Yi = np.meshgrid(xi, yi)
                Zi = kde(np.vstack([Xi.ravel(), Yi.ravel()])).reshape(Xi.shape)
                # Contour lines only — no fill, no color mixing
                ax.contour(Xi, Yi, Zi, levels=3, colors=[color], alpha=0.7, linewidths=0.9)
            except Exception:
                pass

    # Scatter — with explicit marker per category so legend is informative
    seen = set()
    for i, lab in enumerate(labels):
        color = COLORS.get(lab, "gray")
        marker = MARKERS.get(lab, "o")
        kw = dict(label=lab) if lab not in seen else {}
        ax.scatter(cx[i], cy[i], c=color, marker=marker, alpha=0.5, s=20, **kw)
        seen.add(lab)

    # Centroids + ellipses
    for cat in cats:
        idx = [i for i,l in enumerate(labels) if l==cat]
        if not idx: continue
        color = COLORS.get(cat, "gray")
        mx, my = np.mean(cx[idx]), np.mean(cy[idx])
        ax.scatter(mx, my, c=color, marker=MARKERS.get(cat,"o"),
                   s=120, edgecolors="black", linewidths=1.2, zorder=5)
        if show_ellipse and len(idx) >= 3:
            confidence_ellipse(ax, cx[idx], cy[idx], color, n_std=1.0, linestyle="--")

    ax.set_xlabel(f"PC{pc_x+1} ({pca.explained_variance_ratio_[pc_x]*100:.1f}%)", fontsize=8)
    ax.set_ylabel(f"PC{pc_y+1} ({pca.explained_variance_ratio_[pc_y]*100:.1f}%)", fontsize=8)
    tag = "mean-centred" if mean_centre else "raw"
    ax.set_title(f"PC{pc_x+1} vs PC{pc_y+1} ({tag})", fontsize=9)
    ax.grid(alpha=0.2)
    ax.tick_params(labelsize=6)

# ── Separation analysis ───────────────────────────────────────────────────────

def mahalanobis_separation(cat_routing, n_pcs=6):
    """
    Pairwise separation between categories in PCA space.

    Returns dict with:
      'mahal'  : (n_cats, n_cats) Mahalanobis distance between centroids
                 using pooled Ledoit-Wolf covariance (robust, handles n<p)
      'cohen_d': (n_cats, n_cats) simplified (|mu_i - mu_j|_2) / (sigma_i + sigma_j)
                 where sigma = mean of within-class std across PCs
    """
    cats = list(cat_routing.keys())
    # Build mean-centred PCA features
    all_vecs, labels = [], []
    for cat in cats:
        flat = cat_routing[cat].reshape(len(cat_routing[cat]), -1)
        all_vecs.append(flat); labels.extend([cat]*len(flat))
    X = np.vstack(all_vecs)
    X -= X.mean(axis=0, keepdims=True)
    pca = PCA(n_components=min(n_pcs, X.shape[1], X.shape[0]-1))
    coords = pca.fit_transform(X)

    n = len(cats)
    mahal  = np.zeros((n, n))
    cohend = np.zeros((n, n))

    per_cat = {}
    for i, cat in enumerate(cats):
        idx = [j for j,l in enumerate(labels) if l==cat]
        per_cat[cat] = coords[idx]

    # Pooled covariance (Ledoit-Wolf for stability)
    all_centred = []
    for cat in cats:
        X_c = per_cat[cat] - per_cat[cat].mean(0)
        all_centred.append(X_c)
    X_pooled = np.vstack(all_centred)
    lw = LedoitWolf().fit(X_pooled)
    Sigma_inv = np.linalg.inv(lw.covariance_ + 1e-6*np.eye(lw.covariance_.shape[0]))

    for i, ci in enumerate(cats):
        for j, cj in enumerate(cats):
            if i == j: continue
            mu_i = per_cat[ci].mean(0)
            mu_j = per_cat[cj].mean(0)
            diff = mu_i - mu_j

            # Mahalanobis
            mahal[i, j] = np.sqrt(max(0, diff @ Sigma_inv @ diff))

            # Simplified Cohen's d: |mu_i - mu_j|_2 / (sigma_i + sigma_j)
            sig_i = per_cat[ci].std(0).mean()
            sig_j = per_cat[cj].std(0).mean()
            cohend[i, j] = np.linalg.norm(diff) / (sig_i + sig_j + 1e-8)

    return {"mahal": mahal, "cohen_d": cohend, "cats": cats, "pca": pca,
            "per_cat_coords": per_cat}

# ── Plots ────────────────────────────────────────────────────────────────────

def _save(fig, name):
    for ext in (".pdf", ".png"):
        fig.savefig(PLOTS / (name+ext), bbox_inches="tight",
                    dpi=150 if ext==".png" else None)
    plt.close(fig)
    print(f"  saved {name}.pdf")


def plot_pca_deep(cat_routing, model_tag):
    """3×2 grid: raw and mean-centred, each with PC1/2, PC1/3, PC2/3."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    for row, centre in enumerate([False, True]):
        coords, labels, pca, cats = build_pca_features(cat_routing, mean_centre=centre)
        for col, (px, py) in enumerate([(0,1),(0,2),(1,2)]):
            if pca.n_components_ <= max(px, py):
                axes[row, col].set_visible(False)
                continue
            plot_pca_panel(axes[row, col], coords, labels, cats, px, py, pca,
                           centre, show_kde=True, show_ellipse=True)

    # Shared legend with correct marker shapes
    from matplotlib.lines import Line2D
    handles = [Line2D([0],[0], marker=MARKERS.get(c,"o"), color=COLORS.get(c,"gray"),
                      markersize=8, linestyle="None", label=c)
               for c in cat_routing.keys()]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{model_tag} — PCA of routing vectors (top=raw, bottom=mean-centred)",
                 fontweight="bold", fontsize=12)
    plt.tight_layout(rect=[0,0.04,1,0.97])
    _save(fig, f"expert_pca_deep_{model_tag}")


def plot_routing_profiles(cat_routing, model_tag):
    """Mean ± std routing weight per expert, averaged over layers, per category."""
    cats = list(cat_routing.keys())
    n = len(cats)
    fig, axes = plt.subplots(2, (n+1)//2, figsize=(4*((n+1)//2), 6), sharey=False)
    axes = axes.flatten()
    global_mean = np.vstack([cat_routing[c].mean(axis=(0,1)) for c in cats]).mean(0)

    for i, cat in enumerate(cats):
        data = cat_routing[cat]  # (N, n_layers, n_experts)
        flat = data.mean(axis=1)  # (N, n_experts)
        mu = flat.mean(0)
        se = flat.std(0) / np.sqrt(len(flat))
        residual = mu - global_mean

        ax = axes[i]
        color = COLORS.get(cat, "steelblue")
        x = np.arange(N_EXPERTS)
        ax.bar(x, mu, color=color, alpha=0.6, label="mean")
        ax.errorbar(x, mu, yerr=se*1.96, fmt="none", color="black", capsize=2, lw=0.8)
        ax.step(x, residual + 1./N_EXPERTS, where="mid", color="black",
                lw=1.2, ls="--", label="residual+baseline")
        ax.axhline(1./N_EXPERTS, color="gray", ls=":", lw=0.8)
        ax.set_title(cat.replace("MMLU-",""), fontsize=9, fontweight="bold")
        ax.set_xlabel("Expert", fontsize=7); ax.tick_params(labelsize=6)
        ax.set_xlim(-0.5, N_EXPERTS-0.5)

    for j in range(len(cats), len(axes)): axes[j].set_visible(False)
    axes[0].legend(fontsize=6)
    fig.suptitle(f"{model_tag} — mean routing weight per expert by category\n"
                 f"(dashed = residual after subtracting global mean)", fontsize=10)
    plt.tight_layout()
    _save(fig, f"expert_profiles_{model_tag}")


def plot_separation(sep_result, model_tag):
    """Pairwise Mahalanobis distance and Cohen's d heatmaps."""
    cats = sep_result["cats"]
    n = len(cats)
    short = [c.replace("MMLU-","") for c in cats]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for ax, key, title in zip(axes,
            ["mahal", "cohen_d"],
            ["Mahalanobis distance (pooled cov)", "Cohen-style d  |μᵢ−μⱼ|/(σᵢ+σⱼ)"]):
        mat = sep_result[key]
        im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
        ax.set_xticks(range(n)); ax.set_xticklabels(short, rotation=45, ha="right", fontsize=7)
        ax.set_yticks(range(n)); ax.set_yticklabels(short, fontsize=7)
        ax.set_title(title, fontsize=9)
        plt.colorbar(im, ax=ax, shrink=0.8)
        for i in range(n):
            for j in range(n):
                v = mat[i,j]
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6,
                        color="white" if v > mat.max()*0.65 else "black")

    fig.suptitle(f"{model_tag} — routing vector separation between categories\n"
                 f"(higher = more distinct routing, computed in {sep_result['pca'].n_components_}-D PCA space)",
                 fontweight="bold")
    plt.tight_layout()
    _save(fig, f"expert_separation_{model_tag}")


def print_summary(cat_routing, model_tag, sep_result):
    cats = sep_result["cats"]
    print(f"\n{'='*60}")
    print(f"Deep Routing Analysis — {model_tag}")
    print(f"{'='*60}")
    # Top separations
    mahal = sep_result["mahal"]
    pairs = [(mahal[i,j], cats[i], cats[j])
             for i in range(len(cats)) for j in range(i+1, len(cats))]
    pairs.sort(reverse=True)
    print("\nTop-5 most separated category pairs (Mahalanobis):")
    for d, ci, cj in pairs[:5]:
        print(f"  {ci} vs {cj}: {d:.3f}")
    print("\nTop-5 least separated:")
    for d, ci, cj in pairs[-5:]:
        print(f"  {ci} vs {cj}: {d:.3f}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="b1", choices=list(MODELS.keys()))
    parser.add_argument("--n_per_cat", type=int, default=200)
    parser.add_argument("--reuse", action="store_true",
                        help="Load cached routing arrays instead of re-running inference")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    cache_file = CACHE / f"routing_{args.model}.npz"

    if args.reuse and cache_file.exists():
        print(f"Loading cached routing from {cache_file}")
        data = np.load(cache_file, allow_pickle=True)
        cat_routing = {str(k): data[k] for k in data.files}
    else:
        model_path = MODELS[args.model]
        if not model_path.exists():
            print(f"Model not found: {model_path}"); return

        print(f"Loading model {args.model} ...")
        tokenizer = AutoTokenizer.from_pretrained(str(model_path))
        if tokenizer.pad_token is None: tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16,
            device_map=args.device, trust_remote_code=True)
        model.eval()
        attach_hooks(model)

        print(f"Building {args.n_per_cat} samples per category ...")
        categories = build_categories(args.n_per_cat)
        for cat, texts in categories.items():
            print(f"  {cat}: {len(texts)}")

        cat_routing = {}
        for cat, texts in categories.items():
            print(f"  Collecting routing for {cat} ...", end=" ", flush=True)
            routing = collect_routing(model, tokenizer, texts, args.device, batch_size=4)
            cat_routing[cat] = routing
            print(f"shape={routing.shape}")

        # Save routing arrays as npz (fast numpy) and comprehensive pkl (with texts+meta)
        np.savez(cache_file, **cat_routing)

        pkl_path = CACHE / f"routing_{args.model}.pkl"
        import pickle
        save = {
            "model": args.model, "date": "2026-04-29",
            "n_per_cat": args.n_per_cat, "n_layers": N_LAYERS, "n_experts": N_EXPERTS,
            "description": (
                "routing[cat]['routing']: (N, n_layers, n_experts) float32 — "
                "Boltzmann weights averaged over sequence tokens per sample. "
                "routing[cat]['texts']: list of N input strings."
            ),
            "categories": {
                cat: {"texts": categories[cat][:len(routing)], "routing": routing}
                for cat, routing in cat_routing.items()
            },
        }
        with open(pkl_path, "wb") as f:
            pickle.dump(save, f, protocol=4)
        print(f"Cached to {cache_file} and {pkl_path.name}")

    print("\nGenerating plots ...")
    tag = args.model
    plot_pca_deep(cat_routing, tag)
    plot_routing_profiles(cat_routing, tag)
    sep = mahalanobis_separation(cat_routing, n_pcs=min(8, 40))
    plot_separation(sep, tag)
    print_summary(cat_routing, tag, sep)
    print(f"\nDone. Plots in {PLOTS}")


if __name__ == "__main__":
    main()
