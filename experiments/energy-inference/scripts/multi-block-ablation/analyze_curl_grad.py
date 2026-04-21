"""
Compute per-layer curl/grad ratio for proj_attn and proj_mlp weights.

For a projection matrix W (d×d), the Helmholtz decomposition is:
  W = S + A  where  S = (W + W^T)/2  (symmetric = gradient part)
                    A = (W - W^T)/2  (anti-symmetric = curl part)

Ratio = ||A||_F / ||S||_F  (how much curl vs gradient character the proj has)

Saves:
  results/multi-block-ablation/plots/curl_grad_ratio.{png,pdf}
"""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from safetensors import safe_open

RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)


MODELS = {
    "V1 lr=2e-3 (12×1, d=768, 30k steps)": "v1_12x1_d768_lr2e3",
    "V4 lr=2e-3 (shared, d=1152, 10k steps)": "v4_shared_wide_d1152_lr2e3",
}


def load_proj_weights(model_dir: str) -> dict:
    """Returns {layer_idx: {'proj_attn': tensor, 'proj_mlp': tensor}}"""
    unsharded = RESULTS_DIR / model_dir / "unsharded"
    sfiles = sorted(unsharded.glob("*.safetensors"))
    if not sfiles:
        raise FileNotFoundError(f"No safetensors in {unsharded}")

    weights = {}
    for sf in sfiles:
        with safe_open(sf, framework="pt") as st:
            for key in st.keys():
                if "proj_attn.weight" in key or "proj_mlp.weight" in key:
                    # parse layer index from e.g. transformer.h.3.proj_attn.weight
                    parts = key.split(".")
                    idx = int(parts[parts.index("h") + 1])
                    ptype = "proj_attn" if "proj_attn" in key else "proj_mlp"
                    if idx not in weights:
                        weights[idx] = {}
                    weights[idx][ptype] = st.get_tensor(key).float()
    return weights


def curl_grad_ratio(W: torch.Tensor) -> tuple[float, float, float]:
    """Returns (||S||_F, ||A||_F, ||A||_F / ||S||_F)"""
    S = (W + W.T) / 2
    A = (W - W.T) / 2
    s_norm = S.norm(p="fro").item()
    a_norm = A.norm(p="fro").item()
    ratio = a_norm / (s_norm + 1e-12)
    return s_norm, a_norm, ratio


def analyze(model_dir: str) -> dict:
    weights = load_proj_weights(model_dir)
    layers = sorted(weights.keys())
    results = {"layers": layers, "proj_attn": {}, "proj_mlp": {}}
    print(f"\n  Layer  ||S_attn||  ||A_attn||  ratio_attn  ||S_mlp||  ||A_mlp||  ratio_mlp")
    print(f"  {'─'*75}")
    for i in layers:
        row = {}
        for ptype in ["proj_attn", "proj_mlp"]:
            W = weights[i][ptype]
            s, a, r = curl_grad_ratio(W)
            row[ptype] = {"S_norm": s, "A_norm": a, "ratio": r}
        results["proj_attn"][i] = row["proj_attn"]
        results["proj_mlp"][i] = row["proj_mlp"]
        print(f"  {i:>5}  {row['proj_attn']['S_norm']:>9.4f}  {row['proj_attn']['A_norm']:>9.4f}  "
              f"{row['proj_attn']['ratio']:>10.4f}  {row['proj_mlp']['S_norm']:>8.4f}  "
              f"{row['proj_mlp']['A_norm']:>8.4f}  {row['proj_mlp']['ratio']:>9.4f}")
    return results


def plot_ratios(all_results: dict):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), sharey=False)
    colors = plt.cm.tab10.colors

    for ax, ptype, title in zip(axes, ["proj_attn", "proj_mlp"],
                                 ["proj_attn  (curl/grad ratio)", "proj_mlp  (curl/grad ratio)"]):
        for ci, (label, res) in enumerate(all_results.items()):
            layers = res["layers"]
            ratios = [res[ptype][i]["ratio"] for i in layers]
            ax.plot(layers, ratios, "o-", color=colors[ci], label=label, linewidth=1.8, markersize=5)

        ax.set_xlabel("Layer index", fontsize=10)
        ax.set_ylabel(r"$\|A\|_F \;/\; \|S\|_F$", fontsize=10)
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.6, label="ratio=1 (equal)")
        ax.legend(fontsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    plt.suptitle("Curl vs Gradient character of projection matrices", fontsize=12, fontweight="bold")
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"curl_grad_ratio.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"  Saved {out}")
    plt.close(fig)


def main():
    all_results = {}
    for label, model_dir in MODELS.items():
        print(f"\n{'='*60}\n{label}")
        try:
            all_results[label] = analyze(model_dir)
        except FileNotFoundError as e:
            print(f"  SKIP: {e}")

    if all_results:
        plot_ratios(all_results)

    # Print summary stats
    print("\n=== Summary: mean curl/grad ratio ===")
    for label, res in all_results.items():
        layers = res["layers"]
        attn_mean = np.mean([res["proj_attn"][i]["ratio"] for i in layers])
        mlp_mean  = np.mean([res["proj_mlp"][i]["ratio"]  for i in layers])
        print(f"  {label}")
        print(f"    proj_attn: {attn_mean:.4f}   proj_mlp: {mlp_mean:.4f}")

    return all_results


if __name__ == "__main__":
    main()
