"""LM-head alignment analysis for EGPT vs RecGPT (matched pair).

Computes:
  - Full SV decomposition of W_U (LM head)
  - align(write_op, top-k SVs) and align(write_op, bottom-k SVs)
    for k = 512, 256, 128, 64, 32
  - For EGPT: write_op = Π @ J where J = W_Q^T @ W_K (full write operator)
  - For RecGPT/GPT: write_op = W_O @ W_V (standard write operator)
  - Random baseline: sqrt(k/d) for each k

Usage:
    python 03_lk_alignment.py --models 410m_hybrid_s8e4,410m_recgpt_s8e4 \\
                               --base_dir /path/to/unsharded/models

Add your models to the MODELS dict below, then run.
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
import torch.nn.functional as F

# ── Setup ────────────────────────────────────────────────────────────────────
REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM

# ── Model registry ─────────────────────────────────────────────────────────
# Add your models here: model_id -> (path_to_unsharded, is_egpt)
# is_egpt=True: uses Π @ J as write operator
# is_egpt=False: uses W_O @ W_V as write operator
DEFAULT_BASE = REPO / "experiments/energy-inference/results/multi-block-ablation"
BSAHA = Path("/proj/dmfexp/energy-gpt/checkpoints-bsaha/egpt_400m")

MODELS: dict[str, tuple[Path, bool]] = {
    # Matched pair (same arch, only value tying differs)
    "410m_hybrid_s8e4":  (DEFAULT_BASE / "410m_hybrid_s8e4/unsharded",  True),
    "410m_recgpt_s8e4":  (DEFAULT_BASE / "410m_recgpt_s8e4/unsharded",  False),
    # Other baselines
    "v71_rmsray":        (DEFAULT_BASE / "v71_hybrid_8gpt_4egpt_rmsray_d1280/unsharded", True),
    "v9_gpt":            (DEFAULT_BASE / "v9_gpt_baseline_d1024_lr1e3/unsharded", False),
    "v76_final": (DEFAULT_BASE / "v76_4gpt_1egpt6x_rmsray_d1024_reg128/unsharded", True),
    "u4_rmsray":  (DEFAULT_BASE / "u4_2gpt_4egpt3x_rmsray_d1024/unsharded", True),
    # Add your 800M / 1B models here:
    # "my_egpt_800m":    (Path("/path/to/unsharded"), True),
    # "my_recgpt_800m":  (Path("/path/to/unsharded"), False),
}

K_VALUES = [512, 256, 128, 64, 32]


# ── Helpers ──────────────────────────────────────────────────────────────────

def get_lm_svd(model: AutoModelForCausalLM):
    """SVD of LM head W_U. Returns (S, Vh) on CPU."""
    for attr in ("lm_head", "embed_out", "output"):
        mo = getattr(model, attr, None)
        if mo is not None and hasattr(mo, "weight"):
            W = mo.weight.detach().float().cpu()
            _, S, Vh = torch.linalg.svd(W, full_matrices=False)
            return S.numpy(), Vh.numpy()
    # Fallback: tied embedding
    for attr in ("wte",):
        mo = getattr(model, attr, None) or \
             getattr(getattr(model, "transformer", None), attr, None)
        if mo is not None and hasattr(mo, "weight"):
            W = mo.weight.detach().float().cpu()
            _, S, Vh = torch.linalg.svd(W, full_matrices=False)
            return S.numpy(), Vh.numpy()
    raise RuntimeError("Cannot find LM head weight")


def get_blocks(model):
    for attr in ("transformer", "model"):
        t = getattr(model, attr, None)
        if t is not None and hasattr(t, "h"):
            return t.h
    raise AttributeError(f"No transformer blocks on {type(model)}")


def align_k(W_flat: np.ndarray, Vh: np.ndarray, k: int, top: bool = True) -> float:
    """||P_{L_k} W||_F / ||W||_F where L_k = top/bottom-k right SVs of W_U."""
    d = Vh.shape[1]
    Lk = torch.tensor(Vh[:k].T if top else Vh[-k:].T, dtype=torch.float32)
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    proj = Lk @ (Lk.T @ W)
    return (proj.norm() / W.norm()).item()


def get_write_op(blk, is_egpt: bool, layer_iter: int) -> np.ndarray | None:
    """Extract the full write operator for a recurrent block.

    EGPT: Π @ J = proj @ (W_Q^T @ W_K)
          proj may be single 'proj' (old models) or 'proj_attn' (dual models)
    GPT:  W_O @ W_V = c_proj @ c_attn[2d:]
    """
    if layer_iter <= 1:
        return None

    attn = getattr(blk, "attn", None) or getattr(blk, "sequence_mixer", None)
    if attn is None or not hasattr(getattr(attn, "c_attn", None), "weight"):
        return None

    w = attn.c_attn.weight.detach().float().cpu()
    d = w.shape[1]

    if is_egpt and w.shape[0] >= 2 * d:
        # EGPT: J = W_Q^T @ W_K, then Π @ J
        J = w[:d].T @ w[d:2*d]  # [d, d]
        # Find projection matrix Π (try several attribute names)
        Pi = None
        for attr in ("proj", "proj_attn"):
            p = getattr(blk, attr, None)
            if p is not None and hasattr(p, "weight"):
                Pi = p.weight.detach().float().cpu()
                break
        if Pi is not None:
            return (Pi @ J).reshape(-1).numpy()
        else:
            return J.reshape(-1).numpy()  # fallback: J alone

    elif not is_egpt and w.shape[0] >= 3 * d:
        # Standard attention: W_O @ W_V
        c_proj = getattr(attn, "c_proj", None)
        if c_proj is not None and hasattr(c_proj, "weight"):
            WV = w[2*d:]
            WO = c_proj.weight.detach().float().cpu()
            return (WO @ WV).reshape(-1).numpy()

    return None


# ── Main ─────────────────────────────────────────────────────────────────────

def analyze(model_id: str, ckpt: Path, is_egpt: bool):
    print(f"\n{'='*60}\n{model_id}  (is_egpt={is_egpt})\n{'='*60}")
    if not ckpt.exists():
        print(f"  SKIP: {ckpt} not found")
        return None

    m = AutoModelForCausalLM.from_pretrained(
        str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cpu().eval()

    S, Vh = get_lm_svd(m)
    d = Vh.shape[1]
    total = (S**2).sum()
    cum = np.cumsum(S**2) / total
    k50  = int(np.searchsorted(cum, 0.50)) + 1
    k95  = int(np.searchsorted(cum, 0.95)) + 1
    k99  = int(np.searchsorted(cum, 0.99)) + 1
    print(f"  d={d}  SV range=[{S[0]:.1f}, {S[-1]:.4f}]  "
          f"decay={S[0]/S[-1]:.0f}x  k50={k50}  k95={k95}  k99={k99}")

    blocks = get_blocks(m)
    iters  = getattr(m.config, "layer_iterations", [1] * len(blocks))

    block_results = []
    for i, blk in enumerate(blocks):
        op = get_write_op(blk, is_egpt, iters[i])
        if op is None:
            continue
        lbl = f"B{i}(x{iters[i]})"
        row = {"label": lbl, "layer": i, "n_iter": iters[i]}
        print(f"\n  {lbl}:")
        for k in K_VALUES:
            rand = np.sqrt(k / d)
            t = align_k(op, Vh, k, top=True)
            b = align_k(op, Vh, k, top=False)
            row[f"top{k}"] = round(t, 4)
            row[f"bot{k}"] = round(b, 4)
            row[f"top{k}_excess"] = round(t - rand, 4)
            row[f"bot{k}_excess"] = round(b - rand, 4)
            print(f"    k={k:4d}:  top={t:.3f} (rand={rand:.3f}, {t-rand:+.3f})  "
                  f"bot={b:.3f} (rand={rand:.3f}, {b-rand:+.3f})")
        block_results.append(row)

    del m

    # Summary means
    means = {}
    for k in K_VALUES:
        rand = np.sqrt(k / d)
        means[f"top{k}_mean"]        = float(np.mean([r[f"top{k}"] for r in block_results]))
        means[f"bot{k}_mean"]        = float(np.mean([r[f"bot{k}"] for r in block_results]))
        means[f"top{k}_excess_mean"] = round(means[f"top{k}_mean"] - rand, 4)
        means[f"bot{k}_excess_mean"] = round(means[f"bot{k}_mean"] - rand, 4)

    print(f"\n  MEANS:")
    for k in K_VALUES:
        print(f"    k={k:4d}:  top_excess={means[f'top{k}_excess_mean']:+.3f}  "
              f"bot_excess={means[f'bot{k}_excess_mean']:+.3f}")

    return {
        "model_id": model_id,
        "is_egpt": is_egpt,
        "d": d,
        "sv_range": [float(S[0]), float(S[-1])],
        "decay_ratio": float(S[0] / S[-1]),
        "k50": k50, "k95": k95, "k99": k99,
        "blocks": block_results,
        "means": means,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS.keys()),
                        help="Comma-separated model IDs to analyze (default: all)")
    parser.add_argument("--output", default="lk_alignment_results.json")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]
    results = {}
    for mid in model_ids:
        if mid not in MODELS:
            print(f"Unknown model: {mid}. Available: {list(MODELS.keys())}")
            continue
        ckpt, is_egpt = MODELS[mid]
        r = analyze(mid, ckpt, is_egpt)
        if r is not None:
            results[mid] = r

    # Print comparison summary
    print("\n" + "="*70)
    print("COMPARISON SUMMARY (mean excess over random baseline sqrt(k/d))")
    print("="*70)
    header = f"{'Model':30s}" + "".join(f" top{k:4d}" for k in K_VALUES) + " |" + \
             "".join(f" bot{k:4d}" for k in K_VALUES)
    print(header)
    print("-" * len(header))
    for mid, r in results.items():
        row = f"{mid:30s}" + \
              "".join(f" {r['means'][f'top{k}_excess_mean']:+.3f}" for k in K_VALUES) + " |" + \
              "".join(f" {r['means'][f'bot{k}_excess_mean']:+.3f}" for k in K_VALUES)
        print(row)

    out = Path(args.output)
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
