"""A6: rho^{Pi_a J} — projected attention kernel cross-block cosine similarity.

PLANS.md A6 (Tier 1.5, highest <30min priority):
  The existing cross-block table reports rho^J = cos(W_Q^T W_K) across EGPT blocks.
  But in EGPT, attention writes via Pi_a @ J (not J alone).
  This script computes rho^{Pi_a @ J} for EGPT models (dual_unconstrained and
  unconstrained) and compares to the existing rho^J.

  For each EGPT block b:
    J_b       = W_Q^T @ W_K                        (d × d, gauge-fixed combined)
    Pi_b      = proj_attn.weight  (dual) or
                proj.weight       (unconstrained)   (d × d)
    W_b       = Pi_b @ J_b                          (d × d, full write operator)

    rho^{Pi@J}_{ij} = cos(vec(W_i), vec(W_j))

  Expected outcomes (PLANS.md predictions):
    If rho^{Pi@J} >> rho^J  → cross-block sharing was always in the full write op;
                              J alone understates it.
    If rho^{Pi@J} ~ rho^J   → Pi_a does not help align; blocks share J but differ in Pi.
    If rho^{Pi@J} < rho^J   → Pi_a actively differentiates blocks.

Usage:
    python analyze_a6_proj_cosim_20260507.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

import sys
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM

MODELS: dict[str, Path] = {
    "v71_dual":            BASE / "v71_hybrid_8gpt_4egpt_rmsray_d1280" / "unsharded",
    "410m_hybrid_single":  BASE / "410m_hybrid_s8e4" / "unsharded",
    "410m_recgpt":         BASE / "410m_recgpt_s8e4" / "unsharded",
    "v9_gpt":              BASE / "v9_gpt_baseline_d1024_lr1e3" / "unsharded",
    "u1_rmsray":           BASE / "u1_2gpt_4egpt3x_rmsray_d1280" / "unsharded",
    "u4_rmsray":           BASE / "u4_2gpt_4egpt3x_rmsray_d1024" / "unsharded",
}


def get_blocks(model):
    for attr in ("transformer", "model"):
        t = getattr(model, attr, None)
        if t is not None and hasattr(t, "h"):
            return t.h
        if t is not None and hasattr(t, "transformer"):
            return t.transformer.h
    raise AttributeError(f"No transformer blocks on {type(model)}")


def get_egpt_indices(model):
    cfg = model.config
    blocks = get_blocks(model)
    indices = []
    if hasattr(cfg, "layer_iterations"):
        for i, it in enumerate(cfg.layer_iterations):
            if it > 1:
                indices.append(i)
    if not indices:
        for i, blk in enumerate(blocks):
            st = getattr(blk, "sequence_mixer_type", "") or ""
            if "energy" in st:
                indices.append(i)
    return indices


def get_write_op(blk, d: int) -> tuple[np.ndarray | None, np.ndarray | None, str]:
    """Return (J_flat, PiJ_flat, proj_type) for block blk.

    J     = W_Q^T @ W_K  (attention kernel alone)
    PiJ   = Pi @ J        (projected write operator)
    For GPT/RecGPT: WOV = W_O @ W_V instead
    """
    # ── EnergyBlock (EGPT) ──────────────────────────────────────────────────
    attn = getattr(blk, "attn", None)
    if attn is not None and hasattr(getattr(attn, "c_attn", None), "weight"):
        w = attn.c_attn.weight.detach().float().cpu()  # [2d or 3d, d]
        if w.shape[0] < 2 * d:
            return None, None, "unknown"

        J = w[:d].T @ w[d:2*d]  # [d, d]

        Pi = None
        proj_type = "unknown"
        for attr_name in ("proj_attn", "proj"):
            p = getattr(blk, attr_name, None)
            if p is not None and hasattr(p, "weight"):
                Pi = p.weight.detach().float().cpu()  # [d, d]
                proj_type = attr_name
                break

        J_flat = J.reshape(-1).numpy()
        PiJ_flat = (Pi @ J).reshape(-1).numpy() if Pi is not None else None
        return J_flat, PiJ_flat, proj_type

    # ── GPT / RecGPT block ────────────────────────────────────────────────
    seq = getattr(blk, "sequence_mixer", None)
    if seq is not None and hasattr(getattr(seq, "c_attn", None), "weight"):
        w = seq.c_attn.weight.detach().float().cpu()  # [3d, d]
        if w.shape[0] < 3 * d:
            return None, None, "gpt_no_v"
        c_proj = getattr(seq, "c_proj", None)
        if c_proj is not None and hasattr(c_proj, "weight"):
            WO = c_proj.weight.detach().float().cpu()  # [d, d]
            WV = w[2*d:]                               # [d, d]
            WOV = (WO @ WV).reshape(-1).numpy()
            J_flat = (w[:d].T @ w[d:2*d]).reshape(-1).numpy()
            return J_flat, WOV, "WOV"

    return None, None, "unknown"


def flat_cosim(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))


def analyze(model_id: str, path: Path):
    print(f"\n{'='*60}\n{model_id}\n{'='*60}")
    if not path.exists():
        print(f"  SKIP: {path} not found")
        return None

    m = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).cpu().eval()

    cfg = m.config
    d = cfg.hidden_size
    blocks = get_blocks(m)
    egpt_idxs = get_egpt_indices(m)
    iters = getattr(cfg, "layer_iterations", [1] * len(blocks))
    proj_type_str = getattr(cfg, "energy_proj_type", getattr(cfg, "proj_mode", "N/A"))

    print(f"  d={d}  layers={len(blocks)}  egpt_blocks={egpt_idxs}  proj={proj_type_str}")

    # Collect write operators
    J_vecs, PiJ_vecs, labels = [], [], []
    for i in egpt_idxs:
        blk = blocks[i]
        J_flat, PiJ_flat, pt = get_write_op(blk, d)
        if J_flat is None:
            print(f"  B{i}: skipped (no write op)")
            continue
        J_vecs.append(J_flat)
        PiJ_vecs.append(PiJ_flat)
        labels.append(f"B{i}(×{iters[i]})")
        print(f"  B{i}(×{iters[i]}): proj_type={pt}  J={J_flat.shape}  PiJ={PiJ_flat.shape if PiJ_flat is not None else 'None'}")

    n = len(J_vecs)
    if n < 2:
        print("  Not enough blocks for cross-block cosim")
        return None

    # Cross-block cosim matrices
    def cosim_matrix(vecs):
        mat = np.zeros((len(vecs), len(vecs)))
        for i in range(len(vecs)):
            for j in range(len(vecs)):
                if vecs[i] is None or vecs[j] is None:
                    mat[i, j] = float('nan')
                else:
                    mat[i, j] = flat_cosim(vecs[i], vecs[j])
        return mat

    J_mat = cosim_matrix(J_vecs)
    PiJ_mat = cosim_matrix(PiJ_vecs) if any(v is not None for v in PiJ_vecs) else None

    # Off-diagonal means
    mask = ~np.eye(n, dtype=bool)
    rho_J   = float(J_mat[mask].mean())
    rho_PiJ = float(PiJ_mat[mask].mean()) if PiJ_mat is not None else float('nan')

    print(f"\n  rho^J     = {rho_J:.4f}  (kernel alone)")
    print(f"  rho^Pi@J  = {rho_PiJ:.4f}  (projected write operator)")

    # Print matrices
    print("\n  rho^J matrix:")
    for i, row in enumerate(J_mat):
        print(f"    {labels[i]:12s}: " + " ".join(f"{v:+.3f}" for v in row))

    if PiJ_mat is not None:
        print("\n  rho^{Pi@J} matrix:")
        for i, row in enumerate(PiJ_mat):
            print(f"    {labels[i]:12s}: " + " ".join(f"{v:+.3f}" for v in row))

    del m

    return {
        "model_id": model_id,
        "d": d,
        "egpt_blocks": egpt_idxs,
        "proj_type": proj_type_str,
        "labels": labels,
        "rho_J": rho_J,
        "rho_PiJ": rho_PiJ,
        "J_cosim_matrix": J_mat.tolist(),
        "PiJ_cosim_matrix": PiJ_mat.tolist() if PiJ_mat is not None else None,
    }


def main():
    results = {}
    for mid, path in MODELS.items():
        r = analyze(mid, path)
        if r is not None:
            results[mid] = r

    print("\n" + "="*70)
    print("A6 SUMMARY: rho^J vs rho^{Pi@J} across EGPT models")
    print("="*70)
    header = f"{'Model':30s}  rho^J   rho^Pi@J  delta"
    print(header)
    print("-" * len(header))
    for mid, r in results.items():
        rho_J   = r["rho_J"]
        rho_PiJ = r["rho_PiJ"]
        delta = rho_PiJ - rho_J
        print(f"{mid:30s}  {rho_J:+.4f}  {rho_PiJ:+.4f}   {delta:+.4f}")

    # Save
    out = PLOTS_DIR / "a6_proj_cosim_results.json"
    out.write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
