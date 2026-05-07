"""LM-head singular value distribution analysis.

Plots the full SV spectrum of W_U and computes key statistics:
  - SV range, decay ratio
  - k_50, k_95, k_99 (index where cumulative energy crosses 50/95/99%)
  - align(write_op, bottom-k) for k = 256, 128, 64, 32

This is the "near-null space" analysis: does the write operator of each model
concentrate into the weakly-predictive tail of W_U's spectrum?

Usage:
    python 04_sv_distribution.py --models 410m_hybrid_s8e4,410m_recgpt_s8e4
"""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM

from importlib.util import spec_from_file_location, module_from_spec
_spec = spec_from_file_location('lk_alignment', Path(__file__).parent / '03_lk_alignment.py')
_mod = module_from_spec(_spec); _spec.loader.exec_module(_mod)
MODELS = _mod.MODELS; get_lm_svd = _mod.get_lm_svd; get_blocks = _mod.get_blocks
get_write_op = _mod.get_write_op; align_k = _mod.align_k; K_VALUES = _mod.K_VALUES


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default=",".join(MODELS.keys()))
    parser.add_argument("--output_dir", default=".")
    args = parser.parse_args()

    model_ids = [m.strip() for m in args.models.split(",")]
    out_dir = Path(args.output_dir)

    fig, axes = plt.subplots(1, len(model_ids), figsize=(5 * len(model_ids), 4))
    if len(model_ids) == 1:
        axes = [axes]

    all_results = {}
    for ax, mid in zip(axes, model_ids):
        if mid not in MODELS:
            print(f"Unknown: {mid}"); continue
        ckpt, is_egpt = MODELS[mid]
        if not ckpt.exists():
            print(f"SKIP {mid}: {ckpt} not found"); continue

        print(f"\n=== {mid} ===")
        m = AutoModelForCausalLM.from_pretrained(
            str(ckpt), torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cpu().eval()

        S, Vh = get_lm_svd(m)
        d = Vh.shape[1]
        cum = np.cumsum(S**2) / (S**2).sum()
        k95 = int(np.searchsorted(cum, 0.95)) + 1
        k99 = int(np.searchsorted(cum, 0.99)) + 1

        # Plot spectrum
        ax.semilogy(S, "b-", lw=0.8, alpha=0.7)
        ax.axvline(k95, color="r",      ls="--", lw=1.2, label=f"k_95={k95}")
        ax.axvline(k99, color="orange", ls="--", lw=1.2, label=f"k_99={k99}")
        ax.set_title(f"{mid}\ndecay={S[0]/S[-1]:.0f}×", fontsize=8)
        ax.set_xlabel("Singular value index"); ax.set_ylabel("SV (log)")
        ax.legend(fontsize=7); ax.grid(alpha=0.3)

        # Bottom-k alignment
        blocks = get_blocks(m)
        iters  = getattr(m.config, "layer_iterations", [1] * len(blocks))
        bot_aligns = {k: [] for k in K_VALUES}
        for i, blk in enumerate(blocks):
            op = get_write_op(blk, is_egpt, iters[i])
            if op is None: continue
            for k in K_VALUES:
                bot_aligns[k].append(align_k(op, Vh, k, top=False))

        print(f"  d={d}  k95={k95}  k99={k99}  decay={S[0]/S[-1]:.0f}x")
        for k in K_VALUES:
            if bot_aligns[k]:
                mb = np.mean(bot_aligns[k])
                rand = np.sqrt(k/d)
                print(f"  bot-{k:4d}: mean={mb:.3f}  rand={rand:.3f}  excess={mb-rand:+.3f}")

        all_results[mid] = {
            "d": d, "k95": k95, "k99": k99,
            "decay": float(S[0]/S[-1]),
            "bot_align_means": {k: float(np.mean(v)) for k,v in bot_aligns.items() if v},
            "rand_bot": {k: float(np.sqrt(k/d)) for k in K_VALUES},
        }
        del m

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(str(out_dir / f"sv_distribution.{ext}"), dpi=150, bbox_inches="tight")
    print(f"\nSaved plots to {out_dir}/sv_distribution.{{png,pdf}}")

    (out_dir / "sv_results.json").write_text(json.dumps(all_results, indent=2))


if __name__ == "__main__":
    main()
