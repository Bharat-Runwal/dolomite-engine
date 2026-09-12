"""Inference-FLOPs accounting + expert-weight spectral analysis for the
Boltzmann-MoE-Hopfield FET checkpoint.

Target run:
  math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu

Two questions:

1. FLOPs — where does inference compute actually go, and what is the ceiling for
   (a) top-k truncation with the exact energy router, (b) top-k with a free router?
   The Hopfield energy E_k = (1/I_e)||gelu(W_k h)||^2 needs ONLY the first matmul
   (W_k h); the descent gradient needs a second (gated @ W_k). So the router costs
   exactly as much as all K expert-output matmuls combined.

2. Is a low-rank proxy for E_k plausible? For roughly symmetric pre-activations,
   E||relu(z)||^2 = (1/2)||z||^2, so ||W_k h||^2 = h^T (W_k^T W_k) h is a natural
   proxy, and its top-r eigendirections cost 2*d*r per expert instead of 2*d*I_e.
   Here we measure the spectrum of each expert's W_k (CPU, weights only).

Run (CPU is fine):
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  python experiments/boltzmann-moe/scripts/analyze_moe_router_flops_20260912.py
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from safetensors.torch import load_file


RUN = "math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu"
CKPT = Path(
    "/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results"
    f"/multi-block-ablation/{RUN}/unsharded"
)


# --------------------------------------------------------------------------- #
# 1. FLOPs                                                                    #
# --------------------------------------------------------------------------- #


def flop_report(cfg: dict) -> dict:
    """Per-token matmul MACs, from the checkpoint's own shapes.

    Only weight matmuls are counted (exact); attention score/AV terms are
    sequence-length dependent and reported separately.
    """
    d = cfg["hidden_size"]
    vocab = cfg["vocab_size"]
    iters = cfg["layer_iterations"]
    n_gpt = sum(1 for b in cfg["sequence_mixer_blocks"] if b["sequence_mixer_type"] != "energy_attention")
    egpt_idx = [i for i, b in enumerate(cfg["sequence_mixer_blocks"])
                if b["sequence_mixer_type"] == "energy_attention"]
    assert len(egpt_idx) == 1, egpt_idx
    e_i = egpt_idx[0]
    n_iter = iters[e_i]

    moe = cfg["mlp_blocks"][e_i]
    I_tot, K = moe["intermediate_size"], moe["n_experts"]
    I_e = I_tot // K

    gpt_ffn_I = cfg["mlp_blocks"][0]["intermediate_size"]

    # --- one GPT layer -----------------------------------------------------
    gpt_attn = 3 * d * d + d * d            # c_attn (qkv) + c_proj
    gpt_mlp = 2 * d * gpt_ffn_I + gpt_ffn_I * d   # swiglu: gate+up, then down
    gpt_layer = gpt_attn + gpt_mlp

    # --- one energy-block iteration ---------------------------------------
    e_attn = 2 * d * d + d * d              # c_attn (Q,K only; V=K) + out-proj via W_Q
    # psd_anti projection: v -> S(S^T v) + U_a(V_a^T v) - V_a(U_a^T v)
    proj_rank = cfg.get("energy_proj_rank") or d
    a_rank = cfg.get("energy_antisym_rank") or 0
    e_proj = 2 * d * proj_rank + 4 * d * a_rank

    router_matmul = d * I_tot               # W h for ALL experts (gives every E_k)
    expert_out_one = d * I_e                # gated_k @ W_k for ONE expert
    moe_dense = router_matmul + K * expert_out_one

    head = vocab * d                        # tied lm_head

    def moe_cost(k: int, free_router: bool) -> int:
        if free_router:
            # cheap proxy picks k experts; only those experts run BOTH matmuls
            return k * (expert_out_one + expert_out_one)
        return router_matmul + k * expert_out_one

    base_egpt = n_iter * (e_attn + e_proj)
    total_dense = n_gpt * gpt_layer + base_egpt + n_iter * moe_dense + head

    rows = []
    for label, k, free in [
        ("dense soft (current)", K, False),
        ("top-4, energy router", 4, False),
        ("top-2, energy router", 2, False),
        ("top-1, energy router", 1, False),
        ("top-4, free router", 4, True),
        ("top-2, free router", 2, True),
        ("top-1, free router", 1, True),
    ]:
        m = n_iter * moe_cost(k, free)
        egpt = base_egpt + m
        tot = n_gpt * gpt_layer + egpt + head
        rows.append({
            "variant": label,
            "moe_MACs": m,
            "egpt_block_MACs": egpt,
            "total_MACs": tot,
            "moe_vs_dense": m / (n_iter * moe_dense),
            "egpt_vs_dense": egpt / (base_egpt + n_iter * moe_dense),
            "total_vs_dense": tot / total_dense,
        })

    return {
        "dims": dict(d=d, vocab=vocab, n_gpt=n_gpt, n_iter=n_iter, K=K, I_e=I_e, I_tot=I_tot),
        "breakdown_MACs": {
            "gpt_layers_x%d" % n_gpt: n_gpt * gpt_layer,
            "egpt_attn_x%d" % n_iter: n_iter * e_attn,
            "egpt_proj_x%d" % n_iter: n_iter * e_proj,
            "egpt_moe_router_x%d" % n_iter: n_iter * router_matmul,
            "egpt_moe_experts_x%d" % n_iter: n_iter * K * expert_out_one,
            "lm_head": head,
            "TOTAL": total_dense,
        },
        "variants": rows,
    }


# --------------------------------------------------------------------------- #
# 2. Expert-weight spectra                                                    #
# --------------------------------------------------------------------------- #


def spectral_report(ranks: list[int]) -> dict:
    sd = load_file(CKPT / "model.safetensors")
    key = "transformer.h.8.ffwd.expert_holder.W.weight"
    W = sd[key].float()                       # [I_tot, d]
    cfg = json.loads((CKPT / "config.json").read_text())
    moe = cfg["mlp_blocks"][8]
    K, I_tot = moe["n_experts"], moe["intermediate_size"]
    I_e = I_tot // K

    out = {"per_expert": [], "ranks": ranks}
    sv_all = []
    for k in range(K):
        Wk = W[k * I_e:(k + 1) * I_e]         # [I_e, d]
        sv = torch.linalg.svdvals(Wk)         # descending
        sq = sv ** 2
        tot = sq.sum()
        cum = torch.cumsum(sq, 0) / tot
        # participation ratio of the squared spectrum = effective rank
        eff_rank = (tot ** 2 / (sq ** 2).sum()).item()
        out["per_expert"].append({
            "expert": k,
            "fro_norm": Wk.norm().item(),
            "eff_rank_pr": eff_rank,
            "energy_frac_at_rank": {r: cum[min(r, len(cum)) - 1].item() for r in ranks},
        })
        sv_all.append(sv)

    # cross-expert redundancy: cosine between the Gram matrices M_k = W_k^T W_k
    Ms = torch.stack([
        (W[k * I_e:(k + 1) * I_e].T @ W[k * I_e:(k + 1) * I_e]).flatten()
        for k in range(K)
    ])
    Mn = torch.nn.functional.normalize(Ms, dim=-1)
    C = Mn @ Mn.T
    off = C[~torch.eye(K, dtype=torch.bool)]
    out["gram_cosine"] = {
        "mean_offdiag": off.mean().item(),
        "min_offdiag": off.min().item(),
        "max_offdiag": off.max().item(),
        "matrix": [[round(v, 4) for v in row] for row in C.tolist()],
    }
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128, 256])
    ap.add_argument("--skip-spectra", action="store_true")
    args = ap.parse_args()

    cfg = json.loads((CKPT / "config.json").read_text())
    f = flop_report(cfg)

    print(f"=== {RUN} ===")
    print("dims:", f["dims"])
    print("\n--- per-token matmul MACs (dense soft routing) ---")
    tot = f["breakdown_MACs"]["TOTAL"]
    for k, v in f["breakdown_MACs"].items():
        print(f"  {k:28s} {v/1e6:9.2f} M   {100*v/tot:5.1f}%")
    print(f"  (per-token FLOPs = 2x MACs = {2*tot/1e9:.3f} GFLOP/token)")

    print("\n--- routing / sparsity variants ---")
    print(f"  {'variant':24s} {'MoE':>9s} {'EGPT blk':>10s} {'TOTAL':>10s}   "
          f"{'MoE':>7s} {'EGPTblk':>8s} {'TOTAL':>7s}  (x dense)")
    for r in f["variants"]:
        print(f"  {r['variant']:24s} {r['moe_MACs']/1e6:8.2f}M {r['egpt_block_MACs']/1e6:9.2f}M "
              f"{r['total_MACs']/1e6:9.2f}M   {r['moe_vs_dense']:6.3f} "
              f"{r['egpt_vs_dense']:7.3f} {r['total_vs_dense']:6.3f}")

    if not args.skip_spectra:
        s = spectral_report(args.ranks)
        print("\n--- expert W_k spectra (d=1536, I_e=1024) ---")
        print(f"  {'k':>2s} {'||W||_F':>9s} {'eff_rank':>9s}  " +
              " ".join(f"r={r:<4d}" for r in s["ranks"]))
        for e in s["per_expert"]:
            fr = e["energy_frac_at_rank"]
            print(f"  {e['expert']:2d} {e['fro_norm']:9.3f} {e['eff_rank_pr']:9.1f}  " +
                  " ".join(f"{fr[r]:6.3f}" for r in s["ranks"]))
        g = s["gram_cosine"]
        print(f"\n  cross-expert cos(W_i^T W_i, W_j^T W_j): "
              f"mean {g['mean_offdiag']:.4f}  min {g['min_offdiag']:.4f}  max {g['max_offdiag']:.4f}")


if __name__ == "__main__":
    main()
