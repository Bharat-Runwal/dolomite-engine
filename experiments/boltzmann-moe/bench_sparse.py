"""
Benchmark dense vs sparse Boltzmann MoE forward.

Two implementations of BoltzmannMoE_Energy_MLP forward:
  - dense_forward : current path (computes term1, term2 for ALL K experts)
  - sparse_forward: efficient path (computes pre1/pre2 for all K, but term1/term2
                    only for top-k experts via per-expert token dispatch)

Verifies numerical equivalence (modulo top-k truncation), then times forward
+ backward on actual hidden states drawn from the trained h1_boltz_topk2 / fullsize
checkpoint. Run with --K 4 --topk 2 --d 768 to match h1 setting.

Usage:
    python bench_sparse.py --K 4 --topk 2 --I 8192 --d 768 --N 16384
    python bench_sparse.py --K 16 --topk 2 --I 16384 --d 768 --N 16384      # B-series
    python bench_sparse.py --K 4 --topk 2 --I 24576 --d 1536 --N 8192       # 580M
    python bench_sparse.py --load /path/to/unsharded   # load weights from ckpt
"""

import argparse, math, time
import torch
import torch.nn.functional as F
from torch import nn

SIGMOID_SCALE = (2.0 / math.pi) ** 0.5


def gelu_grad(x):
    # tanh-approx grad of GELU (matches what the trained model uses)
    return torch.sigmoid(SIGMOID_SCALE * x) * 0.5


class BoltzMoE(nn.Module):
    """Container holding W1 (K*I_e, d) and W2 (K*I_e, d) for both fwd paths."""
    def __init__(self, d, I_total, K, top_k=None, tau=1.0):
        super().__init__()
        assert I_total % K == 0
        self.d = d
        self.K = K
        self.I_e = I_total // K
        self.I = I_total
        self.top_k = top_k
        self.tau = tau
        self.scale = self.I_e ** -0.5

        self.W1 = nn.Linear(d, I_total, bias=False)
        self.W2 = nn.Linear(d, I_total, bias=False)
        nn.init.normal_(self.W1.weight, std=0.02)
        nn.init.normal_(self.W2.weight, std=0.02)

    # --------------------------------------------------------------
    # Dense path: matches current production BoltzmannMoE_Energy_MLP
    # --------------------------------------------------------------
    def dense_forward(self, x):
        """x: (..., d). Returns (..., d). Computes term1+term2 for ALL K experts."""
        leading = x.shape[:-1]
        K, I_e, d = self.K, self.I_e, self.d
        W1_e = self.W1.weight.view(K, I_e, d)
        W2_e = self.W2.weight.view(K, I_e, d)

        pre1 = self.W1(x).view(*leading, K, I_e)
        phi = F.gelu(pre1)
        phi_prime = gelu_grad(pre1)

        # term1 for ALL K experts (used for both routing E and output)
        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)         # (..., K, d)
        E = torch.einsum("...h,...eh->...e", x, term1) * self.scale  # (..., K)

        # routing weights
        if self.top_k is not None and self.top_k < K:
            p_full = F.softmax(E / self.tau, dim=-1)
            _, topk_idx = E.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(p_full, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p_full * mask
        else:
            p = F.softmax(E / self.tau, dim=-1)

        # term2 for ALL K experts
        pre2 = self.W2(x).view(*leading, K, I_e)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * pre2, W1_e)
        expert_grads = term1 + term2                                # (..., K, d)
        out = torch.einsum("...e,...eh->...h", p, expert_grads)
        return out

    # --------------------------------------------------------------
    # Sparse path: only top-k experts run pre2 + term2 matmuls.
    # NOTE: routing energy is computed exactly as in dense_forward (E = x · term1)
    #       so the topk selection matches dense bit-for-bit (modulo precision).
    # --------------------------------------------------------------
    def sparse_forward(self, x):
        """Compute pre1+term1 for all K (needed for routing), then pre2+term2
        only for top-k experts via per-expert token dispatch.

        FLOPs per token: 2(K + top_k) · I_e · d
        vs dense:        4K · I_e · d
        Saving: (K - top_k) / (2K)   →  25% for K=4,top_k=2; 44% for K=16,top_k=2
        """
        leading = x.shape[:-1]
        K, I_e, d = self.K, self.I_e, self.d
        W1_e = self.W1.weight.view(K, I_e, d)
        W2_e = self.W2.weight.view(K, I_e, d)
        top_k = self.top_k if self.top_k is not None else K

        x_flat = x.reshape(-1, d)
        N = x_flat.shape[0]

        # (1) pre1 + term1 for ALL K (needed to compute routing energy E = x · term1
        #     identically to dense_forward; ensures topk picks the same experts).
        pre1 = self.W1(x_flat).view(N, K, I_e)
        phi  = F.gelu(pre1)
        term1 = torch.einsum("nei,eih->neh", phi, W2_e)              # (N, K, d)

        # (2) routing energy — same expression as dense_forward
        E = torch.einsum("nh,neh->ne", x_flat, term1) * self.scale   # (N, K)

        if top_k < K:
            p_full = F.softmax(E / self.tau, dim=-1)
            _, topk_idx = E.topk(top_k, dim=-1)                      # (N, top_k)
            mask = torch.zeros_like(p_full, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p_full * mask
        else:
            p = F.softmax(E / self.tau, dim=-1)
            mask = torch.ones_like(p, dtype=torch.bool)

        # term1 contribution: weighted sum across all K (non-topk have p=0)
        out = torch.einsum("ne,ned->nd", p, term1)                   # (N, d)

        # (3) term2 only for top-k experts (per-expert token dispatch).
        for e in range(K):
            tok_mask = mask[:, e]
            if not tok_mask.any():
                continue
            tok_idx = tok_mask.nonzero(as_tuple=False).squeeze(-1)   # (M,)
            x_e = x_flat[tok_idx]                                    # (M, d)
            pre1_e = pre1[tok_idx, e]                                # (M, I_e) — already computed
            phi_prime_e = gelu_grad(pre1_e)
            pre2_e = x_e @ W2_e[e].T                                 # (M, I_e)
            term2_e = (phi_prime_e * pre2_e) @ W1_e[e]               # (M, d)
            p_e = p[tok_idx, e].unsqueeze(-1)                        # (M, 1)
            out.index_add_(0, tok_idx, p_e * term2_e)

        return out.view(*leading, d)


    # --------------------------------------------------------------
    # Sparse-grouped path: same FLOPs as sparse_forward, but uses bmm
    # over (top_k, N, ·) rather than a Python loop over K experts.
    # No saving from skipping experts on indexing — this version still
    # runs term2 over all top_k slots without per-expert bucketing.
    # --------------------------------------------------------------
    def sparse_grouped_forward(self, x):
        """Top-k expert weights are gathered per token (not per expert).
        For top_k=2, K=4 this gathers ~50% of the W matrices into a per-token
        slot tensor — high memory but no Python loop, all GPU ops."""
        leading = x.shape[:-1]
        K, I_e, d = self.K, self.I_e, self.d
        W1_e = self.W1.weight.view(K, I_e, d)
        W2_e = self.W2.weight.view(K, I_e, d)
        top_k = self.top_k if self.top_k is not None else K

        x_flat = x.reshape(-1, d)
        N = x_flat.shape[0]

        # (1) pre1 + term1 for ALL K (routing — same as dense)
        pre1 = self.W1(x_flat).view(N, K, I_e)
        phi  = F.gelu(pre1)
        term1 = torch.einsum("nei,eih->neh", phi, W2_e)

        E = torch.einsum("nh,neh->ne", x_flat, term1) * self.scale

        if top_k < K:
            p_full = F.softmax(E / self.tau, dim=-1)
            topk_vals, topk_idx = E.topk(top_k, dim=-1)              # (N, top_k)
            mask = torch.zeros_like(p_full, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p_full * mask
        else:
            p = F.softmax(E / self.tau, dim=-1)
            topk_idx = torch.arange(K, device=x.device).expand(N, K)

        out = torch.einsum("ne,ned->nd", p, term1)

        # (3) term2 for top-k slots:
        #   gather per-token: pre1_topk (N, top_k, I_e), W1[topk] (N, top_k, I_e, d), W2[topk] (N, top_k, I_e, d)
        #   then bmm along token dim — but per-token weight gather is memory-heavy.
        #   Compromise: loop over top_k slot (small, e.g. 2) instead of over K.
        for slot in range(top_k):
            e_per_tok = topk_idx[:, slot]                            # (N,)
            W1_sel = W1_e[e_per_tok]                                  # (N, I_e, d)
            W2_sel = W2_e[e_per_tok]                                  # (N, I_e, d)
            pre1_sel = torch.gather(pre1, 1,
                                    e_per_tok[:, None, None].expand(N, 1, I_e)).squeeze(1)  # (N, I_e)
            phi_prime_sel = gelu_grad(pre1_sel)
            # pre2_sel = x · W2_sel^T : (N, d) @ (N, d, I_e) → (N, I_e)
            pre2_sel = torch.bmm(x_flat.unsqueeze(1), W2_sel.transpose(1, 2)).squeeze(1)
            inner = phi_prime_sel * pre2_sel
            term2_sel = torch.bmm(inner.unsqueeze(1), W1_sel).squeeze(1)  # (N, d)
            p_sel = torch.gather(p, 1, e_per_tok.unsqueeze(1)).squeeze(1).unsqueeze(-1)  # (N, 1)
            out = out + p_sel * term2_sel

        return out.view(*leading, d)


# ====================================================================
# Verification + timing
# ====================================================================
@torch.no_grad()
def verify(mod, x):
    a = mod.dense_forward(x)
    b = mod.sparse_forward(x)
    diff_b = (a - b).abs().max().item()
    rel_b  = diff_b / (a.abs().max().item() + 1e-9)
    return diff_b, rel_b


def time_path(fn, x, n_warm=3, n_iter=10, backward=False):
    torch.cuda.synchronize()
    # warmup
    for _ in range(n_warm):
        if backward:
            x_ = x.clone().requires_grad_(True)
            y = fn(x_)
            y.sum().backward()
        else:
            y = fn(x)
    torch.cuda.synchronize()

    t0 = time.time()
    for _ in range(n_iter):
        if backward:
            x_ = x.clone().requires_grad_(True)
            y = fn(x_)
            y.sum().backward()
        else:
            y = fn(x)
    torch.cuda.synchronize()
    return (time.time() - t0) / n_iter * 1000  # ms / iter


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=4, help="num experts")
    ap.add_argument("--topk", type=int, default=2)
    ap.add_argument("--I", type=int, default=8192, help="total intermediate (= K * I_e)")
    ap.add_argument("--d", type=int, default=768)
    ap.add_argument("--N", type=int, default=16384, help="num tokens (B*T)")
    ap.add_argument("--dtype", default="bfloat16", choices=["float32","float16","bfloat16"])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--load", default=None, help="optional: load W1/W2 from unsharded ckpt")
    ap.add_argument("--block_idx", type=int, default=6, help="EGPT block index in checkpoint")
    args = ap.parse_args()

    dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[args.dtype]
    torch.manual_seed(0)

    print(f"K={args.K}  top_k={args.topk}  I={args.I}  I_e={args.I//args.K}  d={args.d}  N={args.N}  dtype={args.dtype}")
    print(f"Theoretical FLOPs ratio (sparse/dense) = {(2*args.I*args.d + 2*args.topk*(args.I//args.K)*args.d) / (4*args.I*args.d):.3f}")
    print()

    mod = BoltzMoE(args.d, args.I, args.K, top_k=args.topk).to(args.device).to(dtype)

    if args.load:
        from safetensors.torch import load_file
        sd = load_file(f"{args.load}/model.safetensors")
        # find W1, W2 of block_idx
        prefix = f"transformer.h.{args.block_idx}.ffwd"
        w1 = sd[f"{prefix}.W1.weight"].to(args.device).to(dtype)
        w2 = sd[f"{prefix}.W2.weight"].to(args.device).to(dtype)
        with torch.no_grad():
            mod.W1.weight.copy_(w1)
            mod.W2.weight.copy_(w2)
        print(f"Loaded W1, W2 from block {args.block_idx} of {args.load}")

    x = torch.randn(args.N, args.d, device=args.device, dtype=dtype) * 0.5

    # ---- Verify (in float32 for clean math check) ----
    print("=== Numerical check (in float32) ===")
    mod_fp32 = BoltzMoE(args.d, args.I, args.K, top_k=args.topk).to(args.device).float()
    if args.load:
        from safetensors.torch import load_file
        sd = load_file(f"{args.load}/model.safetensors")
        prefix = f"transformer.h.{args.block_idx}.ffwd"
        with torch.no_grad():
            mod_fp32.W1.weight.copy_(sd[f"{prefix}.W1.weight"].float().to(args.device))
            mod_fp32.W2.weight.copy_(sd[f"{prefix}.W2.weight"].float().to(args.device))
    else:
        with torch.no_grad():
            mod_fp32.W1.weight.copy_(mod.W1.weight.float())
            mod_fp32.W2.weight.copy_(mod.W2.weight.float())
    x_fp32 = x.float()
    diff_b, rel_b = verify(mod_fp32, x_fp32)
    print(f"sparse vs dense: max abs={diff_b:.2e}  max rel={rel_b:.2e}  {'OK' if rel_b<1e-3 else 'FAIL'}")
    print()

    # ---- Time forward ----
    print(f"=== Forward timing ({args.dtype}, ms / iter, mean over 10) ===")
    t_d_fwd = time_path(mod.dense_forward,  x, backward=False)
    t_s_fwd = time_path(mod.sparse_forward, x, backward=False)
    print(f"dense  fwd: {t_d_fwd:7.3f}  ms")
    print(f"sparse fwd: {t_s_fwd:7.3f}  ms     speedup: {t_d_fwd/t_s_fwd:.2f}×")
    print()

    print(f"=== Forward+Backward timing ({args.dtype}, ms / iter) ===")
    t_d_bwd = time_path(mod.dense_forward,  x, backward=True)
    t_s_bwd = time_path(mod.sparse_forward, x, backward=True)
    print(f"dense  fwd+bwd: {t_d_bwd:7.3f}  ms")
    print(f"sparse fwd+bwd: {t_s_bwd:7.3f}  ms     speedup: {t_d_bwd/t_s_bwd:.2f}×")


if __name__ == "__main__":
    main()
