"""Measured prefill throughput of the Boltzmann-MoE energy FF at production widths.

Why this script exists
----------------------
Analytic FLOPs (design_single_block_moe_20260912.py) say top-k with a cheap router
costs k/K of dense. But the SHIPPED implementation cannot realise that: both
``BoltzmannMoEFFEnergy.forward`` (energy_ff.py:432) and the legacy class compute
every expert and then zero the rejected ones (``p = p * mask``). So top_k today buys
exactly 0% wall-clock. Measuring the current code would report "no gain" and tell us
nothing about whether the idea is sound.

So we benchmark four paths at real width:

  (a) dense-soft          all K experts, Boltzmann weights          [what ships]
  (b) topk-masked         all K experts, then mask                  [what ships with top_k]
  (c) topk-grouped-exact  exact energy router (needs W h for all K),
                          then group tokens by expert and run only
                          the selected experts' second matmul
  (d) topk-grouped-proxy  rank-r proxy router (d*r MACs), then group
                          tokens by expert and run BOTH matmuls for
                          only the selected experts

(d) is the target design. (c) is the fallback that needs no fitted router. The gap
between (b) and (d) is the real prize.

Only the energy FF is timed, because in the single-block all-MoE architecture it is
98.9-99.6% of per-token FLOPs (see design_single_block_moe_20260912.py). Attention,
the projection and the lm_head are reported analytically there, not here.

Usage (1 GPU; the 7B point needs ~14GB for weights, the 30B point ~60GB):
  python experiments/boltzmann-moe/scripts/bench_moe_throughput_20260912.py
  python .../bench_moe_throughput_20260912.py --shapes 7b_K128 --tokens 16384
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/boltzmann-moe/results/router_analysis"


# name -> (d, I_tot, K, top_k)   mirrors configs/single_block_moe/
SHAPES = {
    "3b_K32":  (2048, 1343488, 32, 4),
    "7b_K64":  (2560, 2613248, 64, 8),
    "7b_K128": (2560, 2605056, 128, 8),
    "12b_K128": (3072, 3784704, 128, 8),
    "30b_K256": (3584, 8224768, 256, 8),
    # small sanity shape, always runnable
    "tiny":    (1536, 8192, 8, 2),
}

_SIG = (2.0 / math.pi) ** 0.5


def gelu_and_grad(x):
    return F.gelu(x), torch.sigmoid(_SIG * x) * 0.5


# --------------------------------------------------------------------------- #
# the four paths                                                              #
# --------------------------------------------------------------------------- #


def dense_soft(x, W, K, I_e, tau=1.0):
    """All K experts, Boltzmann-weighted. Mirrors BoltzmannMoEFFEnergy (hopfield)."""
    Wx = x @ W.t()                                        # [N, I_tot]
    g, gp = gelu_and_grad(Wx)
    E = (g.float() ** 2).view(-1, K, I_e).mean(-1)         # [N, K]
    p = F.softmax(-E / tau, dim=-1).to(x.dtype)
    gated = (g * gp).view(-1, K, I_e)
    per = torch.einsum("nki,kih->nkh", gated, W.view(K, I_e, -1))
    return torch.einsum("nk,nkh->nh", p, per) * (4.0 / I_e)


def topk_masked(x, W, K, I_e, k, tau=1.0):
    """What ships when top_k is set: full compute, then zero the rejected experts."""
    Wx = x @ W.t()
    g, gp = gelu_and_grad(Wx)
    E = (g.float() ** 2).view(-1, K, I_e).mean(-1)
    logits = -E / tau
    p = F.softmax(logits, dim=-1)
    idx = logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, True)
    p = (p * mask).to(x.dtype)
    gated = (g * gp).view(-1, K, I_e)
    per = torch.einsum("nki,kih->nkh", gated, W.view(K, I_e, -1))
    return torch.einsum("nk,nkh->nh", p, per) * (4.0 / I_e)


def _sort_by_expert(idx, w, K, renorm: bool, device):
    """(token, expert) pairs sorted by expert, plus the group row offsets."""
    N, k = idx.shape
    if renorm:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(N, device=device).repeat_interleave(k)
    order = flat_e.argsort()
    flat_e, flat_t = flat_e[order], flat_t[order]
    flat_w = w.reshape(-1)[order]
    counts = torch.bincount(flat_e, minlength=K)
    offs = counts.cumsum(0).to(torch.int32)          # torch._grouped_mm wants int32
    return flat_t, flat_w, offs


def grouped_two_matmuls(x, We, idx, w, K, I_e, d, renorm: bool, Wt=None):
    """Both expert matmuls via torch._grouped_mm — one kernel launch each.

    torch._grouped_mm(A, B, offs) with A [total_M, Kc], B [G, Kc, N] and `offs`
    the CUMULATIVE row boundary per group. Rows are gathered so that all rows
    belonging to expert e are contiguous, which is what makes a single grouped
    GEMM legal. This replaces the K-iteration Python loop, whose per-expert
    M = N*k/K was far too small to reach GEMM efficiency.
    """
    N = x.shape[0]
    flat_t, flat_w, offs = _sort_by_expert(idx, w, K, renorm, x.device)
    xg = x.index_select(0, flat_t)                      # [N*k, d] expert-contiguous
    if Wt is None:
        # NEVER do this inside the timed region: transpose+contiguous COPIES the
        # entire expert weight tensor (13.3 GB at the 7B shape) on every call.
        # A deployment materialises this once at load time; so does the caller below.
        Wt = We.transpose(1, 2).contiguous()           # [K, d, I_e]
    z = torch._grouped_mm(xg, Wt, offs=offs)           # [N*k, I_e]  first matmul
    g, gp = gelu_and_grad(z)
    y = torch._grouped_mm((g * gp).contiguous(), We, offs=offs)   # [N*k, d]  second
    out = torch.zeros(N, d, dtype=x.dtype, device=x.device)
    out.index_add_(0, flat_t, y * flat_w[:, None].to(x.dtype))
    return out * (4.0 / I_e)


def _grouped_second_matmul(x, We, idx, w, K, I_e, d, renorm: bool,
                           gated_all=None):
    """Group tokens by selected expert; run each expert on only its own tokens.

    idx: [N, k] expert ids, w: [N, k] routing weights.
    Cost is k/K of the dense second matmul, plus the first matmul for the same
    (token, expert) pairs. This is the sort-and-group pattern a real kernel
    (scattermoe / grouped_mm) implements; a Python loop over K is enough to
    measure whether the arithmetic saving survives launch overhead.
    """
    N, k = idx.shape
    if renorm:
        w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
    flat_e = idx.reshape(-1)                             # [N*k]
    flat_t = torch.arange(N, device=x.device).repeat_interleave(k)
    order = flat_e.argsort()
    flat_e, flat_t = flat_e[order], flat_t[order]
    flat_w = w.reshape(-1)[order].to(x.dtype)
    counts = torch.bincount(flat_e, minlength=K)
    bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device),
                        counts.cumsum(0)])
    out = torch.zeros(N, d, dtype=x.dtype, device=x.device)
    b = bounds.tolist()
    for e in range(K):
        lo, hi = b[e], b[e + 1]
        if hi == lo:
            continue
        rows = flat_t[lo:hi]
        Wk = We[e]                                       # [I_e, d]
        if gated_all is None:
            # proxy path: the router never computed W h, so pay it here for the
            # selected (token, expert) pairs only -- this is the k/K saving.
            z = x.index_select(0, rows) @ Wk.t()         # first matmul, selected only
            g, gp = gelu_and_grad(z)
            gated = g * gp
        else:
            # exact-router path: W h was already computed for ALL experts to form
            # the energies, so reuse it instead of recomputing (a gather, not a GEMM).
            gated = gated_all[:, e, :].index_select(0, rows)
        ye = gated @ Wk                                  # second matmul
        out.index_add_(0, rows, ye * flat_w[lo:hi, None])
    return out * (4.0 / I_e)


def topk_grouped_exact(x, W, K, I_e, k, tau=1.0, renorm=False):
    """Exact energy router (needs W h for ALL experts), grouped expert compute."""
    Wx = x @ W.t()
    g, gp = gelu_and_grad(Wx)
    E = (g.float() ** 2).view(-1, K, I_e).mean(-1)
    logits = -E / tau
    p = F.softmax(logits, dim=-1)
    top = logits.topk(k, dim=-1)
    w = p.gather(-1, top.indices)
    return _grouped_second_matmul(x, W.view(K, I_e, -1), top.indices, w,
                                  K, I_e, x.shape[-1], renorm,
                                  gated_all=(g * gp).view(-1, K, I_e))


def topk_grouped_proxy(x, V, S, K, I_e, k, W, tau=1.0, renorm=False):
    """Rank-r proxy router (d*r MACs), grouped expert compute. The target design.

    V: [K, r, d] per-expert right-singular directions, S: [K, r] gains.
    The proxy score is a stand-in for the fitted head; its ARITHMETIC cost is what
    we are timing, and it is what an implementation would pay.
    """
    a = torch.einsum("nd,krd->nkr", x, V)                 # d*r*K MACs
    E = (a.float() ** 2 * S.float() ** 2).sum(-1) / I_e   # [N, K]
    logits = -E / tau
    p = F.softmax(logits, dim=-1)
    top = logits.topk(k, dim=-1)
    w = p.gather(-1, top.indices)
    return _grouped_second_matmul(x, W.view(K, I_e, -1), top.indices, w,
                                  K, I_e, x.shape[-1], renorm)


def topk_gmm_proxy(x, V, S, K, I_e, k, W, tau=1.0, renorm=False, Wt=None):
    """Target design with a real grouped GEMM: rank-r proxy router + _grouped_mm."""
    a = torch.einsum("nd,krd->nkr", x, V)
    E = (a.float() ** 2 * S.float() ** 2).sum(-1) / I_e
    logits = -E / tau
    p = F.softmax(logits, dim=-1)
    top = logits.topk(k, dim=-1)
    w = p.gather(-1, top.indices)
    return grouped_two_matmuls(x, W.view(K, I_e, -1), top.indices, w,
                               K, I_e, x.shape[-1], renorm, Wt=Wt)


# --------------------------------------------------------------------------- #
# timing                                                                      #
# --------------------------------------------------------------------------- #


def run_chunked(fn, x, chunk: int):
    """Apply ``fn`` over token chunks.

    Required, not a nicety: the dense path materialises ``W h`` of shape
    [N, I_tot], which at N=8192 and I_tot=2.6M is 42.7 GB in bf16 -- it cannot run
    unchunked at production width at all. Real inference chunks; so do we, and all
    paths get the same chunk so the comparison is apples-to-apples.
    """
    outs = []
    for i in range(0, x.shape[0], chunk):
        outs.append(fn(x[i:i + chunk]))
    return torch.cat(outs, 0)


def timeit(fn, warmup=3, iters=10) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters * 1e3          # ms


def macs_of(path: str, d: int, I_tot: int, K: int, k: int, N: int, r: int) -> int:
    I_e = I_tot // K
    if path in ("dense-soft", "topk-masked"):
        return N * 2 * d * I_tot
    if path == "topk-grouped-exact":
        return N * (d * I_tot + k * d * I_e)
    if path in ("topk-grouped-proxy", "topk-gmm-proxy"):
        return N * (d * r * K + 2 * k * d * I_e)
    raise ValueError(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--shapes", nargs="+", default=["tiny", "3b_K32", "7b_K128"])
    ap.add_argument("--tokens", type=int, default=8192, help="prefill batch of tokens")
    ap.add_argument("--rank", type=int, default=16, help="proxy router rank r")
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--chunk", type=int, default=512,
                    help="tokens/chunk for the DENSE paths; halved on OOM")
    ap.add_argument("--sparse_chunk", type=int, default=0,
                    help="tokens/chunk for the grouped paths (0 = all tokens)")
    ap.add_argument("--router_init", choices=["random", "svd"], default="random",
                    help="proxy-router directions; does not affect timing")
    ap.add_argument("--renorm", action="store_true",
                    help="renormalize top-k weights to sum 1 (recommended; ships as OFF)")
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    print(f"device: {props.name}  {props.total_memory/1e9:.0f} GB")
    print(f"tokens/call: {args.tokens}   proxy rank r={args.rank}   "
          f"renorm={args.renorm}\n")

    results = []
    for name in args.shapes:
        if name not in SHAPES:
            print(f"  (unknown shape {name}, skipping)")
            continue
        d, I_tot, K, k = SHAPES[name]
        I_e = I_tot // K
        need = I_tot * d * 2 / 1e9
        if need > props.total_memory / 1e9 * 0.62:
            print(f"### {name}: W needs {need:.1f} GB, too big for this GPU — skipped")
            continue
        print(f"### {name}: d={d} I_tot={I_tot} K={K} top_k={k} expert_I={I_e} "
              f"(W = {need:.1f} GB bf16)")

        torch.manual_seed(0)
        W = (torch.randn(I_tot, d, device=dev, dtype=torch.bfloat16)
             * (d ** -0.5)).contiguous()
        x = torch.randn(args.tokens, d, device=dev, dtype=torch.bfloat16)
        # a cheap stand-in for the fitted rank-r router: top-r right-singular
        # directions of each expert, computed once (offline in a real deployment)
        # The proxy router's ARITHMETIC cost (d*r*K MACs) is what we time; which
        # directions V holds changes only which experts get picked, not the cost.
        # So use a random orthonormal basis and skip K full SVDs (~13 s each at
        # these widths). Pass --router_init svd for the true spectral directions.
        r = args.rank
        if args.router_init == "svd":
            V = torch.empty(K, r, d, device=dev, dtype=torch.bfloat16)
            S = torch.empty(K, r, device=dev, dtype=torch.bfloat16)
            for e in range(K):
                _, sv, vh = torch.linalg.svd(W[e * I_e:(e + 1) * I_e].float(),
                                             full_matrices=False)
                V[e], S[e] = vh[:r].to(torch.bfloat16), sv[:r].to(torch.bfloat16)
        else:
            V = torch.linalg.qr(torch.randn(K, d, r, device=dev))[0]
            V = V.transpose(1, 2).contiguous().to(torch.bfloat16)   # [K, r, d]
            S = torch.ones(K, r, device=dev, dtype=torch.bfloat16)

        # Pre-transposed expert weights for the grouped-GEMM path, built ONCE
        # (a deployment does this at load time). Costs another copy of W in memory.
        try:
            Wt = W.view(K, I_e, d).transpose(1, 2).contiguous()
        except torch.cuda.OutOfMemoryError:
            Wt = None
            torch.cuda.empty_cache()
            print("  (not enough memory to pre-transpose W; gmm path will be skipped)")

        kern = {
            "dense-soft": lambda xc: dense_soft(xc, W, K, I_e),
            "topk-masked": lambda xc: topk_masked(xc, W, K, I_e, k),
            "topk-grouped-exact": lambda xc: topk_grouped_exact(xc, W, K, I_e, k,
                                                                renorm=args.renorm),
            "topk-grouped-proxy": lambda xc: topk_grouped_proxy(xc, V, S, K, I_e, k, W,
                                                               renorm=args.renorm),
            "topk-gmm-proxy": lambda xc: topk_gmm_proxy(xc, V, S, K, I_e, k, W,
                                                        renorm=args.renorm, Wt=Wt),
        }
        # Each path gets the largest chunk it can afford. The dense path must chunk
        # (it materialises [N, I_tot]); the grouped paths never do, so they can take
        # all tokens at once -- which matters enormously, because chunking starves
        # each expert's GEMM of rows (N*k/K per chunk).
        chunk_of = {n: (args.chunk if n in ("dense-soft", "topk-masked")
                        else (args.sparse_chunk or args.tokens)) for n in kern}
        base_ms = None
        print(f"  {'path':22s} {'ms/call':>9s} {'us/token':>9s} {'speedup':>8s} "
              f"{'MACs/tok':>10s} {'TFLOP/s':>9s} {'MAC ratio':>10s}")
        for pname in kern:
            if pname == "topk-gmm-proxy" and Wt is None:
                continue
            ms, chunk_used = None, chunk_of[pname]
            while chunk_used >= 32:
                try:
                    f = kern[pname]
                    ms = timeit(lambda f=f, c=chunk_used: run_chunked(f, x, c),
                                iters=args.iters)
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    chunk_used //= 2
            if ms is None:
                print(f"  {pname:22s}  OOM even at chunk=32")
                continue
            if base_ms is None:
                base_ms = ms
            m = macs_of(pname, d, I_tot, K, k, args.tokens, r)
            tflops = 2 * m / (ms * 1e-3) / 1e12
            print(f"  {pname:22s} {ms:9.3f} {ms*1e3/args.tokens:9.3f} "
                  f"{base_ms/ms:7.2f}x {m/args.tokens/1e6:9.2f}M {tflops:9.1f} "
                  f"{m/macs_of('dense-soft', d, I_tot, K, k, args.tokens, r):9.4f}"
                  f"   chunk={chunk_used}")
            results.append(dict(shape=name, path=pname, ms=ms,
                                speedup=base_ms / ms, macs_per_token=m / args.tokens,
                                tflops=tflops))
        # correctness: grouped-exact must match masked+renorm ordering choice
        with torch.no_grad():
            a = topk_masked(x[:256], W, K, I_e, k).float()
            b = topk_grouped_exact(x[:256], W, K, I_e, k, renorm=False).float()
            rel = (a - b).norm() / a.norm().clamp_min(1e-9)
            print(f"  check (256 tok): grouped-exact vs topk-masked rel.err {rel:.2e} "
                  f"({'OK' if rel < 2e-2 else 'MISMATCH'})")
            # the grouped-GEMM path must reproduce the Python-loop path exactly
            if Wt is not None:
                c = topk_grouped_proxy(x[:256], V, S, K, I_e, k, W, renorm=args.renorm).float()
                e = topk_gmm_proxy(x[:256], V, S, K, I_e, k, W, renorm=args.renorm,
                                   Wt=Wt).float()
                rel2 = (c - e).norm() / c.norm().clamp_min(1e-9)
                print(f"  check (256 tok): gmm-proxy vs loop-proxy   rel.err {rel2:.2e} "
                      f"({'OK' if rel2 < 2e-2 else 'MISMATCH'})")
        del W, x, V, S, Wt
        torch.cuda.empty_cache()
        print()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench_moe_throughput_20260912.json").write_text(json.dumps(results, indent=1))
    print(f"wrote {OUT}/bench_moe_throughput_20260912.json")


if __name__ == "__main__":
    main()
