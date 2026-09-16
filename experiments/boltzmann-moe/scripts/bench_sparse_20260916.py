#!/usr/bin/env python3
"""Wall-clock cost of the three mixture paths, on the real 134M/400M block shapes.

FLOP arithmetic said 6.4x on the back GEMM and ~6.4-21x on the whole mixture. That is an upper
bound on what the clock will show, and the gap between them is the point: HANDOFF 7.11 measured
GPU-busy 1.71s against 6.75s wall, i.e. this block has been launch-overhead-bound, and the sparse
path adds index/sort/scatter kernels to save GEMM. A per-expert Python loop lost 0.41x that way.

So: eager AND compiled, forward AND forward+backward, since training is what we are paying for.
"""
import argparse
import time

import torch

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe

SHAPES = {
    # name          d     I_total  K   k   iters   what it is
    "pure_T12":   (768,   71680,  16,  2,   12,  "t90k_pure_T12 / iclr pure isoP: I_e=4480"),
    "hyb_K32":    (768,   16384,  32,  2,    6,  "t90k_hybrid_K32top2: I_e=512"),
    "big_hyb":   (1024,   20480,  16,  2,    6,  "400M hybrid scale32B: I_e=1280"),
}


def build(d, I, K, k, mode, r=8, cf=1.25):
    kw = dict(expert_kind="hopfield", hidden_size=d, intermediate_size=I, n_experts=K,
              temperature=1.0, top_k=k, e_sign_override="pos", add_bias=False,
              initializer_range=0.02, m_width=1.0, fused_experts=True,
              routing_norm="zscore", hopfield_grad_scale="sqrt_consistent",
              sparse_capacity_factor=cf)
    if mode == "dense":
        pass
    elif mode == "backproj":
        kw["sparse_backproj"] = True
    elif mode == "forward":
        kw.update(sparse_forward=True, proxy_rank=r, renormalize_topk=True)
    return build_boltzmann_moe(**kw).cuda().to(torch.bfloat16)


def timeit(fn, n=30, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t) / n * 1e3      # ms


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, default=4096, help="micro_batch * sequence_length")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--train", action="store_true", help="time forward+backward")
    args = ap.parse_args()

    print(f"tokens/call = {args.tokens}   compile = {args.compile}   "
          f"mode = {'fwd+bwd' if args.train else 'fwd'}")
    print(f"GPU = {torch.cuda.get_device_name(0)}")
    print()
    hdr = f"{'shape':10s} {'dense':>9s} {'backproj':>9s} {'sp/dn':>7s} {'forward':>9s} {'sp/dn':>7s}"
    print(hdr)
    print("-" * len(hdr))

    for name, (d, I, K, k, iters, what) in SHAPES.items():
        x = torch.randn(args.tokens, d, device="cuda", dtype=torch.bfloat16)
        res = {}
        for mode in ("dense", "backproj", "forward"):
            try:
                m = build(d, I, K, k, mode)
                m.train(args.train)
                f = torch.compile(m) if args.compile else m
                if args.train:
                    xx = x.clone().requires_grad_(True)

                    def call(f=f, xx=xx):
                        out = f(xx)
                        out.sum().backward()
                        xx.grad = None
                else:
                    def call(f=f, x=x):
                        with torch.no_grad():
                            f(x)
                res[mode] = timeit(call)
                del m, f
                torch.cuda.empty_cache()
            except Exception as e:                       # noqa: BLE001
                res[mode] = float("nan")
                print(f"  !! {name}/{mode}: {type(e).__name__}: {str(e)[:90]}")
        dn = res["dense"]
        print(f"{name:10s} {dn:8.3f}m {res['backproj']:8.3f}m "
              f"{dn/res['backproj']:6.2f}x {res['forward']:8.3f}m {dn/res['forward']:6.2f}x")
        print(f"{'':10s} x{iters} iters/step -> per-step mixture: dense {dn*iters:.1f} ms, "
              f"backproj {res['backproj']*iters:.1f} ms, forward {res['forward']*iters:.1f} ms"
              f"   [{what}]")


if __name__ == "__main__":
    main()
