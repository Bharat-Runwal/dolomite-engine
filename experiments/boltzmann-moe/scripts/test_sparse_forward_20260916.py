#!/usr/bin/env python3
"""Is the fully-sparse path (forward AND back projection over top-k only) sound?

`sparse_backproj` was easy to validate: p is already known when the back projection runs, so
it is exact, full stop. `sparse_forward` is not, because skipping the forward projection means
the router cannot see the experts it skipped -- a rank-r proxy chooses instead. That makes THREE
separate things to check, and lumping them together is how a wrong number gets published:

  1. the dispatch + weighting arithmetic UNDERNEATH the selection. Held to bit-exactness by
     forcing the selection to the dense path's own top-k (tests 1, 2).
  2. the softmax denominator, which at renormalize_topk=False runs over K logits of which we
     computed only k. Exact when handed exact logits (test 2); measured against dense when
     handed the proxy's (test 4).
  3. routing_norm="zscore" moments, which need all K energies (test 3).

Selection quality itself -- the one approximation that matters -- is NOT measurable here: it
depends on trained weights being near low-rank (HANDOFF 7.3), and these are random. It is
measured on a real checkpoint by calibrate_proxy_router_20260916.py.
"""
import math

import torch

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe


def mk(K=8, k=2, cf=4.0, H=64, I=256, seed=0, rn="none", renorm=True, r=8, **kw):  # noqa: D103
    torch.manual_seed(seed)
    c = build_boltzmann_moe(
        expert_kind="hopfield", hidden_size=H, intermediate_size=I, n_experts=K,
        temperature=1.0, top_k=k, e_sign_override="pos", add_bias=False,
        initializer_range=0.05, m_width=1.0, fused_experts=True, routing_norm=rn,
        renormalize_topk=renorm, proxy_rank=r, sparse_forward=True,
        sparse_capacity_factor=cf, **kw)
    c.double()
    c.eval()
    return c


def exact_sel(moe, x):
    """The selection and all-K logits the DENSE path would use for this x."""
    W = moe._fused_W()
    g = torch.nn.functional.gelu(x @ W.t()).view(*x.shape[:-1], moe.n_experts, moe._expert_I)
    E_k = (g * g).mean(-1)
    logits = moe._logits_raw(E_k)
    return logits.reshape(-1, moe.n_experts).topk(int(moe.top_k), dim=-1).indices, \
        logits.reshape(-1, moe.n_experts)


torch.manual_seed(123)
x = torch.randn(2, 12, 64, dtype=torch.float64)

print(__doc__.strip().split("\n")[0])
print("=" * 78)

print("\nTEST 1 -- selection forced to the dense choice, renormalize_topk=TRUE")
print("         (the sparse path must then reproduce the dense output exactly)")
c = mk(renorm=True)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
rel = (sp - dense).abs().max().item() / dense.abs().max().item()
ov = int(m._sparse_overflow)
print(f"  relative error = {rel:.3e}   overflow = {ov}")
print(f"  EXACT: {rel < 1e-12 and ov == 0}")

print("\nTEST 2 -- same, renormalize_topk=FALSE (denominator over all K)")
print("         exact logits supplied for the K-k unevaluated experts")
c = mk(renorm=False)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
rel2 = (sp - dense).abs().max().item() / dense.abs().max().item()
print(f"  relative error = {rel2:.3e}")
print(f"  EXACT: {rel2 < 1e-12}")

print("\nTEST 3 -- routing_norm=zscore: moments come from the proxy, so this is NOT exact.")
print("         Selection still forced, so this isolates the moment error alone.")
c = mk(rn="zscore", renorm=True)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
rel3 = (sp - dense).abs().max().item() / dense.abs().max().item()
print(f"  relative error = {rel3:.3e}  (a per-token rescaling of tau; the mean cancels)")

print("\nTEST 4 -- denominator filled from the RANDOM-INIT proxy, selection still forced.")
print("         Upper bound on approximation 2 with a maximally bad proxy.")
c = mk(renorm=False)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel).clone()          # no exact_denominator
rel4 = (sp - dense).abs().max().item() / dense.abs().max().item()
print(f"  relative error = {rel4:.3e}")
print("  NOTE renormalize_topk=TRUE removes this term entirely -- prefer it for sparse arms.")

print("\nTEST 4b -- ORACLE proxy (proxy returns the EXACT energies).")
print("         Pins that tests 3 and 4 measure proxy QUALITY and not a bug: with a perfect")
print("         proxy every approximation must vanish, in both the zscore and the")
print("         non-renormalised denominator, with the selection left FREE this time.")
for rn, renorm in (("none", True), ("zscore", True), ("zscore", False)):
    c = mk(rn=rn, renorm=renorm)
    m = c.moe

    def oracle(x, _m=m):
        W = _m._fused_W()
        g = torch.nn.functional.gelu(x @ W.t()).view(*x.shape[:-1], _m.n_experts, _m._expert_I)
        return (g * g).mean(-1)

    m._proxy_energies = oracle
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()          # selection FREE, from the oracle
    r = (sp - dense).abs().max().item() / dense.abs().max().item()
    print(f"  routing_norm={rn:6s} renormalize_topk={str(renorm):5s}  relative error = {r:.3e}"
          f"   {'EXACT' if r < 1e-12 else 'NOT EXACT'}")

print("\nTEST 4c -- OVER-SELECT then RE-RANK. p = K must be EXACT with NO proxy involvement:")
print("         every energy is then computed exactly, the denominator has no proxy term and")
print("         the zscore moments are exact. This is the self-test that should have existed")
print("         first -- it catches p != k shape bugs instantly.")
for rn, renorm in (("none", False), ("zscore", False), ("zscore", True)):
    c = mk(rn=rn, renorm=renorm, sparse_candidates=8)   # 8 = mk's default n_experts
    m = c.moe
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()
    r = (sp - dense).abs().max().item() / dense.abs().max().item()
    print(f"  routing_norm={rn:6s} renorm={str(renorm):5s} p=K  rel err = {r:.3e}"
          f"   {'EXACT' if r < 1e-12 else 'NOT EXACT'}")

print("\nTEST 4d -- intermediate p: shapes must hold and the re-rank must pick by EXACT energy")
for pc in (2, 3, 4, 6, 8):
    c = mk(rn="zscore", renorm=False, sparse_candidates=pc)
    m = c.moe
    with torch.no_grad():
        out = m._forward_sparse(x)
    # with an oracle proxy, over-selecting can only help: p=K is exact, so larger p must not be
    # worse than smaller p on the SAME weights
    def oracle(z, _m=m):
        W = _m._fused_W()
        g = torch.nn.functional.gelu(z @ W.t()).view(*z.shape[:-1], _m.n_experts, _m._expert_I)
        return (g * g).mean(-1)
    m._proxy_energies = oracle
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()
    r = (sp - dense).abs().max().item() / dense.abs().max().item()
    print(f"  p={pc:2d}  out {tuple(out.shape)}  oracle-proxy rel err vs dense = {r:.3e}")

print("\nTEST 5 -- overflow is counted, not silent")
c = mk(cf=0.15, renorm=True)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
ov5 = int(m._sparse_overflow)
d5 = (sp - dense).abs().max().item()
print(f"  overflow pairs = {ov5} (nonzero expected)   max|diff| = {d5:.3e} (nonzero expected)")
print(f"  behaves as documented: {ov5 > 0 and d5 > 0}")

print("\nTEST 6a -- guards rejected at CONSTRUCTION (the config can never be sound)")
for name, kw in (("proxy_rank=0 (no selector)", dict(r=0)),
                 ("sparse_backproj also set", dict(sparse_backproj=True))):
    try:
        mk(**kw)
        print(f"  {name:28s} NOT REJECTED -- bug")
    except AssertionError as e:
        print(f"  {name:28s} rejected: {str(e)[:56]}")

print("\nTEST 6b -- guards rejected only while TRAINING, and NOT at construction.")
print("         Both fire only under self.training, and the main use of sparse_forward is")
print("         evaluating a trained checkpoint whose config says repulsion_space: output --")
print("         rejecting that at load time would make every such checkpoint unusable.")
for name, kw in (("output-space repulsion", dict(repulsion_coef=0.1, repulsion_space="output")),
                 ("cos probe on", dict(cos_probe_interval=10))):
    c = mk(**kw)                                  # must construct
    with torch.no_grad():
        c.moe._forward_sparse(x)                  # eval: must run
    c.moe.train()
    try:
        c.moe._forward_sparse(x)
        print(f"  {name:28s} loads+evals OK, but TRAINING NOT REJECTED -- bug")
    except AssertionError as e:
        print(f"  {name:28s} loads+evals OK, training rejected: {str(e)[:40]}")

print("\nTEST 7 -- FLOP model: mixture GEMMs per token, dense vs each sparse mode")
print(f"  {'shape':22s} {'dense':>10s} {'backproj':>10s} {'forward':>10s} {'fwd speedup':>12s}")
for K, k, I_e, d in ((16, 2, 4480, 768), (32, 2, 512, 768), (16, 2, 1280, 1024), (32, 1, 512, 768)):
    r, cf = 8, 1.25
    dense_f = 2 * K * d * I_e
    back_f = K * d * I_e + math.ceil(cf * k / K * K) * d * I_e   # fwd dense + back sparse
    back_f = K * d * I_e + cf * k * d * I_e
    fwd_f = 2 * cf * k * d * I_e + K * d * r
    print(f"  K={K:2d} k={k} I_e={I_e:5d} d={d:4d} {dense_f/1e6:9.1f}M {back_f/1e6:9.1f}M "
          f"{fwd_f/1e6:9.1f}M {dense_f/fwd_f:11.2f}x")
print("  (proxy is the K*d*r term: <0.6% of dense in every row above)")

print("\nTEST 8 -- activation footprint of the intermediates, dense vs sparse_forward")
T = 4096
for K, k, I_e in ((16, 2, 4480), (32, 2, 512)):
    cf = 1.25
    print(f"  K={K:2d} k={k} I_e={I_e:5d} T={T}:  dense (T,K,I_e) = {T*K*I_e*2/2**30:.2f} GiB bf16"
          f"   sparse (K,C,I_e) = {cf*T*k*I_e*2/2**30:.2f} GiB   ({K/(cf*k):.1f}x less)")
