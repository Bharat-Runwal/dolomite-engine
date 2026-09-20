#!/usr/bin/env python3
"""Is the fully-sparse path SOUND for w1w2 (bilinear) experts, as it is for Hopfield?

CPU, float64, tiny tensors. No GPU, no bsub. Mirrors
`test_sparse_forward_20260916.py` (the Hopfield version) test for test, and adds three
w1w2-specific ones.

The question this file answers is NOT "is the router good" -- that needs trained weights and a
checkpoint. It is "given the selection, is the DISPATCH ARITHMETIC exact, so that the proxy is
the ONLY approximation". Four separate things, and lumping them together is how a wrong number
gets published:

  0. the FUSED DENSE path must equal the LOOPED per-expert path (new here: Hopfield already had
     a validated `_forward_fused`; w1w2 had none, so the reference itself must be proved).
  1. the dispatch + weighting arithmetic UNDERNEATH the selection -- held to bit-exactness by
     forcing the selection to the dense path's own top-k (tests 1, 2).
  2. the softmax denominator, which at renormalize_topk=False runs over K logits of which only
     k were computed (test 2 exact-fed, test 4 proxy-fed).
  3. routing_norm="zscore" moments, which need all K energies (test 3).

TEST 4b is the headline: with an ORACLE proxy and the selection left FREE, the sparse w1w2 path
must reproduce the dense w1w2 output to ~1e-15. TEST 8 is the w1w2-specific negative result.
"""
import hashlib
import inspect
import math

import torch
import torch.nn.functional as F

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import (
    BoltzmannMoEFFEnergy,
    _gelu_and_grad,
)
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_w1w2_sparse import (
    build_boltzmann_moe_w1w2_sparse,
)


def mk(K=8, k=2, cf=4.0, H=64, I=256, seed=0, rn="none", renorm=True, r=8, sf=True, **kw):  # noqa: D103,E501
    torch.manual_seed(seed)
    c = build_boltzmann_moe_w1w2_sparse(
        hidden_size=H, intermediate_size=I, n_experts=K, temperature=1.0, top_k=k,
        e_sign_override="neg", add_bias=False, initializer_range=0.05, m_width=1.0,
        routing_norm=rn, renormalize_topk=renorm, proxy_rank=r,
        sparse_forward=sf, sparse_capacity_factor=cf, **kw)
    c.double()
    c.eval()
    return c


def exact_E(moe, x):
    """The EXACT per-expert w1w2 energies, computed independently of the paths under test."""
    W1, W2, _, _ = moe._fused_views()
    K, I_e = moe.n_experts, moe._expert_I
    z1, z2 = x @ W1.t(), x @ W2.t()
    phi, _ = _gelu_and_grad(z1, moe._fused_spec["gelu_grad_method"])
    lead = z1.shape[:-1]
    return -(I_e ** -0.5) * (phi.view(*lead, K, I_e) * z2.view(*lead, K, I_e)).sum(-1)


def exact_sel(moe, x):
    """The selection and all-K logits the DENSE path would use for this x."""
    logits = moe._logits_raw(exact_E(moe, x)).reshape(-1, moe.n_experts)
    return logits.topk(int(moe.top_k), dim=-1).indices, logits


def relerr(a, b):
    return (a - b).abs().max().item() / b.abs().max().item()


torch.manual_seed(123)
x = torch.randn(2, 12, 64, dtype=torch.float64)

print(__doc__.strip().split("\n")[0])
print("=" * 82)

# ---------------------------------------------------------------------------------------- #
print("\nTEST 0 -- the FUSED DENSE w1w2 path must equal the LOOPED per-expert path.")
print("         Hopfield's `_forward_fused` was already validated; w1w2's is new, so the")
print("         reference every later test compares against has to be proved first.")
for rn, renorm in (("none", True), ("none", False), ("zscore", True), ("sqrt_width", False)):
    c = mk(rn=rn, renorm=renorm)
    m = c.moe
    with torch.no_grad():
        looped = m._forward_looped(x).clone()
        fused = m._forward_fused(x).clone()
    e = relerr(fused, looped)
    print(f"  routing_norm={rn:10s} renorm={str(renorm):5s}  rel err = {e:.3e}"
          f"   {'EXACT' if e < 1e-12 else 'NOT EXACT'}")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 0b -- the energy the fused path routes on is the `_W1W2Expert` energy")
c = mk()
m = c.moe
with torch.no_grad():
    E_ref = torch.stack([e.energy_per_token(x) for e in m.experts], dim=-1)
    e0 = relerr(exact_E(m, x), E_ref)
print(f"  rel err = {e0:.3e}   {'EXACT' if e0 < 1e-12 else 'NOT EXACT'}")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 1 -- selection forced to the dense choice, renormalize_topk=TRUE")
print("         (the sparse path must then reproduce the dense output exactly)")
c = mk(renorm=True)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
rel = relerr(sp, dense)
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
rel2 = relerr(sp, dense)
print(f"  relative error = {rel2:.3e}   EXACT: {rel2 < 1e-12}")

print("\nTEST 3 -- routing_norm=zscore: moments come from the proxy, so NOT exact.")
print("         Selection still forced, so this isolates the moment error alone.")
c = mk(rn="zscore", renorm=True)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel, exact_denominator=lg_all).clone()
print(f"  relative error = {relerr(sp, dense):.3e}  (a per-token rescaling of tau; mean cancels)")

print("\nTEST 4 -- denominator filled from the RANDOM-INIT proxy, selection still forced.")
print("         Upper bound on approximation 2 with a maximally bad proxy.")
c = mk(renorm=False)
m = c.moe
sel, lg_all = exact_sel(m, x)
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m._forward_sparse(x, sel_idx=sel).clone()
print(f"  relative error = {relerr(sp, dense):.3e}")
print("  NOTE renormalize_topk=TRUE removes this term entirely -- prefer it for sparse arms.")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 4b -- *** THE HEADLINE *** ORACLE proxy (returns the EXACT w1w2 energies),")
print("         selection left FREE. Every approximation must vanish: this is what proves")
print("         the dispatch is exact and isolates the proxy as the ONLY approximation.")


def attach_oracle(m):
    def oracle(z, _m=m):
        return exact_E(_m, z)
    m._proxy_energies = oracle


worst = 0.0
for rn, renorm in (("none", True), ("none", False), ("zscore", True), ("zscore", False),
                   ("sqrt_width", True)):
    c = mk(rn=rn, renorm=renorm)
    m = c.moe
    attach_oracle(m)
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()
    e = relerr(sp, dense)
    worst = max(worst, e)
    print(f"  routing_norm={rn:10s} renormalize_topk={str(renorm):5s}  rel err = {e:.3e}"
          f"   {'EXACT' if e < 1e-12 else 'NOT EXACT'}")
print(f"  worst over all five configurations = {worst:.3e}")

print("\nTEST 4c -- OVER-SELECT then RE-RANK. p = K must be EXACT with NO proxy involvement:")
print("         every energy is computed exactly, the denominator has no proxy term and the")
print("         zscore moments are exact. Catches p != k shape bugs instantly.")
for rn, renorm in (("none", False), ("zscore", False), ("zscore", True)):
    c = mk(rn=rn, renorm=renorm, sparse_candidates=8)      # 8 = mk's default n_experts
    m = c.moe
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()
    e = relerr(sp, dense)
    print(f"  routing_norm={rn:6s} renorm={str(renorm):5s} p=K  rel err = {e:.3e}"
          f"   {'EXACT' if e < 1e-12 else 'NOT EXACT'}")

print("\nTEST 4d -- intermediate p: shapes hold and the re-rank picks by EXACT energy")
for pc in (2, 3, 4, 6, 8):
    c = mk(rn="zscore", renorm=False, sparse_candidates=pc)
    m = c.moe
    with torch.no_grad():
        out = m._forward_sparse(x)
    attach_oracle(m)
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m._forward_sparse(x).clone()
    print(f"  p={pc:2d}  out {tuple(out.shape)}  oracle-proxy rel err vs dense = "
          f"{relerr(sp, dense):.3e}")

print("\nTEST 4e -- gradients flow through the sparse path and match the dense path's")
c = mk(renorm=True)
m = c.moe
attach_oracle(m)
gs = []
for fn in (m._forward_fused, m._forward_sparse):
    for pp in c.parameters():
        pp.grad = None
    fn(x).pow(2).sum().backward()
    gs.append(torch.cat([pp.grad.reshape(-1) for pp in c.expert_holder.parameters()]))
print(f"  ||d/dW dense|| = {gs[0].norm():.6e}   ||d/dW sparse|| = {gs[1].norm():.6e}")
print(f"  rel err = {((gs[1]-gs[0]).abs().max()/gs[0].abs().max()).item():.3e}")

# ---------------------------------------------------------------------------------------- #
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

print("\nTEST 6 -- guards")
for name, fn in (
    ("proxy_rank=0 (no selector)", lambda: mk(r=0)),
    ("sparse_backproj + sparse_forward", lambda: mk(sparse_backproj=True)),
    ("sparse_backproj alone (not impl.)", lambda: mk(sparse_backproj=True, sf=False)),
    ("add_bias=True (silently unread)", lambda: build_boltzmann_moe_w1w2_sparse(
        hidden_size=16, intermediate_size=64, n_experts=4, top_k=2, add_bias=True)),
    ("proxy_kind='quad' (no w1w2 form)", lambda: mk(proxy_kind="quad")),
):
    try:
        fn()
        print(f"  {name:34s} NOT REJECTED -- bug")
    except AssertionError as e:
        print(f"  {name:34s} rejected: {str(e)[:50]}")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 7 -- FLOP model. w1w2 pays 4 GEMMs where Hopfield pays 2, so the DENSE cost")
print("         doubles at equal I_e -- but the sparse/dense RATIO is identical, K/(cf*k).")
print(f"  {'shape':26s} {'dense':>10s} {'sparse+proxy':>13s} {'speedup':>9s} {'proxy %':>8s}")
for K, k, I_e, d in ((16, 2, 2240, 768), (32, 2, 256, 768), (16, 2, 640, 1024)):
    r, cf, m_sub = 8, 1.25, 512
    dense_f = 4 * K * d * I_e
    prox_f = 2 * K * r * (d + m_sub)
    fwd_f = 4 * cf * k * d * I_e + prox_f
    print(f"  K={K:2d} k={k} I_e={I_e:5d} d={d:4d} {dense_f/1e6:9.1f}M {fwd_f/1e6:12.1f}M "
          f"{dense_f/fwd_f:8.2f}x {100*prox_f/dense_f:7.2f}%")
print("  NOTE at m = I_e (the only UNBIASED setting -- TEST 8) the proxy's ELEMENTWISE work is")
print("       K*I_e/token, i.e. K/(cf*k) = 6.4x MORE than the sparse mixture's own cf*k*I_e.")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 8 -- *** THE NON-MECHANICAL STEP, MEASURED ***")
print("         `proxy_out_dim` (m of I_e rows) is an UNBIASED, LOW-VARIANCE estimator of the")
print("         HOPFIELD energy and is NOT one for the w1w2 energy. Hopfield sums I_e")
print("         NONNEGATIVE terms (rel err ~ 1/sqrt(m)); w1w2 sums I_e SIGNED terms, which")
print("         cancel to O(sqrt(I_e)) (rel err ~ sqrt(I_e/m)). Same code, opposite verdict.")
torch.manual_seed(7)
I_e, H, N = 2240, 64, 512
W1 = torch.randn(I_e, H, dtype=torch.float64) * 0.05
W2 = torch.randn(I_e, H, dtype=torch.float64) * 0.05
Wh = torch.randn(I_e, H, dtype=torch.float64) * 0.05
xs = torch.randn(N, H, dtype=torch.float64)
u, v, zh = xs @ W1.t(), xs @ W2.t(), xs @ Wh.t()
t_w1w2 = F.gelu(u) * v                          # the w1w2 summands (signed)
t_hop = F.gelu(zh) ** 2                         # the Hopfield summands (nonnegative)
E_w = -(I_e ** -0.5) * t_w1w2.sum(-1)
E_h = t_hop.mean(-1)
print(f"  summand sign structure: w1w2 fraction negative = "
      f"{(t_w1w2 < 0).double().mean():.3f}   hopfield = {(t_hop < 0).double().mean():.3f}")
print(f"  cancellation: |sum| / sum|term|  w1w2 = {(t_w1w2.sum(-1).abs()/t_w1w2.abs().sum(-1)).median():.4f}"
      f"   hopfield = {(t_hop.sum(-1).abs()/t_hop.abs().sum(-1)).median():.4f}"
      f"   (1/sqrt(I_e) = {I_e**-0.5:.4f})")
print(f"  {'m':>6s} {'m/I_e':>7s} {'w1w2 med|relerr|':>17s} {'hopfield med|relerr|':>21s} "
      f"{'pred sqrt(I/m)':>15s} {'pred 1/sqrt(m)':>15s}")
for m_sub in (32, 128, 512, 1120, 2240):
    g = torch.Generator().manual_seed(11)
    rows = torch.randperm(I_e, generator=g)[:m_sub]
    Ew_hat = -(I_e ** 0.5) * t_w1w2[:, rows].mean(-1)
    Eh_hat = t_hop[:, rows].mean(-1)
    ew = ((Ew_hat - E_w).abs() / E_w.abs()).median().item()
    eh = ((Eh_hat - E_h).abs() / E_h.abs()).median().item()
    print(f"  {m_sub:6d} {m_sub/I_e:7.3f} {ew:17.4f} {eh:21.4f} "
          f"{math.sqrt(I_e/m_sub):15.2f} {1/math.sqrt(m_sub):15.4f}")
print("  READ THIS AS: at m=512 the w1w2 subsample's error is ~2x the energy it estimates,")
print("  while the Hopfield subsample's is a few percent. A ROUTER built on the first is noise.")
print("  Consequence: the faithful w1w2 subspace proxy is forced to m = I_e, where it is no")
print("  cheaper elementwise than the dense forward projection (TEST 7's NOTE). m < I_e is a")
print("  LEARNED head that must be distilled, not an estimator an SVD can warm-start exactly.")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 9 -- the w1w2 subspace proxy IS the exact energy restricted to a subspace:")
print("         at r = d (full basis) and m = I_e the SVD warm start must reproduce the exact")
print("         energies. This pins the FUNCTION CLASS as correct, so the only error in a")
print("         real run is the rank truncation -- the same statement HANDOFF 7.6 makes for")
print("         Hopfield, and the reason `subspace` exists instead of a fitted quad head.")
torch.manual_seed(5)
c = build_boltzmann_moe_w1w2_sparse(
    hidden_size=8, intermediate_size=64, n_experts=4, top_k=2, temperature=1.0,
    e_sign_override="neg", initializer_range=0.05, m_width=1.0, proxy_rank=8,
    proxy_kind="subspace", proxy_out_dim=0, sparse_forward=True, renormalize_topk=True)
c.double(); c.eval()
m = c.moe
xx = torch.randn(3, 5, 8, dtype=torch.float64)
with torch.no_grad():
    before = relerr(m._proxy_energies(xx), exact_E(m, xx))
    m._svd_refit_proxy()
    after = relerr(m._proxy_energies(xx), exact_E(m, xx))
print(f"  I_e={m._expert_I} r=d=8 m=I_e   random init rel err = {before:.3e}"
      f"   after SVD warm start = {after:.3e}   {'EXACT' if after < 1e-12 else 'NOT EXACT'}")
print("  ... and truncating the rank degrades it smoothly (r < d, m = I_e):")
for r in (1, 2, 4, 6, 8):
    torch.manual_seed(5)
    cc = build_boltzmann_moe_w1w2_sparse(
        hidden_size=8, intermediate_size=64, n_experts=4, top_k=2, temperature=1.0,
        e_sign_override="neg", initializer_range=0.05, m_width=1.0, proxy_rank=r,
        proxy_kind="subspace", proxy_out_dim=0, sparse_forward=True, renormalize_topk=True)
    cc.double(); cc.eval()
    mm = cc.moe
    with torch.no_grad():
        mm._svd_refit_proxy()
        E_hat, E_tr = mm._proxy_energies(xx), exact_E(mm, xx)
        agree = (E_hat.reshape(-1, 4).topk(2, -1).indices.unsqueeze(-1)
                 == E_tr.reshape(-1, 4).topk(2, -1).indices.unsqueeze(-2)).any(-1).double().mean()
    print(f"    r={r}  energy rel err = {relerr(E_hat, E_tr):.3e}   top-2 set agreement = "
          f"{agree:.3f}  (chance = 0.500)")
print("  (K=4/k=2 makes chance 0.5 -- this is a FUNCTION-CLASS check, NOT a routing-quality")
print("   result. Quality needs trained weights: calibrate_proxy_router_20260916.py.)")

print("\nTEST 9b -- proxy_kind='bilinear' (no I_e axis at all) constructs and warm-starts.")
print("         UNMEASURED for quality by design: it drops the gelu gate, which is the same")
print("         failure mode that made Hopfield's ||Wx||^2 proxy score ~0% top-1.")
torch.manual_seed(5)
cb = build_boltzmann_moe_w1w2_sparse(
    hidden_size=8, intermediate_size=64, n_experts=4, top_k=2, temperature=1.0,
    e_sign_override="neg", initializer_range=0.05, m_width=1.0, proxy_rank=8,
    proxy_kind="bilinear", sparse_forward=True, renormalize_topk=True)
cb.double(); cb.eval()
mb = cb.moe
with torch.no_grad():
    mb._svd_refit_proxy()
    Eb, Et = mb._proxy_energies(xx), exact_E(mb, xx)
    agree = (Eb.reshape(-1, 4).topk(2, -1).indices.unsqueeze(-1)
             == Et.reshape(-1, 4).topk(2, -1).indices.unsqueeze(-2)).any(-1).double().mean()
    sp = mb._forward_sparse(xx)
print(f"  eigh warm start: energy rel err = {relerr(Eb, Et):.3e}   top-2 agreement = {agree:.3f}"
      f"   (chance 0.500)   sparse out {tuple(sp.shape)}")
print("  -> the ENERGY is ~30% off even with an exact rank-d factorisation of W1^T W2 (the")
print("     dropped gate is an irreducible error of this form, not a truncation error), while")
print("     the top-2 SET survives better than the energy does. Both readings are consistent")
print("     with a gate-free head being usable for RANKING and useless for the denominator --")
print("     but K=4 with random weights is a 2-of-4 choice at chance 0.5, so this is a")
print("     SMOKE TEST, not evidence. The real number is top-k agreement at K=16/32 on TRAINED")
print("     weights, which needs calibrate_proxy_router_20260916.py. Do not ship this head on")
print("     the strength of its FLOP count.")

# ---------------------------------------------------------------------------------------- #
print("\nTEST 10 -- DRIFT GUARD on the one block copied out of energy_ff.py")
src = inspect.getsource(BoltzmannMoEFFEnergy._forward_sparse)
h = hashlib.sha256(src.encode()).hexdigest()[:16]
EXPECTED = "813e883ace01f8be"
print(f"  sha256(BoltzmannMoEFFEnergy._forward_sparse)[:16] = {h}")
if h == EXPECTED:
    print("  UNCHANGED since the weighting block was copied -- no drift.")
else:
    print(f"  *** CHANGED (recorded {EXPECTED}). Re-read the COPIED BLOCK in")
    print("      energy_ff_w1w2_sparse.py against energy_ff.py and update this hash. ***")
# 2026-09-19: the base class's assert now accepts "w1w2" (and additionally requires that
# `_forward_fused` really was overridden), so the `_KIND_SHIM` is gone and `fused_spec["kind"]`
# carries the true kind. If either line below prints False, the shim has come back.
src = inspect.getsource(BoltzmannMoEFFEnergy.__init__)
print("  base class accepts fused w1w2:", 'in ("hopfield", "w1w2")' in src)
import importlib
print("  _KIND_SHIM removed from the w1w2 module:", "_KIND_SHIM = " not in inspect.getsource(
    importlib.import_module(build_boltzmann_moe_w1w2_sparse.__module__)))
