#!/usr/bin/env python3
"""Does the KL-distilled surrogate head work as the SPARSE selector?

CPU, float64, tiny tensors. No GPU, no bsub. Companion to
`test_surrogate_router_20260918.py` (which validated the head in the DENSE path) and to
`test_sparse_forward_20260916.py` (which validated the sparse dispatch under the rank-r proxy).

The wiring is ONE override: `surrogate_replaces_proxy: true` makes the head the module's cheap
all-K router (`_proxy_energies`), so the inherited `_forward_sparse` takes its NOMINATION, its
zscore moments and its denominator tail from the head while the EXACT energies of the nominated
p still re-rank to the final top-k and set every weight. Section (d) of `energy_ff_surrogate.py`
is the design record. Three separate claims, and conflating them is how a wrong number gets
published:

  1. THE DISPATCH IS STILL EXACT AND THE HEAD IS THE ONLY APPROXIMATION. With an ORACLE head
     the sparse output must reproduce the dense output to fp roundoff -- with the Sinkhorn
     tilt, top-k, renormalize_topk and routing_norm all still applied (TESTS 1-3).
  2. NOMINATION RECALL -- the fraction of the true top-k that lands in the head's top-p -- is
     the metric that decides whether the idea works at all. Measured against an oracle head, a
     random head, the OLS-OPTIMAL linear head (the ceiling for `surrogate_kind: linear`),
     trained MLP heads, and the INCUMBENT rank-r subspace proxy on identical data (TEST 4).
  3. WITH THE HEAD DISABLED NOTHING CHANGES: the module is the base class, and every unsafe
     combination is refused at construction rather than silently ignored (TESTS 5-6).

TEST 7 runs the real training loop end to end (the candidate-restricted KL inside
`_forward_sparse`, with exploration), TEST 8 prints the per-token MAC comparison, and TEST 9
repeats the exactness check on the W1W2 expert kind through the sibling module, which needs no
additional code at all.

HONEST SCOPE. These weights are RANDOM. The rank-r proxy's published 0.94 top-1 / 0.90 top-2
depends on TRAINED expert weights being near low-rank (HANDOFF 7.3), so TEST 4's absolute
numbers are a lower bound for BOTH routers and only their RELATIVE standing on identical data is
meaningful here. The real curve needs a checkpoint and a GPU.
"""
import logging
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_surrogate import (
    build_surrogate_boltzmann_moe,
    surrogate_head_macs,
)


H, I_TOT, K, TOPK = 32, 256, 8, 2          # I_e = 32
DT = torch.float64

# The module warns (correctly) whenever `sparse_explore: 0` is combined with sparse selection --
# see (d.1): candidate-restricted distillation is then self-reinforcing. TESTS 1-3 set explore=0
# ON PURPOSE, because exploration perturbs the candidate set and would break the exactness
# comparison, so the warning is expected there and only obscures the table. The HARD refusals
# (which are what protect a real run) are exercised in TEST 6.
for _lg in ("energy_ff_surrogate", "energy_ff_w1w2_sparse"):
    logging.getLogger(
        f"lm_engine.hf_models.modeling_utils.mlp_blocks.{_lg}").setLevel(logging.ERROR)


def mk(*, p=TOPK, sparse=True, replaces=True, rn="none", renorm=True, sink=0,
       explore=0, cf=6.0, seed=0, **kw):
    """A surrogate MoE whose sparse path selects with the head."""
    torch.manual_seed(seed)
    base = dict(
        expert_kind="hopfield", hidden_size=H, intermediate_size=I_TOT, n_experts=K,
        temperature=1.0, top_k=TOPK, initializer_range=0.05, m_width=1.0, add_bias=False,
        fused_experts=True, e_sign_override="pos", routing_norm=rn, renormalize_topk=renorm,
        surrogate_coef=0.0, use_surrogate=False,
        surrogate_replaces_proxy=replaces,
        proxy_mu_convention="pre_mu" if replaces else "legacy",
        proxy_loss_coef=0.01,
        sinkhorn_iters=sink, sinkhorn_persist_mu=sink > 0, sinkhorn_mu_iters=1,
    )
    if sparse:
        base.update(sparse_forward=True, sparse_candidates=p, sparse_explore=explore,
                    sparse_capacity_factor=cf)
    if not replaces:
        base.update(proxy_rank=4, proxy_kind="subspace")
    base.update(kw)
    c = build_surrogate_boltzmann_moe(**base)
    c.double()
    c.eval()
    return c


def exact_E(moe, x):
    """Exact per-expert Hopfield energies, computed independently of every path under test."""
    W = moe._fused_W()
    g = F.gelu(x @ W.t()).view(*x.shape[:-1], moe.n_experts, moe._expert_I)
    return (g * g).mean(-1)


def attach_oracle(moe):
    """The head becomes an ORACLE: it returns the EXACT energies. This isolates the PLUMBING
    from the head's accuracy -- if the sparse path is the dense path with one tensor swapped,
    an oracle head must reproduce the dense output to roundoff."""
    moe.surrogate_energies = lambda xx: exact_E(moe, xx)


def attach_random(moe, seed=7):
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(K, H, generator=g, dtype=DT)
    moe.surrogate_energies = lambda xx: xx @ A.t()


def relerr(a, b):
    d = b.abs().max().item()
    return (a - b).abs().max().item() / (d if d > 0 else 1.0)


torch.manual_seed(123)
x = torch.randn(2, 12, H, dtype=DT)

print(__doc__.strip().split("\n")[0])
print("=" * 94)

# ========================================================================================== #
print("\nTEST 1 -- ORACLE head: surrogate-SELECTED sparse output == dense output.")
print("          Selection is FREE (the head's own top-p), so this pins the whole path:")
print("          nomination, exact-energy re-rank, weights, denominator and dispatch.")
rows = []
for p in (TOPK, 4, K):
    for rn in ("none", "zscore"):
        for renorm in (True, False):
            kw = {} if (renorm or p == K) else {"surrogate_sparse_allow_proxy_denominator": True}
            c = mk(p=p, rn=rn, renorm=renorm, **kw)
            m = c.moe
            attach_oracle(m)
            with torch.no_grad():
                dense = m._forward_fused(x).clone()
                sp = m(x).clone()
            rows.append((p, rn, renorm, relerr(sp, dense), int(m._sparse_overflow)))
print(f"   {'p':>3} {'routing_norm':>13} {'renorm':>7} {'rel err':>11} {'overflow':>9}  verdict")
ok1 = True
for p, rn, renorm, e, ov in rows:
    good = e < 1e-13 and ov == 0
    ok1 &= good
    print(f"   {p:>3} {rn:>13} {str(renorm):>7} {e:>11.3e} {ov:>9}  "
          f"{'EXACT' if good else 'NOT EXACT'}")
print(f"  -> dispatch+weighting exact under an oracle head in every combination: {ok1}")

# ========================================================================================== #
print("\nTEST 2 -- same, with the SINKHORN dual on (iters=3, persist, mu_iters=1).")
print("          The dual is solved on the HEAD's all-K logits; with an oracle head those are")
print("          the exact logits, so the tilt must be identical on both paths.")
c = mk(p=4, sink=3, rn="zscore")
m = c.moe
attach_oracle(m)
m.train()                                  # training solves the dual; also fills the buffer
with torch.no_grad():
    _ = m(x)
m.eval()
with torch.no_grad():
    dense = m._forward_fused(x).clone()
    sp = m(x).clone()
e2 = relerr(sp, dense)
print(f"  mu running absmax = {float(m.sinkhorn_mu.abs().max()):.4f}   "
      f"mu_count = {float(m.sinkhorn_mu_count.sum()):.0f}")
print(f"  relative error = {e2:.3e}   EXACT: {e2 < 1e-13}")

# ========================================================================================== #
print("\nTEST 3 -- a RANDOM head on the same shapes: the error is now the head's, not the")
print("          dispatch's. Bounds how much the SELECTION alone can cost.")
for p in (TOPK, 4, 6, K):
    c = mk(p=p, rn="zscore", renorm=True)
    m = c.moe
    attach_random(m)
    with torch.no_grad():
        dense = m._forward_fused(x).clone()
        sp = m(x).clone()
    print(f"   p={p:<2}  rel err vs dense = {relerr(sp, dense):.3e}"
          f"{'   (p=K: every energy exact -> must be 0)' if p == K else ''}")

# ========================================================================================== #
print("\nTEST 4 -- NOMINATION RECALL: fraction of the true top-k that lands in the head's")
print("          top-p, as a function of p. THIS IS THE NUMBER THAT DECIDES THE IDEA.")
print("          K=16, k=2, d=64, I_e=128, RANDOM expert weights (see HONEST SCOPE).")

Kr, kr, dr, Ie = 16, 2, 64, 128
torch.manual_seed(11)
Wr = torch.randn(Kr, Ie, dr, dtype=DT) * 0.05          # the expert bank


def energies(xx):
    g = F.gelu(torch.einsum("nd,kid->nki", xx, Wr))
    return (g * g).mean(-1)                             # (N, K) -- the module's hopfield form


def zlogits(E):                                         # routing_norm="zscore", e_sign="pos"
    return (E - E.mean(-1, keepdim=True)) / E.std(-1, keepdim=True).clamp_min(1e-12)


def recall(pred_E, tgt_E, ps, k=kr):
    """|top-k(exact) INTERSECT top-p(pred)| / k, averaged over tokens. The same set-overlap
    estimator `_proxy_step` and `_add_surrogate_loss` use (HANDOFF 7.9: `torch.isin` fakes it)."""
    tk = zlogits(tgt_E).topk(k, dim=-1).indices
    out = {}
    for p in ps:
        tp = zlogits(pred_E).topk(p, dim=-1).indices
        hit = (tk.unsqueeze(-1) == tp.unsqueeze(-2)).any(-1).double().sum(-1)
        out[p] = (hit / k).mean().item()
    return out


def make_x(n, gen, q=None):
    if q is None:
        return torch.randn(n, dr, generator=gen, dtype=DT)
    A = torch.randn(dr, q, generator=torch.Generator().manual_seed(5), dtype=DT) / math.sqrt(q)
    return torch.randn(n, q, generator=gen, dtype=DT) @ A.t()


def fit_kl(module, xtr, Etr, steps=800, lr=3e-2):
    """Forward KL(router || head) on the z-scored logits -- the module's own objective."""
    tgt = F.softmax(zlogits(Etr), dim=-1)
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    for _ in range(steps):
        opt.zero_grad()
        pred = module(xtr)
        loss = F.kl_div(F.log_softmax(zlogits(pred), dim=-1), tgt, reduction="batchmean")
        loss.backward()
        opt.step()
    return float(loss)


class Lin(nn.Module):
    def __init__(s):
        super().__init__(); s.l = nn.Linear(dr, Kr).double()

    def forward(s, xx):
        return s.l(xx)


class Mlp(nn.Module):
    def __init__(s, h):
        super().__init__()
        s.a = nn.Linear(dr, h).double(); s.b = nn.Linear(h, Kr).double()

    def forward(s, xx):
        return s.b(F.gelu(s.a(xx)))


class Subspace(nn.Module):
    """The INCUMBENT: `_proxy_energies(proxy_kind="subspace")` -- the exact energy form on a
    rank-r projection, mean(gelu(B_k V_k^T x)^2)*scale + bias. Same arithmetic as energy_ff.py."""

    def __init__(s, r, m):
        super().__init__()
        g = torch.Generator().manual_seed(3)
        s.V = nn.Parameter(torch.randn(Kr, dr, r, generator=g, dtype=DT) / math.sqrt(dr))
        s.B = nn.Parameter(torch.randn(Kr, m, r, generator=g, dtype=DT) / math.sqrt(r))
        s.sc = nn.Parameter(torch.ones(Kr, dtype=DT))
        s.bi = nn.Parameter(torch.zeros(Kr, dtype=DT))

    def forward(s, xx):
        a = torch.einsum("nd,kdr->nkr", xx, s.V)
        z = torch.einsum("nkr,kmr->nkm", a, s.B)
        gz = F.gelu(z)
        return (gz * gz).mean(-1) * s.sc + s.bi


PS = (2, 3, 4, 6, 8)
for tag, q in (("isotropic x", None), ("x on a rank-8 subspace", 8)):
    gtr = torch.Generator().manual_seed(21)
    gte = torch.Generator().manual_seed(22)
    xtr, xte = make_x(4096, gtr, q), make_x(2048, gte, q)
    Etr, Ete = energies(xtr), energies(xte)

    print(f"\n  --- {tag} ---")
    pred = {"ORACLE (exact energies)": Ete}
    torch.manual_seed(31)
    pred["random head (untrained linear)"] = torch.randn(2048, Kr, dtype=DT)
    # OLS-optimal linear head: a CEILING for surrogate_kind="linear" under squared error on the
    # z-scored logits. NOT a ceiling for the KL objective, which optimises ranking -- and the
    # distilled linear head below duly beats it on the structured distribution.
    Xa = torch.cat([xtr, torch.ones(xtr.shape[0], 1, dtype=DT)], 1)
    Wls = torch.linalg.lstsq(Xa, zlogits(Etr)).solution
    pred["linear head, OLS (sq-err ceiling)"] = \
        torch.cat([xte, torch.ones(xte.shape[0], 1, dtype=DT)], 1) @ Wls
    for name, mod in (("linear head, KL-distilled", Lin()),
                      ("mlp head h=64, KL-distilled", Mlp(64)),
                      ("mlp head h=256, KL-distilled", Mlp(256)),
                      ("rank-8 subspace proxy m=32", Subspace(8, 32)),
                      ("rank-8 subspace proxy m=I_e", Subspace(8, Ie))):
        fit_kl(mod, xtr, Etr)
        with torch.no_grad():
            pred[name] = mod(xte)
    for kk, lbl in ((kr, f"recall of the true TOP-{kr} (k = top_k)"),
                    (1, "recall of the true TOP-1 (the damaging miss)")):
        print(f"   {lbl}")
        print(f"   {'selector':<34}" + "".join(f"  p={p:<5}" for p in PS))
        for name, pe in pred.items():
            r = recall(pe, Ete, PS, k=kk)
            print(f"   {name:<34}" + "".join(f"  {r[p]:.3f} " for p in PS))
        print(f"   {'chance (random p-subset) = p/K':<34}"
              + "".join(f"  {p / Kr:.3f} " for p in PS))

# ========================================================================================== #
print("\nTEST 5 -- FALLBACK: with the head disabled nothing changes.")
c = mk(sparse=False, replaces=False)
m = c.moe
torch.manual_seed(0)        # mk() seeds internally; the twin must see the same stream
cb = build_boltzmann_moe(
    expert_kind="hopfield", hidden_size=H, intermediate_size=I_TOT, n_experts=K,
    temperature=1.0, top_k=TOPK, initializer_range=0.05, m_width=1.0, add_bias=False,
    fused_experts=True, e_sign_override="pos", routing_norm="none", renormalize_topk=True,
    proxy_rank=4, proxy_kind="subspace", proxy_loss_coef=0.01, proxy_mu_convention="legacy")
cb.double(); cb.eval()
torch.manual_seed(1)
with torch.no_grad():
    a = m(x).clone()
torch.manual_seed(1)
with torch.no_grad():
    b = cb.moe(x).clone()
print(f"  surrogate_replaces_proxy=False, dense: max|diff| vs base class = "
      f"{(a - b).abs().max().item():.3e}  BITWISE: {torch.equal(a, b)}")
print("  _proxy_energies resolves to:",
      "the rank-r proxy" if not m.surrogate_replaces_proxy else "the head")
# and with the flag ON, the SAME construction must resolve to the head instead
c2 = mk(sparse=False, replaces=True, proxy_rank=4, proxy_kind="subspace",
        surrogate_free_proxy=False)
m2 = c2.moe
with torch.no_grad():
    E_head = m2.surrogate_energies(x)
    E_got = m2._proxy_energies(x)
print(f"  surrogate_replaces_proxy=True:  _proxy_energies IS the head: "
      f"{torch.equal(E_head, E_got)}   "
      f"(rank-r tensors kept as parameters: {not m2._surr_proxy_freed})")
c3 = mk(sparse=False, replaces=True)
print(f"  surrogate_free_proxy default: proxy params in model.parameters() = "
      f"{[n for n, _ in c3.moe.named_parameters() if n.startswith('proxy')]}")

# ========================================================================================== #
print("\nTEST 6 -- every unsafe combination is REFUSED at construction, not ignored.")
cases = [
    ("sparse_forward without surrogate_replaces_proxy",
     dict(sparse=True, replaces=False)),
    ("surrogate_replaces_proxy + use_surrogate (would be a Switch gate)",
     dict(sparse=True, replaces=True, use_surrogate=True)),
    ("surrogate_replaces_proxy + proxy_mu_convention: legacy",
     dict(sparse=True, replaces=True, proxy_mu_convention="legacy")),
    ("surrogate_replaces_proxy + proxy_iters: 2",
     dict(sparse=True, replaces=True, proxy_iters=2)),
    ("sparse + renormalize_topk: false, p < K, no explicit opt-in",
     dict(sparse=True, replaces=True, renorm=False)),
    ("sparse_backproj with the head",
     dict(sparse=False, replaces=True, sparse_backproj=True)),
]
for name, kw in cases:
    try:
        mk(**kw)
        print(f"  {name:<62}  NOT REFUSED  <-- BUG")
    except AssertionError as e:
        print(f"  {name:<62}  refused: {str(e).split('.')[0][:44]}")

# ========================================================================================== #
print("\nTEST 7 -- the real loop: 400 Adam steps of the CANDIDATE-RESTRICTED KL inside")
print("          _forward_sparse (with exploration), through the actual module.")
c = mk(p=4, rn="zscore", renorm=True, explore=2, sink=0, seed=0, surrogate_kind="mlp",
       surrogate_hidden=64, proxy_loss_coef=1.0, cf=2.0)
m = c.moe
m.train()
xtr = torch.randn(64, 16, H, dtype=DT)
head_params = [p for n, p in m.named_parameters() if n.startswith("surrogate_")]
opt = torch.optim.Adam(head_params, lr=3e-3)
# The EXPERT weights are owned by the container's holder (make_experts hands each expert a
# closure over a slice), so `moe.named_parameters()` alone would find nothing to check.
frozen = {n: p.detach().clone() for n, p in c.named_parameters()
          if "surrogate_" not in n}
assert frozen, "nothing to check -- the non-head parameter list came out empty"


def sel_agree():
    mm = m.pop_load_metrics()
    return mm.get("surrogate_sel_agree"), mm.get("proxy_topk_agree")


for step in range(401):
    opt.zero_grad()
    clear_aux_loss()
    out = m(xtr)
    aux = get_aux_loss()
    aux.backward()
    opt.step()
    if step % 100 == 0:
        a, b = sel_agree()
        print(f"   step {step:>4}  candidate-restricted KL = {float(aux):.5f}   "
              f"surrogate_sel_agree = {a if a is None else round(a, 4)}  "
              f"(== proxy_topk_agree: {a == b})")
drift = max((p.detach() - frozen[n]).abs().max().item()
            for n, p in c.named_parameters() if n in frozen)
print(f"  max drift of every NON-head parameter = {drift:.3e}  "
      f"(the head is distilled, the model is not reshaped: {drift == 0.0})")

# ========================================================================================== #
print("\nTEST 8 -- PER-TOKEN MAC COST of the SELECTION, head vs rank-r proxy.")
print("          The rank-r proxy's cost carries I_e; the head's does not. Selection only --")
print("          the sparse mixture itself is listed for scale.")
for tag, d, Kx, kx, Ie_x, r in (("134M hybrid", 768, 16, 2, 1024, 16),
                                ("400M hybrid", 1024, 32, 2, 5871, 16)):
    mix = int(2 * 1.25 * kx * d * Ie_x)                 # cf=1.25, p=k, fwd+back projection
    ent = [
        ("exact all-K router (= dense fwd projection)", Kx * d * Ie_x),
        (f"subspace proxy m=512  (K*d*r + K*m*r)", Kx * d * r + Kx * min(512, Ie_x) * r),
        (f"subspace proxy m=I_e  (K*d*r + K*I_e*r)", Kx * d * r + Kx * Ie_x * r),
        ("surrogate head, linear      (d*K)", surrogate_head_macs(d, Kx, "linear")),
        ("surrogate head, mlp h=64    (d*h + h*K)", surrogate_head_macs(d, Kx, "mlp", 64)),
        ("surrogate head, mlp h=256   (d*h + h*K)", surrogate_head_macs(d, Kx, "mlp", 256)),
    ]
    print(f"\n  {tag}: d={d}, K={Kx}, k={kx}, I_e={Ie_x}, r={r}")
    print(f"   {'selector':<44} {'MACs/token':>13} {'x under exact':>14} {'% of mixture':>13}")
    for nm, v in ent:
        print(f"   {nm:<44} {v:>13,} {ent[0][1]/v:>14.1f} {100.0*v/mix:>12.3f}%")
    print(f"   {'(sparse mixture at p=k, cf=1.25: 2*cf*k*d*I_e)':<44} {mix:>13,}")
    print(f"   head parameters per block: linear {d*Kx + Kx:,}, mlp h=256 "
          f"{d*256 + 256 + 256*Kx + Kx:,}")

# ========================================================================================== #
print("\nTEST 9 -- W1W2 SPARSE, the second expert kind, with ZERO additional code.")
print("          SurrogateBoltzmannMoEW1W2 = the same mixin over the sibling's")
print("          BoltzmannMoEW1W2Sparse, whose _forward_sparse step 1 also calls")
print("          self._proxy_energies -- so the override lands there unchanged.")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_surrogate import (  # noqa: E402
    SurrogateBoltzmannMoEW1W2, _HAVE_W1W2,
)
print(f"  sibling module importable: {_HAVE_W1W2}")


def mk_w1w2(*, p=TOPK, rn="none", renorm=True, sink=0, cf=6.0, seed=0, **kw):
    torch.manual_seed(seed)
    c = build_surrogate_boltzmann_moe(
        expert_kind="w1w2", hidden_size=H, intermediate_size=I_TOT, n_experts=K,
        temperature=1.0, top_k=TOPK, initializer_range=0.05, m_width=1.0, add_bias=False,
        fused_experts=True, e_sign_override="neg", routing_norm=rn, renormalize_topk=renorm,
        surrogate_coef=0.0, use_surrogate=False, surrogate_replaces_proxy=True,
        proxy_mu_convention="pre_mu", proxy_loss_coef=0.01, proxy_kind="subspace",
        sinkhorn_iters=sink, sinkhorn_persist_mu=sink > 0, sinkhorn_mu_iters=1,
        sparse_forward=True, sparse_candidates=p, sparse_explore=0,
        sparse_capacity_factor=cf, **kw)
    c.double(); c.eval()
    return c


def exact_E_w1w2(moe, x):
    """E_k = -I_e^-0.5 * sum_j gelu((W1_k x)_j) * (W2_k x)_j -- W1W2FFEnergy's form."""
    W1, W2, W1v, W2v = moe._fused_views()
    z1 = torch.einsum("...d,kid->...ki", x, W1v)
    z2 = torch.einsum("...d,kid->...ki", x, W2v)
    return -(moe._expert_I ** -0.5) * (F.gelu(z1) * z2).sum(-1)


c = mk_w1w2()
print("  built:", type(c.moe).__name__,
      "| proxy params freed:", c.moe._surr_proxy_freed,
      "| proxy_rank autoset:", c.moe._surr_proxy_rank_autoset)
ok9 = True
for p in (TOPK, 4, K):
    for rn in ("none", "zscore"):
        c = mk_w1w2(p=p, rn=rn)
        m = c.moe
        m.surrogate_energies = lambda xx, _m=m: exact_E_w1w2(_m, xx)
        with torch.no_grad():
            dense = m._forward_fused(x).clone()
            sp = m(x).clone()
        e = relerr(sp, dense)
        good = e < 1e-13 and int(m._sparse_overflow) == 0
        ok9 &= good
        print(f"   p={p:<2} routing_norm={rn:<7} rel err = {e:.3e}  "
              f"{'EXACT' if good else 'NOT EXACT'}")
print(f"  -> the head is EXPERT-KIND-AGNOSTIC in the sparse path too: {ok9}")
print("  NOTE the head's advantage is LARGER here: the w1w2 subspace proxy is forced to")
print("       m = I_e (its m-row subsample has relative error ~sqrt(I_e/m) because the")
print("       bilinear energy's terms cancel), at which point the proxy costs MORE per token")
print("       than the sparse mixture it exists to make cheap. The head's cost has no I_e.")
# RESOLVED 2026-09-19. The claim below ("not registered in get_mlp_block") was half right:
# SurrogateBoltzmannMoEW1W2 was ALREADY reachable, because get_mlp_block's
# EnergyFF_SurrogateBoltzmannMoE branch forwards expert_kind + fused_experts and
# build_surrogate_boltzmann_moe picks the class from that pair. It was the NON-surrogate
# BoltzmannMoEW1W2Sparse that no mlp_type could reach: build_boltzmann_moe emitted
# fused_spec={"kind": "w1w2"} and the base class asserted "hopfield". Both are now reachable --
# the assert accepts "w1w2" and get_mlp_block dispatches EnergyFF_BoltzmannMoE with
# expert_kind: w1w2 + fused_experts: true to build_boltzmann_moe_w1w2_sparse. Ready-to-launch
# config: configs/cmix/cmix_134M_hyb_w1w2_sparse_surr_32B.yml. Checked by
# test_w1w2_sparse_registration_20260919.py.
print("  w1w2 SPARSE ARM IS NOW REACHABLE FROM YAML (2026-09-19):")
print("       mlp_type: EnergyFF_SurrogateBoltzmannMoE + expert_kind: w1w2 + fused_experts:")
print("       true + sparse_forward: true + surrogate_replaces_proxy: true  ->  this class.")
print("       The non-surrogate BoltzmannMoEW1W2Sparse is reached the same way from")
print("       mlp_type: EnergyFF_BoltzmannMoE. See configs/cmix/cmix_134M_hyb_w1w2_sparse_surr_32B.yml.")

print("\n" + "=" * 94)
print("done")
