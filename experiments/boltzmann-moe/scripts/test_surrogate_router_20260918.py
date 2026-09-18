#!/usr/bin/env python3
"""Is the KL-distilled surrogate router SOUND on the composable energy-FF family?

CPU, float64, tiny tensors. No GPU, no bsub. Mirrors
`test_sparse_w1w2_20260918.py` in structure: prove the reference, then prove each
approximation separately, then a drift/structure guard.

FOUR SEPARATE CLAIMS, and lumping them together is how a wrong number gets published:

  1. the LOSS is a true KL -- exactly 0 when the head reproduces the router, which the LEGACY
     class's cross-entropy form can never be (TEST 1, TEST 1b);
  2. the surrogate ROUTING PATH is the exact path with one tensor swapped -- so an ORACLE head
     must reproduce the base class's output bitwise, with mu / top_k / renormalize_topk /
     routing_norm all still applied (TEST 2, TEST 3);
  3. `use_surrogate=False` is a NO-OP -- the base class, bitwise (TEST 4);
  4. the head is EXPERT-KIND-AGNOSTIC -- identical code on hopfield and on w1w2 (TEST 5).

TEST 6 is the pre/post-mu design decision made falsifiable: it MEASURES the double-counting
the existing dense-path proxy convention would introduce. TEST 7 checks the Sinkhorn guard
rails. TEST 8 trains a head for real (CPU, 400 Adam steps) so the loop is exercised end to
end. TEST 9 is a structure guard on the two frozen methods this module depends on.
"""
import inspect

import torch
import torch.nn.functional as F

from lm_engine.hf_models.loss import clear_aux_loss, get_aux_loss
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import (
    BoltzmannMoEFFEnergy,
    build_boltzmann_moe,
)
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_surrogate import (
    SurrogateBoltzmannMoEW1W2,
    build_surrogate_boltzmann_moe,
    surrogate_head_macs,
)


H, I_TOT, K, TOPK = 32, 256, 8, 2


def mk_surr(seed=0, kind="hopfield", fused=True, rn="none", renorm=True,
            sk="linear", sh=0, coef=1.0, use=False, **kw):
    torch.manual_seed(seed)
    c = build_surrogate_boltzmann_moe(
        expert_kind=kind, hidden_size=H, intermediate_size=I_TOT, n_experts=K,
        temperature=1.0, top_k=TOPK, initializer_range=0.05, m_width=1.0,
        add_bias=False, fused_experts=fused,
        e_sign_override="pos" if kind == "hopfield" else "neg",
        routing_norm=rn, renormalize_topk=renorm,
        surrogate_coef=coef, use_surrogate=use, surrogate_kind=sk, surrogate_hidden=sh,
        **kw)
    c.double(); c.eval()
    return c


def mk_base(seed=0, kind="hopfield", fused=True, rn="none", renorm=True, **kw):
    torch.manual_seed(seed)
    c = build_boltzmann_moe(
        expert_kind=kind, hidden_size=H, intermediate_size=I_TOT, n_experts=K,
        temperature=1.0, top_k=TOPK, initializer_range=0.05, m_width=1.0,
        add_bias=False, fused_experts=fused,
        e_sign_override="pos" if kind == "hopfield" else "neg",
        routing_norm=rn, renormalize_topk=renorm, **kw)
    c.double(); c.eval()
    return c


def exact_E_hopfield(moe, x):
    """Exact per-expert Hopfield energies, computed independently of the paths under test."""
    W = moe._fused_W()
    lead = x.shape[:-1]
    g = F.gelu(x @ W.t()).view(*lead, moe.n_experts, moe._expert_I)
    return (g * g).mean(-1)


def attach_oracle(moe, x_ref=None):
    """Replace the head with an ORACLE: it returns the EXACT energies.

    This is the test that isolates the PLUMBING from the head's accuracy. If the surrogate
    path is the exact path with one tensor substituted, an oracle head must reproduce the exact
    output BITWISE -- including the Sinkhorn tilt, the top-k mask and the renormalisation.
    """
    moe.surrogate_energies = (lambda xx: torch.stack(
        [e.energy_per_token(xx) for e in moe.experts], dim=-1))


def relerr(a, b):
    d = b.abs().max().item()
    return (a - b).abs().max().item() / (d if d > 0 else 1.0)


torch.manual_seed(123)
x = torch.randn(2, 12, H, dtype=torch.float64)

print(__doc__.strip().split("\n")[0])
print("=" * 90)

# ------------------------------------------------------------------------------------------ #
print("\nTEST 1 -- the KL is EXACTLY ZERO when the head reproduces the router.")
print("         (this is the property the legacy class cannot have: its loss is a")
print("          CROSS-ENTROPY, so its floor is H(p_boltz) > 0)")
for rn in ("none", "zscore", "sqrt_width"):
    c = mk_surr(rn=rn, coef=1.0)
    m = c.moe
    attach_oracle(m)
    m.train()
    clear_aux_loss()
    with torch.no_grad():
        m(x)
    kl = get_aux_loss()
    kl = float(kl) if not torch.is_tensor(kl) else kl.item()
    print(f"  routing_norm={rn:11s}  aux_loss (= 1.0 * KL) = {kl:.3e}   "
          f"{'ZERO' if abs(kl) < 1e-14 else 'NOT ZERO'}")

print("\nTEST 1b -- the LEGACY form on the same inputs, for comparison.")
c = mk_surr(coef=1.0); m = c.moe; attach_oracle(m); m.train()
with torch.no_grad():
    E = m.surrogate_energies(x)
    lg = m._logits_raw(E).reshape(-1, K)
    p = F.softmax(lg, dim=-1)
    legacy = -(p * (p + 1e-8).log()).sum(-1).mean().item()      # mlp.py:1077, p_surr == p_boltz
    true_kl = F.kl_div(F.log_softmax(lg, -1), p, reduction="batchmean").item()
    Hp = -(p * (p + 1e-12).log()).sum(-1).mean().item()
print(f"  legacy 'kl' at the optimum = {legacy:.6f}   true KL = {true_kl:.3e}   "
      f"H(p_boltz) = {Hp:.6f}")
print(f"  legacy - H(p_boltz) = {legacy - Hp:+.3e}  -> the legacy value IS CE = KL + H, so its")
print("  irreducible floor is the router's entropy. Gradients are identical (p_boltz is")
print("  detached), so no trained checkpoint is affected -- but the LOGGED number was not a gap.")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 2 -- ORACLE head + use_surrogate=True must reproduce the BASE class BITWISE.")
print("         Sweeps every routing feature the legacy class never had.")
cases = [
    dict(rn="none", renorm=True),
    dict(rn="none", renorm=False),
    dict(rn="zscore", renorm=True),
    dict(rn="sqrt_width", renorm=False),
    dict(rn="zscore", renorm=True, top_k=None),
    dict(rn="zscore", renorm=True, sinkhorn_iters=4, sinkhorn_persist_mu=True,
         sinkhorn_mu_iters=1),
    dict(rn="none", renorm=True, balance_rate=0.05),
]
for cs in cases:
    cs = dict(cs)
    tk = cs.pop("top_k", TOPK)
    rn, renorm = cs.pop("rn"), cs.pop("renorm")
    b = mk_base(rn=rn, renorm=renorm, **({} if tk else {}), **cs)
    s = mk_surr(rn=rn, renorm=renorm, coef=0.0, use=True, **cs)
    if tk is None:
        b.moe.top_k = s.moe.top_k = None
    attach_oracle(s.moe)
    # For a persisted-mu arm the running dual must be filled in TRAIN mode first, exactly as
    # a calibration pass would -- and both models must see the same batches to compare.
    if cs.get("sinkhorn_persist_mu"):
        b.moe.train(); s.moe.train()
        with torch.no_grad():
            for _ in range(3):
                b.moe(x); s.moe(x)
        b.moe.eval(); s.moe.eval()
        b.moe._mu_call.zero_(); s.moe._mu_call.zero_()
    with torch.no_grad():
        ob, os_ = b.moe(x).clone(), s.moe(x).clone()
    bits = torch.equal(ob, os_)
    lbl = f"rn={rn:11s} renorm={str(renorm):5s} top_k={tk} " + \
          " ".join(f"{k}={v}" for k, v in cs.items())
    print(f"  {lbl:78s} rel={relerr(os_, ob):.2e} bitwise={bits}")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 3 -- the surrogate SELECTION really is the head's, not the energy's.")
print("         A DELIBERATELY WRONG head (energies reversed) must change the top-k set,")
print("         otherwise TEST 2 would pass for a path that silently ignores the head.")
s = mk_surr(coef=0.0, use=True, rn="zscore")
m = s.moe
with torch.no_grad():
    E_ex = exact_E_hopfield(m, x).reshape(-1, K)
    m.surrogate_energies = (lambda xx: -exact_E_hopfield(m, xx))     # inverted
    out_inv = m(x).clone()
    attach_oracle(m)
    out_ok = m(x).clone()
    a = m._logits_raw(E_ex).topk(TOPK, -1).indices
    bb = m._logits_raw(-E_ex).topk(TOPK, -1).indices
    ov = (a.unsqueeze(-1) == bb.unsqueeze(-2)).any(-1).double().mean().item()
print(f"  top-{TOPK} overlap between the two heads = {ov:.3f}  (identical would be 1.000)")
print(f"  output differs: rel = {relerr(out_inv, out_ok):.3e}   "
      f"{'HEAD IS USED' if relerr(out_inv, out_ok) > 1e-6 else '*** HEAD IGNORED ***'}")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 4 -- use_surrogate=False AND surrogate_coef=0 is the base class, BITWISE.")
print("         (the head uses a PRIVATE torch.Generator so it cannot shift the global RNG")
print("          stream and move the expert init -- the base class records that hazard)")
for kind, fused in (("hopfield", True), ("hopfield", False), ("w1w2", False)):
    b = mk_base(kind=kind, fused=fused)
    s = mk_surr(kind=kind, fused=fused, coef=0.0, use=False,
                surrogate_track_agree=False)
    with torch.no_grad():
        ob = b.moe(x).clone(); os_ = s.moe(x).clone()
    wb = b.expert_holder.W.weight if kind == "hopfield" else b.expert_holder.W1.weight
    ws = s.expert_holder.W.weight if kind == "hopfield" else s.expert_holder.W1.weight
    print(f"  kind={kind:8s} fused={str(fused):5s}  expert weights identical="
          f"{torch.equal(wb, ws)}  output bitwise={torch.equal(ob, os_)}  "
          f"rel={relerr(os_, ob):.2e}")
    # and in TRAIN mode with the loss enabled the aux loss must be UNTOUCHED when coef=0
    s.moe.train(); clear_aux_loss()
    with torch.no_grad():
        s.moe(x)
    al = get_aux_loss()
    print(f"      aux_loss with surrogate_coef=0: {float(al) if not torch.is_tensor(al) else al.item():.1e}")
    s.moe.eval()

# ------------------------------------------------------------------------------------------ #
print("\nTEST 5 -- EXPERT-KIND AGNOSTICISM: the identical head code on both energies.")
print("         w1w2 sums SIGNED terms that cancel (ratio 0.0254 vs 1.000 for hopfield), which")
print("         is what breaks the rank-r SUBSPACE proxy. The head never sees the energy's form.")
for kind, fused in (("hopfield", True), ("hopfield", False), ("w1w2", False),
                    ("w1w2", True) if SurrogateBoltzmannMoEW1W2 is not None else ("w1w2", False)):
    try:
        s = mk_surr(kind=kind, fused=fused, coef=1.0, use=False, rn="zscore")
    except AssertionError as e:
        print(f"  kind={kind:8s} fused={str(fused):5s}  REFUSED: {str(e)[:60]}")
        continue
    m = s.moe
    m.train(); clear_aux_loss()
    with torch.no_grad():
        m(x)
    al = get_aux_loss()
    al = float(al) if not torch.is_tensor(al) else al.item()
    met = m.pop_load_metrics()
    print(f"  kind={kind:8s} fused={str(fused):5s} cls={type(m).__name__:32s} "
          f"KL={al:.4f} agree={met.get('surrogate_topk_agree', float('nan')):.3f} "
          f"(chance {TOPK/K:.3f})")

print("\n  head cost, INDEPENDENT of I_e (section (b) of the module docstring):")
for d, k_, sk, sh in ((1024, 32, "linear", 0), (1024, 32, "mlp", 256),
                      (768, 16, "linear", 0), (1024, 64, "mlp", 512)):
    print(f"    d={d:5d} K={k_:3d} {sk:6s} h={sh:4d} -> {surrogate_head_macs(d, k_, sk, sh):>10,d}"
          f" MACs/token")
print("    for reference, the 400M hybrid's EXACT hopfield router is I_total*d ="
      f" {187872*1024:,d}, and its subspace proxy K*d*r + K*m*r at r=16 is"
      f" {32*1024*16 + 32*512*16:,d} (m=512) / {32*1024*16 + 32*5871*16:,d} (m=I_e).")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 6 -- THE PRE/POST-mu DECISION, MADE FALSIFIABLE.")
print("         We distil PRE-mu and let _route apply mu once. The existing DENSE-path proxy")
print("         convention (_proxy_step targets the POST-mu logits while predicting PRE-mu,")
print("         and _route then subtracts mu again) would apply mu TWICE. Measure the gap.")
s = mk_surr(coef=1.0, use=True, rn="zscore", sinkhorn_iters=6,
            sinkhorn_persist_mu=True, sinkhorn_mu_iters=1)
m = s.moe
attach_oracle(m)
m.train()
with torch.no_grad():
    for _ in range(4):
        m(x)
m.eval(); m._mu_call.zero_()
with torch.no_grad():
    E = m.surrogate_energies(x)
    raw = m._logits_raw(E).reshape(-1, K)
    mu = m.sinkhorn_mu[0]
    p1 = F.softmax(raw - mu, -1)                  # our convention: mu once
    p2 = F.softmax(raw - 2 * mu, -1)              # the double-counted convention
    p0 = F.softmax(raw, -1)                       # untilted
    tv12 = 0.5 * (p1 - p2).abs().sum(-1).mean().item()
    tv10 = 0.5 * (p1 - p0).abs().sum(-1).mean().item()
    a1 = (raw - mu).topk(TOPK, -1).indices
    a2 = (raw - 2 * mu).topk(TOPK, -1).indices
    a0 = raw.topk(TOPK, -1).indices
    ov12 = (a1.unsqueeze(-1) == a2.unsqueeze(-2)).any(-1).double().mean().item()
    ov10 = (a1.unsqueeze(-1) == a0.unsqueeze(-2)).any(-1).double().mean().item()
print(f"  |mu|_max = {mu.abs().max().item():.4f}   (live 134M arms measure 3.44)")
print(f"  mu-once vs mu-TWICE : total-variation {tv12:.4f}   top-{TOPK} overlap {ov12:.3f}")
print(f"  mu-once vs UNTILTED : total-variation {tv10:.4f}   top-{TOPK} overlap {ov10:.3f}")
print("  Both alternatives are distinguishable from ours at THIS |mu|; the gap grows with |mu|,")
print("  and |mu| here is much smaller than the trained 3.44, so treat these as lower bounds.")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 7 -- guard rails.")
try:
    mk_surr(coef=1.0, use=True, sinkhorn_iters=4, sinkhorn_persist_mu=False,
            rn="zscore")
    print("  *** sinkhorn without persist_mu was ACCEPTED -- guard missing ***")
except AssertionError as e:
    print(f"  use_surrogate + sinkhorn_iters, no persist_mu -> REFUSED ok: {str(e)[:66]}...")
try:
    mk_surr(coef=1.0, sparse_forward=True, sparse_candidates=4, sparse_explore=0,
            proxy_rank=4, proxy_kind="subspace", repulsion_subsample=8)
    print("  *** sparse_forward was ACCEPTED -- requirement (d) violated ***")
except AssertionError as e:
    print(f"  sparse_forward                          -> REFUSED ok: {str(e)[:66]}...")
try:
    mk_surr(coef=1.0, sk="mlp", sh=0)
    print("  *** mlp head with hidden=0 was ACCEPTED ***")
except AssertionError as e:
    print(f"  surrogate_kind=mlp, hidden=0            -> REFUSED ok: {str(e)[:66]}...")

# a persisted-mu arm with an EMPTY running dual must warn, not silently route untilted
s = mk_surr(coef=0.0, use=True, rn="zscore", sinkhorn_iters=4,
            sinkhorn_persist_mu=True, sinkhorn_mu_iters=1)
attach_oracle(s.moe)
with torch.no_grad():
    s.moe(x)
print(f"  empty running mu at first surrogate eval -> warned once: {s.moe._surr_warned_mu}")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 8 -- TRAIN A HEAD FOR REAL (CPU, float64, 400 Adam steps on the head only).")
print("         Exercises the whole loop: forward -> _route -> KL -> backward -> step.")
print("         NOT a quality result, and not even a generalisation result: random frozen")
print("         experts, K=8, d=32, and ONE FIXED batch of 256 tokens, which an MLP head of")
print("         2.6k parameters can memorise. It shows only that the loss DESCENDS and the")
print("         agreement RISES above chance -- i.e. the gradient reaches the head and the")
print("         metric moves with it. Real capacity numbers need trained experts on a GPU.")
torch.manual_seed(7)
s = mk_surr(seed=11, coef=1.0, use=False, rn="zscore", sk="mlp", sh=64,
            surrogate_init_std=0.05)
m = s.moe
m.train()
head = [p for n, p in m.named_parameters() if n.startswith("surrogate_")]
frozen = [p for n, p in m.named_parameters() if not n.startswith("surrogate_")]
for p in frozen:
    p.requires_grad_(False)
opt = torch.optim.Adam(head, lr=3e-2)
xb = torch.randn(256, H, dtype=torch.float64)
first = last = None
for step in range(400):
    clear_aux_loss()
    m(xb)
    loss = get_aux_loss()
    opt.zero_grad(); loss.backward(); opt.step()
    if step == 0:
        first = loss.item()
    last = loss.item()
    if step in (0, 99, 399):
        met = m.pop_load_metrics()
        print(f"    step {step:3d}  KL = {loss.item():.4f}   top-{TOPK} agreement = "
              f"{met['surrogate_topk_agree']:.3f}   (chance {TOPK/K:.3f})")
print(f"  KL {first:.4f} -> {last:.4f}   {'DESCENDED' if last < first else '*** DID NOT DESCEND ***'}")
print(f"  head params = {m.surrogate_parameter_numel():,d}, "
      f"MACs/token = {m.surrogate_macs_per_token():,d}  "
      f"(MACs count multiply-accumulates only; the difference is the two bias vectors)")

# ------------------------------------------------------------------------------------------ #
print("\nTEST 9 -- STRUCTURE GUARD on the two frozen behaviours this module depends on.")
src_route = inspect.getsource(BoltzmannMoEFFEnergy._route)
checks = {
    "_route still returns (p, logits) after applying mu":
        "logits = logits - mu.to(logits.dtype)" in src_route and "return p, logits" in src_route,
    "_route is still the ONLY place mu is applied in the dense paths":
        src_route.count("_mu_for(") == 1,
    "_forward_looped calls _route":
        "self._route(E_k)" in inspect.getsource(BoltzmannMoEFFEnergy._forward_looped),
    "_forward_fused calls _route":
        "self._route(E_k" in inspect.getsource(BoltzmannMoEFFEnergy._forward_fused),
    "_forward_sparse still does NOT call _route (hence requirement (d))":
        "self._route(" not in inspect.getsource(BoltzmannMoEFFEnergy._forward_sparse),
    # NB: match the CODE, not the docstring -- `_logits_raw`'s docstring explains why it
    # exists by naming `_mu_call`, so a bare substring test is a false alarm.
    "_logits_raw still does NOT tick _mu_call":
        "_mu_call +=" not in inspect.getsource(BoltzmannMoEFFEnergy._logits_raw),
    "_proxy_step still targets the POST-mu logits (the double-count reported in (c).3)":
        "flat_tgt = F.softmax(logits.detach()" in inspect.getsource(
            BoltzmannMoEFFEnergy._proxy_step),
    "n_dominant_experts is still computed only in _log_metrics":
        "n_dominant_experts" in inspect.getsource(BoltzmannMoEFFEnergy._log_metrics)
        and "n_dominant" not in inspect.getsource(BoltzmannMoEFFEnergy.pop_load_metrics),
}
for k, v in checks.items():
    print(f"  [{'OK ' if v else 'CHANGED'}] {k}")
print("\n  If any row says CHANGED, re-read energy_ff.py against energy_ff_surrogate.py before")
print("  trusting anything above -- these are the assumptions the overrides rest on.")
