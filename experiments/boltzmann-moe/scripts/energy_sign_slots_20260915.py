#!/usr/bin/env python3
"""Verify, numerically, each place a minus sign can enter the Boltzmann-MoE energy,
against the PHYSICS convention:
    Boltzmann weights   p_k = exp(-E_k/tau)/Z          Z = sum_k exp(-E_k/tau)
    free energy         E_FF = -tau*log Z_FF
    full energy         E = E_AT + s*E_FF
    forward             h <- h - proj(grad E)           (descent)
Four slots can each carry a sign: (1) the definition of E_FF, (2) the Boltzmann weight,
(3) the free-energy/total, (4) the forward update. This script measures 1, 3 and 4 for the
hopfield expert and reports which are consistent with the convention above.
"""
import torch
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe
torch.manual_seed(0)

def mk(K, e_sign, H=32, I=64):
    c = build_boltzmann_moe(expert_kind="hopfield", hidden_size=H, intermediate_size=I,
                            n_experts=K, temperature=1.0, top_k=None,
                            e_sign_override=e_sign, add_bias=False,
                            initializer_range=0.05, m_width=1.0)
    c.double(); c.train(); c._capture_energy = True
    return c

print("SLOT 1+4 -- what does one expert STORE, and what does it EMIT?")
c = mk(2, "pos"); ex = c.moe.experts[0]   # K>=2 required by the class; one expert suffices here
x = torch.randn(3, 32, dtype=torch.float64, requires_grad=True)
S = ex.energy_per_token(x)
gS, = torch.autograd.grad(S.sum(), x, retain_graph=True)
with torch.no_grad(): out = ex(x.detach())
cos = ((out*gS).sum()/(out.norm()*gS.norm())).item()
print(f"  stores S = mean(gelu(Wx)^2) in [{S.min():.5f},{S.max():.5f}]  -> S >= 0, GROWS with overlap")
print(f"  physics energy is therefore E_FF = -S   (low energy = good match)")
print(f"  cos(expert_out, +grad S) = {cos:+.10f}   |out|/|grad S| = {(out.norm()/gS.norm()).item():.4f}")
print(f"  => expert emits +c*grad S = -c*grad E_FF   (c>0)")
print()

print("SLOT 2+3 -- Boltzmann weight and free energy, for each e_sign")
for s in ("pos","neg"):
    c2 = mk(4, s); h = torch.randn(3, 32, dtype=torch.float64)
    o = c2(h); exported = c2._last_energy_per_token
    with torch.no_grad():
        Sk = torch.stack([e.energy_per_token(h) for e in c2.moe.experts], dim=-1)
        Ek = -Sk                                   # physics energy
        p_phys = torch.softmax(-Ek, dim=-1)        # exp(-E)/Z
        p_code = torch.softmax(Sk if s=="pos" else -Sk, dim=-1)
        E_phys = -torch.logsumexp(-Ek, dim=-1)     # -tau*log Z_FF
    print(f"  e_sign={s!r}")
    print(f"    p_code == exp(-E)/Z (physics)? {torch.allclose(p_code, p_phys)}")
    print(f"    exported {exported.mean():+.8f}   vs  -tau*logZ_FF {E_phys.mean():+.8f}"
          f"   match={torch.allclose(exported, E_phys)}")
print()

print("SLOT 4 -- does the layer's h <- h - proj(out) DESCEND the physics E_FF?")
for s in ("pos","neg"):
    c3 = mk(4, s); h = torch.randn(3, 32, dtype=torch.float64)
    out = c3(h); E0 = c3._last_energy_per_token.detach().clone()
    with torch.no_grad():
        c3(h - 1e-5*out); E1 = c3._last_energy_per_token.detach().clone()
    d = (E1-E0).mean().item()
    print(f"  e_sign={s!r}: dE_FF along -out = {d:+.4e}  -> {'DESCENDS' if d<0 else 'ASCENDS (inconsistent)'}")
print()
print("BOUNDS -- hopfield E_FF = -tau*log sum_k exp(S_k/tau), S_k in [0, S_max]:")
print("  E_FF <= -tau*log K  (upper)   and   E_FF >= -tau*log K - S_max  (lower)")
print("  S_max is finite on the RMSNorm sphere (||ln_x||^2=d, finite ||W||), so E_FF is")
print("  bounded BOTH ways -- an action loss on it cannot run to -inf.")
