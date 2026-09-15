# ⚠ SUSPECTED ROUTING SIGN INVERSION in the composable Boltzmann MoE

**Found 2026-09-15, prompted by a question about `e_sign`. Not yet fixed — the
decision is the user's, because fixing it changes what every trained checkpoint
computes.**

## The measurement

On a trained checkpoint (`iclr_flops/iclr_hop_K32_top2`, K=32), reproducing the
deployed routing exactly (zscore, `e_sign="neg"`, tau=0.35, top-2):

```
mean ||gelu(W_k x)||^2 / I_e of the experts the router SELECTS : 1.135776
mean over ALL experts                                          : 1.845705
mean of the HIGHEST-overlap 2 experts                          : 2.837024
mean of the LOWEST-overlap  2 experts                          : 1.135776   <-- identical
```

**The router selects exactly the LOWEST-overlap experts** — 0.62x the average
expert's overlap. It is choosing the *worst*-matching experts, every token.

## Why, in code

`_HopfieldExpert` stores as its "energy" a quantity that INCREASES with overlap:

    _last_energy_per_token = (gelu_Wx ** 2).mean(-1)        # >= 0, larger = better match

and the wrapper then applies `e_sign="neg"`:

    s = -E_k  ->  logits = s / tau  ->  p = softmax(-E_k / tau)   # favours SMALLEST E_k

Composing those two gives `p ∝ exp(-overlap/tau)`: anti-routing. The intended
convention is the opposite — `p ∝ exp(-E/tau)` with `E = -overlap`, i.e.
`p ∝ exp(+overlap/tau)`, so a better match gets a higher weight.

## The legacy class is the smoking gun

Legacy `BoltzmannMoE_Energy_MLP` (`mlp.py`), which these runs were meant to
reproduce, does:

    E = einsum("...h,...eh->...e", x, term1) * routing_scale   # E = +phi(W1x).(W2x) = +OVERLAP
    p = F.softmax(E / temperature)                             # p ∝ exp(+overlap/tau)  CORRECT

The composable `_W1W2Expert` instead stores

    _last_energy_per_token = -inv_sqrt_d * (phi * W2x).sum(-1)  # E = -OVERLAP  (minus ADDED)

and is constructed with `e_sign="pos"`, i.e. `p = softmax(+E/tau) = softmax(-overlap/tau)`.

**So the refactor added a minus sign to the energy AND kept the legacy softmax
sign — double negation.** The docstring's claim that `e_sign="pos"` "matches legacy
`BoltzmannMoE_Energy_MLP`" is therefore incorrect: legacy's `E` has no minus.
Both composable forms (Hopfield via `e_sign="neg"` on a positive energy, W1W2 via
`e_sign="pos"` on a negated energy) end up anti-routing, consistently.

## Scope

Affects **every composable (`EnergyFF_BoltzmannMoE`) run**: the whole ICLR grid
(all 22 `iclr_*` arms) and the live 400M `scale32B_boltz_hop`. Does NOT affect the
learned-gate / Switch arms (their router has no energy sign), `gptswitch`, or the
legacy `h1_*`/`b*`/`c*` series, which use the legacy class shown above.

## It may explain several standing puzzles

1. **"The energy-FF branch is essentially dead"** (HANDOFF 7.5): routing to the
   worst-matching experts would produce a small, incoherent FF contribution.
2. **Routing is near-uniform** (7.4, `effective_n_experts` ~ K): selecting on an
   anti-correlated criterion carries little usable signal.
3. **"Energy routing merely matches a learned gate, never beats it"** — the paper's
   central routing claim. It was measured with the energy sign inverted, so the
   correctly-signed router is **untested**.
4. It compounds the separate Hopfield energy-scale problem (E ~ 1e-2 << tau).

## What I did NOT do

I did not change it. Flipping the sign changes what every existing checkpoint
computes and would invalidate the comparability of the ICLR grid, so it is a
research decision, not a bug-fix I should make unilaterally. The live 400M arm is
still training with the current (inverted) convention, which at least keeps it
comparable with every prior arm.

## Decisive next experiment

One 4-GPU, 300-step A/B on the probe harness, identical except the routing sign:
`e_sign="pos"` for Hopfield (equivalently, negate the stored energy). Read
`load_effective_n_experts`, `expert_cos_abs_mean`, `lm_loss`, and `||ffwd_out||`.
If the corrected sign sharpens routing and revives the FF branch, that is both the
explanation for finding 1-3 above and a likely quality win — and it would need to be
stated in the paper.

Note also: the cheap-router proxy tests run today ranked by LOWEST energy, i.e. they
faithfully proxy the DEPLOYED (inverted) router. If the sign is corrected, those
agreement numbers must be re-measured in the corrected direction.
