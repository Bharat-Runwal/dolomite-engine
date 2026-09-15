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

---

# A/B RESULT (early, step 20-30): the sign is confirmed causal — and correcting it collapses routing

4 GPU, 300 steps, identical seed/data/knobs except the routing sign.
S1 = deployed (`e_sign=neg`, anti-routing). S2 = `e_sign_override: pos` (best-match).

| metric | S1 deployed | S2 corrected | change |
|---|---:|---:|---|
| `ffwd/output_norm` | 0.0294 | **3.6562** | **124x larger** |
| `load_effective_n_experts` (of 32) | 9.18 | **1.03** | collapsed |
| `ffwd/n_dominant_experts` | 2.0 | **1.0** | a single expert |
| `ffwd/mean_token_entropy_norm` | 0.1014 | **0.0002** | ~zero |
| `expert_cos_abs_mean` | 0.1812 | 0.2217 | — |
| `lm_loss` | 7.79 @ step 30 | 8.02 @ step 20 | not yet matched-step |

## 1. The dead FF branch is CAUSED by the inverted sign — confirmed

HANDOFF 7.5 measured `||ffwd_out|| = 0.0050` against `||attn_out|| = 18.53` and
concluded "the energy-FF branch is essentially dead", treating it as a property of
the architecture / energy scale. It is not: correcting the routing sign makes the
branch **124x stronger** with nothing else changed. Routing to the worst-matching
experts produced a small incoherent FF contribution, exactly as predicted.

## 2. But the inverted sign was ACCIDENTALLY ACTING AS A LOAD BALANCER

S2 collapses to one expert within 20 steps. The mechanism is a sign flip in the
feedback loop:

- **Anti-routing (deployed)** is self-LIMITING: the worst-matching expert is
  selected, receives gradient, becomes better-matched, and therefore stops being
  selected. Negative feedback -> load spreads on its own.
- **Correct routing** is self-REINFORCING: the best-matching expert is selected,
  receives gradient, becomes even better-matched, and is selected more. Positive
  feedback -> winner-take-all. This is the standard MoE collapse that
  load-balancing losses exist to prevent.

**This puts a second paper claim at risk.** Not only "the FF branch is dead", but
the formulation's headline property — *"Boltzmann routing does not collapse and needs
no gate parameters and no load-balancing loss"* (measured: effective experts
12.3-12.8 of 16, max share 0.17-0.18) — may hold **only because the sign was
inverted**, i.e. because anti-routing is inherently self-balancing. The
correctly-signed router collapses immediately with no balancing.

## 3. Caveats — this is step 20-30, not a verdict

- Early-training collapse sometimes resolves; needs a few hundred steps.
- `temperature: 0.35` and `routing_norm: zscore` were BOTH tuned under the inverted
  sign. Under the corrected sign the effective sharpness differs, so tau may simply
  be too low now.
- `balance_rate` (the aux-loss-free DeepSeek-V3-style balancing already implemented,
  default OFF) is available and untested here. The collapse says it is now
  load-bearing rather than optional.

## 4. The configuration that has never been tested

Corrected sign + retuned temperature (and/or `balance_rate > 0`). Neither the
paper nor any run has explored it: every arm to date used the inverted sign. A small
sweep — `e_sign_override: pos` x `temperature in {0.35, 1.0, 3.0}` x
`balance_rate in {0, 0.001}` — would establish whether the corrected router is usable
and whether the no-load-balancing claim survives.

Do NOT restate the paper's routing-health results until this is settled.
