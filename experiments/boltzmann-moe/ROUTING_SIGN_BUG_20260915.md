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

---

# PRINCIPLED REFRAMING: load balancing is a CHEMICAL POTENTIAL, not an auxiliary loss

Proposed 2026-09-15 in response to "I want to convert this accidental finding into a
principled approach". This is **theory plus a testable prediction**, not a measured
result — the sweep (`tb_T2..T5`) is the test.

## The derivation

The Boltzmann weights are not an arbitrary choice: `p_k ∝ exp(E_k/tau)` is exactly the
solution of the entropy-regularised assignment problem

    max_p  sum_k p_k E_k  +  tau H(p)

i.e. the softmax IS the entropy-regularised argmax. Now impose load balance as a
constraint on the batch-marginal occupancy rather than as a penalty:

    subject to   sum_{tokens} p_k(x)  ~  N/K   for every k

Introducing a Lagrange multiplier `mu_k` for that constraint gives

    p_k  ∝  exp( (E_k - mu_k) / tau )

In statistical mechanics `mu_k` is a **chemical potential**: the quantity conjugate to
occupancy, and the thermodynamically standard way to constrain an average particle
number. It is a dual variable, not a loss term.

## Why this is the right frame for THIS paper

- It uses the vocabulary the paper is already built on (Boltzmann weights, partition
  function, free energy `E_total = -tau LSE_k(E_k/tau)`). Balancing becomes another
  thermodynamic quantity rather than an imported MoE heuristic.
- **It strengthens rather than concedes the "no auxiliary loss, no gate parameters"
  claim.** `mu_k` has no gradient pathway and introduces no learned gate — exactly the
  property `load_balance_bias` was implemented with (and the reason the existing code
  comment argues the claim survives it).
- It cleanly separates two roles that the sign bug had collapsed into one:
  **`E_k` decides which expert FITS; `mu_k` decides how CROWDED it is.**

## The accident, re-read

Anti-routing (the inverted sign) is a **fixed, crude chemical potential**. "Prefer the
expert you match least" is a static proxy for "prefer the expert that is
under-occupied" — the two correlate, because under-used experts tend to be poorly
matched, which is why routing stayed balanced for the whole project. But it buys
balance by inverting the SELECTION, which destroys what the router is for (measured:
the FF branch was 20-100x weaker). The chemical potential achieves the same balance
without touching the selection.

So the finding converts from "we had a bug that accidentally helped" into "the
balance property has a derivation, and the bug was implementing a degenerate case
of it".

## Testable prediction (this is what tb_T2..T5 measures)

Corrected sign + adaptive `mu_k` should keep **both**:
- the FF-branch revival (from correct `E_k`): `ffwd/output_norm` ~ 10, not ~0.03
- the load balance (from `mu_k`): `load_effective_n_experts` back toward S1's 4-8,
  not S2's 1.0-2.3

If T4/T5 (`balance_rate: 0.001`) show that combination, the reframing is supported and
the paper can state balance as a constrained-variational result. If `mu_k` saturates
at the +-1.0 `_BIAS_MAX` clamp without restoring `effK`, the bias is too weak for
K=32 and needs a larger clamp or a Sinkhorn-style exact normalisation instead.
Watch `load_bias_absmax` for exactly that.

## Two orthogonal refinements

- **Sinkhorn** enforces the marginal EXACTLY (a few normalisation sweeps; the
  canonical ensemble), where the adaptive bias tracks it approximately (grand
  canonical, `mu` estimated online). The bias is cheaper and needs no inner loop; the
  Sinkhorn version is the one to reach for if the bias proves too soft.
- **Deterministic annealing on tau** is a separate lever: large `tau` gives
  near-uniform occupancy for free. Plausibly the inverted sign was ALSO mimicking
  this, since the Hopfield energy scale left `E << tau` (HANDOFF 7.4). `tb_T2/T3`
  separate the tau effect from the `mu` effect.

## What is NOT claimed here

That the corrected router is better for quality. Across 300 steps the sign made no
resolvable difference to `lm_loss` (mean delta +0.006 to +0.013 against a 0.038
noise floor), consistent with 7.5's finding that deleting the FF branch entirely
moved perplexity by +0.0003. At this scale the branch is not load-bearing for the LM
loss in either sign. The chemical-potential frame is a claim about the ROUTING
MECHANISM being principled, not yet about it winning.

---

# SWEEP RESULT (step 30, early): the chemical potential works, and tau is NOT needed

All arms below are CORRECTED sign (`e_sign_override: "pos"`), 4 GPU, same seed/data.
`mu_k` = `balance_rate: 0.001` (the aux-loss-free per-expert bias).

| arm | tau | mu_k | effK (of 32) | ffwd/output_norm | bias_absmax |
|---|---:|:---:|---:|---:|---:|
| S1 deployed (anti-routing) | 0.35 | — | 9.18 | 0.029 | — |
| S2 corrected | 0.35 | off | 1.54 | 2.78 | — |
| T2 corrected | 1.0 | off | 3.05 | 5.28 | — |
| T3 corrected | 3.0 | off | 3.47 | **0.34** | — |
| **T4 corrected** | **0.35** | **on** | **5.00** | **6.94** | **1.0000** |
| **T5 corrected** | 1.0 | on | **7.08** | 2.00 | **1.0000** |

## The prediction held

**T4 — the ORIGINAL tau plus `mu_k` — dominates both temperature arms on BOTH axes.**
Against S2 (same tau, no bias): effK 1.54 -> 5.00 AND ffwd_norm 2.78 -> 6.94. Balance
improves 3.2x while the branch gets 2.5x STRONGER. It also beats T3 (tau=3.0) on
balance and by 20x on branch magnitude.

So the two mechanisms are cleanly separated, as the derivation says:
- **tau** buys occupancy spread by BLURRING the selection — and self-defeats, because
  masked top-k weights shrink as routing softens, so T3's branch collapses to 0.34.
- **mu_k** redistributes occupancy WITHOUT touching the selection.

**tau does not need retuning after all.** The corrected sign plus a chemical potential
at the original tau=0.35 is the configuration that gets both properties. This converts
the accidental anti-routing balance into a principled mechanism whose principled form
is strictly better than the temperature workaround.

Frontier: T4 = best branch (239x S1's ffwd_norm), T5 = best balance (7.08 against
S1's 9.18, still 69x S1's branch).

## Actionable caveat: `mu_k` is CLAMP-SATURATED

Both bias arms report `bias_absmax = 1.0000`, exactly `_BIAS_MAX`. So `mu_k` achieves
this while pinned, and wants to push harder — the +-1.0 bound is now the binding
constraint at K=32. Raising the clamp is the obvious move but risky: the bound exists
because an earlier unclamped `sign()`-based version reached |bias| = 1482 and
destabilised `pure_hop_isoP_bal` (loss 4.05 -> 5.18, 77 upward jumps).

**Preferred fix: Sinkhorn-style exact marginal normalisation**, which enforces the
occupancy constraint without an unbounded multiplier — the canonical-ensemble version
of the same variational problem. A few normalisation sweeps per step, no clamp, no
tuning knob.

## Still open

- Step 30 only, and S2 showed non-monotone early dynamics (1.03 -> 1.47 -> 2.32), so
  the ordering needs confirming at 300 steps.
- No quality claim. Every sign/balance arm so far sits inside the 0.038 lm_loss noise
  floor at this scale, consistent with 7.5's "deleting the FF branch moved perplexity
  by +0.0003". These are mechanism results, not quality results.
