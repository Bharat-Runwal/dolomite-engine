# Making the energy signs universal (an inconsistency, not a bug)

**Status: PROPOSAL, not applied.** Written 2026-09-17. Nothing here changes a running arm.
Companion to `HANDOFF.md` §11.1 / §12.28 and to `ROUTING_SIGN_BUG_20260915.md`.

## The situation in one paragraph

The physics is consistent and the code is consistent, but they use **opposite sign conventions for
the word "energy"**, and every arm has to paper over the gap with a per-kind `e_sign_override`. The
paper's `sec/theory.tex` now states the convention explicitly (Overleaf `18c1e3d`):

```
E_i     = expert OVERLAP            e.g. mean(gelu(Wx)^2)  or  phi(W1 x)^T (W2 x)
S_i     = -E_i                      the STATE ENERGY (Hopfield sign, unbounded below)
Z       = sum_i exp(-S_i/tau)     = sum_i exp(+E_i/tau)
E^FF    = -tau log Z                the FREE ENERGY (physics sign)
p_i     = softmax(-S_i/tau)       = softmax(+E_i/tau)      favours the BEST match
forward = -grad E^FF = -sum p_i grad S_i = +sum p_i grad E_i
```

The code computes exactly this. The awkwardness is that the code's *variable* named `E_k` holds
`E_i` (the overlap) for hopfield and `-E_i` for w1w2 — i.e. `-S_i` and `+S_i` respectively — so
`e_sign` has to differ per kind to land on the same `logits = -S_i/tau`.

## Why the per-kind override exists

| expert kind | `energy_per_token` returns | that is | needed `e_sign` |
|---|---|---|---|
| `hopfield` | `+(gelu(Wx)**2).mean(-1)` | `+E_i = -S_i` | **`"pos"`** |
| `w1w2` | `-d^{-1/2} (phi * W2x).sum(-1)` | `-E_i = +S_i` | **`"neg"`** |

Both then give `logits = -S_i/tau`. And the kind-based DEFAULTS at `energy_ff.py:2387/2396` are
inverted relative to these, which is why **every** config in the corrected grid carries
`e_sign_override`, and why CLAUDE.md pre-flight 8 exists as a standing trap.

**This is not a bug in the trained models.** With the overrides in place the arms implement the
Boltzmann form correctly. It is a naming/defaults inconsistency that costs a config line per arm and
has already cost real time: `ROUTING_SIGN_BUG_20260915.md` had to argue from "which sign picks the
best match" instead of just applying `p ∝ exp(-S/tau)`, and this session I initially concluded the
code was wrong because the variable called "energy" is really the negative energy.

## Proposal: make `S_i` the thing the code stores

1. **Redefine both experts' `energy_per_token` to return the STATE ENERGY `S_i = -overlap`.**
   - `hopfield`: `return -(F.gelu(Wx) ** 2).mean(-1)`   (flips today's sign)
   - `w1w2`: unchanged — it already returns `-overlap`
2. **Set the default `e_sign = "neg"` for BOTH kinds** and delete the per-kind defaults, so
   `logits = -S_i/tau` falls out with no override.
3. **Keep `e_sign_override` as an escape hatch** for reproducing old checkpoints, but make its
   docstring say the only physically consistent value is `"neg"`.
4. **Rename for honesty**: `energy_per_token` → `state_energy_per_token`, and the metrics
   `energy_abs_mean` / `energy_abs_max` → `state_energy_*`. The logged block energy
   (`-tau*logsumexp(logits)`) is already `E^FF` and needs no change.

Net effect: `logits = -E_stored/tau` universally, one code path, no per-kind trap, and the variable
named "energy" is an energy.

## What this must NOT break, and how to check

* **Every existing checkpoint's `config.json` carries `e_sign_override`.** Under the proposal those
  values become WRONG for hopfield (`"pos"` on a now-negated stored energy inverts the router).
  So the change needs a **config-version guard**: if `e_sign_override` is present and the checkpoint
  predates the change, interpret it against the old storage sign. Without that guard, every
  published number silently re-evaluates with an inverted router — the exact 1.6–1.8 nat class of
  failure as §12.1.
* **Equivalence test before anything else.** On one trained checkpoint, assert bit-identical
  `p`, `E^FF` and block output between old (`hopfield`+`"pos"`, `E=+overlap`) and new
  (`hopfield`+`"neg"`, `E=-overlap`). This is a pure relabelling, so the tolerance is 0, not 1e-6.
  Pattern to copy: `scripts/test_logits_refactor_equiv_20260916.py`, which did exactly this for the
  `_logits` split and verified 0.00e+00 on both logits and buffers across 5 routing configs.
* **Do not touch the live arms.** `t32B_sandwich_sparse` and `t32B_pure_it4_sparse` are mid-run
  (~61035 steps each). They must finish on the current code.

## Recommended sequencing

1. **Now (free):** the paper states the convention explicitly — done, Overleaf `18c1e3d`.
2. **Now (free):** new configs keep carrying the correct `e_sign_override`. No change needed; the
   arms are already right.
3. **After the deadline:** implement 1–4 above behind the config-version guard, with the
   bit-identity test as the gate. Estimated half a day, most of it on the guard and the test.
4. **Never:** flip the storage sign without the guard. That would silently invert the router on
   every existing checkpoint.

## Why it is worth doing at all

Three separate sessions have now spent time re-deriving which sign is correct, and one of them
(2026-09-15) rewrote the entire 22-arm grid on the strength of that reasoning. The models were fine
either way once the overrides were set; what the inconsistency actually costs is **reviewer and
author confidence**. A paper whose central claim is "the router is a free-energy gradient" should
not need a per-expert-kind sign table to reconcile itself with its own implementation.
