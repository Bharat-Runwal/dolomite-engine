# Register-decode bug (2026-06-29)

## TL;DR

Every register-equipped checkpoint trained in this repo (V0, V1, V73, V41, V56, V76, h1,
Hopfield variants — all use `model_type: register_energy`) has its **generation** path
corrupted by an off-by-R RoPE position bug in the cached-decode branch of
`register_energy/model.py`. Likelihood-based evaluations (perplexity, avg10_norm,
log-prob accuracy) are **unaffected** because they run a single prefill forward — no
cached decode. Generation-based evaluations (BBH multiple-choice via free-form
generation, GSM8K, anything that uses `model.generate()`) are **catastrophically
degraded**.

**Empirical confirmation** — `hop_r256` step 24000 (Hopfield-MEAN + 256 registers,
8G+1Ex6 d=1536):
- `avg10_norm` = 0.5112 (matches base 0.5135 — registers don't hurt logp scoring)
- BBH = **0.0738** (way below random ~0.23-0.25)
- GSM8K = **0.0%**

## The bug

The model's `forward()` does the following during a cached-decode step
(`use_cache=True`, `T_q=1`, prefill already populated):

1. `register_energy/main.py:prepare_inputs_for_generation` pads the
   `attention_mask` by `R` ones at the front so the new token can attend to the
   `R+T` cached register+content KVs.
2. `register_energy/model.py:forward` falls through to `super().forward(...)`
   (the plain `EnergyModel.forward`).  No register re-prepending.
3. Inside `_prepare_a_bunch_of_stuff`, `_get_position_ids` computes
   `position_ids = attention_mask.cumsum(-1) - 1`, then slices
   `[past_length:key_length]`.  With `past_length = R + T` and `key_length = R + T + 1`,
   the new token gets `position_id = R + T + k`.
4. But during **training/prefill**, the content tokens were assigned positions
   `0..T-1` (in `_make_extended`: `pos_ids_exp` from `_prepare_a_bunch_of_stuff` is
   `0..T-1`, and registers got `0..R-1`).  So the "next content position" should be
   `T + k`, not `R + T + k`.
5. RoPE rotations on the new token are therefore off by R — its attention with the
   cached register KVs (and with cached content KVs) uses wrong relative offsets,
   producing meaningless logits.

Additionally, `config.register_generation_mode` was documented (lines 183-190 of
the original file) as having `"bypass"` and `"no_cache"` modes — but **the code
never read it**.  Patching a config from `"bypass"` to `"no_cache"` did nothing.

## Why training is fine

Training runs only `forward(full_ids, use_cache=False)` — i.e. the "Prefill or
training" branch in model.py.  That branch correctly calls `_make_extended` which
prepends register embeddings and rotates the full sequence.  Trained checkpoints
are therefore valid; only the inference-time decode path is wrong.

## Why V1 (uniform EGPT) was anecdotally "spared"

The user's prior memory note says:
> Registers raise zero-shot avg by ~2-3pp but kill GSM8K generation in most archs
> (V0 GPT, h1, V73, V56, V41, V76); flex eval does NOT recover them. Safe in
> uniform EGPT (V1 12x1, V1-400M)

All these architectures share `model_type: register_energy`, so they share this
bug.  The V1 robustness is probably a model-class effect (uniform deep recurrence
washes out RoPE position errors over many iterations), not an architectural
immunity.  **All such results should be re-verified with the fix applied.**

## The fix

Wire up `register_generation_mode` properly.  Default kept as `"bypass"` for
backward-compat reproduction of prior eval numbers, but the user can switch to
`"no_cache"` at inference time (config.json patch — no retraining needed).

| Mode | Behaviour | Speed | Correctness |
| --- | --- | --- | --- |
| `"bypass"` (default) | Cached decode with off-by-R bug | Fast (~1x) | **WRONG** |
| `"no_cache"` (recommended) | Drops cache each step; re-prefills full R+T sequence | ~50x slower for BBH | Matches training |
| `"persistent_cache"` | Not implemented; raises | — | — |

Two places are touched:

- **`register_energy/main.py`** (`prepare_inputs_for_generation`): in `no_cache`
  mode, drops `past_key_values` to None so HF generate passes the full input_ids
  back into `forward`.
- **`register_energy/model.py`** (`forward`): validates the mode; provides a
  safety-net that drops the cache if a non-HF caller invokes `forward(full_ids,
  use_cache=True)` with `no_cache` mode.  Also raises on unknown modes / on the
  unimplemented `persistent_cache`.

Trained-time behaviour is **unchanged**: no_cache only kicks in when
`not self.training`.

## How to use the fix

For an existing checkpoint, edit `config.json` (no retraining):

```json
{
  "model_type": "register_energy",
  "n_registers": 256,
  "register_generation_mode": "no_cache",
  ...
}
```

Then re-run any generation-based eval (BBH, GSM8K, anything via `.generate()`).
This should be ~50x slower per BBH step but produce correct logits.

## Validation

CPU test (n_registers=8, hidden=32, n_layers=2):

```
                       step 0     step 1     step 2     step 3
bypass mode (bug)      0.0e+0     3.8e-3     6.9e-3     9.4e-3   <-- diverges, compounds
no_cache mode (fix)    0.0e+0     0.0e+0     0.0e+0     0.0e+0   <-- matches reference
```

Test script: `/tmp/test_register_decode.py` (kept locally; re-runnable).

Reference is "full recompute": each step calls `forward(full_ids, use_cache=False)`,
which is the same path training uses — guaranteed to match the prefill numerics
exactly because there is no cache for either run.

## What still needs to happen

1. Re-run BBH / GSM8K on every register checkpoint with `register_generation_mode:
   "no_cache"` and update the numbers in RESULTS.md / paper.
2. The user's memory note "Registers raise zero-shot avg by ~2-3pp but kill GSM8K
   generation in most archs" needs re-verification: if the fix recovers GSM8K,
   the conclusion "registers kill generation" is invalidated and registers
   become a real win.
3. Consider switching the default to `"no_cache"` after the historical eval-number
   sweep is recorded — bypass is the wrong default for any future work.
