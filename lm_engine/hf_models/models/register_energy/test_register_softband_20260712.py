"""Smoke test for the SOFT-BAND register attention-balance auxiliary loss.

Verifies, for register_balance_mode="softband":
  1. register_attn_balance_loss is not None, finite, and has grad (backward runs,
     gradient reaches register_embeddings).
  2. Arm activation via band-straddle (fixed model/inputs ⇒ fixed per-layer
     lr=log(rho); we MOVE the band relative to lr):
       - floor config (rho_lo >> rho ⇒ lr below floor)   → lower arm large
       - ceiling config (rho_hi << rho ⇒ lr above ceiling)→ upper arm large
       - in-band config (lr strictly inside band)         → loss ≈ 0
     This directly demonstrates "high per-key ratio → upper (anti-bypass) arm,
     low per-key ratio → lower (floor) arm", which is the intended U-shape.
  3. register_balance_hi_weight scales the upper (anti-bypass) arm.
  4. Supplementary: report per-layer log-ratio for default/×100/×0.001 register
     embeddings (RMSNorm largely absorbs a scalar rescale — reported, not asserted).

Uses a tiny model (n_registers=16, hidden=256, 2 layers: 1 softmax-attn GPT + 1
energy-attn block) so BOTH the attention.py and energy_attention.py capture paths
are exercised.
"""
import math
import sys

import torch

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')

from lm_engine.hf_models.models.register_energy.config import RegisterEnergyConfig
from lm_engine.hf_models.models.register_energy.main import RegisterEnergyForCausalLM


def make_config(coef=1.0, mode="softband", rho_lo=1.0, rho_hi=4.0, beta=1.0,
                hi_weight=1.0, register_start_layer=0,
                n_registers=16, hidden_size=256, n_heads=4):
    return RegisterEnergyConfig(
        n_registers=n_registers,
        register_generation_mode="bypass",
        register_start_layer=register_start_layer,
        register_attn_balance_coef=coef,
        register_attn_balance_threshold=0.1,
        register_balance_mode=mode,
        register_balance_rho_lo=rho_lo,
        register_balance_rho_hi=rho_hi,
        register_balance_beta=beta,
        register_balance_hi_weight=hi_weight,
        vocab_size=128,
        hidden_size=hidden_size,
        num_layers=2,
        layer_iterations=[1, 1],
        num_iterations=1,
        position_embedding_type="rope",
        rope_dim=hidden_size // n_heads,
        rope_theta=10000.0,
        max_position_embeddings=128,
        normalization_function="rmsnorm",
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        num_pre_layers=0,
        num_post_layers=0,
        init_method="normal",
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=0,
        pad_token_id=1,
        sequence_mixer_blocks=[
            {"sequence_mixer_type": "softmax_attention", "num_attention_heads": n_heads,
             "num_key_value_heads": n_heads, "add_bias": False,
             "attention_multiplier": 1.0 / (hidden_size // n_heads) ** 0.5},
            {"sequence_mixer_type": "energy_attention", "num_attention_heads": n_heads,
             "num_key_value_heads": n_heads, "add_bias": False,
             "attention_multiplier": 1.0 / (hidden_size // n_heads) ** 0.5},
        ],
        mlp_blocks=[
            {"mlp_type": "MLP", "intermediate_size": 512, "activation_function": "swiglu", "add_bias": False},
            {"mlp_type": "Energy_MLP", "intermediate_size": 512, "activation_function": "gelu", "add_bias": False},
        ],
    )


def build_model(seed=42, **kw):
    torch.manual_seed(seed)
    return RegisterEnergyForCausalLM(make_config(**kw))


def run_one(model, input_ids, backward=False):
    model.train()
    attn = torch.ones_like(input_ids)
    labels = input_ids.clone()
    out = model(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    return out


def read_logratios(model):
    """Read per-block captured log-ratio (after a training forward)."""
    tf = model.transformer
    vals = []
    for i in range(len(tf.h)):
        ba = tf._get_block_attn(tf.h[i])
        lr = getattr(ba, '_register_logratio', None)
        vals.append(None if lr is None else lr.item())
    return vals


def main():
    B, T = 2, 16
    torch.manual_seed(123)
    input_ids = torch.randint(2, 100, (B, T))
    failures = []

    # ---- 1. wiring: loss present, finite, grad ------------------------------
    m = build_model(seed=7, mode="softband", coef=1.0, rho_lo=1.0, rho_hi=4.0)
    out = run_one(m, input_ids)
    loss = out.register_attn_balance_loss
    print("== 1. softband wiring ==")
    if loss is None:
        failures.append("softband register_attn_balance_loss is None")
        print("  FAIL: loss is None")
    else:
        finite = bool(torch.isfinite(loss).all())
        rg = bool(loss.requires_grad)
        print(f"  loss={loss.item():.6f}  finite={finite}  requires_grad={rg}")
        if not finite:
            failures.append("softband loss non-finite")
        if not rg:
            failures.append("softband loss has no grad")
    lrs = read_logratios(m)
    print(f"  per-block captured log-ratio (lr=log rho): {lrs}")

    # gradient reaches register_embeddings
    m_g = build_model(seed=7, mode="softband", coef=1.0)
    m_g.train()
    attn = torch.ones_like(input_ids)
    out_g = m_g(input_ids=input_ids, attention_mask=attn, use_cache=False)
    out_g.register_attn_balance_loss.backward()
    g = m_g.transformer.register_embeddings.grad
    gnorm = 0.0 if g is None else g.norm().item()
    print(f"  grad norm on register_embeddings = {gnorm:.6f}")
    if not (gnorm > 0):
        failures.append("no grad flows to register_embeddings in softband")

    # ---- 2. arm activation via band-straddle --------------------------------
    # Fixed model/inputs ⇒ fixed lr (near 0 for random init, |lr|<5). Move band.
    print("\n== 2. arm activation (band straddle, fixed lr) ==")
    L = 5.0
    # floor active: put band well ABOVE lr → lower arm ≈ beta*(log(rho_lo)-lr)
    m_floor = build_model(seed=7, mode="softband", coef=1.0,
                          rho_lo=math.exp(L), rho_hi=math.exp(4 * L))
    loss_floor = run_one(m_floor, input_ids).register_attn_balance_loss.item()
    # ceiling active: put band well BELOW lr → upper arm ≈ beta*(lr-log(rho_hi))
    m_ceil = build_model(seed=7, mode="softband", coef=1.0,
                         rho_lo=math.exp(-4 * L), rho_hi=math.exp(-L))
    loss_ceil = run_one(m_ceil, input_ids).register_attn_balance_loss.item()
    # in band: lr strictly inside → both arms ≈ 0
    m_band = build_model(seed=7, mode="softband", coef=1.0,
                         rho_lo=math.exp(-L), rho_hi=math.exp(L))
    loss_band = run_one(m_band, input_ids).register_attn_balance_loss.item()
    print(f"  floor-active   (band above lr) loss = {loss_floor:.4f}   (expect large, lower arm)")
    print(f"  ceiling-active (band below lr) loss = {loss_ceil:.4f}   (expect large, upper arm)")
    print(f"  in-band        (lr inside band) loss = {loss_band:.6f}  (expect ~0)")
    if not (loss_floor > 2.0):
        failures.append(f"floor arm did not activate (loss_floor={loss_floor:.4f})")
    if not (loss_ceil > 2.0):
        failures.append(f"ceiling arm did not activate (loss_ceil={loss_ceil:.4f})")
    if not (loss_band < 0.1):
        failures.append(f"in-band loss not ~0 (loss_band={loss_band:.6f})")

    # ---- 3. hi_weight scales the upper arm ----------------------------------
    print("\n== 3. hi_weight scales upper (anti-bypass) arm ==")
    m_ceil2 = build_model(seed=7, mode="softband", coef=1.0, hi_weight=2.0,
                          rho_lo=math.exp(-4 * L), rho_hi=math.exp(-L))
    loss_ceil2 = run_one(m_ceil2, input_ids).register_attn_balance_loss.item()
    ratio = loss_ceil2 / max(loss_ceil, 1e-9)
    print(f"  hi_weight=1 → {loss_ceil:.4f} ;  hi_weight=2 → {loss_ceil2:.4f} ;  ratio = {ratio:.3f} (expect ~2)")
    if not (1.8 < ratio < 2.2):
        failures.append(f"hi_weight did not scale upper arm ~2x (ratio={ratio:.3f})")

    # ---- 4. supplementary: register magnitude → log-ratio -------------------
    print("\n== 4. supplementary: register-magnitude vs captured log-ratio (energy block) ==")
    def lr_energy_for_scale(scale):
        mm = build_model(seed=7, mode="softband", coef=1.0)
        if scale != 1.0:
            with torch.no_grad():
                mm.transformer.register_embeddings.data.mul_(scale)
        run_one(mm, input_ids)
        return read_logratios(mm)[1]  # energy block (index 1)
    lr_default = lr_energy_for_scale(1.0)
    lr_huge = lr_energy_for_scale(100.0)
    lr_tiny = lr_energy_for_scale(0.001)
    print(f"  lr(×0.001)={lr_tiny:.4f}   lr(default)={lr_default:.4f}   lr(×100)={lr_huge:.4f}")
    print("  (RMSNorm largely absorbs a scalar register rescale — reported, not asserted)")

    # ---- 5. ceiling mode default reproduces OLD loss exactly ----------------
    # A softband coef>0 model in ceiling mode must equal the legacy ceiling hinge.
    print("\n== 5. default ceiling mode == legacy hinge (behaviour unchanged) ==")
    m_ceilmode = build_model(seed=7, mode="ceiling", coef=1.0)
    out_ceilmode = m_ceilmode(input_ids=input_ids, attention_mask=torch.ones_like(input_ids),
                              labels=input_ids.clone(), use_cache=False)
    ceil_loss = out_ceilmode.register_attn_balance_loss
    # recompute the legacy hinge sum from captured masses over active layers
    masses = []
    for i in range(len(m_ceilmode.transformer.h)):
        ba = m_ceilmode.transformer._get_block_attn(m_ceilmode.transformer.h[i])
        mval = getattr(ba, '_register_attn_mass', None)
        if mval is not None:
            masses.append(mval.item())
    thr = 0.1
    expect = 1.0 * sum(max(0.0, mm - thr) for mm in masses)  # coef * sum hinge
    got = None if ceil_loss is None else ceil_loss.item()
    print(f"  ceiling loss = {got}   recomputed coef*sum(max(0,mass-thr)) = {expect:.6f}")
    print(f"  per-block masses = {masses}")
    if got is None or abs(got - expect) > 1e-5:
        failures.append(f"ceiling mode loss mismatch (got={got}, expect={expect:.6f})")

    print("\n" + "=" * 60)
    if failures:
        print("SMOKE TEST: FAIL")
        for f in failures:
            print("  - " + f)
        sys.exit(1)
    else:
        print("SMOKE TEST: PASS (softband wiring + grad + both arms + hi_weight + ceiling-default unchanged)")


if __name__ == "__main__":
    main()
