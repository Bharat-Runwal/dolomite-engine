"""CPU validation for the register_generation_mode fix.

We instantiate a tiny RegisterEnergyModel and compare three forward modes
across multiple decode steps:

  M1: use_cache=False from start (recompute everything each call). The reference.
  M2: use_cache=True with register_generation_mode='no_cache' (the fix).
       Goes through HF prepare_inputs_for_generation path → drops past_kv each step.
  M3: use_cache=True with register_generation_mode='bypass' (the bug).

Expect M1 ≈ M2 (token-by-token logit match, modulo numerics).
Expect M3 to diverge from M1 (the bug we found).
"""
import os
import sys
import torch

# Use deterministic settings
torch.manual_seed(42)
torch.set_default_dtype(torch.float32)

sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')

from lm_engine.hf_models.models.register_energy.config import RegisterEnergyConfig
from lm_engine.hf_models.models.register_energy.main import RegisterEnergyForCausalLM


def make_tiny_config(n_registers: int = 16, gen_mode: str = "bypass"):
    return RegisterEnergyConfig(
        n_registers=n_registers,
        register_generation_mode=gen_mode,
        register_start_layer=0,
        # base config
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        layer_iterations=[1, 1],
        num_iterations=1,
        position_embedding_type="rope",
        rope_dim=8,
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
            {"sequence_mixer_type": "softmax_attention", "num_attention_heads": 4,
             "num_key_value_heads": 4, "add_bias": False, "attention_multiplier": 0.125}
        ] * 2,
        mlp_blocks=[
            {"mlp_type": "MLP", "intermediate_size": 128, "activation_function": "swiglu", "add_bias": False}
        ] * 2,
    )


def build_model(gen_mode: str, seed: int = 42, n_registers: int = 16):
    torch.manual_seed(seed)
    cfg = make_tiny_config(n_registers=n_registers, gen_mode=gen_mode)
    model = RegisterEnergyForCausalLM(cfg).eval()
    return model


def get_logits_full_recompute(model, input_ids, n_steps=5):
    """Mode M1: never use cache.  Re-run forward(full_ids) each step."""
    logits_list = []
    cur = input_ids.clone()
    attn = torch.ones_like(cur)
    with torch.no_grad():
        for _ in range(n_steps):
            out = model(input_ids=cur, attention_mask=attn, use_cache=False)
            last_logits = out.logits[:, -1, :]
            logits_list.append(last_logits.clone())
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
    return torch.stack(logits_list, dim=0)  # [n_steps, B, V]


def get_logits_with_kvcache(model, input_ids, n_steps=5):
    """Simulate the HF generate() loop: prefill once, then decode one token at a time
    using past_key_values; consult the model's prepare_inputs_for_generation to
    decide whether to slice input_ids or pass them in full.
    """
    logits_list = []
    cur_ids = input_ids.clone()
    attn = torch.ones_like(cur_ids)

    with torch.no_grad():
        # --- Prefill ---
        out = model(input_ids=cur_ids, attention_mask=attn, use_cache=True)
        last_logits = out.logits[:, -1, :]
        logits_list.append(last_logits.clone())
        next_tok = last_logits.argmax(dim=-1, keepdim=True)
        pkv = out.past_key_values

        for _ in range(n_steps - 1):
            cur_ids = torch.cat([cur_ids, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)

            # Determine what input_ids to pass: simulate HF generate's default
            # which slices to the last token when past_key_values exists.
            # Our override of prepare_inputs_for_generation may discard pkv
            # (for no_cache mode), in which case we should pass the full ids.
            n_reg = getattr(model.config, 'n_registers', 0)
            register_start = getattr(model.config, 'register_start_layer', 0)
            gen_mode = getattr(model.config, 'register_generation_mode', 'bypass')
            force_recompute = (
                (n_reg > 0 and register_start > 0)
                or (n_reg > 0 and gen_mode == 'no_cache')
            )
            if force_recompute:
                fwd_pkv = None
                fwd_input_ids = cur_ids
                fwd_attn = attn
            else:
                fwd_pkv = pkv
                fwd_input_ids = next_tok  # last token only
                # In bypass mode the override pads the mask by n_reg
                mask = attn
                if n_reg > 0 and register_start == 0 and pkv is not None:
                    pad = torch.ones(mask.shape[0], n_reg, dtype=mask.dtype, device=mask.device)
                    mask = torch.cat([pad, mask], dim=1)
                fwd_attn = mask

            out = model(
                input_ids=fwd_input_ids,
                past_key_values=fwd_pkv,
                attention_mask=fwd_attn,
                use_cache=True,
            )
            last_logits = out.logits[:, -1, :]
            logits_list.append(last_logits.clone())
            next_tok = last_logits.argmax(dim=-1, keepdim=True)
            pkv = out.past_key_values
    return torch.stack(logits_list, dim=0)


def main():
    n_registers = 16
    seed = 7
    B = 2
    T_prompt = 6
    n_steps = 4

    torch.manual_seed(0)
    prompt = torch.randint(2, 100, (B, T_prompt))

    # --- Build three models with IDENTICAL weights but different modes ---
    m1 = build_model("bypass", seed=seed, n_registers=n_registers)  # weights frozen
    m2 = build_model("no_cache", seed=seed, n_registers=n_registers)
    m3 = build_model("bypass", seed=seed, n_registers=n_registers)
    # Verify weights are identical (same seed)
    for (n1, p1), (n2, p2) in zip(m1.named_parameters(), m2.named_parameters()):
        assert torch.allclose(p1, p2), f"weight mismatch: {n1}"
    print("[setup] m1=m2=m3 weights confirmed identical (same seed)")
    print(f"[setup] n_registers={n_registers}, hidden=64, n_layers=2, prompt T={T_prompt}, n_decode_steps={n_steps}")

    # --- M1: full-recompute reference ---
    logits_m1 = get_logits_full_recompute(m1, prompt, n_steps=n_steps)

    # --- M2: cached decode with no_cache mode (the fix) ---
    logits_m2 = get_logits_with_kvcache(m2, prompt, n_steps=n_steps)

    # --- M3: cached decode with bypass mode (the bug) ---
    logits_m3 = get_logits_with_kvcache(m3, prompt, n_steps=n_steps)

    # --- Report ---
    print("\nMax abs diff vs reference M1 (use_cache=False, full recompute):")
    print(f"{'step':>4} {'M2 (no_cache fix)':>22} {'M3 (bypass / bug)':>22}")
    for k in range(n_steps):
        d12 = (logits_m1[k] - logits_m2[k]).abs().max().item()
        d13 = (logits_m1[k] - logits_m3[k]).abs().max().item()
        print(f"{k:>4} {d12:>22.6e} {d13:>22.6e}")

    overall12 = (logits_m1 - logits_m2).abs().max().item()
    overall13 = (logits_m1 - logits_m3).abs().max().item()
    print(f"\nOverall: M1↔M2 max-abs={overall12:.3e}, M1↔M3 max-abs={overall13:.3e}")

    if overall12 < 1e-4:
        print("PASS: no_cache mode matches reference (registers re-prepended each step).")
    else:
        print(f"FAIL: no_cache mode diverges from reference by {overall12:.3e}.")
    if overall13 > 1e-3:
        print(f"CONFIRMED BUG: bypass mode diverges from reference by {overall13:.3e}.")
    else:
        print(f"WARNING: bypass mode does not appear to diverge much — bug may not reproduce on this tiny config.")


if __name__ == "__main__":
    main()
