"""Validate persistent_cache mode bit-by-bit against the no_cache reference.

Compares per-step logits (not just argmax tokens, which can mask divergences in
an untrained tiny model) for the three gen modes.
"""
import sys
import torch

sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')

from lm_engine.hf_models.models.register_energy.config import RegisterEnergyConfig
from lm_engine.hf_models.models.register_energy.main import RegisterEnergyForCausalLM


def make_tiny_config(n_registers=16, gen_mode="no_cache", max_pos=512):
    return RegisterEnergyConfig(
        n_registers=n_registers,
        register_generation_mode=gen_mode,
        register_start_layer=0,
        vocab_size=128,
        hidden_size=64,
        num_layers=2,
        layer_iterations=[1, 1],
        num_iterations=1,
        position_embedding_type="rope",
        rope_dim=8,
        rope_theta=10000.0,
        max_position_embeddings=max_pos,
        normalization_function="rmsnorm",
        initializer_range=0.02,
        layer_norm_epsilon=1e-5,
        num_pre_layers=0,
        num_post_layers=0,
        init_method="normal",
        tie_word_embeddings=True,
        bos_token_id=0,
        eos_token_id=2,
        pad_token_id=1,
        sequence_mixer_blocks=[
            {"sequence_mixer_type": "softmax_attention", "num_attention_heads": 4,
             "num_key_value_heads": 4, "add_bias": False, "attention_multiplier": 0.125}
        ] * 2,
        mlp_blocks=[
            {"mlp_type": "MLP", "intermediate_size": 128, "activation_function": "swiglu", "add_bias": False}
        ] * 2,
    )


def build_model(gen_mode, seed=42, n_registers=16):
    torch.manual_seed(seed)
    cfg = make_tiny_config(n_registers=n_registers, gen_mode=gen_mode)
    model = RegisterEnergyForCausalLM(cfg).eval()
    return model


def driven_logits(model, prompt, n_steps, forced_next_tokens):
    """Walk the model through forced tokens, returning the per-step logits[:, -1, :].

    The model's prepare_inputs_for_generation is consulted each step (so the
    behaviour matches HF generate exactly).  Returns [n_steps+1, B, V].
    """
    model_inputs = {
        "input_ids": prompt,
        "attention_mask": torch.ones_like(prompt),
        "past_key_values": None,
    }
    all_logits = []
    with torch.no_grad():
        for k in range(n_steps + 1):
            # Build inputs via prepare_inputs_for_generation
            inputs = model.prepare_inputs_for_generation(
                model_inputs["input_ids"],
                past_key_values=model_inputs["past_key_values"],
                attention_mask=model_inputs["attention_mask"],
            )
            out = model(**inputs)
            last_logits = out.logits[:, -1, :].clone()
            all_logits.append(last_logits)

            if k == n_steps:
                break
            # Append next FORCED token (same across all modes for fair comparison)
            next_tok = forced_next_tokens[:, k:k+1]
            model_inputs["input_ids"] = torch.cat(
                [model_inputs["input_ids"], next_tok], dim=1
            )
            model_inputs["attention_mask"] = torch.cat(
                [model_inputs["attention_mask"],
                 torch.ones_like(next_tok)], dim=1
            )
            model_inputs["past_key_values"] = out.past_key_values
    return torch.stack(all_logits, dim=0)  # [n_steps+1, B, V]


def reference_logits(model, prompt, n_steps, forced_next_tokens):
    """Reference: run forward(full_ids, use_cache=False) each step."""
    cur = prompt.clone()
    attn = torch.ones_like(cur)
    out_logits = []
    with torch.no_grad():
        for k in range(n_steps + 1):
            out = model(input_ids=cur, attention_mask=attn, use_cache=False)
            out_logits.append(out.logits[:, -1, :].clone())
            if k == n_steps:
                break
            cur = torch.cat([cur, forced_next_tokens[:, k:k+1]], dim=1)
            attn = torch.cat([attn, torch.ones_like(forced_next_tokens[:, k:k+1])], dim=1)
    return torch.stack(out_logits, dim=0)


def main():
    n_registers = 32
    seed = 13
    B = 2
    T_prompt = 16
    n_steps = 10

    torch.manual_seed(2)
    prompt = torch.randint(3, 120, (B, T_prompt))
    forced = torch.randint(3, 120, (B, n_steps))

    m_ref = build_model("no_cache", seed=seed, n_registers=n_registers)
    ref = reference_logits(m_ref, prompt, n_steps, forced)

    print(f"Setup: n_registers={n_registers}, T_prompt={T_prompt}, n_decode={n_steps}, B={B}\n")

    for mode in ("no_cache", "persistent_cache", "bypass"):
        m = build_model(mode, seed=seed, n_registers=n_registers)
        log = driven_logits(m, prompt, n_steps, forced)
        # Compute per-step max-abs diff vs reference logits
        per_step = (log - ref).abs().reshape(log.shape[0], -1).max(dim=-1).values
        overall = per_step.max().item()
        print(f"[{mode:>18}] per-step max-abs diff: " +
              " ".join(f"{v:.3e}" for v in per_step.tolist()))
        print(f"    overall: {overall:.3e}  " +
              ("PASS (matches ref)" if overall < 1e-4 else
               "FAIL — diverges from ref" if overall > 1e-3 else
               "near-pass"))
        print()


if __name__ == "__main__":
    main()
