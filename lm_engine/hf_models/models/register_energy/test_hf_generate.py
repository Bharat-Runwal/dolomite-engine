"""Reproduce the HF .generate() OOB on a tiny RegisterEnergyModel.

The CPU smoke (test_decode_bug_fix.py) confirms our manual loop emulating HF's
generate works.  But when lm-eval-harness invokes HF's actual generate() through
HFLM, the no_cache path crashes with CUDA OOB ("vectorized gather kernel index
out of bounds").

This script reproduces on CPU (no CUDA needed) by calling the model's HF
generate() directly.  CPU won't actually OOB the same way (no fused gather
kernel), but indexing bugs will surface as wrong outputs or Python-level
IndexErrors / wrong-shape errors.  To confirm OOB-style bugs we should also
run a GPU pass.
"""
import sys
import torch

sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')

from lm_engine.hf_models.models.register_energy.config import RegisterEnergyConfig
from lm_engine.hf_models.models.register_energy.main import RegisterEnergyForCausalLM


def make_tiny_config(n_registers=16, gen_mode="no_cache", max_pos=256):
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


def manual_reference_generate(model, input_ids, max_new_tokens=10):
    """Reference: re-run forward(full_ids, use_cache=False) each step."""
    cur = input_ids.clone()
    attn = torch.ones_like(cur)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            out = model(input_ids=cur, attention_mask=attn, use_cache=False)
            next_tok = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            cur = torch.cat([cur, next_tok], dim=1)
            attn = torch.cat([attn, torch.ones_like(next_tok)], dim=1)
    return cur


def build_model(gen_mode, seed=42, n_registers=16):
    torch.manual_seed(seed)
    cfg = make_tiny_config(n_registers=n_registers, gen_mode=gen_mode)
    model = RegisterEnergyForCausalLM(cfg).eval()
    return model


def run_hf_generate(model, input_ids, max_new_tokens=10):
    """Use HF transformers generate() — this is what lm-eval-harness uses."""
    from transformers import GenerationConfig
    # Avoid the custom dolomite generate(); we want HF's
    # GenerationMixin.generate which uses prepare_inputs_for_generation.
    # The mixin's `generate()` (custom) shadows it — patch it out for the test:
    import lm_engine.hf_models.mixins.dense.main as m
    original = m.CausalLMModelMixin.generate
    try:
        # Replace with HF's generate
        from transformers import GenerationMixin
        m.CausalLMModelMixin.generate = GenerationMixin.generate
        attn = torch.ones_like(input_ids)
        # set min/max so HF doesn't stop early
        gen_cfg = GenerationConfig(
            max_new_tokens=max_new_tokens, do_sample=False,
            pad_token_id=1, eos_token_id=None,  # disable eos to force max_new_tokens
        )
        out_ids = model.generate(
            input_ids=input_ids, attention_mask=attn,
            generation_config=gen_cfg, use_cache=True,
        )
        return out_ids
    finally:
        m.CausalLMModelMixin.generate = original


def main():
    # Use more registers and a longer prompt so bypass's off-by-R bug
    # (RoPE rotation error) accumulates enough to surface in token-id space.
    n_registers = 32
    seed = 13
    B = 2
    T_prompt = 12
    n_steps = 10

    torch.manual_seed(2)
    prompt = torch.randint(3, 100, (B, T_prompt))

    print(f"Prompt shape: {prompt.shape}")
    print(f"Prompt: {prompt.tolist()}")

    # M0: reference (no cache, manual recompute)
    m_ref = build_model("no_cache", seed=seed, n_registers=n_registers)
    ref_out = manual_reference_generate(m_ref, prompt, max_new_tokens=n_steps)
    print(f"\n[REF manual no_cache] output ids: {ref_out.tolist()}")

    # M1: HF generate with no_cache mode
    m_nc = build_model("no_cache", seed=seed, n_registers=n_registers)
    try:
        out_nc = run_hf_generate(m_nc, prompt, max_new_tokens=n_steps)
        print(f"\n[HF gen no_cache  ] output ids: {out_nc.tolist()}")
        diff_nc = (ref_out != out_nc).sum().item()
        print(f"   token-id diff vs ref: {diff_nc}")
    except Exception as e:
        print(f"\n[HF gen no_cache  ] CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # M2: HF generate with persistent_cache mode (currently NotImplementedError)
    m_pc = build_model("persistent_cache", seed=seed, n_registers=n_registers)
    try:
        out_pc = run_hf_generate(m_pc, prompt, max_new_tokens=n_steps)
        print(f"\n[HF gen persistent] output ids: {out_pc.tolist()}")
        diff_pc = (ref_out != out_pc).sum().item()
        print(f"   token-id diff vs ref: {diff_pc}")
    except Exception as e:
        print(f"\n[HF gen persistent] CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()

    # M3: HF generate with bypass (buggy reference)
    m_bp = build_model("bypass", seed=seed, n_registers=n_registers)
    try:
        out_bp = run_hf_generate(m_bp, prompt, max_new_tokens=n_steps)
        print(f"\n[HF gen bypass    ] output ids: {out_bp.tolist()}")
        diff_bp = (ref_out != out_bp).sum().item()
        print(f"   token-id diff vs ref: {diff_bp}")
    except Exception as e:
        print(f"\n[HF gen bypass    ] CRASHED: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
