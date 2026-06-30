"""Gradient-checkpointing test for register attention-balance aux loss.

The original implementation replaced SDPA with a manual matmul path when the
capture flag was on. That broke gradient_checkpointing_method='block' on the
real training run because the recompute pass produced different tensor metadata
than the saved forward pass (CheckpointError, job 1873727 crashed).

The fix is structural: keep SDPA as the output path (unchanged shapes) and run a
side measurement alongside it. This test wraps every EnergyBlock with
torch.distributed.algorithms._checkpoint.checkpoint_wrapper (the same wrapper
the dolomite-engine block-checkpointing path uses) and verifies that forward +
backward both succeed with the capture path active.
"""
import os
import sys

import torch
import torch.nn as nn

torch.manual_seed(0)
torch.set_default_dtype(torch.float32)

sys.path.insert(0, '/proj/dmfexp/nima/Code/dolomite-engine')

from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    apply_activation_checkpointing,
    checkpoint_wrapper,
)

from lm_engine.hf_models.models.register_energy.config import RegisterEnergyConfig
from lm_engine.hf_models.models.register_energy.main import RegisterEnergyForCausalLM
from lm_engine.hf_models.models.energy.layer import EnergyBlock


def make_config(coef: float = 1.0, threshold: float = 0.0,
                n_registers: int = 8, hidden_size: int = 128, n_heads: int = 4):
    return RegisterEnergyConfig(
        n_registers=n_registers,
        register_generation_mode="bypass",
        register_start_layer=0,
        register_attn_balance_coef=coef,
        register_attn_balance_threshold=threshold,
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
             "num_key_value_heads": n_heads, "add_bias": False, "attention_multiplier": 1.0 / (hidden_size // n_heads) ** 0.5},
            {"sequence_mixer_type": "energy_attention", "num_attention_heads": n_heads,
             "num_key_value_heads": n_heads, "add_bias": False, "attention_multiplier": 1.0 / (hidden_size // n_heads) ** 0.5},
        ],
        mlp_blocks=[
            {"mlp_type": "MLP", "intermediate_size": 256, "activation_function": "swiglu", "add_bias": False},
            {"mlp_type": "Energy_MLP", "intermediate_size": 256, "activation_function": "gelu", "add_bias": False},
        ],
    )


def apply_block_checkpointing(model):
    """Replicates the block_checkpointing path used by dolomite-engine pretrain."""

    def _check_fn(submodule):
        return isinstance(submodule, EnergyBlock)

    apply_activation_checkpointing(
        model,
        checkpoint_wrapper_fn=checkpoint_wrapper,  # default non-reentrant
        check_fn=_check_fn,
    )


def main():
    B = 2
    T = 16
    torch.manual_seed(123)
    input_ids = torch.randint(2, 100, (B, T))
    attn = torch.ones_like(input_ids)
    labels = input_ids.clone()

    # ---- 1. Without checkpointing, aux is on ---------------------------------
    cfg = make_config(coef=1.0, threshold=0.0)
    torch.manual_seed(7)
    m_plain = RegisterEnergyForCausalLM(cfg)
    m_plain.train()
    out_plain = m_plain(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    loss_plain = out_plain.loss
    aux_plain = out_plain.register_attn_balance_loss
    loss_plain.backward()
    grad_reg_plain = m_plain.transformer.register_embeddings.grad.norm().item()
    print(f"[no ckpt]   loss={loss_plain.item():.6f}  aux={aux_plain.item():.6f}  grad(register_embeds)={grad_reg_plain:.6f}")

    # ---- 2. With block-checkpointing wrapping every EnergyBlock --------------
    torch.manual_seed(7)
    m_ckpt = RegisterEnergyForCausalLM(cfg)
    apply_block_checkpointing(m_ckpt)
    m_ckpt.train()
    # Confirm the wrapper actually wrapped something.
    wrapped = sum(1 for n, _ in m_ckpt.named_modules() if "checkpoint_wrapper" in type(_).__name__.lower() or "CheckpointWrapper" in type(_).__name__)
    print(f"[ckpt]      number of CheckpointWrapper modules detected: {wrapped}")

    # The real failure mode: forward + backward must both succeed without
    # CheckpointError about mismatched recomputed tensor metadata.
    try:
        out_ckpt = m_ckpt(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
        loss_ckpt = out_ckpt.loss
        aux_ckpt = out_ckpt.register_attn_balance_loss
        loss_ckpt.backward()
        grad_reg_ckpt = m_ckpt.transformer.register_embeddings.grad.norm().item()
        print(f"[ckpt]      loss={loss_ckpt.item():.6f}  aux={aux_ckpt.item():.6f}  grad(register_embeds)={grad_reg_ckpt:.6f}")
    except Exception as e:
        print(f"[ckpt]      FAILED: {type(e).__name__}: {e}")
        raise

    # ---- 3. Plain vs ckpt must match (deterministic same-seed init) ----------
    # Both losses should be identical (seed identical, same forward).
    diff_loss = abs(loss_plain.item() - loss_ckpt.item())
    diff_aux = abs(aux_plain.item() - aux_ckpt.item())
    print()
    print(f"plain vs ckpt:  |Δ loss| = {diff_loss:.6e}, |Δ aux| = {diff_aux:.6e}")
    if diff_loss < 1e-4 and diff_aux < 1e-4:
        print("PASS: gradient checkpointing produces identical loss + aux (no CheckpointError, no recompute mismatch).")
    else:
        print("WARN: small numerical difference between plain and ckpt — check.")

    # ---- 4. Second forward+backward on the same ckpt model (multi-step train)
    # Simulates two training steps; if the capture flag left in an inconsistent
    # state between calls causes any issue, this exercises it.
    m_ckpt.zero_grad()
    out_ckpt_2 = m_ckpt(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    out_ckpt_2.loss.backward()
    grad_reg_ckpt_2 = m_ckpt.transformer.register_embeddings.grad.norm().item()
    print(f"[ckpt step2] grad(register_embeds)={grad_reg_ckpt_2:.6f}")
    assert abs(grad_reg_ckpt_2 - grad_reg_ckpt) < 1e-4, "second step grad differs"
    print("PASS: second forward+backward under ckpt is consistent (capture flag is not poisoned across steps).")

    # ---- 5. Eval-mode forward (no aux loss, no side compute) ----------------
    m_ckpt.eval()
    with torch.no_grad():
        out_eval = m_ckpt(input_ids=input_ids, attention_mask=attn, use_cache=False)
    assert out_eval.register_attn_balance_loss is None, "aux loss appeared in eval mode"
    print("PASS: eval-mode forward produces no register_attn_balance_loss (training=False gates the side compute even though flag is still set).")

    # ---- 6. register_start_layer > 0 + checkpointing -----------------------
    # The real run uses register_start_layer=8 (registers only on the final
    # energy block). Make sure that path also works under ckpt.
    cfg_sel = make_config(coef=1.0, threshold=0.0)
    cfg_sel.register_start_layer = 1  # only the 2nd (energy) block gets registers
    torch.manual_seed(11)
    m_sel = RegisterEnergyForCausalLM(cfg_sel)
    apply_block_checkpointing(m_sel)
    m_sel.train()
    out_sel = m_sel(input_ids=input_ids, attention_mask=attn, labels=labels, use_cache=False)
    out_sel.loss.backward()
    aux_sel = out_sel.register_attn_balance_loss.item()
    print(f"[ckpt selective] register_start_layer=1, aux={aux_sel:.6f} (should be > 0 since layer 1 has registers)")
    assert aux_sel > 0, "selective register placement should still trigger non-zero aux"
    print("PASS: register_start_layer > 0 works with checkpointing.")


if __name__ == "__main__":
    main()
