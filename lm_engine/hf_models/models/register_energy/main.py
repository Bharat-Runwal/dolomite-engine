# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from ...mixins import CausalLMModelMixin
from .model import RegisterEnergyModel, RegisterEnergyPreTrainedModel


class RegisterEnergyForCausalLM(RegisterEnergyPreTrainedModel, CausalLMModelMixin):
    base_model_class = RegisterEnergyModel

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                       attention_mask=None, **kwargs):
        """Prepare inputs for generation, handling register-token KV-cache offsets.

        Three regimes:

        1. ``register_generation_mode == "no_cache"`` (RECOMMENDED, CORRECT):
           Discard past_kv at every step so HF generate passes the full
           (prompt + generated_so_far) input_ids each call. The model then
           re-runs prefill, re-prepending registers and re-rotating RoPE for
           every content position. Slow (~T x prefill cost per step) but
           produces training-consistent hidden states. Fixes the 2026-06-29
           position-offset bug — see REGISTER_DECODE_BUG.md.

        2. ``register_generation_mode == "bypass"`` (default, BUGGY):
           Allows KV caching, with the attention_mask padded by R to cover the
           cached register KVs. BUT the new decode token's position_id (via
           _get_position_ids on the padded mask) becomes R+T+k, whereas during
           training the content's next position would be T+k.  RoPE rotations
           are then off-by-R, corrupting attention with the cached register
           KVs.  Empirically this kills generation: hop_r256 step 24000 (256
           registers) showed BBH=0.074, GSM8K=0.0, while logp-based avg10_norm
           was unaffected (=base perf) because logp uses a single prefill, not
           cached decode.  Kept as default only for backward-compat reproduction
           of prior eval numbers.

        3. Selective registers (``register_start_layer > 0``):
           Different layers have different KV cache sizes.  Always force full
           recompute (same behaviour as no_cache).
        """
        n_reg = getattr(self.config, 'n_registers', 0)
        register_start = getattr(self.config, 'register_start_layer', 0)
        gen_mode = getattr(self.config, 'register_generation_mode', 'bypass')

        # Selective registers OR no_cache mode: force full recompute each step.
        force_recompute = (
            (n_reg > 0 and register_start > 0 and past_key_values is not None)
            or (n_reg > 0 and gen_mode == 'no_cache' and past_key_values is not None)
        )
        if force_recompute:
            past_key_values = None  # discard stale cache; recompute from scratch

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, **kwargs
        )

        # Standard full-register w/ bypass: extend mask by R to match R+T cache entries.
        # (no_cache path doesn't need this because past_key_values is None.)
        if (n_reg > 0 and register_start == 0 and gen_mode != 'no_cache'
                and past_key_values is not None
                and inputs.get('attention_mask') is not None):
            try:
                mask = inputs['attention_mask']
                pad = torch.ones(mask.shape[0], n_reg,
                                 dtype=mask.dtype, device=mask.device)
                inputs['attention_mask'] = torch.cat([pad, mask], dim=1)
            except Exception:
                pass
        return inputs
