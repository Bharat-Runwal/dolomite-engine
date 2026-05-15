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

        For selective-register models (register_start_layer > 0), different layers
        have different KV cache sizes (layers before start: T entries; layer at start
        onward: T+R entries). HF's single-step decode can't reconcile this, so we
        force full-sequence recompute (pass all input_ids, no past_kv).

        For standard full-register models: extend attention_mask by R to match the
        R+T entries in the KV cache.
        """
        n_reg = getattr(self.config, 'n_registers', 0)
        register_start = getattr(self.config, 'register_start_layer', 0)

        # Selective registers: force full recompute at each step (no caching)
        if n_reg > 0 and register_start > 0 and past_key_values is not None:
            past_key_values = None  # discard stale cache; recompute from scratch

        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, **kwargs
        )

        # Standard full-register: extend mask by R to match R+T cache entries
        if (n_reg > 0 and register_start == 0 and past_key_values is not None
                and inputs.get('attention_mask') is not None):
            try:
                mask = inputs['attention_mask']
                pad = torch.ones(mask.shape[0], n_reg,
                                 dtype=mask.dtype, device=mask.device)
                inputs['attention_mask'] = torch.cat([pad, mask], dim=1)
            except Exception:
                pass
        return inputs
