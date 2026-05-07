# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
from ...mixins import CausalLMModelMixin
from .model import RegisterEnergyModel, RegisterEnergyPreTrainedModel


class RegisterEnergyForCausalLM(RegisterEnergyPreTrainedModel, CausalLMModelMixin):
    base_model_class = RegisterEnergyModel

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                       attention_mask=None, **kwargs):
        """Fix HF's attention_mask to include register positions in the KV cache.

        HF's generate() tracks attention_mask based on the original sequence length T.
        After register-prepended prefill, the KV cache has R+T entries.
        At each decode step, HF creates mask of length T+step, but cache has R+T+step.
        We extend the mask by R here to keep them in sync.
        """
        inputs = super().prepare_inputs_for_generation(
            input_ids, past_key_values=past_key_values,
            attention_mask=attention_mask, **kwargs
        )
        n_reg = getattr(self.config, 'n_registers', 0)
        if (n_reg > 0 and past_key_values is not None
                and inputs.get('attention_mask') is not None):
            try:
                kv_len = past_key_values.key_cache[0].shape[2]
                mask = inputs['attention_mask']
                if mask.shape[-1] < kv_len:
                    n_extra = kv_len - mask.shape[-1]
                    pad = torch.ones(mask.shape[0], n_extra,
                                     dtype=mask.dtype, device=mask.device)
                    inputs['attention_mask'] = torch.cat([pad, mask], dim=1)
            except Exception:
                pass
        return inputs
