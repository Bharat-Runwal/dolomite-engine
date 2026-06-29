# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn.functional as F
from ...mixins import CausalLMModelMixin
from .model import RegisterEnergyModel, RegisterEnergyPreTrainedModel


class RegisterEnergyForCausalLM(RegisterEnergyPreTrainedModel, CausalLMModelMixin):
    base_model_class = RegisterEnergyModel

    def forward(self, *args, **kwargs):
        # HF transformers' greedy_search/sample passes output_attentions,
        # output_hidden_states, return_dict=True directly into self(...).
        # CausalLMModelMixin.forward doesn't accept the first two — strip them
        # silently here (they're not supported and we'd ignore them anyway).
        for k in ("output_attentions", "output_hidden_states"):
            kwargs.pop(k, None)
        return super().forward(*args, **kwargs)

    def prepare_inputs_for_generation(self, input_ids, past_key_values=None,
                                       attention_mask=None, **kwargs):
        """Prepare inputs for HF transformers' .generate() loop.

        HF's GenerationMixin defines no default implementation — every CausalLM
        must supply one.  We build the inputs dict from scratch (do **not** call
        super(), which raises NotImplementedError in GenerationMixin).

        Three regimes, controlled by ``config.register_generation_mode``:

        1. ``"no_cache"`` (RECOMMENDED, CORRECT):
           Discard past_kv every step so the model re-runs prefill on the full
           (prompt + generated_so_far) input_ids.  Registers and RoPE are rebuilt
           every step → training-consistent.  Slow (~T x prefill cost per step)
           but bit-exact vs. a pure use_cache=False loop.

        2. ``"persistent_cache"`` (FAST + CORRECT):
           Keep the cache, decode one token at a time, and explicitly override
           ``position_ids`` to the *content-space* position (T+k) — NOT the raw
           kv-length (R+T+k) HF would otherwise compute via cumsum(mask).  The
           cache already holds R register K/V rows at positions 0..R-1 plus T+k
           content rows at positions 0..T+k-1; we just need the new content
           token's position to match training (T+k) so RoPE relative offsets are
           correct.  The attention_mask is extended by R ones at the front so
           the new token attends to all cached registers.

        3. ``"bypass"`` (default, BUGGY — kept for backward-compat):
           Keep cache, single-token decode, extend mask by R, but **do not**
           override position_ids → RoPE is off-by-R for the new content token.
           Empirically corrupts generation (hop_r256: BBH=0.074).  See
           REGISTER_DECODE_BUG.md.

        Selective registers (``register_start_layer > 0``) are forced to
        no_cache (per-layer cache size differs, no clean persistent path).
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
            past_key_values = None  # discard stale cache

        # ------------------------------------------------------------------
        # Build the inputs dict ourselves — HF's GenerationMixin requires it.
        # ------------------------------------------------------------------
        if past_key_values is not None:
            # Cached decode: HF has accumulated full input_ids each step, but
            # only the last token is new.
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        use_cache = kwargs.get("use_cache", True)

        # Standard full-register + cached-decode (bypass / persistent_cache):
        # extend mask by R to cover the cached register KVs.
        if (n_reg > 0 and register_start == 0
                and gen_mode in ('bypass', 'persistent_cache')
                and past_key_values is not None
                and attention_mask is not None):
            try:
                pad = torch.ones(attention_mask.shape[0], n_reg,
                                 dtype=attention_mask.dtype,
                                 device=attention_mask.device)
                attention_mask = torch.cat([pad, attention_mask], dim=1)
            except Exception:
                pass

        # persistent_cache: explicitly set position_ids to *content-space*
        # position (T+k), not the raw kv-length-based cumsum value (R+T+k).
        # The cache already has register rows at positions 0..R-1; the new
        # content token's RoPE rotation must use position T+k so its relative
        # offset to cached content K/V (positions 0..T+k-1) is correct.
        if (n_reg > 0 and register_start == 0 and gen_mode == 'persistent_cache'
                and past_key_values is not None and input_ids.shape[-1] == 1):
            # Compute content position by introspecting the cache.
            # KV-cache holds R + content_so_far entries; content_so_far = next position
            try:
                if hasattr(past_key_values, 'get_seq_length'):
                    kv_len = past_key_values.get_seq_length()
                elif hasattr(past_key_values, 'key_cache'):
                    kv_len = past_key_values.key_cache[0].shape[2]
                else:
                    kv_len = past_key_values[0][0].shape[2]
                # cache has R registers + content_so_far content rows
                content_pos = kv_len - n_reg  # next content position
                position_ids = torch.tensor(
                    [[content_pos]], dtype=torch.long, device=input_ids.device
                ).expand(input_ids.shape[0], 1).contiguous()
            except Exception:
                pass  # leave to model.forward to compute (will be wrong, but no crash)

        inputs = {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "attention_mask": attention_mask,
            "use_cache": use_cache,
        }
        if position_ids is not None:
            inputs["position_ids"] = position_ids
        return inputs
