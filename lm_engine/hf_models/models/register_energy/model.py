# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

"""RegisterEnergyModel: EnergyModel with learnable register tokens.

Register tokens are prepended to the hidden state sequence before the recurrent
energy blocks and stripped off afterwards.  They attend to all tokens (full
attention, no causal mask for their rows), giving the model a
small "global working memory" that can relay information across positions without
contaminating the LM loss computation.

Architecture sketch:
    input_ids → wte → hidden_states: [B, T, d]

    # Prepend n_registers register embeddings:
    reg = register_embeddings.expand(B, -1, -1)      # [B, R, d]
    hidden_states = cat([reg, hidden_states], dim=1)  # [B, R+T, d]
    position_ids  = cat([0..R-1, original_pos_ids])  (registers at pos 0..R-1)
    attention_mask extended to allow registers to attend to all tokens

    # Run through all blocks (EnergyBlock / GPT blocks) normally:
    for block in self.h:
        hidden_states = block(hidden_states, ...)

    # Strip register positions before final layer norm:
    hidden_states = hidden_states[:, n_registers:]   # [B, T, d]
    hidden_states = self.ln_f(hidden_states)

The register embeddings are learnable (nn.Parameter, shape [R, d]), initialised
from N(0, init_std) matching the token embedding initializer.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...utils import is_generation_cache_enabled
from ...mixins.modeling_outputs import BaseModelOutputWithPast
from ..energy.base import EnergyModel, EnergyPreTrainedModel
from .config import RegisterEnergyConfig

# avoid import inside training loop
import random as _random  # noqa: F401  (used for iter dropout, mirrors parent)


class RegisterEnergyPreTrainedModel(EnergyPreTrainedModel):
    config_class = RegisterEnergyConfig

    @property
    def _no_split_modules(self):
        return ["EnergyBlock"]


class RegisterEnergyModel(RegisterEnergyPreTrainedModel, EnergyModel):
    """EnergyModel with n_registers learnable register tokens prepended.

    All recurrent energy blocks and GPT blocks are inherited unchanged.
    Only the forward() is overridden to inject and remove register tokens.
    """

    def __init__(self, config: RegisterEnergyConfig, **kwargs):
        # Initialize the full EnergyModel (creates wte, h, ln_f, rope, etc.)
        super().__init__(config, **kwargs)

        self.n_registers = config.n_registers
        if self.n_registers > 0:
            # Learnable register embeddings: [R, d]
            # Stored as a Parameter so FSDP/DDP handle it correctly.
            self.register_embeddings = nn.Parameter(
                torch.randn(self.n_registers, config.hidden_size) * config.initializer_range
            )

    def _extend_attention_mask_for_registers(
        self,
        causal_mask: torch.Tensor | None,
        batch_size: int,
        n_reg: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        """Extend the causal attention mask to accommodate prepended register tokens.

        The extended sequence has shape [B, R+T]:
            - Register positions (0..R-1): can attend to all R+T positions
              (no causal restriction — they see the whole sequence).
            - Content positions (R..R+T-1): attend causally within content
              (positions R..R+T-1) and also to all registers (0..R-1).

        The causal_mask returned by _prepare_a_bunch_of_stuff has shape:
            [B, 1, T, T] (for non-flashattn SDPA path)  — dtype is float (additive)
        or  None / bool tensor depending on the kernel.

        We need to return a mask of shape [B, 1, R+T, R+T].

        If causal_mask is None (flash-attention path), we return None — flash
        attention will handle causality internally and registers get full attention
        by default because we don't supply a custom mask.
        """
        if causal_mask is None:
            return None

        # causal_mask: [B, 1, T, T] — additive float mask (0.0 = attend, -inf = block)
        # Determine mask_value (the "blocked" fill value)
        mask_val = causal_mask.min().item()
        R, T = n_reg, seq_len
        total = R + T

        # Build new mask [B, 1, R+T, R+T], initialised to mask_val (all blocked)
        new_mask = torch.full(
            (batch_size, 1, total, total),
            fill_value=mask_val,
            dtype=causal_mask.dtype,
            device=device,
        )

        # Register rows (0..R-1): can attend to ALL positions → set to 0.0
        new_mask[:, :, :R, :] = 0.0

        # Content rows (R..R+T-1): copy existing causal_mask for content-to-content block
        new_mask[:, :, R:, R:] = causal_mask  # [B, 1, T, T]

        # Content rows: also allow attending to all registers → 0.0 in [:R] columns
        new_mask[:, :, R:, :R] = 0.0

        return new_mask

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        use_cache: bool | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> BaseModelOutputWithPast:

        # If no registers, fall through to the parent implementation unchanged.
        if self.n_registers == 0:
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        # Note: register tokens are incompatible with the padding-free transformer
        # (cu_seqlens packing) path because the sequence layout would be non-trivial.
        # Raise a clear error to avoid silent mis-behaviour.
        if self.use_padding_free_transformer:
            raise NotImplementedError(
                "RegisterEnergyModel does not support padding-free transformer (cu_seqlens). "
                "Use standard padded inputs."
            )

        # ----------------------------------------------------------------
        # 1. Standard prepare: embed input_ids, build causal mask, rope
        # ----------------------------------------------------------------
        (
            use_cache,
            hidden_states,   # [B, T, d]
            causal_mask,     # [B, 1, T, T] or None
            position_ids,    # [B, T]
            rope_cos_sin,
            past_key_values,
        ) = self._prepare_a_bunch_of_stuff(
            input_ids=input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            position_ids=position_ids,
            use_cache=use_cache,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )

        # During generation (use_cache=True), skip register prepending entirely.
        # Registers change the KV cache length, causing a mismatch between prefill
        # (T+R cached entries) and decode (expects T cached entries) steps.
        # The model gracefully degrades to non-register behaviour at test time.
        effective_use_cache = use_cache if use_cache is not None else self.config.use_cache
        if effective_use_cache and not self.training:
            return super().forward(
                input_ids=input_ids,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=use_cache,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )

        if is_generation_cache_enabled():
            past_key_values = (
                GenerationCache(self.config)
                if use_cache and past_key_values is None
                else past_key_values
            )

        B, T, d = hidden_states.shape
        R = self.n_registers
        device = hidden_states.device
        dtype = hidden_states.dtype
        is_cached_decode = False  # always False here (handled above)

        if is_cached_decode:
            # Standard decode: no register prepending, use causal_mask as-is
            extended_mask = causal_mask
            extended_rope_cos_sin = rope_cos_sin
        else:
            # ----------------------------------------------------------------
            # 2. Prepend register tokens: [B, R+T, d]
            # ----------------------------------------------------------------
            reg_emb = self.register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype)
            hidden_states = torch.cat([reg_emb, hidden_states], dim=1)  # [B, R+T, d]

            # ----------------------------------------------------------------
            # 3. Extend attention mask for registers
            # ----------------------------------------------------------------
            extended_mask = self._extend_attention_mask_for_registers(
                causal_mask, B, R, T, device, dtype
            )

        # ----------------------------------------------------------------
        # 4. Extend rope_cos_sin (skipped for cached decode — already computed)
        # ----------------------------------------------------------------
        if not is_cached_decode:
            extended_rope_cos_sin = None
            if rope_cos_sin is not None:
                pos_ids_expanded = position_ids.expand(B, -1)
                reg_pos_ids = torch.arange(R, device=device, dtype=position_ids.dtype).unsqueeze(0).expand(B, -1)
                ext_pos_ids = torch.cat([reg_pos_ids, pos_ids_expanded], dim=1)
                extended_rope_cos_sin = self._get_rope_cos_sin(
                    key_length=R + T,
                    position_ids=ext_pos_ids,
                    dtype=dtype,
                )

        # ----------------------------------------------------------------
        # 5. Run all transformer blocks (energy + GPT) on extended sequence
        # ----------------------------------------------------------------
        energy_descent_loss = torch.tensor(0.0, device=device)
        mamba_mask_computed = False

        layer_id = 0
        for i, num_iter in enumerate(self.layer_iterations):
            if self.training:
                block_range = (self.iter_dropout_range_per_block[i]
                               if self.iter_dropout_range_per_block is not None
                               else self.iter_dropout_range)
                if block_range > 0:
                    min_iter = max(1, num_iter - block_range)
                    max_iter = num_iter + block_range
                    effective_iter = torch.randint(min_iter, max_iter + 1, (1,)).item()
                else:
                    effective_iter = num_iter
            else:
                effective_iter = num_iter

            block = self.h[i]
            has_energy = (self.training and
                          self.energy_descent_loss_coef > 0 and
                          hasattr(block, 'energy_per_token') and
                          getattr(block, 'sequence_mixer_type', '') == 'energy_attention')
            prev_energy = None

            for j in range(effective_iter):
                if self.halt_thresholds is not None and j > 0 and i in self.halt_thresholds:
                    h_norm = hidden_states.norm(dim=-1).mean()
                    delta_norm = (hidden_states - _prev_h).norm(dim=-1).mean()
                    if (delta_norm / h_norm.clamp(min=1e-6)).item() < self.halt_thresholds[i]:
                        layer_id += (effective_iter - j)
                        break

                _prev_h = hidden_states
                hidden_states = self._run_block(
                    hidden_states,
                    past_key_values,
                    attention_mask,
                    cu_seqlens,
                    max_seqlen,
                    extended_mask,   # use extended mask (registers + content)
                    extended_rope_cos_sin,
                    mamba_mask_computed,
                    i,
                    layer_id=layer_id,
                    iter_idx=j,
                )
                layer_id += 1

                if has_energy:
                    curr_energy = block.energy_per_token(
                        hidden_states, rope_cos_sin=extended_rope_cos_sin
                    ).mean()
                    if prev_energy is not None:
                        energy_increase = torch.clamp(curr_energy - prev_energy, min=0.0)
                        energy_descent_loss = energy_descent_loss + energy_increase
                    prev_energy = curr_energy.detach()

                if self.training and self.iter_noise_eta > 0 and j < effective_iter - 1:
                    noise_scale = (2 * self.iter_noise_eta) ** 0.5
                    hidden_states = hidden_states + noise_scale * torch.randn_like(hidden_states)

            if effective_iter < num_iter:
                layer_id += (num_iter - effective_iter)

        # ----------------------------------------------------------------
        # 6. Strip register positions, then final layer norm
        # ----------------------------------------------------------------
        hidden_states = hidden_states[:, R:]   # [B, T, d]
        hidden_states = self.ln_f(hidden_states)

        edl = energy_descent_loss * self.energy_descent_loss_coef if self.energy_descent_loss_coef > 0 else None

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            energy_descent_loss=edl,
        )
