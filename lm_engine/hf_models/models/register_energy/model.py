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

        # Capture caller-supplied position_ids before _prepare_a_bunch_of_stuff
        # overwrites them with cumsum-derived defaults.  Used below for the
        # persistent_cache decode path.
        _caller_position_ids = position_ids
        _gen_mode = getattr(self.config, 'register_generation_mode', 'bypass')
        _register_start = getattr(self.config, 'register_start_layer', 0)

        # Fast-path detection: cached single-token decode with full registers.
        # When dolomite-engine's generate() (and lm-eval-harness's HFLM
        # dispatching through it) does a decode step, it passes:
        #   input_ids = next_token        (shape [B, 1])
        #   attention_mask = full content amask (shape [B, T+k])    (no R-pad)
        #   past_key_values = populated cache (R + T + k-1 entries)
        # The default _prepare_a_bunch_of_stuff path then computes
        #   key_length = past_length + 1 = R+T+k
        #   position_ids = position_ids[:, past_length:key_length]   # OOB slice!
        # Slicing an amask of length T+k by [past_length:past_length+1] when
        # past_length > T+k gives an empty tensor, and the subsequent
        # cos[position_ids] gather goes OOB on CUDA.
        #
        # To avoid this entirely, when we detect cached single-token decode with
        # full-registers (register_start == 0), extend the attention_mask BEFORE
        # calling _prepare_a_bunch_of_stuff so positions line up.  Then we
        # short-circuit into super().forward() with the correct content-space
        # position_id supplied explicitly.
        if (self.n_registers > 0 and _register_start == 0
                and not self.training
                and past_key_values is not None
                and input_ids is not None and input_ids.shape[-1] == 1
                and attention_mask is not None):
            try:
                if hasattr(past_key_values, 'get_seq_length'):
                    _kv_len = past_key_values.get_seq_length()
                elif hasattr(past_key_values, 'key_cache') and past_key_values.key_cache:
                    _kv_len = past_key_values.key_cache[0].shape[2]
                else:
                    _kv_len = past_key_values[0][0].shape[2]
            except Exception:
                _kv_len = None
            if _kv_len is not None and _kv_len >= self.n_registers:
                _target_len = _kv_len + 1
                if attention_mask.shape[-1] < _target_len:
                    _pad = torch.ones(
                        attention_mask.shape[0],
                        _target_len - attention_mask.shape[-1],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                    attention_mask = torch.cat([_pad, attention_mask], dim=1)
                # Compute correct content-space position_id (T+k = kv_len - R).
                _pos_to_use = None
                if _gen_mode in ('persistent_cache', 'no_cache') and _caller_position_ids is None:
                    _content_pos = _kv_len - self.n_registers
                    _pos_to_use = torch.tensor(
                        [[_content_pos]], dtype=torch.long, device=input_ids.device
                    ).expand(input_ids.shape[0], 1).contiguous()
                elif _gen_mode == 'persistent_cache' and _caller_position_ids is not None:
                    _pos_to_use = _caller_position_ids
                # bypass: leave _pos_to_use = None → super computes from cumsum
                # (off-by-R buggy behaviour, kept for backward compat).
                return super().forward(
                    input_ids=input_ids,
                    past_key_values=past_key_values,
                    attention_mask=attention_mask,
                    position_ids=_pos_to_use,
                    use_cache=use_cache,
                    cu_seqlens=cu_seqlens,
                    max_seqlen=max_seqlen,
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

        # During cached generation (use_cache=True, not training), registers create a
        # KV-cache length mismatch: prefill caches R+T entries, decode expects T.
        # Three options controlled by config.register_generation_mode:
        #   "bypass" (default, BUGGY for backward compat): cached-decode path runs
        #     super().forward() (no register re-prepend). The new token attends to
        #     cached register KVs but its own position_id is R+T+k (off-by-R vs.
        #     training where content positions are 0..T-1). RoPE rotations are wrong,
        #     causing catastrophic generation degradation (hop_r256: BBH=0.0738,
        #     GSM8K=0.0). avg10_norm (logp scoring) is unaffected because logp uses
        #     a single prefill forward, not cached decode. See REGISTER_DECODE_BUG.md.
        #   "no_cache" / "persistent_cache" (CORRECT, identical implementation):
        #     Keep cached register KVs from prefill, decode one token at a time, and
        #     manually compute content-space position_id = T+k for the new token so
        #     RoPE matches training. Extend the attention_mask by R ones so the new
        #     token attends to all cached registers.  Both modes resolve to this same
        #     code path here (the legacy "no_cache" name is preserved for config
        #     compatibility).  See REGISTER_DECODE_BUG.md.
        effective_use_cache = use_cache if use_cache is not None else self.config.use_cache

        register_start = getattr(self.config, 'register_start_layer', 0)
        R = self.n_registers  # convenience alias

        # --------------------------------------------------------------
        # register_generation_mode: validate.
        # --------------------------------------------------------------
        # Both "no_cache" and "persistent_cache" share the same correct decode
        # path below (extended mask + content-space position_ids).  The old
        # "no_cache" implementation that dropped the cache each step relied on
        # HF's prepare_inputs_for_generation re-feeding the full input_ids; but
        # dolomite-engine's CausalLMModelMixin.generate() (which lm-evaluation-
        # harness's HFLM dispatches to) only passes the NEW token at each decode
        # step, so re-prefilling is impossible without the original prompt.
        # Using persistent_cache-style decode is correct, fast, and works for
        # both HF .generate() and dolomite generate().
        gen_mode = getattr(self.config, 'register_generation_mode', 'bypass')
        if gen_mode not in ('bypass', 'no_cache', 'persistent_cache'):
            raise ValueError(
                f"register_generation_mode='{gen_mode}' not recognized. "
                "Valid options: 'bypass', 'no_cache', 'persistent_cache'."
            )

        # ── Cached-decode step (T=1, past_kv already populated from prefill) ──
        if (effective_use_cache and not self.training
                and past_key_values is not None
                and input_ids is not None
                and input_ids.shape[-1] == 1):

            if register_start == 0:
                # Standard full-register decode: single mask extended by R covers all layers.
                # Always extend amask by R for bypass / no_cache / persistent_cache
                # (registers occupy positions 0..R-1 in the KV cache).
                if attention_mask is not None:
                    try:
                        if hasattr(past_key_values, 'key_cache'):
                            kv_len = past_key_values.key_cache[0].shape[2]
                        elif hasattr(past_key_values, 'get_seq_length'):
                            kv_len = past_key_values.get_seq_length()
                        else:
                            kv_len = past_key_values[0][0].shape[2]
                        target_len = kv_len + 1
                        if attention_mask.shape[-1] < target_len:
                            pad = torch.ones(attention_mask.shape[0], target_len - attention_mask.shape[-1],
                                             dtype=attention_mask.dtype, device=attention_mask.device)
                            attention_mask = torch.cat([pad, attention_mask], dim=1)
                    except Exception:
                        pass

                # no_cache / persistent_cache: explicitly compute content-space
                # position_id for the new token so RoPE rotation matches training.
                # The KV cache holds R register rows at positions 0..R-1 plus
                # content_so_far rows at positions 0..content_so_far-1.  The new
                # content token's position is content_so_far = kv_len - R.
                # bypass: pass position_ids=None so super computes from cumsum(mask),
                # which gives the off-by-R buggy position (kept for backward compat).
                pos_to_use = None
                if gen_mode in ('persistent_cache', 'no_cache'):
                    try:
                        if hasattr(past_key_values, 'get_seq_length'):
                            kv_len = past_key_values.get_seq_length()
                        elif hasattr(past_key_values, 'key_cache'):
                            kv_len = past_key_values.key_cache[0].shape[2]
                        else:
                            kv_len = past_key_values[0][0].shape[2]
                        content_pos = kv_len - R  # next content position = T+k
                        pos_to_use = torch.tensor(
                            [[content_pos]], dtype=torch.long, device=input_ids.device
                        ).expand(input_ids.shape[0], 1).contiguous()
                    except Exception:
                        pos_to_use = None
                # If caller supplied an explicit position_ids (HF prepare_inputs_for_generation
                # in persistent_cache mode does this), respect it.
                if gen_mode == 'persistent_cache' and _caller_position_ids is not None:
                    pos_to_use = _caller_position_ids
                return super().forward(
                    input_ids=input_ids, past_key_values=past_key_values,
                    attention_mask=attention_mask, position_ids=pos_to_use,
                    use_cache=use_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
            else:
                # Selective-register decode: per-layer mask selection.
                # Layers 0..register_start-1: KV has T entries  → normal mask (T+step length)
                # Layers register_start..:   KV has T+R entries → extended mask (T+R+step length)
                (
                    use_cache_eff, hidden_states, causal_mask, position_ids,
                    rope_cos_sin, past_key_values,
                ) = self._prepare_a_bunch_of_stuff(
                    input_ids=input_ids, past_key_values=past_key_values,
                    attention_mask=attention_mask, position_ids=position_ids,
                    use_cache=use_cache, cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                )
                # causal_mask for decode: [B,1,1,T+step] (float, 0=attend, -inf=block).
                # Extended version: prepend R zeros to the key dimension.
                if causal_mask is not None:
                    causal_mask_ext = torch.nn.functional.pad(
                        causal_mask, (R, 0), value=0.0)  # [B,1,1,R+T+step]
                else:
                    causal_mask_ext = None

                energy_descent_loss = torch.tensor(0.0, device=hidden_states.device)
                layer_id = 0
                for i, num_iter in enumerate(self.layer_iterations):
                    mask_to_use = causal_mask_ext if i >= register_start else causal_mask
                    for j in range(num_iter):
                        hidden_states = self._run_block(
                            hidden_states, past_key_values, attention_mask,
                            cu_seqlens, max_seqlen, mask_to_use, rope_cos_sin,
                            False, i, layer_id=layer_id, iter_idx=j,
                        )
                        layer_id += 1
                    if num_iter < self.layer_iterations[i]:
                        layer_id += (self.layer_iterations[i] - num_iter)

                hidden_states = self.ln_f(hidden_states)
                return BaseModelOutputWithPast(
                    last_hidden_state=hidden_states,
                    past_key_values=past_key_values,
                )

        # ── Prefill or training ──
        # Do NOT modify attention_mask before _prepare_a_bunch_of_stuff.
        # The causal mask is built from [T_q=input_ids.shape, T_k=attention_mask.shape]
        # and then _extend_attention_mask_for_registers extends it to [R+T, R+T].
        # (Extending before would create a [T, R+T] mask — T_q vs T_k mismatch.)

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
        register_start = getattr(self.config, 'register_start_layer', 0)

        # Pre-compute the extended mask and rope for when registers ARE active.
        # For selective registers (register_start > 0), we apply this only from
        # layer register_start onward; layers 0..register_start-1 use the plain mask.
        def _make_extended(hs, T_):
            reg_emb = self.register_embeddings.unsqueeze(0).expand(B, -1, -1).to(dtype)
            hs_ext = torch.cat([reg_emb, hs], dim=1)
            ext_mask = self._extend_attention_mask_for_registers(causal_mask, B, R, T_, device, dtype)
            ext_rope = None
            if rope_cos_sin is not None:
                pos_ids_exp = position_ids.expand(B, -1)
                reg_pos = torch.arange(R, device=device, dtype=position_ids.dtype).unsqueeze(0).expand(B, -1)
                ext_pos = torch.cat([reg_pos, pos_ids_exp], dim=1)
                ext_rope = self._get_rope_cos_sin(key_length=R + T_, position_ids=ext_pos, dtype=dtype)
            return hs_ext, ext_mask, ext_rope

        registers_injected = False
        if register_start == 0:
            # Original behaviour: prepend registers before all layers
            hidden_states, extended_mask, extended_rope_cos_sin = _make_extended(hidden_states, T)
            registers_injected = True
        else:
            # Selective: layers 0..register_start-1 run without registers
            extended_mask = causal_mask
            extended_rope_cos_sin = rope_cos_sin

        # ----------------------------------------------------------------
        # Run all transformer blocks (energy + GPT) on extended sequence
        # ----------------------------------------------------------------
        energy_descent_loss = torch.tensor(0.0, device=device)
        mamba_mask_computed = False

        layer_id = 0
        for i, num_iter in enumerate(self.layer_iterations):
            # Selective injection: prepend registers at layer register_start
            if not registers_injected and i == register_start:
                hidden_states, extended_mask, extended_rope_cos_sin = _make_extended(hidden_states, hidden_states.shape[1])
                registers_injected = True
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
