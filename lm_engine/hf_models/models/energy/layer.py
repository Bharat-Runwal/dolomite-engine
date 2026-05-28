# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from .config import EnergyConfig



class EnergyBlock(nn.Module):
    """Energy Transformer block with customizable attention and feedforward.

    Unlike standard Transformer blocks that use additive residual connections,
    EnergyBlock uses a subtractive update inspired by energy-based models:
        x = x - alpha[j] * proj(attn(ln(x)) + scale_ff * ffwd(ln(x)))

    When iter_step_scales is enabled, alpha[j] is a learnable per-iteration
    step size initialized with a decaying schedule to prevent oscillation.
    """

    def __init__(
        self,
        config: EnergyConfig,
        use_padding_free_transformer: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        self.sequence_mixer_type = config.sequence_mixer_blocks[layer_idx].sequence_mixer_type
        if self.sequence_mixer_type=="energy_attention":
            self.ln = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
            self.attn = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)

            self.ffwd = get_mlp_block(
                config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
            )

            # Diamond-shaped scale_ff init: edges strong, middle weak (breaks symmetry)
            # e.g. for 6 blocks: [4, 3, 1, 1, 3, 4]
            n = config.num_layers
            mid = (n - 1) / 2.0
            dist = abs(layer_idx - mid) / mid  # 1.0 at edges, 0.0 at center
            scale_ff_init = 1.0 + 3.0 * dist   # center=1.0, edges=4.0
            self.scale_ff = nn.Parameter(torch.ones(1) * scale_ff_init, requires_grad=True)

            self.proj_mode = getattr(config, 'proj_mode', 'unconstrained')
            if self.proj_mode in ("unconstrained", "riemannian"):
                self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            elif self.proj_mode in ("split", "riemannian_split"):
                self.proj_attn = nn.Linear(hidden_size, hidden_size, bias=False)
                self.proj_ff = nn.Linear(hidden_size, hidden_size, bias=False)
            else:
                raise ValueError(f"unexpected proj_mode ({self.proj_mode})")

            # Per-iteration learnable step size: alpha[j] scales the energy update
            num_iters = config.layer_iterations[layer_idx]
            use_step_scales = getattr(config, 'iter_step_scales', False)
            if use_step_scales and num_iters > 1:
                # Initialize with linear decay from 1.0 to 0.3
                init_vals = torch.linspace(1.0, 0.3, num_iters)
                self.iter_step_scales = nn.Parameter(init_vals)
            else:
                self.iter_step_scales = None
        elif self.sequence_mixer_type in ("parallel_softmax_attention", "egrad_attention"):
            # PaLM-style parallel block: single LN, attn + MLP computed in parallel
            # x = x + attn(ln(x)) + mlp(ln(x))
            self.ln = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
            self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
            self.mlp_block = get_mlp_block(
                config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
            )
        else:

            hidden_size = config.hidden_size
            self.m_residual = config.m_residual
            self.ln_1 = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
            self.sequence_mixer = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)
            self.ln_2 = get_normalization_function(
                config.normalization_function, hidden_size, eps=config.layer_norm_epsilon
            )
            self.mlp_block = get_mlp_block(
                config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
            )


    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_id: int | None = None, #TODO: Handle KV Caching for Energy Models
        iteration_idx: int = 0,
    ) -> torch.Tensor:

        if self.sequence_mixer_type=="energy_attention":

            ln_x = self.ln(hidden_states)
            attn_out = self.attn(
                ln_x,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                layer_id=layer_id,
            )
            ffwd_out = self.ffwd(ln_x)

            if self.proj_mode == "unconstrained":
                update = self.proj(attn_out + self.scale_ff * ffwd_out)
            elif self.proj_mode == "riemannian":
                combined = attn_out + self.scale_ff * ffwd_out
                # Project out radial component (tangent to layer norm sphere)
                h_norm = hidden_states / hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                combined = combined - (combined * h_norm).sum(dim=-1, keepdim=True) * h_norm
                update = self.proj(combined)
            elif self.proj_mode == "split":
                update = self.proj_attn(attn_out) + self.proj_ff(self.scale_ff * ffwd_out)
            elif self.proj_mode == "riemannian_split":
                h_norm = hidden_states / hidden_states.norm(dim=-1, keepdim=True).clamp(min=1e-6)
                attn_corr = attn_out - (attn_out * h_norm).sum(dim=-1, keepdim=True) * h_norm
                ff_scaled = self.scale_ff * ffwd_out
                ff_corr = ff_scaled - (ff_scaled * h_norm).sum(dim=-1, keepdim=True) * h_norm
                update = self.proj_attn(attn_corr) + self.proj_ff(ff_corr)

            # Apply per-iteration step scale if available
            if self.iter_step_scales is not None:
                alpha = self.iter_step_scales[iteration_idx]
                update = alpha * update

            hidden_states = hidden_states - update
            return hidden_states
        elif self.sequence_mixer_type in ("parallel_softmax_attention", "egrad_attention"):
            return self.forward_parallel_gpt(hidden_states,past_key_values,attention_mask,rope_cos_sin,cu_seqlens,max_seqlen,layer_id=layer_id)
        else:
            return self.forward_gpt(hidden_states,past_key_values,attention_mask,rope_cos_sin,cu_seqlens,max_seqlen,layer_id=layer_id)


    def energy_per_token(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        """Compute total energy per token: E = E_attn + scale_ff * E_ff."""
        ln_x = self.ln(x)
        return self.attn.energy_per_token(ln_x, rope_cos_sin=rope_cos_sin) + self.scale_ff * self.ffwd.energy_per_token(ln_x)



    def forward_parallel_gpt(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        ln_x = self.ln(hidden_states)
        attn_out = self._sequence_mixer_forward(
            hidden_states=ln_x,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            layer_id=layer_id,
        )
        mlp_out = self.mlp_block(ln_x)
        hidden_states = hidden_states + attn_out + mlp_out
        return hidden_states

    def forward_gpt(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)

        hidden_states = self._sequence_mixer_forward(
            hidden_states=hidden_states,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            layer_id=layer_id,
        )

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)

        hidden_states = self.mlp_block(hidden_states)

        if self.m_residual is not None:
            hidden_states = hidden_states * self.m_residual

        hidden_states = hidden_states + residual

        return hidden_states

    def _sequence_mixer_forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
        layer_id: int | None = None,
    ) -> torch.Tensor:
        if self.sequence_mixer_type in ["softmax_attention", "parallel_softmax_attention", "multihead_latent_attention", "egrad_attention"]:
            # Use iteration-aware layer_id for KV cache indexing when available
            orig_layer_idx = self.sequence_mixer.layer_idx
            if layer_id is not None:
                self.sequence_mixer.layer_idx = layer_id
            hidden_states = self.sequence_mixer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
            self.sequence_mixer.layer_idx = orig_layer_idx
        elif self.sequence_mixer_type in ["causal_convolution", "mamba2"]:
            hidden_states = self.sequence_mixer(
                hidden_states, cache_params=past_key_values, attention_mask=attention_mask
            )
        elif self.sequence_mixer_type in ["gru", "rnn"]:
            hidden_states = self.sequence_mixer(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        elif self.sequence_mixer_type == "gated_deltanet":
            # GatedDeltaNet returns (output, attentions, past_key_values)
            hidden_states = self.sequence_mixer(
                hidden_states,
                cache_params=past_key_values,
                attention_mask=attention_mask,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
            )
        else:
            raise ValueError(f"unexpected sequence_mixer_type ({self.sequence_mixer_type})")

        return hidden_states



# class EnergyBlock_QK_FF2W_manual(EnergyBlock):
#     """Energy Transformer block with standard QK attention and GradFF_2W_manual.

#     This is the main energy-based block combining:
#     - EnergyAttention_QK: Energy-based Q/K attention
#     - GradFF_2W_manual: Feedforward with manual gradient computation
#     - BareLayerNorm: LayerNorm without learnable weights
#     """

#     def __init__(
#         self,
#         config: CommonConfig,
#         use_padding_free_transformer: bool = False,
#         layer_idx: int | None = None,
#     ) -> None:

#         super().__init__(
#             config,
#             use_padding_free_transformer=use_padding_free_transformer,
#             layer_idx=layer_idx,
#         )


