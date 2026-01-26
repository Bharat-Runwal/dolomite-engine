# **************************************************
# Copyright (c) 2025
# Energy-based Transformer Blocks
# **************************************************

from __future__ import annotations

import math
import numbers

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, is_torch_xla_available
from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils.dropout import Dropout
from ...modeling_utils.sequence_mixer_blocks.utils import flash_attention


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


class BareLayerNorm(nn.Module):
    """LayerNorm without learnable weights, only bias."""

    def __init__(self, normalized_shape: int | tuple[int, ...], eps: float = 1e-5) -> None:
        super().__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        self.normalized_shape = tuple(normalized_shape)
        self.eps = eps
        self.bias = nn.Parameter(torch.zeros(self.normalized_shape))
        self.weight = nn.Parameter(torch.ones(self.normalized_shape), requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x, normalized_shape=self.normalized_shape, eps=self.eps, bias=self.bias, weight=self.weight
        )



###################### With regression from class Attention
# class EnergyAttention_QK(nn.Module):
#     """Energy-based Q/K attention for Energy Transformer.

#     This attention mechanism uses tied Q/K weights and sets V=K.
#     Compatible with flash attention 2/3 and the standard sequence mixer interface.

#     Key differences from standard attention:
#     - Q and K are computed via a single learnable parameter with special initialization
#     - Value is always equal to Key (V = K)
#     - Output projection uses Q weights divided by initial std

#     Core forward logic (preserved exactly):
#         q, k = einsum('btc,ahcs->bahts', x, qk_weights).unbind(1)
#         v = k
#         y = scaled_dot_product_attention(q, k, v, is_causal=True)
#         y = einsum('bhtd,hcd->btc', y, W_Q)  # where W_Q = qk_weights[0] / INIT_STD
#     """

#     def __init__(
#         self,
#         hidden_size: int,
#         num_attention_heads: int,
#         num_key_value_heads: int,  # Ignored - energy attention uses num_attention_heads
#         attention_multiplier: float,  # Ignored - using default 1/sqrt(head_dim)
#         sliding_window: int | None,
#         position_embedding_type: str,  # Ignored - energy attention doesn't use RoPE
#         add_bias: bool,  # Ignored
#         qkv_bias: bool,  # Ignored
#         softmax_dropout: float,
#         dropout: float,
#         init_method: str,  # Ignored - using energy-specific initialization
#         initializer_range: float,  # Ignored
#         m_width: float,  # Ignored
#         num_layers: int,  # Ignored
#         causal: bool,
#         layer_idx: int,
#         use_padding_free_transformer: bool,
#     ) -> None:
#         super().__init__()

#         self.causal = causal
#         self.hidden_size = hidden_size
#         self.num_heads = num_attention_heads
#         self.use_padding_free_transformer = use_padding_free_transformer
#         self.sliding_window = sliding_window
#         self.layer_idx = layer_idx

#         assert hidden_size % num_attention_heads == 0, "hidden_size must be divisible by num_attention_heads"
#         self.head_dim = hidden_size // num_attention_heads

#         # Energy attention uses tied Q/K weights with special initialization
#         # Shape: (2, num_heads, hidden_size, head_dim)
#         self.qk_weights = nn.Parameter(torch.randn(2, num_attention_heads, hidden_size, self.head_dim))
#         nn.init.xavier_uniform_(self.qk_weights, gain=8.0)
#         self.INIT_STD = self.qk_weights.std().item()

#         self.softmax_dropout_p = softmax_dropout
#         self.softmax_dropout = Dropout(softmax_dropout)
#         self.resid_dropout = Dropout(dropout)

#     def forward(
#         self,
#         hidden_states: torch.Tensor,
#         past_key_values: GenerationCache | None = None,
#         attention_mask: torch.Tensor | None = None,
#         rope_cos_sin: torch.Tensor | None = None,
#         cu_seqlens: torch.Tensor | None = None,
#         max_seqlen: int | None = None,
#     ) -> torch.Tensor:
#         """
#         Energy attention forward pass.

#         Preserves exact original logic:
#             q, k = einsum('btc,ahcs->bahts', x, qk_weights).unbind(1)
#             v = k
#             y = scaled_dot_product_attention(q, k, v, is_causal=True)
#             y = einsum('bhtd,hcd->btc', y, W_Q)
#         """
#         use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
#         use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
#         accelerator = Accelerator.get_accelerator()

#         # Compute Q and K via einsum with tied weights, V = K (energy attention specific)
#         if self.use_padding_free_transformer:
#             assert use_flash_attention_2 or use_flash_attention_3
#             assert past_key_values is None

#             # hidden_states: (total_tokens, hidden_size)
#             # einsum: 'tc,ahcs->ahts' produces (2, num_heads, total_tokens, head_dim)
#             qk = torch.einsum("tc,ahcs->ahts", hidden_states, self.qk_weights)
#             query, key = qk.unbind(0)  # Each: (num_heads, total_tokens, head_dim)
#             value = key.clone()  # V = K

#             # Flash attention expects (total_tokens, num_heads, head_dim)
#             query = query.permute(1, 0, 2).contiguous()
#             key = key.permute(1, 0, 2).contiguous()
#             value = value.permute(1, 0, 2).contiguous()
#         else:
#             batch_size, query_length = hidden_states.shape[:-1]

#             # hidden_states: (B, T, C), qk_weights: (2, H, C, S) -> (B, 2, H, T, S)
#             qk = torch.einsum("btc,ahcs->bahts", hidden_states, self.qk_weights)
#             query, key = qk.unbind(1)  # Each: (B, H, T, S)
#             value = key.clone()  # V = K

#         # W_Q for output projection (energy attention specific)
#         W_Q = self.qk_weights[0] / self.INIT_STD  # (num_heads, hidden_size, head_dim)

#         # KV cache support for generation
#         if past_key_values is not None:
#             key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

#         if use_flash_attention_2 or use_flash_attention_3:
#             assert accelerator == Accelerator.cuda

#             if not self.use_padding_free_transformer:
#                 # Flash attention expects (B, T, H, S)
#                 query = query.transpose(1, 2).contiguous()
#                 key = key.transpose(1, 2).contiguous()
#                 value = value.transpose(1, 2).contiguous()

#             query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
#             key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
#             value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

#             attn_output = flash_attention(
#                 query=query,
#                 key=key,
#                 value=value,
#                 cu_seqlens=cu_seqlens,
#                 max_seqlen=max_seqlen,
#                 attention_mask=attention_mask,
#                 use_padding_free_transformer=self.use_padding_free_transformer,
#                 causal=self.causal,
#                 dropout=self.softmax_dropout_p if self.training else 0,
#                 softmax_scale=None,  # Use default 1/sqrt(head_dim) to match original
#                 sliding_window=self.sliding_window,
#             )

#             del query, key, value
#             attn_output = wait_for_ACT(attn_output, wait_in_forward=False, wait_in_backward=True)

#             if self.use_padding_free_transformer:
#                 # attn_output: (total_tokens, num_heads, head_dim)
#                 # Transpose for output einsum: (num_heads, total_tokens, head_dim)
#                 attn_output = attn_output.permute(1, 0, 2)
#                 # Output projection: 'hts,hcs->tc' (energy attention specific)
#                 hidden_states = torch.einsum("hts,hcs->tc", attn_output, W_Q)
#             else:
#                 # attn_output: (B, T, H, S) -> (B, H, T, S) for einsum
#                 attn_output = attn_output.transpose(1, 2)
#                 # Output projection: 'bhts,hcs->btc' (energy attention specific)
#                 hidden_states = torch.einsum("bhts,hcs->btc", attn_output, W_Q)
#         else:
#             assert self.sliding_window is None

#             if accelerator == Accelerator.tpu:
#                 assert attention_mask is None
#                 assert self.softmax_dropout_p == 0

#                 attn_output = flash_attention_tpu(
#                     query,
#                     key,
#                     value,
#                     causal=self.causal if attention_mask is None else False,
#                     sm_scale=1 / math.sqrt(self.head_dim),
#                 )
#             else:
#                 # Standard scaled_dot_product_attention (same as original energy attention)
#                 attn_output = F.scaled_dot_product_attention(
#                     query,
#                     key,
#                     value,
#                     attn_mask=attention_mask,
#                     dropout_p=self.softmax_dropout_p if self.training else 0,
#                     is_causal=self.causal if attention_mask is None else False,
#                 )

#             del query, key, value

#             # Output projection via einsum (energy attention specific)
#             # attn_output: (B, H, T, S), W_Q: (H, C, S) -> (B, T, C)
#             hidden_states = torch.einsum("bhts,hcs->btc", attn_output, W_Q)

#         hidden_states = self.resid_dropout(hidden_states)
#         return hidden_states

#     def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
#         """Compute energy per token for this attention layer."""
#         B, T, C = x.shape
#         q, k = torch.einsum("btc,ahcs->bahts", x, self.qk_weights).unbind(1)
#         attn_weights = torch.einsum("bhts,bhks->bhtk", q, k) / (self.head_dim**0.5)
#         return attn_weights.sum(dim=-1).mean(dim=1)


class GradFF_2W_manual(nn.Module):
    """Feedforward network with manual gradient computation for Energy Transformer."""

    def __init__(self, config: CommonConfig, layer_idx: int | None = None) -> None:
        super().__init__()
        hidden_size = config.hidden_size
        ff_hid_factor = getattr(config, "ff_hid_factor", 4)
        intermediate_size = getattr(config, "intermediate_size", ff_hid_factor * hidden_size)

        self.W = nn.Parameter(torch.randn(2, hidden_size, intermediate_size))
        nn.init.xavier_uniform_(self.W, gain=8.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x, W2x = torch.einsum("bti, hio -> hbto", x, self.W).unbind(0)
        y1 = F.gelu(W1x)
        # The second term from grad of sigma(x W1) W2^T x^T
        y2 = F.sigmoid((2 / torch.pi) ** 0.5 * W1x) * 0.5
        y2 = torch.einsum("bto,io,bto->bti", y2, self.W[0], W2x)
        out = torch.einsum("bto, io -> bti", y1, self.W[1]) + y2
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy per token for this feedforward layer."""
        W1x, W2x = torch.einsum("bti, hio -> hbto", x, self.W).unbind(0)
        return F.gelu(W1x).sum(dim=-1)


class EnergyBlock(nn.Module):
    """Energy Transformer block with customizable attention and feedforward.

    Unlike standard Transformer blocks that use additive residual connections,
    EnergyBlock uses a subtractive update inspired by energy-based models:
        x = x - proj(attn(ln(x)) + scale_ff * ffwd(ln(x)))
    """

    def __init__(
        self,
        config: CommonConfig,
        attn_class: type,
        ffwd_class: type,
        layernorm_class: type = nn.LayerNorm,
        use_padding_free_transformer: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        hidden_size = config.hidden_size

        # Build sequence mixer kwargs for EnergyAttention_QK (same as get_sequence_mixer)
        sequence_mixer_kwargs = {
            "hidden_size": hidden_size,
            "num_attention_heads": config.num_attention_heads,
            "num_key_value_heads": config.num_key_value_heads,
            "attention_multiplier": config.attention_multiplier,
            "sliding_window": getattr(config, "sliding_window", None),
            "position_embedding_type": config.position_embedding_type,
            "add_bias": config.add_bias,
            "qkv_bias": getattr(config.sequence_mixer_blocks[layer_idx], "qkv_bias", False) if layer_idx is not None else False,
            "softmax_dropout": getattr(config.sequence_mixer_blocks[layer_idx], "softmax_dropout", 0.0) if layer_idx is not None else 0.0,
            "dropout": config.hidden_dropout,
            "init_method": config.init_method,
            "initializer_range": config.initializer_range,
            "m_width": config.m_width,
            "num_layers": config.num_layers,
            "causal": True,
            "layer_idx": layer_idx,
            "use_padding_free_transformer": use_padding_free_transformer,
        }

        self.attn = attn_class(**sequence_mixer_kwargs)
        self.ffwd = ffwd_class(config, layer_idx=layer_idx)
        self.ln = layernorm_class(hidden_size)
        self.scale_ff = nn.Parameter(torch.ones(1) * 4, requires_grad=True)
        self.proj = nn.Linear(hidden_size, hidden_size, bias=False)

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        ln_x = self.ln(hidden_states)
        attn_out = self.attn(
            ln_x,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            rope_cos_sin=rope_cos_sin,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        ffwd_out = self.ffwd(ln_x)
        hidden_states = hidden_states - self.proj(attn_out + self.scale_ff * ffwd_out)
        return hidden_states

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total energy per token."""
        ln_x = self.ln(x)
        return self.attn.energy_per_token(ln_x) + self.scale_ff * self.ffwd.energy_per_token(ln_x)


class EnergyBlock_QK_FF2W_manual(EnergyBlock):
    """Energy Transformer block with standard QK attention and GradFF_2W_manual.

    This is the main energy-based block combining:
    - EnergyAttention_QK: Energy-based Q/K attention (with flash attention support)
    - GradFF_2W_manual: Feedforward with manual gradient computation
    - BareLayerNorm: LayerNorm without learnable weights
    """

    def __init__(
        self,
        config: CommonConfig,
        use_padding_free_transformer: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__(
            config,
            attn_class=EnergyAttention_QK,
            ffwd_class=GradFF_2W_manual,
            layernorm_class=BareLayerNorm,
            use_padding_free_transformer=use_padding_free_transformer,
            layer_idx=layer_idx,
        )
