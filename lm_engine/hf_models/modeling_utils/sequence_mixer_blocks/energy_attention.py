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
from ....utils import Accelerator, divide_if_divisible, is_torch_xla_available
from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils.dropout import Dropout
from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.position_embedding import apply_rotary_pos_emb
from ...modeling_utils.sequence_mixer_blocks.utils import flash_attention
from ...parameter import mark_parameter_as_mup_learning_rate


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


class EnergyAttention_QK(nn.Module):
    """Energy-based Q/K attention with mup initialization, RoPE, and flash attention support.

    Core energy attention properties preserved:
    - Q and K projected together via c_attn (no separate V projection)
    - V = K (value equals key)
    - Output projection uses Q weights scaled by initialization std
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        attention_multiplier: float,
        sliding_window: int | None,
        position_embedding_type: str,
        add_bias: bool,
        qkv_bias: bool,
        softmax_dropout: float,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        causal: bool,
        layer_idx: int,
        use_padding_free_transformer: bool,
    ) -> EnergyAttention_QK:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.qkv_bias = qkv_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sliding_window = sliding_window

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        # Compute initialization std (mup-style)
        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_attn_init_std = std  # Store for output projection scaling

        # c_attn projects to Q and K only (V = K in energy attention)
        self.c_attn = ParameterizedLinear(
            self.hidden_size,
            2 * self.hidden_size,
            bias=self.qkv_bias,
            std=std,
        )

        self.softmax_dropout_p = softmax_dropout
        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_attn.weight)

    def extra_repr(self) -> str:
        return f"sliding_window={self.sliding_window}, energy_attention=True"

    def _get_q_weight_for_output(self) -> torch.Tensor:
        """Extract Q projection weights for energy attention output projection."""
        # c_attn.weight shape: (2*hidden_size, hidden_size)
        # Q portion is first hidden_size rows
        q_weight = self.c_attn.weight[: self.hidden_size]  # (H*D, C)
        q_weight = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
        q_weight = q_weight.permute(0, 2, 1).contiguous()  # (H, C, D)
        return q_weight / self.c_attn_init_std

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
        accelerator = Accelerator.get_accelerator()

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None
            total_q = hidden_states.shape[0]
            input_shape = (total_q, 2, self.num_heads, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]
            input_shape = (batch_size, query_length, 2, self.num_heads, self.head_dim)

        hidden_states = self.c_attn(hidden_states)
        hidden_states = hidden_states.view(*input_shape)

        if self.use_padding_free_transformer:
            query, key = hidden_states.unbind(1)
            query = query.contiguous()
            key = key.contiguous()
        else:
            query, key = hidden_states.unbind(2)
            query = query.transpose(1, 2).contiguous()
            key = key.transpose(1, 2).contiguous()

        # V = K (core energy attention property)
        value = key

        # Apply RoPE if configured
        if self.position_embedding_type == "rope" and rope_cos_sin is not None:
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)
            value = key

        if past_key_values is not None:
            key, value = past_key_values.update(
                key_states=key, value_states=value, layer_idx=self.layer_idx
            )

        W_Q = self._get_q_weight_for_output()

        if use_flash_attention_2 or use_flash_attention_3:
            assert accelerator == Accelerator.cuda

            if not self.use_padding_free_transformer:
                query = query.transpose(1, 2).contiguous()
                key = key.transpose(1, 2).contiguous()
                value = value.transpose(1, 2).contiguous()

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            attn_output = flash_attention(
                query=query,
                key=key,
                value=value,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
                sliding_window=self.sliding_window,
            )

            del query, key, value
            attn_output = wait_for_ACT(attn_output, wait_in_forward=False, wait_in_backward=True)

            if self.use_padding_free_transformer:
                attn_output = attn_output.permute(1, 0, 2)
                hidden_states = torch.einsum("hts,hcs->tc", attn_output, W_Q)
            else:
                attn_output = attn_output.transpose(1, 2)
                hidden_states = torch.einsum("bhts,hcs->btc", attn_output, W_Q)
        else:
            assert self.sliding_window is None

            if accelerator == Accelerator.tpu:
                assert attention_mask is None
                assert self.softmax_dropout_p == 0

                attn_output = flash_attention_tpu(
                    query,
                    key,
                    value,
                    causal=self.causal if attention_mask is None else False,
                    sm_scale=(
                        1 / math.sqrt(self.head_dim)
                        if self.attention_multiplier is None
                        else self.attention_multiplier
                    ),
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    is_causal=self.causal if attention_mask is None else False,
                    scale=self.attention_multiplier,
                )

            del query, key, value
            hidden_states = torch.einsum("bhts,hcs->btc", attn_output, W_Q)

        hidden_states = self.dropout(hidden_states)
        return hidden_states

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Compute energy per token for this attention layer."""
        batch_size, seq_len = x.shape[:2]
        qk = self.c_attn(x)
        qk = qk.view(batch_size, seq_len, 2, self.num_heads, self.head_dim)
        query, key = qk.unbind(2)
        query = query.transpose(1, 2)
        key = key.transpose(1, 2)
        attn_weights = torch.einsum("bhts,bhks->bhtk", query, key) / (self.head_dim**0.5)
        return attn_weights.sum(dim=-1).mean(dim=1)
