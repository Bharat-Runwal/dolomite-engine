# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed, wait_for_ACT
from ....utils import Accelerator, divide_if_divisible, is_flex_attention_available, is_torch_xla_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate
from ..chunk import contiguous_split
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from ..position_embedding import apply_rotary_pos_emb
from .utils import flash_attention


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu

if is_flex_attention_available():
    import functools

    from torch.nn.attention.flex_attention import create_block_mask, flex_attention as torch_flex_attention

    def _causal_mask(b, h, q_idx, kv_idx):
        return q_idx >= kv_idx

    @functools.lru_cache
    def _sliding_window_causal_mask(window_size: int):
        def mask_fn(b, h, q_idx, kv_idx):
            causal = q_idx >= kv_idx
            windowed = q_idx - kv_idx < window_size
            return causal & windowed
        mask_fn.__name__ = f"sliding_window_{window_size}"
        return mask_fn


def interleave_query_key_value_tensor_for_attention(
    query_weight: torch.Tensor,
    key_weight: torch.Tensor,
    value_weight: torch.Tensor,
    num_heads: int,
    num_key_value_heads: int,
    head_dim: int,
) -> torch.Tensor:
    query_heads_per_group = num_heads // num_key_value_heads

    interleaved = []
    for i in range(num_key_value_heads):
        start_index = i * query_heads_per_group * head_dim
        end_index = start_index + query_heads_per_group * head_dim
        interleaved.append(query_weight[start_index:end_index])

        start_index = i * head_dim
        end_index = start_index + head_dim
        interleaved.append(key_weight[start_index:end_index])
        interleaved.append(value_weight[start_index:end_index])

    return torch.cat(interleaved)


def split_query_key_value_tensor_for_attention(
    query_key_value_weight: torch.Tensor, num_heads: int, num_key_value_heads: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    query_heads_per_group = num_heads // num_key_value_heads
    original_shape = query_key_value_weight.shape

    query_key_value_weight = query_key_value_weight.view(num_key_value_heads, (query_heads_per_group + 2), -1)

    query_weight, key_weight, value_weight = query_key_value_weight.split((query_heads_per_group, 1, 1), 1)

    query_weight = query_weight.reshape(-1, *original_shape[1:])
    key_weight = key_weight.reshape(-1, *original_shape[1:])
    value_weight = value_weight.reshape(-1, *original_shape[1:])

    return query_weight, key_weight, value_weight


class Attention(nn.Module):
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
        use_attention_sink: bool = False,
    ) -> Attention:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.add_bias = add_bias
        self.qkv_bias = qkv_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sliding_window = sliding_window
        self.use_attention_sink = use_attention_sink

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by `num_heads` ({self.num_heads})",
        )

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        divide_if_divisible(
            self.num_heads,
            self.num_key_value_heads,
            f"`num_heads` ({self.num_heads}) should be a multiple of `num_key_value_heads` ({self.num_key_value_heads})",
        )

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_attn = ParameterizedLinear(
            self.hidden_size,
            self.hidden_size + 2 * self.num_key_value_heads * self.head_dim,
            bias=self.qkv_bias,
            std=std,
        )

        std = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(self.hidden_size, self.hidden_size, bias=self.add_bias, std=std)

        self.softmax_dropout_p = softmax_dropout

        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_attn.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

        # Learnable attention sink (MIMO-V2 / GPT-OSS style)
        # Per-head bias that learns to scale attention output based on logsumexp
        if self.use_attention_sink:
            self.sinks = nn.Parameter(torch.zeros(self.num_heads))


    def extra_repr(self):
        return f"sliding_window={self.sliding_window}, {super().extra_repr()}"


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
        use_flex_attention = is_kernel_allowed(Kernel.flex_attention)
        accelerator = Accelerator.get_accelerator()

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None

            total_q = hidden_states.shape[0]
            input_shape = (total_q, self.num_key_value_heads, -1)
            output_shape = (total_q, -1, self.head_dim)
        else:
            batch_size, query_length = hidden_states.shape[:-1]

            input_shape = (batch_size, query_length, self.num_key_value_heads, -1)
            output_shape = (batch_size, query_length, -1, self.head_dim)

        hidden_states = self.c_attn(hidden_states)
        hidden_states = hidden_states.view(*input_shape)

        query, key, value = (
            contiguous_split if Accelerator.get_accelerator() == Accelerator.trainium else torch.split
        )(
            hidden_states,
            ((self.num_heads // self.num_key_value_heads) * self.head_dim, self.head_dim, self.head_dim),
            dim=-1,
        )

        query = query.reshape(*output_shape)

        if not self.use_padding_free_transformer:
            query = query.transpose(1, 2)
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)

        if self.position_embedding_type == "rope":
            query = apply_rotary_pos_emb(query, rope_cos_sin)
            key = apply_rotary_pos_emb(key, rope_cos_sin)

        if past_key_values is not None:
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=self.layer_idx)

        lse = None  # logsumexp for attention sink

        # flex_attention path (PyTorch native)
        if use_flex_attention:
            assert accelerator == Accelerator.cuda
            assert not self.use_padding_free_transformer, "flex_attention doesn't support padding-free mode"
            assert past_key_values is None, "flex_attention doesn't support KV cache"

            # flex_attention expects (B, H, S, D) format - query/key/value are already in this format
            # build block mask: causal with optional sliding window
            block_mask = None
            if self.causal:
                if self.sliding_window is not None:
                    mask_fn = _sliding_window_causal_mask(self.sliding_window)
                else:
                    mask_fn = _causal_mask
                block_mask = create_block_mask(
                    mask_fn, B=None, H=None, Q_LEN=query.shape[2], KV_LEN=key.shape[2],
                    device=query.device,
                )

            if self.use_attention_sink:
                hidden_states, lse = torch_flex_attention(
                    query,
                    key,
                    value,
                    block_mask=block_mask,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                    return_lse=True,
                )
            else:
                hidden_states = torch_flex_attention(
                    query,
                    key,
                    value,
                    block_mask=block_mask,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                )

            del query, key, value

            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        elif use_flash_attention_2 or use_flash_attention_3:
            assert accelerator == Accelerator.cuda

            if self.use_padding_free_transformer:
                output_shape = (-1, self.hidden_size)
            else:
                query = query.transpose(1, 2)
                key = key.transpose(1, 2)
                value = value.transpose(1, 2)

                output_shape = (batch_size, query_length, -1)

            query = wait_for_ACT(query, wait_in_forward=True, wait_in_backward=False)
            key = wait_for_ACT(key, wait_in_forward=True, wait_in_backward=False)
            value = wait_for_ACT(value, wait_in_forward=True, wait_in_backward=False)

            # Request lse when using FA3 with attention sink
            if self.use_attention_sink and use_flash_attention_3:
                hidden_states, lse = flash_attention(
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
                    return_lse=True,
                )
            else:
                hidden_states = flash_attention(
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

            hidden_states = wait_for_ACT(hidden_states, wait_in_forward=False, wait_in_backward=True)
            hidden_states = hidden_states.view(*output_shape)
        else:
            assert self.sliding_window is None

            if accelerator == Accelerator.tpu:
                assert attention_mask is None
                assert self.softmax_dropout_p == 0

                hidden_states = flash_attention_tpu(
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
                hidden_states = F.scaled_dot_product_attention(
                    query,
                    key,
                    value,
                    attn_mask=attention_mask,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    is_causal=self.causal if attention_mask is None else False,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                )

            del query, key, value

            batch_size = hidden_states.shape[0]
            hidden_states = hidden_states.transpose(1, 2)
            hidden_states = hidden_states.reshape(batch_size, -1, self.num_heads * self.head_dim)

        # Apply attention sink scaling if enabled and lse is available
        if self.use_attention_sink and lse is not None:
            hidden_states = self._apply_attention_sink(hidden_states, lse)

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states

    def _apply_attention_sink(self, attn_output: torch.Tensor, lse: torch.Tensor) -> torch.Tensor:
        """Apply implicit attention sink via output scaling (MIMO-V2 / GPT-OSS style).

        The attention sink mechanism scales the attention output based on the logsumexp
        of attention scores. This allows the model to learn when to "sink" attention
        to reduce the impact of uncertain/diffuse attention patterns.

        Formula: sink_scale = sigmoid(logsumexp - sinks)
                 attn_output = attn_output * sink_scale

        Args:
            attn_output: Attention output tensor
                - Standard mode: (B, S, H*D)
                - Padding-free mode: (total_tokens, H*D)
            lse: Logsumexp from attention computation
                - Standard mode: (B, H, S) or (B, S, H)
                - Padding-free mode: (total_tokens, H)

        Returns:
            Scaled attention output with same shape as input
        """
        if self.use_padding_free_transformer:
            # lse: (total_tokens, H), attn_output: (total_tokens, H*D)
            sink_scale = torch.sigmoid(lse - self.sinks)  # (total_tokens, H)
            total_tokens = attn_output.shape[0]
            attn_output = attn_output.view(total_tokens, self.num_heads, self.head_dim)
            attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
            attn_output = attn_output.view(total_tokens, -1)
        else:
            # lse from FA3: (B, H, S), from flex_attention: (B, H, S)
            # sinks: (H,) -> (1, H, 1) for broadcast
            sink_scale = torch.sigmoid(lse - self.sinks.view(1, -1, 1))  # (B, H, S)
            sink_scale = sink_scale.transpose(1, 2)  # (B, S, H)
            # attn_output: (B, S, H*D)
            B, S, _ = attn_output.shape
            attn_output = attn_output.view(B, S, self.num_heads, self.head_dim)
            attn_output = attn_output * sink_scale.unsqueeze(-1).to(attn_output.dtype)
            attn_output = attn_output.view(B, S, -1)

        return attn_output
