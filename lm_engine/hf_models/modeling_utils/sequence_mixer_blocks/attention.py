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
from ....utils import Accelerator, divide_if_divisible, is_torch_xla_available
from ...cache import GenerationCache
from ...parameter import mark_parameter_as_mup_learning_rate
from ..chunk import contiguous_split
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from ..position_embedding import apply_rotary_pos_emb
from .utils import flash_attention


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


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

        # ----- register attention-balance aux loss hooks ---------------------
        # Set externally by RegisterEnergyModel.forward() before each block forward.
        # When _capture_register_attn_mass is True (and _n_registers > 0 and
        # self.layer_idx >= _register_start_layer), the forward will fall back to
        # a manual Q @ K^T / sqrt(d_h) path so the attention probabilities are
        # available, and cache mean content→register attention mass on
        # _register_attn_mass. Otherwise the regular fused path runs (zero cost).
        self._capture_register_attn_mass: bool = False
        self._n_registers: int = 0
        self._register_start_layer: int = 0
        self._register_attn_mass: torch.Tensor | None = None


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
        layer_id: int | None = None,
    ) -> torch.Tensor:
        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
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
            cache_idx = layer_id if layer_id is not None else self.layer_idx
            key, value = past_key_values.update(key_states=key, value_states=value, layer_idx=cache_idx)

        # ---- register attention-balance capture (side measurement) ----------
        # Compute a side softmax over (content_queries × all_keys) JUST to read
        # the content→register attention mass. This DOES NOT replace the main
        # attention output (SDPA/FA2/FA3 still produces the output below with
        # unchanged shapes/dtypes), so gradient checkpointing recompute sees
        # identical tensor metadata. Gradient still flows through Q, K so the
        # aux loss is trainable. Inert when capture flag is off.
        _capture_reg = (
            getattr(self, '_capture_register_attn_mass', False)
            and getattr(self, '_n_registers', 0) > 0
            and self.layer_idx >= getattr(self, '_register_start_layer', 0)
            and not self.use_padding_free_transformer
            and self.training
        )
        if _capture_reg:
            R = self._n_registers
            # query / key here are [B, H, T_q, d_h] / [B, H_kv, T_k, d_h] (the
            # non-FA branch transposes happen below). Use them as-is.
            T_q = query.shape[-2]
            T_k = key.shape[-2]
            if T_q > R and T_k >= R:
                scale = (
                    1.0 / math.sqrt(self.head_dim)
                    if self.attention_multiplier is None
                    else self.attention_multiplier
                )
                qheads_per_kv = self.num_heads // self.num_key_value_heads
                # GQA broadcast over the key-head axis without materialising a
                # full repeated copy: repeat_interleave is fine here because we
                # only slice content rows (T_q - R), and this side path is
                # already cheaper than the main attention.
                if qheads_per_kv > 1:
                    k_for_mass = key.repeat_interleave(qheads_per_kv, dim=1)
                else:
                    k_for_mass = key
                # Only content queries (rows R..T_q-1) participate in the loss.
                q_content = query[..., R:, :]                                  # [B, H, T_q-R, d_h]
                scores = torch.matmul(q_content, k_for_mass.transpose(-2, -1)) * scale  # [B, H, T_q-R, T_k]
                if attention_mask is not None:
                    # attention_mask: [B, 1, T_q, T_k] additive float
                    scores = scores + attention_mask[..., R:, :]
                elif self.causal:
                    row = torch.arange(R, T_q, device=scores.device).unsqueeze(-1)
                    col = torch.arange(T_k, device=scores.device).unsqueeze(0)
                    causal_mask = (col > (row + (T_k - T_q)))
                    scores = scores.masked_fill(causal_mask, float('-inf'))
                probs = F.softmax(scores, dim=-1)
                # Mass on register keys (columns 0..R-1):
                self._register_attn_mass = probs[..., :R].sum(-1).mean()
            else:
                self._register_attn_mass = None

        if use_flash_attention_2 or use_flash_attention_3:
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

        hidden_states = self.c_proj(hidden_states)
        hidden_states = self.dropout(hidden_states)

        return hidden_states
