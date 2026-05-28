# **************************************************
# Copyright (c) 2025
# EGrad Attention: Mixed energy + standard heads
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
from ...modeling_utils.dropout import Dropout
from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.position_embedding import apply_rotary_pos_emb
from ...modeling_utils.sequence_mixer_blocks.utils import flash_attention
from ...parameter import mark_parameter_as_mup_learning_rate


if is_torch_xla_available():
    from torch_xla.experimental.custom_kernel import flash_attention as flash_attention_tpu


def _build_head_mask(num_heads: int, num_energy_heads: int, placement: str) -> list[bool]:
    """Return a boolean mask: True = energy head, False = standard head."""
    if placement == "first":
        return [True] * num_energy_heads + [False] * (num_heads - num_energy_heads)
    elif placement == "last":
        return [False] * (num_heads - num_energy_heads) + [True] * num_energy_heads
    elif placement == "interleaved":
        mask = [False] * num_heads
        step = num_heads / num_energy_heads
        for i in range(num_energy_heads):
            mask[int(i * step)] = True
        return mask
    else:
        raise ValueError(f"unexpected energy_head_placement ({placement})")


class EGradAttention(nn.Module):
    """EGrad-style attention with mixed energy and standard heads.

    Energy heads use V=K and output via W_Q^T (reuse Q weights transposed).
    Standard heads use learned V and separate c_proj output projection.
    Both share a single flash attention call for efficiency.

    Head placement is controlled by `energy_head_placement`:
    - "first": energy heads are heads 0..E-1
    - "last": energy heads are heads (H-E)..H-1
    - "interleaved": energy heads are evenly spaced
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_key_value_heads: int,
        num_energy_heads: int,
        energy_head_placement: str,
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
    ) -> EGradAttention:
        super().__init__()

        self.causal = causal
        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_key_value_heads = num_key_value_heads
        self.num_energy_heads = num_energy_heads
        self.num_standard_heads = num_attention_heads - num_energy_heads
        self.add_bias = add_bias
        self.qkv_bias = qkv_bias
        self.use_padding_free_transformer = use_padding_free_transformer
        self.sliding_window = sliding_window
        self.energy_head_placement = energy_head_placement

        assert 0 < num_energy_heads < num_attention_heads, (
            f"num_energy_heads ({num_energy_heads}) must be between 1 and "
            f"num_attention_heads-1 ({num_attention_heads - 1})"
        )

        self.head_dim = divide_if_divisible(
            self.hidden_size,
            self.num_heads,
            f"`hidden_size` ({self.hidden_size}) must be divisible by "
            f"`num_heads` ({self.num_heads})",
        )

        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.layer_idx = layer_idx

        self.energy_dim = self.num_energy_heads * self.head_dim
        self.standard_dim = self.num_standard_heads * self.head_dim

        # Build head assignment mask
        head_mask = _build_head_mask(self.num_heads, self.num_energy_heads, energy_head_placement)
        self.energy_head_indices = [i for i, is_e in enumerate(head_mask) if is_e]
        self.standard_head_indices = [i for i, is_e in enumerate(head_mask) if not is_e]
        self._energy_idx: torch.Tensor | None = None
        self._standard_idx: torch.Tensor | None = None

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        # Q and K for ALL heads + V only for standard heads
        # Layout: [Q_all (hidden_size), K_all (hidden_size), V_standard (standard_dim)]
        qkv_size = self.hidden_size + self.hidden_size + self.standard_dim
        self.c_attn = ParameterizedLinear(
            self.hidden_size, qkv_size, bias=self.qkv_bias, std=std,
        )

        # Output projection for standard heads only
        std_proj = initializer_range / math.sqrt(2 * num_layers)
        if init_method == "mup":
            std_proj /= math.sqrt(m_width)
        self.c_proj = ParameterizedLinear(
            self.standard_dim, self.hidden_size, bias=self.add_bias, std=std_proj
        )

        self.softmax_dropout_p = softmax_dropout
        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_attn.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def _ensure_index_buffers(self) -> None:
        """Build index buffers on the correct device (lazy, so meta-device init is safe)."""
        device = self.c_attn.weight.device
        if self._energy_idx is not None and self._energy_idx.device == device:
            return
        self._energy_idx = torch.tensor(
            self.energy_head_indices, dtype=torch.long, device=device
        )
        self._standard_idx = torch.tensor(
            self.standard_head_indices, dtype=torch.long, device=device
        )

    def extra_repr(self) -> str:
        return (
            f"num_energy_heads={self.num_energy_heads}, "
            f"num_standard_heads={self.num_standard_heads}, "
            f"placement={self.energy_head_placement}, "
            f"sliding_window={self.sliding_window}"
        )

    def _get_energy_q_weight(self) -> torch.Tensor:
        """Extract Q weights for energy heads to use as output projection."""
        q_weight = self.c_attn.weight[: self.hidden_size]  # (hidden_size, hidden_size)
        q_weight = q_weight.view(self.num_heads, self.head_dim, self.hidden_size)
        q_energy = q_weight[self._energy_idx]  # (E, head_dim, hidden_size)
        q_energy = q_energy.permute(0, 2, 1).contiguous()  # (E, hidden_size, head_dim)
        return q_energy

    def forward(
        self,
        hidden_states: torch.Tensor,
        past_key_values: GenerationCache | None = None,
        attention_mask: torch.Tensor | None = None,
        rope_cos_sin: torch.Tensor | None = None,
        cu_seqlens: torch.Tensor | None = None,
        max_seqlen: int | None = None,
    ) -> torch.Tensor:
        self._ensure_index_buffers()

        use_flash_attention_2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_flash_attention_3 = is_kernel_allowed(Kernel.flash_attention_3)
        accelerator = Accelerator.get_accelerator()

        if self.use_padding_free_transformer:
            assert use_flash_attention_2 or use_flash_attention_3
            assert past_key_values is None
            total_q = hidden_states.shape[0]
        else:
            batch_size, query_length = hidden_states.shape[:-1]

        # Project Q, K, V_standard
        qkv = self.c_attn(hidden_states)
        q_all, k_all, v_std = qkv.split(
            [self.hidden_size, self.hidden_size, self.standard_dim], dim=-1
        )

        # Reshape to head format
        if self.use_padding_free_transformer:
            q_all = q_all.view(total_q, self.num_heads, self.head_dim)
            k_all = k_all.view(total_q, self.num_heads, self.head_dim)
            v_std = v_std.view(total_q, self.num_standard_heads, self.head_dim)
        else:
            q_all = q_all.view(batch_size, query_length, self.num_heads, self.head_dim)
            k_all = k_all.view(batch_size, query_length, self.num_heads, self.head_dim)
            v_std = v_std.view(batch_size, query_length, self.num_standard_heads, self.head_dim)
            q_all = q_all.transpose(1, 2)  # (B, H, T, D)
            k_all = k_all.transpose(1, 2)
            v_std = v_std.transpose(1, 2)

        # Apply RoPE
        if self.position_embedding_type == "rope" and rope_cos_sin is not None:
            q_all = apply_rotary_pos_emb(q_all, rope_cos_sin)
            k_all = apply_rotary_pos_emb(k_all, rope_cos_sin)

        # Build V tensor: energy heads get V=K, standard heads get learned V
        # Place them in the correct head positions
        if self.use_padding_free_transformer:
            v_all = torch.empty_like(q_all)  # (total_q, H, D)
            v_all[:, self._energy_idx, :] = k_all[:, self._energy_idx, :]
            v_all[:, self._standard_idx, :] = v_std
        else:
            v_all = torch.empty_like(q_all)  # (B, H, T, D)
            v_all[:, self._energy_idx, :, :] = k_all[:, self._energy_idx, :, :]
            v_all[:, self._standard_idx, :, :] = v_std

        # KV cache update
        if past_key_values is not None:
            k_all, v_all = past_key_values.update(
                key_states=k_all, value_states=v_all, layer_idx=self.layer_idx
            )

        W_Q_energy = self._get_energy_q_weight()

        # Run attention (single call for all heads)
        if use_flash_attention_2 or use_flash_attention_3:
            assert accelerator == Accelerator.cuda

            if not self.use_padding_free_transformer:
                q_all = q_all.transpose(1, 2)
                k_all = k_all.transpose(1, 2)
                v_all = v_all.transpose(1, 2)

            q_all = wait_for_ACT(q_all, wait_in_forward=True, wait_in_backward=False)
            k_all = wait_for_ACT(k_all, wait_in_forward=True, wait_in_backward=False)
            v_all = wait_for_ACT(v_all, wait_in_forward=True, wait_in_backward=False)

            attn_output = flash_attention(
                query=q_all, key=k_all, value=v_all,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                softmax_scale=self.attention_multiplier,
                sliding_window=self.sliding_window,
            )

            del q_all, k_all, v_all
            attn_output = wait_for_ACT(attn_output, wait_in_forward=False, wait_in_backward=True)

            if self.use_padding_free_transformer:
                # (total_q, H, D)
                attn_energy = attn_output[:, self._energy_idx, :]
                attn_std = attn_output[:, self._standard_idx, :]
                energy_out = torch.einsum("tes,ecs->tc", attn_energy, W_Q_energy)
                std_out = self.c_proj(attn_std.reshape(total_q, self.standard_dim))
            else:
                # (B, T, H, D)
                attn_energy = attn_output[:, :, self._energy_idx, :]
                attn_std = attn_output[:, :, self._standard_idx, :]
                attn_energy_p = attn_energy.permute(2, 0, 1, 3)  # (E, B, T, D)
                energy_out = torch.einsum("ebts,ecs->btc", attn_energy_p, W_Q_energy)
                std_out = self.c_proj(attn_std.reshape(batch_size, query_length, self.standard_dim))
        else:
            assert self.sliding_window is None

            if accelerator == Accelerator.tpu:
                assert attention_mask is None
                assert self.softmax_dropout_p == 0
                attn_output = flash_attention_tpu(
                    q_all, k_all, v_all,
                    causal=self.causal if attention_mask is None else False,
                    sm_scale=(
                        1 / math.sqrt(self.head_dim)
                        if self.attention_multiplier is None
                        else self.attention_multiplier
                    ),
                )
            else:
                attn_output = F.scaled_dot_product_attention(
                    q_all, k_all, v_all,
                    attn_mask=attention_mask,
                    dropout_p=self.softmax_dropout_p if self.training else 0,
                    is_causal=self.causal if attention_mask is None else False,
                    scale=self.attention_multiplier,
                    enable_gqa=True,
                )

            del q_all, k_all, v_all

            # (B, H, T, D)
            attn_energy = attn_output[:, self._energy_idx, :, :]
            attn_std = attn_output[:, self._standard_idx, :, :]
            energy_out = torch.einsum("bhts,hcs->btc", attn_energy, W_Q_energy)
            attn_std = attn_std.transpose(1, 2).reshape(batch_size, -1, self.standard_dim)
            std_out = self.c_proj(attn_std)

        hidden_states = energy_out + std_out
        hidden_states = self.dropout(hidden_states)

        return hidden_states
