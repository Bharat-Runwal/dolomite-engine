from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ....enums import Kernel
from ....kernels import is_kernel_allowed
from ....utils import Accelerator, divide_if_divisible
from ...cache import GenerationCache
from ...config import CommonConfig
from ...modeling_utils.dropout import Dropout
from ...modeling_utils.linear import ParameterizedLinear
from ...modeling_utils.position_embedding import apply_rotary_pos_emb
from ...modeling_utils.sequence_mixer_blocks.utils import flash_attention
from ...parameter import mark_parameter_as_mup_learning_rate


class MixedHeadAttention(nn.Module):
    """Per-head mix of energy-style heads (V=K) and standard GPT heads (independent V).

    Energy heads perform nearest-neighbor retrieval in key-space (V=K).
    GPT heads perform arbitrary value lookup (V = W_V * x).
    All heads share a single W_O output projection.
    Uses a single flash-attention kernel call: build V as (K_energy | V_gpt).
    """

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        num_energy_heads: int,
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
    ) -> MixedHeadAttention:
        super().__init__()

        assert 0 < num_energy_heads < num_attention_heads, (
            f"num_energy_heads ({num_energy_heads}) must be in (0, num_attention_heads={num_attention_heads})"
        )

        self.hidden_size = hidden_size
        self.num_heads = num_attention_heads
        self.num_energy_heads = num_energy_heads
        self.num_gpt_heads = num_attention_heads - num_energy_heads
        self.causal = causal
        self.layer_idx = layer_idx
        self.sliding_window = sliding_window
        self.position_embedding_type = position_embedding_type
        self.attention_multiplier = attention_multiplier
        self.use_padding_free_transformer = use_padding_free_transformer
        self.softmax_dropout_p = softmax_dropout

        self.head_dim = divide_if_divisible(
            hidden_size, num_attention_heads,
            f"hidden_size ({hidden_size}) must be divisible by num_attention_heads ({num_attention_heads})",
        )

        std = initializer_range
        if init_method == "mup":
            std /= math.sqrt(m_width)

        # Energy heads: project to Q and K only (no W_V; V = K)
        self.c_attn_energy = ParameterizedLinear(
            hidden_size, 2 * self.num_energy_heads * self.head_dim,
            bias=qkv_bias, std=initializer_range,
        )
        mark_parameter_as_mup_learning_rate(self.c_attn_energy.weight)

        # GPT heads: project to Q, K, V independently
        self.c_attn_gpt = ParameterizedLinear(
            hidden_size, 3 * self.num_gpt_heads * self.head_dim,
            bias=qkv_bias, std=initializer_range,
        )
        mark_parameter_as_mup_learning_rate(self.c_attn_gpt.weight)

        # Shared output projection for all heads
        self.W_O = ParameterizedLinear(
            hidden_size, hidden_size, bias=add_bias, std=std,
        )
        mark_parameter_as_mup_learning_rate(self.W_O.weight)

        self.softmax_dropout = Dropout(softmax_dropout)
        self.dropout = Dropout(dropout)

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
        use_fa2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_fa3 = is_kernel_allowed(Kernel.flash_attention_3)

        batch_size, query_length = hidden_states.shape[:2]
        ne, ng, dh = self.num_energy_heads, self.num_gpt_heads, self.head_dim

        # ── Energy heads: Q_e, K_e ──────────────────────────────────────────
        qk_e = self.c_attn_energy(hidden_states)                  # (B,T, 2*ne*dh)
        qk_e = qk_e.view(batch_size, query_length, 2, ne, dh)
        Q_e, K_e = qk_e.unbind(2)                                  # (B,T,ne,dh) each
        Q_e = Q_e.transpose(1, 2).contiguous()                     # (B,ne,T,dh)
        K_e = K_e.transpose(1, 2).contiguous()

        # ── GPT heads: Q_g, K_g, V_g ────────────────────────────────────────
        qkv_g = self.c_attn_gpt(hidden_states)                    # (B,T, 3*ng*dh)
        qkv_g = qkv_g.view(batch_size, query_length, 3, ng, dh)
        Q_g, K_g, V_g = qkv_g.unbind(2)                           # (B,T,ng,dh) each
        Q_g = Q_g.transpose(1, 2).contiguous()                     # (B,ng,T,dh)
        K_g = K_g.transpose(1, 2).contiguous()
        V_g = V_g.transpose(1, 2).contiguous()

        # ── RoPE ────────────────────────────────────────────────────────────
        if self.position_embedding_type == "rope" and rope_cos_sin is not None:
            Q_e = apply_rotary_pos_emb(Q_e, rope_cos_sin)
            K_e = apply_rotary_pos_emb(K_e, rope_cos_sin)
            Q_g = apply_rotary_pos_emb(Q_g, rope_cos_sin)
            K_g = apply_rotary_pos_emb(K_g, rope_cos_sin)

        # ── Build combined Q, K, V for a single attention call ───────────────
        # V for energy heads = K_e (nearest-neighbor retrieval)
        Q = torch.cat([Q_e, Q_g], dim=1)   # (B, num_heads, T, dh)
        K = torch.cat([K_e, K_g], dim=1)
        V = torch.cat([K_e, V_g], dim=1)   # energy heads: V=K; GPT heads: V=W_V*x

        if past_key_values is not None:
            cache_idx = layer_id if layer_id is not None else self.layer_idx
            K, V = past_key_values.update(key_states=K, value_states=V, layer_idx=cache_idx)

        if use_fa2 or use_fa3:
            # flash_attention expects (B, T, H, dh)
            Q = Q.transpose(1, 2).contiguous()
            K = K.transpose(1, 2).contiguous()
            V = V.transpose(1, 2).contiguous()

            attn_out = flash_attention(
                query=Q, key=K, value=V,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                sliding_window=self.sliding_window,
            )
            # attn_out: (B, T, num_heads, dh) → (B, T, H*dh)
            attn_out = attn_out.reshape(batch_size, query_length, self.hidden_size)
        else:
            attn_out = F.scaled_dot_product_attention(
                Q, K, V,
                attn_mask=attention_mask,
                dropout_p=self.softmax_dropout_p if self.training else 0,
                is_causal=self.causal if attention_mask is None else False,
            )  # (B, num_heads, T, dh)
            attn_out = attn_out.transpose(1, 2).contiguous()
            attn_out = attn_out.reshape(batch_size, query_length, self.hidden_size)

        out = self.W_O(attn_out)
        out = self.dropout(out)
        return out


class EnergyGradMixedHeadAttention(MixedHeadAttention):
    """Mixed-head attention where energy heads use the true energy gradient output.

    Energy heads: output = W_Q_e^T @ (softmax(Q_e K_e^T) @ K_e)  ← energy gradient
    GPT heads:    output = W_O_gpt(softmax(Q_g K_g^T) @ V_g)     ← standard
    Final: energy_grad_out + gpt_out  (both projected to hidden_size)

    Unlike V10 (which uses V=K with a shared W_O), energy heads here apply W_Q^T
    as the output projection (reusing the query weight — no new parameters from attn).
    GPT heads get their own W_O_gpt (ng*dh → hidden_size).

    Net: slightly fewer params than V10 due to W_O_gpt being smaller than full W_O.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        ne, ng, dh = self.num_energy_heads, self.num_gpt_heads, self.head_dim
        # Replace shared W_O with a GPT-only output projection
        del self.W_O
        self.W_O_gpt = ParameterizedLinear(
            ng * dh, self.hidden_size,
            bias=False,
            std=kwargs.get("initializer_range", 0.02),
        )
        mark_parameter_as_mup_learning_rate(self.W_O_gpt.weight)

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
        batch_size, query_length = hidden_states.shape[:2]
        ne, ng, dh = self.num_energy_heads, self.num_gpt_heads, self.head_dim

        # ── Energy heads: Q_e, K_e ──────────────────────────────────────────
        qk_e = self.c_attn_energy(hidden_states)
        qk_e = qk_e.view(batch_size, query_length, 2, ne, dh)
        Q_e, K_e = qk_e.unbind(2)
        Q_e = Q_e.transpose(1, 2).contiguous()
        K_e = K_e.transpose(1, 2).contiguous()

        # ── GPT heads: Q_g, K_g, V_g ────────────────────────────────────────
        qkv_g = self.c_attn_gpt(hidden_states)
        qkv_g = qkv_g.view(batch_size, query_length, 3, ng, dh)
        Q_g, K_g, V_g = qkv_g.unbind(2)
        Q_g = Q_g.transpose(1, 2).contiguous()
        K_g = K_g.transpose(1, 2).contiguous()
        V_g = V_g.transpose(1, 2).contiguous()

        # ── RoPE ────────────────────────────────────────────────────────────
        if self.position_embedding_type == "rope" and rope_cos_sin is not None:
            Q_e = apply_rotary_pos_emb(Q_e, rope_cos_sin)
            K_e = apply_rotary_pos_emb(K_e, rope_cos_sin)
            Q_g = apply_rotary_pos_emb(Q_g, rope_cos_sin)
            K_g = apply_rotary_pos_emb(K_g, rope_cos_sin)

        # ── Energy heads: true energy gradient output ────────────────────────
        # Use SDPA (flash_attention doesn't return weights; we need the post-V output)
        # attn_e: (B, ne, T, dh) — softmax(Q_e K_e^T) @ K_e
        scale = self.attention_multiplier if self.attention_multiplier is not None else (dh ** -0.5)
        attn_e = F.scaled_dot_product_attention(
            Q_e, K_e, K_e,
            dropout_p=self.softmax_dropout_p if self.training else 0,
            is_causal=self.causal,
            scale=scale,
        )  # (B, ne, T, dh)

        # W_Q^T projection: c_attn_energy.weight[:ne*dh] shape (ne*dh, D)
        # → view (ne, dh, D) → permute (ne, D, dh) for einsum "bhts,hcs->btc"
        W_Q_e = (
            self.c_attn_energy.weight[: ne * dh]
            .view(ne, dh, self.hidden_size)
            .permute(0, 2, 1)
            .contiguous()
        )  # (ne, D, dh)
        energy_grad_out = torch.einsum("bhtd,hcd->btc", attn_e, W_Q_e)  # (B, T, D)

        # ── GPT heads: standard flash_attention ─────────────────────────────
        use_fa2 = is_kernel_allowed(Kernel.flash_attention_2)
        use_fa3 = is_kernel_allowed(Kernel.flash_attention_3)

        if past_key_values is not None:
            cache_idx = layer_id if layer_id is not None else self.layer_idx
            K_g, V_g = past_key_values.update(key_states=K_g, value_states=V_g, layer_idx=cache_idx)

        if use_fa2 or use_fa3:
            Q_g = Q_g.transpose(1, 2).contiguous()
            K_g = K_g.transpose(1, 2).contiguous()
            V_g = V_g.transpose(1, 2).contiguous()
            attn_g = flash_attention(
                query=Q_g, key=K_g, value=V_g,
                cu_seqlens=cu_seqlens, max_seqlen=max_seqlen,
                attention_mask=attention_mask,
                use_padding_free_transformer=self.use_padding_free_transformer,
                causal=self.causal,
                dropout=self.softmax_dropout_p if self.training else 0,
                sliding_window=self.sliding_window,
            )
            attn_g = attn_g.reshape(batch_size, query_length, ng * dh)
        else:
            attn_g = F.scaled_dot_product_attention(
                Q_g, K_g, V_g,
                attn_mask=attention_mask,
                dropout_p=self.softmax_dropout_p if self.training else 0,
                is_causal=self.causal if attention_mask is None else False,
                scale=scale,
            )
            attn_g = attn_g.transpose(1, 2).contiguous().reshape(batch_size, query_length, ng * dh)

        gpt_out = self.W_O_gpt(attn_g)  # (B, T, D)

        out = self.dropout(energy_grad_out + gpt_out)
        return out
