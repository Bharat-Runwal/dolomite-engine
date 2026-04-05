# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import torch
import torch.nn as nn

from ...cache import GenerationCache
from ...modeling_utils import get_mlp_block, get_normalization_function, get_sequence_mixer
from .config import EnergyConfig


class PositiveScalarProjection(nn.Module):
    """Projection that guarantees descent: proj(x) = alpha^2 * x."""

    def __init__(self, init_value: float = 1.0):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1) * init_value)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.weight ** 2) * x


class AntisymmetricProjection(nn.Module):
    """V1: Pure rotation via explicit antisymmetric matrix J = A - A^T.

    Update rule: h := h + J @ grad_E
    By construction, J is antisymmetric, preserving norm on the RMSNorm hypersphere.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        # Parametrize via unrestricted A; J = A - A^T is antisymmetric by design
        self.A = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        J = self.A - self.A.T
        return torch.matmul(x, J.T)  # x @ J^T = (J @ x^T)^T, efficient for batch

    def get_J(self) -> torch.Tensor:
        """Return the current antisymmetric matrix."""
        return self.A - self.A.T


class PortHamiltonianProjection(nn.Module):
    """V2: Rotation + dissipation via J - R where J is antisymmetric, R = LL^T is PSD.

    Update rule: h := h + (J - R) @ grad_E
    Learns how much dissipation is beneficial. If R → 0, rotation is optimal.
    """

    def __init__(self, hidden_size: int):
        super().__init__()
        self.A = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)
        # Parametrize R = L @ L^T via Cholesky factor L
        self.L = nn.Parameter(torch.randn(hidden_size, hidden_size) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        J = self.A - self.A.T
        R = torch.matmul(self.L, self.L.T)  # PSD by construction
        K = J - R
        return torch.matmul(x, K.T)

    def get_J(self) -> torch.Tensor:
        return self.A - self.A.T

    def get_R(self) -> torch.Tensor:
        return torch.matmul(self.L, self.L.T)


class LowRankAntisymmetricProjection(nn.Module):
    """V3: Low-rank explicit rotation J = U V^T - V U^T with rank k << hidden_size.

    Parameter count: 2 * hidden_size * rank instead of hidden_size^2.
    """

    def __init__(self, hidden_size: int, rank: int = 32):
        super().__init__()
        self.rank = rank
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # J = U V^T - V U^T
        J = torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)
        return torch.matmul(x, J.T)

    def get_J(self) -> torch.Tensor:
        return torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)


class LowRankPortHamiltonianProjection(nn.Module):
    """V5: Low-rank rotation + low-rank dissipation.

    J = U V^T - V U^T  (rank-k antisymmetric, rotation)
    R = L L^T           (rank-r PSD, dissipation)
    Update: h := h + (J - R) @ grad_E

    Combines V3's convergence properties with V2's learnable dissipation,
    at a fraction of the parameter cost:
      2*d*k + d*r  vs  2*d^2  (full Port-Hamiltonian)
    For d=768, k=32, r=16:  61k vs 1.18M params per block.
    """

    def __init__(self, hidden_size: int, rank: int = 32, dissipation_rank: int = 16):
        super().__init__()
        self.rank = rank
        self.dissipation_rank = dissipation_rank
        # Rotation: J = U V^T - V U^T
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 0.01)
        # Dissipation: R = L L^T (low-rank PSD)
        self.L = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * 0.01)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        J = torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)
        R = torch.matmul(self.L, self.L.T)
        K = J - R
        return torch.matmul(x, K.T)

    def get_J(self) -> torch.Tensor:
        return torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)

    def get_R(self) -> torch.Tensor:
        return torch.matmul(self.L, self.L.T)



class EnergyBlock(nn.Module):
    """Energy Transformer block with customizable attention and feedforward.

    Unlike standard Transformer blocks that use additive residual connections,
    EnergyBlock uses a subtractive update inspired by energy-based models:
        x = x - proj(attn(ln(x)) + scale_ff * ffwd(ln(x)))
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
            # Use energy-specific norm if configured, otherwise fall back to global
            norm_type = getattr(config, 'energy_norm_type', None) or config.normalization_function
            self.ln = get_normalization_function(
                norm_type, hidden_size, eps=config.layer_norm_epsilon
            )
            self.attn = get_sequence_mixer(config, True, use_padding_free_transformer, layer_idx)

            self.ffwd = get_mlp_block(
                config, use_padding_free_transformer=use_padding_free_transformer, layer_idx=layer_idx
            )

            # scale_ff init: per-block list or scalar (default 4.0)
            scale_ff_init_cfg = getattr(config, 'scale_ff_init', None)
            per_block_val = None
            if isinstance(scale_ff_init_cfg, list) and layer_idx < len(scale_ff_init_cfg):
                per_block_val = scale_ff_init_cfg[layer_idx]
            elif isinstance(scale_ff_init_cfg, (int, float)):
                per_block_val = scale_ff_init_cfg
            if per_block_val is not None:
                init_val = float(per_block_val)
            else:
                init_val = 4.0
            self.scale_ff = nn.Parameter(torch.ones(1) * init_val, requires_grad=True)

            # Projection type: controls energy descent guarantee
            proj_type = getattr(config, 'energy_proj_type', 'unconstrained')
            if proj_type == "unconstrained":
                self.proj = nn.Linear(hidden_size, hidden_size, bias=False)
            elif proj_type == "pos_scalar":
                self.proj = PositiveScalarProjection()
            elif proj_type == "identity":
                self.proj = nn.Identity()
            elif proj_type == "antisymmetric":
                # V1: Pure rotation via explicit J = A - A^T
                self.proj = AntisymmetricProjection(hidden_size)
            elif proj_type == "port_hamiltonian":
                # V2: Rotation + dissipation via (J - R)
                self.proj = PortHamiltonianProjection(hidden_size)
            elif proj_type == "low_rank_antisymmetric":
                # V3: Low-rank antisymmetric J = U V^T - V U^T
                rank = getattr(config, 'energy_proj_rank', 32)
                self.proj = LowRankAntisymmetricProjection(hidden_size, rank=rank)
            elif proj_type == "low_rank_port_hamiltonian":
                # V5: Low-rank rotation + low-rank dissipation
                rank = getattr(config, 'energy_proj_rank', 32)
                dissipation_rank = getattr(config, 'energy_dissipation_rank', 16)
                self.proj = LowRankPortHamiltonianProjection(hidden_size, rank=rank, dissipation_rank=dissipation_rank)
            elif proj_type == "dual_unconstrained":
                # Dual projection: separate unconstrained matrices for attn and MLP
                # h := h - proj_attn(attn_out) - proj_mlp(ffwd_out)
                # Motivation: attn gradient has causal bias, MLP gradient is exact
                # scale_ff is redundant (proj_mlp absorbs scaling), so freeze it at 1.0
                self.proj_attn = nn.Linear(hidden_size, hidden_size, bias=False)
                self.proj_mlp = nn.Linear(hidden_size, hidden_size, bias=False)
                self.proj = None  # not used; forward handles separately
                self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=False)
            else:
                raise ValueError(f"Unknown energy_proj_type: {proj_type}")

            # Store proj_type for reference
            self.proj_type = proj_type
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

            # Dual projection: separate matrices for attn and MLP (scale_ff frozen at 1.0)
            if self.proj_type == "dual_unconstrained":
                hidden_states = hidden_states - self.proj_attn(attn_out) - self.proj_mlp(ffwd_out)
            # Rotation projections (antisymmetric, etc.): h := h + proj(grad_E)
            elif self.proj_type in ["antisymmetric", "port_hamiltonian", "low_rank_antisymmetric", "low_rank_port_hamiltonian"]:
                grad_E = attn_out + self.scale_ff * ffwd_out
                hidden_states = hidden_states + self.proj(grad_E)
            # Descent projections (unconstrained, pos_scalar, identity): h := h - proj(grad_E)
            else:
                grad_E = attn_out + self.scale_ff * ffwd_out
                hidden_states = hidden_states - self.proj(grad_E)
            return hidden_states
        else:
            return self.forward_gpt(hidden_states,past_key_values,attention_mask,rope_cos_sin,cu_seqlens,max_seqlen,layer_id=layer_id)


    def energy_per_token(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        """Compute total energy per token: E = E_attn + scale_ff * E_ff."""
        ln_x = self.ln(x)
        return self.attn.energy_per_token(ln_x, rope_cos_sin=rope_cos_sin) + self.scale_ff * self.ffwd.energy_per_token(ln_x)

    def forward_gradient(self, x: torch.Tensor, rope_cos_sin=None) -> torch.Tensor:
        """Return the pre-projection gradient: attn(ln_x) + scale_ff * ffwd(ln_x).

        This is exactly what the forward pass computes and treats as the gradient,
        using the causal partial gradient (query-role only) — no autograd key-role bias.
        Use this instead of autograd(energy_per_token) for causal-correct Helmholtz analysis.
        """
        with torch.no_grad():
            ln_x = self.ln(x)
            attn_out = self.attn(ln_x, past_key_values=None, attention_mask=None,
                                 rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None)
            ffwd_out = self.ffwd(ln_x)
        return attn_out + self.scale_ff * ffwd_out

    def get_projection_diagnostics(self, grad_E: torch.Tensor) -> dict:
        """Compute geometry diagnostics for the projection.

        Returns:
            dict with keys:
            - 'proj_update': the projected update J @ grad_E
            - 'cos_theta': cosine similarity between update and gradient (should be ~0 for rotation)
            - 'antisymmetry_norm': ||J - J^T|| / ||J|| (should be ~0 for antisymmetric)
            - 'norm_change': ||update|| / ||grad_E|| (change in magnitude per iteration)
        """
        with torch.no_grad():
            proj_update = self.proj(grad_E)

            # Flatten for easier computation
            grad_flat = grad_E.reshape(-1, grad_E.shape[-1])  # [T, D]
            update_flat = proj_update.reshape(-1, proj_update.shape[-1])

            # cos θ: should be ≈ 0 for pure rotation
            cos_theta = torch.nn.functional.cosine_similarity(
                grad_flat, update_flat, dim=-1
            ).mean().item()

            # Antisymmetry: ||J - J^T|| / ||J||
            antisymmetry_norm = 0.0
            if hasattr(self.proj, 'get_J'):
                J = self.proj.get_J()
                asymmetry = torch.norm(J - J.T) / (torch.norm(J) + 1e-8)
                antisymmetry_norm = asymmetry.item()

            # Norm change
            grad_norm = torch.norm(grad_flat)
            update_norm = torch.norm(update_flat)
            norm_change = (update_norm / (grad_norm + 1e-8)).item()

            return {
                'proj_update': proj_update,
                'cos_theta': cos_theta,
                'antisymmetry_norm': antisymmetry_norm,
                'norm_change': norm_change,
            }



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
        if self.sequence_mixer_type in ["softmax_attention", "multihead_latent_attention"]:
            hidden_states = self.sequence_mixer(
                hidden_states,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                rope_cos_sin=rope_cos_sin,
                cu_seqlens=cu_seqlens,
                max_seqlen=max_seqlen,
                layer_id=layer_id,
            )
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


