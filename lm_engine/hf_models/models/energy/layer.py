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



class DualLowRankPortHamiltonianProjection(nn.Module):
    """Split port-Hamiltonian projection: separate J-R matrices for attn and FFN.

    For each stream (attn, ff):
        J = U V^T - V U^T   (antisymmetric / curl / rotation,   U,V: d x r_j)
        R = L L^T            (PSD / symmetric / dissipation,     L:   d x r_d)

    Update: h := h + ((1-α)*J_attn - α*R_attn) @ attn_out
                   + ((1-α)*J_ff   - α*R_ff  ) @ ffwd_out

    where α ∈ [0,1]: α=0 → pure rotation (J only), α=1 → pure descent (R only).
    Default (α=0.5) recovers the original equal mix.

    If learnable_alpha=True, α is a per-stream learned scalar (sigmoid-parametrised),
    initialised at 0.5 so training starts from the balanced mix.

    Rationale for splitting: the attention gradient has a causal asymmetry that
    the FFN gradient lacks; giving each its own J-R pair lets them learn different
    curl/descent balances.  scale_ff is frozen at 1.0 (absorbed by the ff matrices).

    Parameter count per block: 2 * (2*d*r_j + d*r_d)
      e.g. d=1280, r_j=32, r_d=16 → 2*(2*1280*32 + 1280*16) = 204,800
      vs unconstrained d^2 = 1,638,400  (~8x fewer)
    """

    def __init__(
        self,
        hidden_size: int,
        rank: int = 32,
        dissipation_rank: int = 16,
        learnable_alpha: bool = False,
        base_weight: torch.Tensor | None = None,
        base_scale_ff: float = 1.0,
        single_stream: bool = False,
        num_iters: int = 1,
    ):
        """
        Args:
            base_weight: if provided, stores a frozen copy of the trained proj weight W.
                In residual mode, the forward computes:
                    delta = -W @ (attn_out + base_scale_ff * ffwd_out)   [frozen base]
                          + structured correction                          [trained]
                This guarantees the initial output equals the trained model.
            base_scale_ff: scale_ff value used in the trained model (for the base pass).
            single_stream: if True, applies ONE K to the combined
                (attn_out + base_scale_ff * ffwd_out) input — mirrors the original
                single-W architecture but with J-R structure.  Only U/V/L_attn
                params are created; U/V/L_ff are absent.
        """
        super().__init__()
        self.rank = rank
        self.dissipation_rank = dissipation_rank
        self.learnable_alpha = learnable_alpha
        self.single_stream = single_stream
        self.num_iters = num_iters
        std = 1e-3  # tiny init so residual mode starts at ~0 correction
        # Attention stream (also serves as the single stream when single_stream=True)
        self.U_attn = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.V_attn = nn.Parameter(torch.randn(hidden_size, rank) * std)
        self.L_attn = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * std)
        # FFN stream (only created for dual-stream mode)
        if not single_stream:
            self.U_ff = nn.Parameter(torch.randn(hidden_size, rank) * std)
            self.V_ff = nn.Parameter(torch.randn(hidden_size, rank) * std)
            self.L_ff = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * std)
        # Learnable alpha: sigmoid(0) = 0.5 → balanced mix at init.
        # If num_iters > 1, one alpha per iteration (shape (num_iters,));
        # otherwise a single scalar (shape (1,)) — backwards compatible.
        if learnable_alpha:
            self.log_alpha_attn = nn.Parameter(torch.zeros(num_iters))
            if not single_stream:
                self.log_alpha_ff = nn.Parameter(torch.zeros(num_iters))
        # Residual mode: frozen base contribution from trained W
        if base_weight is not None:
            self.register_buffer('base_weight', base_weight.detach().clone())
            self.base_scale_ff = base_scale_ff
        else:
            self.base_weight = None
            self.base_scale_ff = base_scale_ff

    def _J(self, U: torch.Tensor, V: torch.Tensor) -> torch.Tensor:
        """Antisymmetric rotation matrix: J = UV^T - VU^T."""
        return torch.matmul(U, V.T) - torch.matmul(V, U.T)

    def _R(self, L: torch.Tensor) -> torch.Tensor:
        """PSD dissipation matrix: R = LL^T."""
        return torch.matmul(L, L.T)

    def _K(self, U, V, L, alpha: float = 0.5):
        """Compute K = (1-α)*J - α*R.  α=0.5 → equal mix (default)."""
        return (1.0 - alpha) * self._J(U, V) - alpha * self._R(L)

    def forward(self, attn_out: torch.Tensor, ffwd_out: torch.Tensor,
                iter_idx: int = 0) -> torch.Tensor:
        """Compute the structured proj update.

        Args:
            attn_out: attention stream output  (B, T, d)
            ffwd_out: FFN stream output        (B, T, d)
            iter_idx: which recurrence iteration we are in (0-indexed).
                      Used to select the per-iteration alpha when num_iters > 1.
                      Ignored when num_iters == 1 (always uses index 0).
        """
        combined = attn_out + self.base_scale_ff * ffwd_out
        # Clamp iter_idx so we never index out of bounds (e.g. during eval with
        # effective_iter < num_iters due to iter dropout).
        t = min(iter_idx, self.num_iters - 1)

        if self.single_stream:
            # One K applied to combined input — mirrors original single-W architecture.
            if self.learnable_alpha:
                alpha = torch.sigmoid(self.log_alpha_attn[t])
                K = (1.0 - alpha) * self._J(self.U_attn, self.V_attn) \
                    - alpha * self._R(self.L_attn)
            else:
                K = self._K(self.U_attn, self.V_attn, self.L_attn)
            struct_delta = torch.matmul(combined, K.T)
        else:
            # Dual stream: independent K for attn and FFN.
            if self.learnable_alpha:
                alpha_attn = torch.sigmoid(self.log_alpha_attn[t])
                alpha_ff   = torch.sigmoid(self.log_alpha_ff[t])
                K_attn = (1.0 - alpha_attn) * self._J(self.U_attn, self.V_attn) \
                         - alpha_attn * self._R(self.L_attn)
                K_ff   = (1.0 - alpha_ff)   * self._J(self.U_ff,   self.V_ff) \
                         - alpha_ff   * self._R(self.L_ff)
            else:
                K_attn = self._K(self.U_attn, self.V_attn, self.L_attn)
                K_ff   = self._K(self.U_ff,   self.V_ff,   self.L_ff)
            struct_delta = torch.matmul(attn_out, K_attn.T) + torch.matmul(ffwd_out, K_ff.T)

        if self.base_weight is not None:
            # Residual mode: subtract frozen trained base, add structured correction.
            # At init (J≈0, R≈0): struct_delta ≈ 0 → output = trained model.
            base_delta = torch.matmul(combined, self.base_weight.T)
            return -base_delta + struct_delta

        return struct_delta

    def get_alpha_attn(self, iter_idx: int | None = None):
        """Return learned alpha for the attention stream.

        If num_iters == 1 (or iter_idx is None and num_iters > 1), returns a
        float (single-iter case) or a list of floats (multi-iter case).
        If iter_idx is given, returns the scalar alpha for that iteration.
        """
        if self.learnable_alpha:
            vals = torch.sigmoid(self.log_alpha_attn).detach()
            if iter_idx is not None:
                return vals[min(iter_idx, self.num_iters - 1)].item()
            if self.num_iters == 1:
                return vals[0].item()
            return vals.tolist()
        return 0.5

    def get_alpha_ff(self, iter_idx: int | None = None):
        if self.learnable_alpha and not self.single_stream:
            vals = torch.sigmoid(self.log_alpha_ff).detach()
            if iter_idx is not None:
                return vals[min(iter_idx, self.num_iters - 1)].item()
            if self.num_iters == 1:
                return vals[0].item()
            return vals.tolist()
        return self.get_alpha_attn(iter_idx)  # single_stream: same alpha for both

    def get_J_attn(self):
        return self._J(self.U_attn, self.V_attn)

    def get_R_attn(self):
        return self._R(self.L_attn)

    def get_J_ff(self):
        if self.single_stream:
            return self.get_J_attn()
        return self._J(self.U_ff, self.V_ff)

    def get_R_ff(self):
        if self.single_stream:
            return self.get_R_attn()
        return self._R(self.L_ff)


class HelmholtzFactoredProjection(nn.Module):
    """V6: dx = -M @ (attn + A @ ffn), M=LL^T PSD, A=UV^T-VU^T antisym.

    Shared PSD preconditioner M sets the time-scale/learning-rate; antisymmetric A
    rotates the FFN contribution before combining. User's formulation:
        dx = -proj_psd(attn + proj_anti(ffn))
    """

    def __init__(self, hidden_size: int, rank: int = 32, dissipation_rank: int = 16):
        super().__init__()
        self.L = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * 1e-3)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)

    def forward(self, attn_out: torch.Tensor, ffwd_out: torch.Tensor) -> torch.Tensor:
        M = torch.matmul(self.L, self.L.T)
        A = torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)
        combined = attn_out + torch.matmul(ffwd_out, A.T)
        return torch.matmul(combined, M.T)


class HelmholtzDualProjection(nn.Module):
    """V7: dx = -(R @ attn + A @ ffn), R=LL^T PSD, A=UV^T-VU^T antisym.

    Clean Helmholtz split: attn is the gradient component (descent via PSD R),
    FFN is the curl component (rotation via antisymmetric A).
    Hypothesis: attn output already approximates -grad_E from causal attention;
    FFN output is the non-gradient (curl) part of the vector field.
    """

    def __init__(self, hidden_size: int, rank: int = 32, dissipation_rank: int = 16):
        super().__init__()
        self.L = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * 1e-3)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)

    def forward(self, attn_out: torch.Tensor, ffwd_out: torch.Tensor) -> torch.Tensor:
        R = torch.matmul(self.L, self.L.T)
        A = torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)
        return torch.matmul(attn_out, R.T) + torch.matmul(ffwd_out, A.T)


class HelmholtzDualReversedProjection(nn.Module):
    """V8: dx = -(A @ attn + R @ ffn) — reversed from V7 (control experiment).

    If V7 outperforms V8, it supports the Helmholtz interpretation:
    attn = gradient, FFN = curl. If both perform similarly, the assignment is arbitrary.
    """

    def __init__(self, hidden_size: int, rank: int = 32, dissipation_rank: int = 16):
        super().__init__()
        self.L = nn.Parameter(torch.randn(hidden_size, dissipation_rank) * 1e-3)
        self.U = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)
        self.V = nn.Parameter(torch.randn(hidden_size, rank) * 1e-3)

    def forward(self, attn_out: torch.Tensor, ffwd_out: torch.Tensor) -> torch.Tensor:
        R = torch.matmul(self.L, self.L.T)
        A = torch.matmul(self.U, self.V.T) - torch.matmul(self.V, self.U.T)
        return torch.matmul(attn_out, A.T) + torch.matmul(ffwd_out, R.T)


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
        if self.sequence_mixer_type in ("energy_attention", "mixed_head_energy_descent"):
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
            elif proj_type == "dual_low_rank_port_hamiltonian":
                # Split port-Hamiltonian: separate low-rank J-R for attn and FFN streams.
                # h := h + (J_attn - R_attn) @ attn_out + (J_ff - R_ff) @ ffwd_out
                # scale_ff absorbed into ff matrices; freeze at 1.0.
                rank = getattr(config, 'energy_proj_rank', 32)
                dissipation_rank = getattr(config, 'energy_dissipation_rank', 16)
                self.proj = DualLowRankPortHamiltonianProjection(
                    hidden_size, rank=rank, dissipation_rank=dissipation_rank
                )
                self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=False)
            elif proj_type == "attn_only_energy":
                # V5: standard additive residual for FFN, energy-projected subtraction for attn.
                # h := h + ffwd_out - proj_attn(attn_out)
                self.proj_attn = nn.Linear(hidden_size, hidden_size, bias=False)
                self.proj = None
                self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=False)
            elif proj_type == "helmholtz_factored":
                # V6: h := h - M(attn + A*ffn), M=LL^T PSD, A=antisym
                rank = getattr(config, 'energy_proj_rank', 32)
                dissipation_rank = getattr(config, 'energy_dissipation_rank', 16)
                self.proj = HelmholtzFactoredProjection(hidden_size, rank=rank, dissipation_rank=dissipation_rank)
                self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=False)
            elif proj_type == "helmholtz_dual":
                # V7: h := h - (R*attn + A*ffn), R=PSD, A=antisym — clean Helmholtz split
                rank = getattr(config, 'energy_proj_rank', 32)
                dissipation_rank = getattr(config, 'energy_dissipation_rank', 16)
                self.proj = HelmholtzDualProjection(hidden_size, rank=rank, dissipation_rank=dissipation_rank)
                self.scale_ff = nn.Parameter(torch.ones(1), requires_grad=False)
            elif proj_type == "helmholtz_dual_reversed":
                # V8: h := h - (A*attn + R*ffn) — reversed from V7, control experiment
                rank = getattr(config, 'energy_proj_rank', 32)
                dissipation_rank = getattr(config, 'energy_dissipation_rank', 16)
                self.proj = HelmholtzDualReversedProjection(hidden_size, rank=rank, dissipation_rank=dissipation_rank)
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
        iter_idx: int = 0,
    ) -> torch.Tensor:

        if self.sequence_mixer_type in ("energy_attention", "mixed_head_energy_descent"):

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
            # Store for cosine-similarity regularizer (activation-based, avoids FSDP weight gather)
            self._attn_out = attn_out
            ffwd_out = self.ffwd(ln_x)

            # Dual projection: separate matrices for attn and MLP (scale_ff frozen at 1.0)
            if self.proj_type == "dual_unconstrained":
                hidden_states = hidden_states - self.proj_attn(attn_out) - self.proj_mlp(ffwd_out)
            # Split port-Hamiltonian: h := h + proj(attn_out, ffwd_out)
            elif self.proj_type == "dual_low_rank_port_hamiltonian":
                hidden_states = hidden_states + self.proj(attn_out, ffwd_out, iter_idx=iter_idx)
            # V5: standard residual for FFN, projected subtraction for attn
            elif self.proj_type == "attn_only_energy":
                hidden_states = hidden_states + ffwd_out - self.proj_attn(attn_out)
            # V6/V7/V8 Helmholtz variants: h := h - proj(attn_out, ffwd_out)
            elif self.proj_type in ["helmholtz_factored", "helmholtz_dual", "helmholtz_dual_reversed"]:
                hidden_states = hidden_states - self.proj(attn_out, ffwd_out)
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
        if self.sequence_mixer_type in ["softmax_attention", "multihead_latent_attention", "mixed_head_attention", "energy_grad_mixed_head_attention"]:
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


