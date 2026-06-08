# **************************************************
# Boltzmann-weighted Mixture-of-Experts Energy FFN.
# VENDORED VERBATIM from Nima Dehmamy's dolomite-engine fork
#   (branch main_merge, lm_engine/hf_models/modeling_utils/mlp_blocks/mlp.py,
#    class BoltzmannMoE_Energy_MLP), fetched 2026-06.
#
# Key difference vs. the earlier port: the routing energy is scaled by
# 1/sqrt(expert_I) (self._routing_scale) BEFORE the Boltzmann softmax — the
# "1/sqrt(d)" stabilization Nima reported as fixing routing spikes / instabilities.
# This is applied in CODE; configs keep temperature: 1.0.
# **************************************************

from __future__ import annotations

import itertools
import math
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss import add_aux_loss
from ...parameter import mark_parameter_as_mup_learning_rate
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from .mlp import _get_std_for_linear


class BoltzmannMoE_Energy_MLP(nn.Module):
    """Boltzmann-weighted Mixture-of-Experts Energy FFN.

    E_moe(h) = log( Σᵢ exp(Eᵢ(h)) )   where Eᵢ(h) = −φ(W1ᵢh)ᵀ(W2ᵢh)
    ∂E_moe/∂h = Σᵢ pᵢ(h) · ∂Eᵢ/∂h   where pᵢ(h) = softmax_i(E(h) / τ)

    Iso-parameter with Energy_MLP of the same intermediate_size: n_experts experts
    each with (intermediate_size // n_experts) hidden units, same total FLOPs and
    parameters as one large Energy_MLP.

    Stochastic contrastive repulsion (optional): at each training step, n_repulsion_pairs
    random expert pairs (i, j) are sampled and their output cosine similarity is penalized
    via add_aux_loss.  This is functional repulsion — it pushes experts apart based on
    their actual outputs on the current batch, not on weight geometry.
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        temperature: float,
        repulsion_coef: float,
        n_repulsion_pairs: int,
        top_k: int | None = None,
        gelu_grad_method: str = "sigmoid",
        init_method: str = "normal",
        activation_function: str = "gelu",
        dropout: float = 0.0,
        initializer_range: float = 0.02,
        m_width: float | None = None,
        num_layers: int = 1,
        add_bias: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        assert intermediate_size % n_experts == 0, (
            f"intermediate_size ({intermediate_size}) must be divisible by n_experts ({n_experts})"
        )
        assert gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact"), (
            f"gelu_grad_method must be one of 'sigmoid' / 'tanh_exact' / 'erf_exact', "
            f"got {gelu_grad_method}"
        )

        # top_k == 0 is treated as "dense" (None) for backward-compat with configs
        # that used the earlier port's `top_k: 0` convention.
        if top_k == 0:
            top_k = None

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        self._routing_scale = self.expert_I ** -0.5  # 1/sqrt(expert_I): prevents routing energy growth → spikes
        self.temperature = temperature
        self.top_k = top_k  # None=soft; int=sparse top-k Boltzmann routing
        self.repulsion_coef = repulsion_coef
        self.n_repulsion_pairs = n_repulsion_pairs
        # "sigmoid"    = legacy φ' = sigmoid(c·x)·0.5 (uniformly half the true GELU' magnitude
        #                — partly absorbed by W2 scale during training; matches all checkpoints
        #                trained before 2026-06)
        # "tanh_exact" = self-consistent φ/φ' pair: φ(x) = 0.5·x·(1+tanh(c·x)), and
        #                φ'(x) = 0.5·(1+tanh(c·x)) + 0.5·c·x·(1−tanh²(c·x))
        #                (c = √(2/π)). Strictly correct ∂E/∂h.
        self.gelu_grad_method = gelu_grad_method
        self.layer_idx = layer_idx
        self._cached_metrics: dict[str, float] | None = None

        # Precompute all expert pairs for stochastic repulsion sampling
        self._all_pairs: list[tuple[int, int]] = list(itertools.combinations(range(n_experts), 2))

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # Fused projections over all experts (same pattern as Compositional_Energy_MLP):
        # W1 maps hidden → n_experts * expert_I; chunking recovers per-expert slices.
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)

        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]  # (B, T) for standard or (BT,) for padding-free

        # W1 path: apply dropout to intermediate activations
        W1x = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.expert_I)

        # Expert weight blocks: (n_experts, expert_I, hidden_size)
        W1_e = self.W1.weight.view(self.n_experts, self.expert_I, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.expert_I, self.hidden_size)

        # Activation φ and its derivative φ'. Three branches:
        #
        #   sigmoid    : φ = F.gelu(W1x);  φ' ≈ sigmoid(c·W1x)·0.5
        #                LEGACY default. φ' is uniformly half of the true gelu'
        #                magnitude and is not a faithful derivative of F.gelu.
        #                Kept as default so all pre-2026-06 checkpoints reproduce
        #                bit-identically.
        #
        #   erf_exact  : φ = F.gelu(W1x);  φ' = analytic d/dx F.gelu(x)
        #                = 0.5·(1 + erf(x/√2)) + x·exp(−x²/2)/√(2π)
        #                Same φ shape as legacy (so the energy landscape is
        #                identical to sigmoid), but the gradient term2 is the
        #                true derivative — isolates the magnitude-of-φ' question
        #                from any φ-shape change.
        #
        #   tanh_exact : φ = 0.5·W1x·(1+tanh(c·W1x));
        #                φ' = 0.5·(1+tanh(c·W1x)) + 0.5·c·W1x·(1−tanh²(c·W1x))
        #                Self-consistent matched pair using tanh-approx GELU
        #                (φ shape ≠ F.gelu). Tested at h1 scale; lost −2.2pp avg.
        if self.gelu_grad_method == "erf_exact":
            phi = F.gelu(W1x)
            inv_sqrt_2 = 0.7071067811865476           # 1/√2
            inv_sqrt_2pi = 0.3989422804014327         # 1/√(2π)
            phi_prime = 0.5 * (1.0 + torch.erf(W1x * inv_sqrt_2)) \
                      + W1x * torch.exp(-0.5 * W1x * W1x) * inv_sqrt_2pi
        elif self.gelu_grad_method == "tanh_exact":
            t = torch.tanh(self._SIGMOID_SCALE * W1x)
            phi = 0.5 * W1x * (1.0 + t)                                  # tanh-approx GELU
            phi_prime = 0.5 * (1.0 + t) + 0.5 * self._SIGMOID_SCALE * W1x * (1.0 - t * t)
        else:  # "sigmoid" (legacy)
            phi = F.gelu(W1x)                                            # (..., n_experts, expert_I)
            phi_prime = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5

        # First gradient term: φ(W1ᵢh) @ W2ᵢᵀ   shape (..., n_experts, hidden)
        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)

        # Routing energy: Eᵢ(h) = h · term1ᵢ / sqrt(expert_I)
        # Scaled like attention (1/sqrt(d_k)) to prevent ||E_i|| from growing as
        # O(||W||² · expert_I) during training, which causes routing spikes.
        # Without this scaling, E_i grows unboundedly → softmax saturates → hard
        # routing shift → loss spikes (observed at steps ~1000-4000).
        E = torch.einsum("...h,...eh->...e", x, term1) * self._routing_scale  # (..., n_experts)

        # Boltzmann routing weights (optionally sparse top-k)
        if self.top_k is not None and self.top_k < self.n_experts:
            # Sparse: full Boltzmann softmax then TRUNCATE (zero non-top-k, no renorm).
            # Avoids abrupt weight redistribution at routing boundaries → fewer spikes.
            p = F.softmax(E / self.temperature, dim=-1)
            _, topk_idx = E.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p * mask                           # sum < 1 intentionally; sparse Boltzmann approx
        else:
            p = F.softmax(E / self.temperature, dim=-1)

        # Second gradient term needs W2(x); independent dropout for regularisation
        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.expert_I)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2                                    # (..., n_experts, hidden)

        # Boltzmann-weighted sum: ∂E_moe/∂h = Σᵢ pᵢ · (term1ᵢ + term2ᵢ)
        # For sparse top-k, non-selected experts have p=0 so contribute nothing.
        out = torch.einsum("...e,...eh->...h", p, expert_grads)         # (..., hidden)

        if self.training and self.repulsion_coef > 0:
            self._add_repulsion_loss(expert_grads)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        """Penalise cosine similarity between random expert output pairs.

        Stochastic: only n_repulsion_pairs random pairs per step — cheap (K dot-products
        of size hidden_size) relative to the main forward pass.  Functional: repulsion
        is based on actual expert outputs for the current batch, not weight geometry.
        """
        # Flatten leading dims: (total_tokens, n_experts, hidden_size)
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)  # (T, n_experts, hidden_size)

        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, k)

        i_idx = [p[0] for p in sampled]
        j_idx = [p[1] for p in sampled]

        # out_i / out_j: (T, k, hidden_size) — gather sampled expert outputs
        out_i = eg_norm[:, i_idx, :]   # (T, k, hidden_size)
        out_j = eg_norm[:, j_idx, :]

        # Mean cosine similarity over tokens and sampled pairs → scalar
        cos_sim = (out_i * out_j).sum(-1).mean()

        add_aux_loss(self.repulsion_coef * cos_sim)

    def _log_metrics(self, p: torch.Tensor, out: torch.Tensor) -> None:
        with torch.no_grad():
            p_flat = p.reshape(-1, self.n_experts)  # (T, n_experts)
            max_H = math.log(self.n_experts) if self.n_experts > 1 else 1.0

            # Per-token entropy: H(p(h)) averaged over tokens.
            # Correct collapse measure: H(E[p]) (entropy of mean) overestimates diversity
            # because it looks uniform even when each token hard-routes to a different expert.
            per_token_H = -(p_flat * (p_flat + 1e-8).log()).sum(-1)  # (T,)
            mean_token_H = per_token_H.mean().item()

            # Effective number of experts = exp(mean per-token entropy).
            # Ranges from 1.0 (fully collapsed) to n_experts (perfectly uniform).
            effective_n = math.exp(mean_token_H)

            # Dominant expert per token: which expert has the highest routing weight.
            dominant = p_flat.argmax(-1)  # (T,)
            expert_counts = dominant.bincount(minlength=self.n_experts).float()
            n_dominant = int((expert_counts > 0).sum().item())  # how many experts ever win
            max_load = (expert_counts / p_flat.shape[0]).max().item()  # top expert's token share

            self._cached_metrics = {
                "effective_n_experts": effective_n,          # 1.0=collapsed, n_experts=uniform
                "n_dominant_experts": float(n_dominant),     # 1=collapsed, n_experts=all used
                "max_expert_load": max_load,                 # 1.0=collapsed, 1/n=uniform
                "mean_token_entropy_norm": mean_token_H / max_H,
                "output_norm": out.norm(dim=-1).mean().item(),
            }

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Returns −E_moe(h) = −τ·log(Σᵢ exp(φ(W1ᵢh)ᵀ(W2ᵢh)/τ)), consistent with Energy_MLP
        convention (negative = lower energy = more aligned)."""
        W1x = self.W1(x).view(*x.shape[:-1], self.n_experts, self.expert_I)
        W2x = self.W2(x).view(*x.shape[:-1], self.n_experts, self.expert_I)
        # Apply same 1/sqrt(expert_I) scaling as forward() for consistency
        E = (F.gelu(W1x) * W2x).sum(dim=-1) * self._routing_scale
        return -torch.logsumexp(E / self.temperature, dim=-1) * self.temperature
