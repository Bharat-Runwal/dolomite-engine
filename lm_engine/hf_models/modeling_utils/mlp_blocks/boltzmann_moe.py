# **************************************************
# Boltzmann-weighted Mixture-of-Experts Energy FFN.
# Port of Nima Dehmamy's BoltzmannMoE_Energy_MLP — no linear router, no Switch
# aux/z-loss, no KL distillation. Routing comes directly from per-expert
# energies; only optional auxiliary signal is stochastic cosine repulsion on
# expert outputs. This avoids the squared-logsumexp gradient-explosion vector
# that drove early-training spikes in MoE_Energy_F5.
#
# E_moe(h)     = log( Σᵢ exp(Eᵢ(h)) )       Eᵢ(h) = φ(W1ᵢh)ᵀ(W2ᵢh)
# ∂E_moe/∂h    = Σᵢ pᵢ(h) · ∂Eᵢ/∂h          pᵢ(h) = softmax(E(h)/τ)
#
# Iso-parameter with Energy_MLP: each of n_experts experts has
# intermediate_size // n_experts hidden units.
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
    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        temperature: float,
        repulsion_coef: float,
        n_repulsion_pairs: int,
        init_method: str,
        activation_function: str,
        dropout: float,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        add_bias: bool = False,
        layer_idx: int | None = None,
        top_k: int = 0,
        normalized_topk: bool = True,
    ) -> None:
        super().__init__()

        assert intermediate_size % n_experts == 0, (
            f"intermediate_size ({intermediate_size}) must be divisible by n_experts ({n_experts})"
        )
        assert 0 <= top_k <= n_experts, f"top_k ({top_k}) must be in [0, n_experts={n_experts}]"

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        self.temperature = temperature
        self.repulsion_coef = repulsion_coef
        self.n_repulsion_pairs = n_repulsion_pairs
        # top_k=0 keeps full-dense behavior (Nima's original). top_k>0 zeros out
        # all-but-top-k routing weights using the same logic as F5's _get_topk +
        # normalized_topk. Iso-param weights still compute every step (no FLOPs
        # saved); only the OUTPUT mixing becomes sparse.
        self.top_k = top_k
        self.normalized_topk = normalized_topk
        self.layer_idx = layer_idx
        self._cached_metrics: dict[str, float] | None = None

        self._all_pairs: list[tuple[int, int]] = list(itertools.combinations(range(n_experts), 2))

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)

        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]

        W1x = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.expert_I)

        W1_e = self.W1.weight.view(self.n_experts, self.expert_I, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.expert_I, self.hidden_size)

        phi = F.gelu(W1x)
        phi_prime = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5

        # term1ᵢ = φ(W1ᵢh) @ W2ᵢᵀ — used for routing energy and as half the gradient
        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)

        # Routing energy uses only term1 (positive alignment +φᵀW2h). Including
        # the full gradient (term1+term2) here would inject a Hessian-like term.
        E = torch.einsum("...h,...eh->...e", x, term1)

        p = F.softmax(E / self.temperature, dim=-1)

        if self.top_k > 0 and self.top_k < self.n_experts:
            # F5-style top-k mask: take top-k routing weights, optionally
            # renormalize the kept ones, scatter back into a sparse weights
            # tensor. All expert outputs still computed (iso-param fused).
            topk_w, topk_idx = p.topk(self.top_k, dim=-1)
            if self.normalized_topk:
                topk_w = F.softmax(topk_w.float(), dim=-1).type_as(p)
            p = torch.zeros_like(p).scatter_(-1, topk_idx, topk_w)

        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.expert_I)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2

        out = torch.einsum("...e,...eh->...h", p, expert_grads)

        if self.training and self.repulsion_coef > 0:
            self._add_repulsion_loss(expert_grads)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)

        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, k)
        i_idx = [pp[0] for pp in sampled]
        j_idx = [pp[1] for pp in sampled]

        out_i = eg_norm[:, i_idx, :]
        out_j = eg_norm[:, j_idx, :]
        cos_sim = (out_i * out_j).sum(-1).mean()

        add_aux_loss(self.repulsion_coef * cos_sim)

    def _log_metrics(self, p: torch.Tensor, out: torch.Tensor) -> None:
        with torch.no_grad():
            p_flat = p.reshape(-1, self.n_experts)
            max_H = math.log(self.n_experts) if self.n_experts > 1 else 1.0

            per_token_H = -(p_flat * (p_flat + 1e-8).log()).sum(-1)
            mean_token_H = per_token_H.mean().item()
            effective_n = math.exp(mean_token_H)

            dominant = p_flat.argmax(-1)
            expert_counts = dominant.bincount(minlength=self.n_experts).float()
            n_dominant = int((expert_counts > 0).sum().item())
            max_load = (expert_counts / p_flat.shape[0]).max().item()

            self._cached_metrics = {
                "effective_n_experts": effective_n,
                "n_dominant_experts": float(n_dominant),
                "max_expert_load": max_load,
                "mean_token_entropy_norm": mean_token_H / max_H,
                "output_norm": out.norm(dim=-1).mean().item(),
            }

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W1x = self.W1(x).view(*x.shape[:-1], self.n_experts, self.expert_I)
        W2x = self.W2(x).view(*x.shape[:-1], self.n_experts, self.expert_I)
        E = (F.gelu(W1x) * W2x).sum(dim=-1)
        return -torch.logsumexp(E / self.temperature, dim=-1) * self.temperature
