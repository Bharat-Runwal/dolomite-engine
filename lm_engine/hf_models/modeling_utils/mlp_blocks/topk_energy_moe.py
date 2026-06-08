# Top-K Energy MoE — port of Nima's TopK_Energy_MoE_MLP.
#
# Each of n_experts is a FULL-SIZE Energy_MLP (intermediate_size NOT divided by
# n_experts). Linear router selects top_k experts per token; outputs are softmax-
# normalized over the selected experts. Switch-style load-balance aux loss only
# (no z-loss, no KL distillation, no expert repulsion).

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss import add_aux_loss
from ...parameter import mark_parameter_as_mup_learning_rate
from ..dropout import Dropout
from ..linear import ParameterizedLinear
from .mlp import _get_std_for_linear


class TopK_Energy_MoE_MLP(nn.Module):
    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,         # per-expert; NOT divided by n_experts
        n_experts: int,
        top_k: int,
        load_balance_coef: float,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        assert top_k <= n_experts, f"top_k ({top_k}) must be <= n_experts ({n_experts})"
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.top_k = top_k
        self.load_balance_coef = load_balance_coef
        self.layer_idx = layer_idx
        self._cached_metrics: dict[str, float] | None = None

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W1 = ParameterizedLinear(hidden_size, n_experts * intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, n_experts * intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        self.router = nn.Linear(hidden_size, n_experts, bias=False)
        torch.nn.init.normal_(self.router.weight, std=0.01)

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]

        logits = self.router(x)
        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        W1_e = self.W1.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)

        W1x = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.intermediate_size)
        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.intermediate_size)
        phi = F.gelu(W1x)
        phi_prime = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5
        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2

        if self.training and self.load_balance_coef > 0:
            all_probs = F.softmax(logits, dim=-1)
            tokens = all_probs.reshape(-1, self.n_experts)
            T = tokens.shape[0]
            one_hot = torch.zeros_like(tokens)
            one_hot.scatter_(-1, topk_indices.reshape(T, self.top_k), 1.0 / self.top_k)
            f = one_hot.mean(0)
            P = tokens.mean(0)
            lb_loss = self.n_experts * (f.detach() * P).sum()
            add_aux_loss(self.load_balance_coef * lb_loss)

        idx = topk_indices.unsqueeze(-1).expand(*leading, self.top_k, self.hidden_size)
        selected = expert_grads.gather(dim=-2, index=idx)
        out = (topk_weights.unsqueeze(-1) * selected).sum(-2)

        if not torch.compiler.is_compiling():
            self._log_metrics(topk_weights, topk_indices)

        return out

    def _log_metrics(self, topk_weights: torch.Tensor, topk_indices: torch.Tensor) -> None:
        with torch.no_grad():
            idx_flat = topk_indices.reshape(-1)
            counts = idx_flat.bincount(minlength=self.n_experts).float()
            n_dominant = int((counts > 0).sum().item())
            max_load = (counts / counts.sum().clamp(min=1)).max().item()
            self._cached_metrics = {
                "n_dominant_experts": float(n_dominant),
                "max_expert_load": max_load,
            }

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics

    def get_num_active_parameters(self) -> int:
        # Per-token: only top_k of n_experts is used in W1 and W2; router is fully used.
        active = 0
        for parameter in self.W1.parameters():
            active += (parameter.numel() * self.top_k) // self.n_experts
        for parameter in self.W2.parameters():
            active += (parameter.numel() * self.top_k) // self.n_experts
        for parameter in self.router.parameters():
            active += parameter.numel()
        return active
