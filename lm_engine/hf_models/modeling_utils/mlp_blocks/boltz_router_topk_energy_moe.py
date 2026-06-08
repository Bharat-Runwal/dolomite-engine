# BoltzRouter Top-K Energy MoE — fh1 with a Boltzmann router.
#
# Same expert form as TopK_Energy_MoE_MLP (each of n_experts is a FULL-SIZE
# Energy_MLP, intermediate_size NOT divided by n_experts; output is the analytic
# energy gradient phi @ W2_e^T + (phi' * W2 x) @ W1_e^T).
#
# Difference vs fh1: routing logits come from per-expert energies
#     E_e(x) = phi(W1_e x)^T (W2_e x)
# routed by p_e ~ exp(-E_e / T) (Boltzmann). Top-k is over -E_e/T. No linear
# router. Learnable temperature T (init 1.0). No KL distillation.
#
# energy_scale_mode controls how energies are scaled into logits:
#   "temperature" : logits = -E_e / T, with T = exp(log_temperature)
#                   (learnable if learnable_temperature else fixed buffer).
#   "sqrt_inv_d"  : logits = -E_e * sqrt(1/intermediate_size), a FIXED scale.
#                   The energy is a dot product summed over intermediate_size,
#                   so its std grows ~sqrt(intermediate_size); this normalizes
#                   the logit scale the way attention uses 1/sqrt(d_k), with no
#                   learnable temperature. (Mentor suggestion: works better than
#                   a learnable T.)
#
# Stability discipline (lessons from F5 spike incident):
#   - Switch load-balance only; NO z-loss.
#   - Detach E_e in the load-balance path so squared-logsumexp gradient does
#     not flow into expert weights via the aux loss.

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


class BoltzRouter_TopK_Energy_MoE_MLP(nn.Module):
    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,         # per-expert; NOT divided by n_experts
        n_experts: int,
        top_k: int,
        load_balance_coef: float,
        boltzmann_temperature: float,
        learnable_temperature: bool,
        energy_scale_mode: str,
        sqrt_inv_d_dim: str,
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
        assert boltzmann_temperature > 0
        assert energy_scale_mode in ("temperature", "sqrt_inv_d"), (
            f"energy_scale_mode must be 'temperature' or 'sqrt_inv_d', got {energy_scale_mode!r}"
        )
        assert sqrt_inv_d_dim in ("intermediate", "hidden"), (
            f"sqrt_inv_d_dim must be 'intermediate' or 'hidden', got {sqrt_inv_d_dim!r}"
        )
        self.energy_scale_mode = energy_scale_mode
        self.sqrt_inv_d_dim = sqrt_inv_d_dim
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

        # FSDP-2 (fully_shard) rejects 0-D parameters; use a 1-D tensor of size 1.
        log_T_init = math.log(boltzmann_temperature)
        if learnable_temperature:
            self.log_temperature = nn.Parameter(torch.tensor([log_T_init], dtype=torch.float32))
        else:
            self.register_buffer("log_temperature", torch.tensor([log_T_init], dtype=torch.float32))

        # Fixed sqrt(1/d) scale used when energy_scale_mode == "sqrt_inv_d".
        # "intermediate": d = intermediate_size — the dim the energy dot-product sums
        #   over, so std(E_e) ~ sqrt(intermediate_size); this is the variance-normalizing
        #   choice (cf. attention's 1/sqrt(d_k)).
        # "hidden": d = hidden_size — scale by the model width instead.
        _d = intermediate_size if sqrt_inv_d_dim == "intermediate" else hidden_size
        self.sqrt_inv_d_scale = (1.0 / _d) ** 0.5

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]

        W1_e = self.W1.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)

        W1x = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.intermediate_size)
        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.intermediate_size)
        phi = F.gelu(W1x)
        phi_prime = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5

        # Per-expert scalar energy E_e(x) = phi(W1_e x) . (W2_e x)
        energies = (phi * W2x).sum(dim=-1)                      # (..., n_experts)

        # Boltzmann logits: lower energy -> higher probability.
        if self.energy_scale_mode == "temperature":
            # Cast T to the input dtype so logits/output preserve bf16 under autocast.
            # log_temperature is fp32 for numerical stability of the log/exp; energies
            # follow x's dtype. Without this cast, division upcasts everything to fp32
            # and breaks downstream bf16 Linear ops at inference.
            T = self.log_temperature.exp().clamp(min=1e-4).to(energies.dtype)  # shape (1,)
            logits = -energies / T
        else:
            # Fixed sqrt(1/intermediate_size) scale; no learnable temperature.
            T = energies.new_tensor([self.sqrt_inv_d_scale])  # for metric logging only
            logits = -energies * self.sqrt_inv_d_scale

        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)
        topk_weights = F.softmax(topk_logits, dim=-1)

        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2

        if self.training and self.load_balance_coef > 0:
            # Switch load-balance. Detach energies in the aux path so the
            # squared-logsumexp gradient does NOT flow into expert weights.
            all_probs = F.softmax(logits.detach(), dim=-1)
            tokens = all_probs.reshape(-1, self.n_experts)
            T_tok = tokens.shape[0]
            one_hot = torch.zeros_like(tokens)
            one_hot.scatter_(-1, topk_indices.reshape(T_tok, self.top_k), 1.0 / self.top_k)
            f = one_hot.mean(0)
            P = tokens.mean(0)
            lb_loss = self.n_experts * (f.detach() * P).sum()
            add_aux_loss(self.load_balance_coef * lb_loss)

        idx = topk_indices.unsqueeze(-1).expand(*leading, self.top_k, self.hidden_size)
        selected = expert_grads.gather(dim=-2, index=idx)
        out = (topk_weights.unsqueeze(-1) * selected).sum(-2)

        if not torch.compiler.is_compiling():
            self._log_metrics(topk_weights, topk_indices, T)

        return out

    def _log_metrics(
        self,
        topk_weights: torch.Tensor,
        topk_indices: torch.Tensor,
        temperature: torch.Tensor,
    ) -> None:
        with torch.no_grad():
            idx_flat = topk_indices.reshape(-1)
            counts = idx_flat.bincount(minlength=self.n_experts).float()
            n_dominant = int((counts > 0).sum().item())
            max_load = (counts / counts.sum().clamp(min=1)).max().item()
            self._cached_metrics = {
                "n_dominant_experts": float(n_dominant),
                "max_expert_load": max_load,
                "boltz_temperature": float(temperature.flatten()[0].item()),
            }

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics

    def get_num_active_parameters(self) -> int:
        active = 0
        for parameter in self.W1.parameters():
            active += (parameter.numel() * self.top_k) // self.n_experts
        for parameter in self.W2.parameters():
            active += (parameter.numel() * self.top_k) // self.n_experts
        # log_temperature is a single scalar; ignore in active-param count.
        return active
