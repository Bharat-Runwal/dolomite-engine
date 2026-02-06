# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import math

import torch
import torch.nn as nn

from ...parameter import mark_parameter_as_mup_learning_rate
from ..activations import get_activation_function, is_glu
from ..dropout import Dropout
from ..linear import ParameterizedLinear

import torch.nn.functional as F


class Energy_MLP(nn.Module):
    """Feedforward with manual gradient computation for Energy Transformer."""

    # Precompute constant for sigmoid scaling (avoids recomputation each forward)
    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        init_method: str,
        activation_function: str,
        dropout: float,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        add_bias: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx

        # Metrics storage for tracking (updated each forward pass)
        self._cached_metrics: dict[str, float] | None = None

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # Two projection layers: hidden -> intermediate (no bias for energy model)
        # W1: used for GELU path and sigmoid-gated path
        # W2: used for gating and output projection
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)

        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Project to intermediate size using ParameterizedLinear layers
        W1x = self.W1(x)  # (b, t, intermediate_size)
        W2x = self.W2(x)  # (b, t, intermediate_size)

        # GELU path
        y1 = F.gelu(W1x)

        # Sigmoid-gated path (energy gradient term)
        # W1.weight has shape (intermediate_size, hidden_size), so it acts as the projection back
        y2 = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5 * W2x
        y2 = y2 @ self.W1.weight  # project back to hidden_size

        # Combine paths: y1 projected back via W2.weight
        out = y1 @ self.W2.weight + y2

        self._log_norms(out)

        return out

    def _log_norms(self, out: torch.Tensor) -> None:
        """Cache weight matrix norms and output norm for external tracking."""
        with torch.no_grad():
            # Weight norms
            w1_norm = self.W1.weight.norm().item()
            w2_norm = self.W2.weight.norm().item()
            w_total_norm = math.sqrt(w1_norm**2 + w2_norm**2)

            # Output norm (mean over batch)
            out_norm = out.norm(dim=-1).mean().item()

            self._cached_metrics = {
                "W1_norm": w1_norm,
                "W2_norm": w2_norm,
                "W_total_norm": w_total_norm,
                "output_norm": out_norm,
            }

    def get_metrics(self) -> dict[str, float] | None:
        """Return cached metrics for external tracking."""
        return self._cached_metrics


    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     # Use @ instead of einsum (faster BLAS operations)
    #     W1x = x @ self.W[0]  # (b, t, intermediate_size)
    #     W2x = x @ self.W[1]  # (b, t, intermediate_size)

    #     y1 = F.gelu(W1x)

    #     # Fuse sigmoid computation with multiplication
    #     y2 = F.sigmoid((2 / torch.pi) ** 0.5 * W1x) * 0.5 * W2x  # element-wise ops only
    #     y2 = y2 @ self.W[0].T  # project back to hidden_size

    #     # Combine paths and project back to hidden_size
    #     out = y1 @ self.W[1].T + y2
    #     return out
    
    # # def forward(self, x: torch.Tensor) -> torch.Tensor:
    # #     # Efficient: single batched einsum for both projections
    # #     W1x, W2x = torch.einsum("bti,hio->hbto", x, self.W).unbind(0)
        
    # #     y1 = F.gelu(W1x)
    # #     y2 = F.sigmoid((2 / torch.pi) ** 0.5 * W1x) * 0.5
    # #     # y2 = torch.einsum("bto,io,bto->bti", y2, self.W[0], W2x)
    # #     y2 = (y2 * W2x) @ self.W[0].T  # shape: (b, t, i)

    # #     out = torch.einsum("bto,io->bti", y1, self.W[1]) + y2
    # #     return out

    # def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
    #     W1x = torch.einsum("bti,io->bto", x, self.W[0])
    #     return F.gelu(W1x).sum(dim=-1)


    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W1x = self.W1(x)
        return F.gelu(W1x).sum(dim=-1)


class MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        activation_function: str,
        add_bias: bool,
        dropout: float,
        init_method: str,
        initializer_range: float,
        m_width: float,
        num_layers: int,
    ) -> MLP:
        super().__init__()

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        self.c_fc = ParameterizedLinear(
            hidden_size,
            2 * intermediate_size if is_glu(activation_function) else intermediate_size,
            bias=add_bias,
            std=std,
        )

        self.act = get_activation_function(activation_function)

        #TODO: Issue here when is_glu case is there
        self.c_proj = ParameterizedLinear(
            intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )

        self.dropout = Dropout(dropout)

        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.act(x)
        x = self.c_proj(x)
        x = self.dropout(x)
        return x


def _get_std_for_linear(initializer_range: float, init_method: str, m_width: float | None) -> float:
    std = initializer_range
    if init_method == "mup":
        std /= math.sqrt(m_width)
    elif init_method != "normal":
        raise ValueError(f"unexpected init_method ({init_method})")

    return std


def interleave_up_gate_tensor_for_mlp(up_weight: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    return torch.cat([up_weight, gate_weight])


def split_up_gate_tensor_for_mlp(c_fc_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return c_fc_weight.chunk(2)
