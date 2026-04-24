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

        if not torch.compiler.is_compiling():
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
        """Compute energy per token: E(h) = -phi(W1h)^T(W2h)."""
        W1x = self.W1(x)
        W2x = self.W2(x)
        return -(F.gelu(W1x) * W2x).sum(dim=-1)


class Compositional_Energy_MLP(nn.Module):
    """Compositional Energy MLP with K parallel energy paths (Product-of-Experts).

    Total energy: E_FF = sum_k alpha_k * E_k(h)
    Each path uses a different activation phi_k for heterogeneous energy landscapes.
    Iso-parameter design: K paths with intermediate_size/K each = same total params.
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5
    _DEFAULT_ACTIVATIONS = ["gelu", "silu", "tanh", "relu"]

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
        num_paths: int = 4,
        path_activations: list[str] | None = None,
        add_bias: bool = False,
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()

        self.layer_idx = layer_idx
        self.num_paths = num_paths
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size

        assert intermediate_size % num_paths == 0, (
            f"intermediate_size ({intermediate_size}) must be divisible by num_paths ({num_paths})"
        )
        self.path_intermediate_size = intermediate_size // num_paths

        # Activation per path
        if path_activations is not None and len(path_activations) > 0:
            assert len(path_activations) == num_paths
            self.path_activations = path_activations
        else:
            self.path_activations = self._DEFAULT_ACTIVATIONS[:num_paths]

        self._cached_metrics: dict[str, float] | None = None

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # Fused projections: all K paths stacked into one matmul
        # W1: hidden_size -> intermediate_size (all paths concatenated)
        # W2: hidden_size -> intermediate_size (all paths concatenated)
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)

        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        # Learnable per-path scales alpha_k, initialized to 1/K
        self.alphas = nn.Parameter(torch.full((num_paths,), 1.0 / num_paths))

    def _activation_and_derivative(self, x: torch.Tensor, name: str):
        """Return (phi(x), phi'(x)) for the given activation name."""
        if name == "gelu":
            return F.gelu(x), torch.sigmoid(self._SIGMOID_SCALE * x) * 0.5
        elif name == "silu":
            sig = torch.sigmoid(x)
            return x * sig, sig * (1.0 + x * (1.0 - sig))
        elif name == "tanh":
            t = torch.tanh(x)
            return t, 1.0 - t * t
        elif name == "relu":
            mask = (x > 0).to(x.dtype)
            return F.relu(x), mask
        else:
            raise ValueError(f"Unsupported activation: {name}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Fused matmul then chunk into K paths
        W1x = self.W1(x)  # (b, t, intermediate_size)
        W2x = self.W2(x)  # (b, t, intermediate_size)

        W1x_paths = W1x.chunk(self.num_paths, dim=-1)  # K chunks of (b, t, path_intermediate_size)
        W2x_paths = W2x.chunk(self.num_paths, dim=-1)

        # Chunk weight matrices for per-path back-projection
        W1_weights = self.W1.weight.chunk(self.num_paths, dim=0)  # K chunks of (path_intermediate_size, hidden_size)
        W2_weights = self.W2.weight.chunk(self.num_paths, dim=0)

        out = torch.zeros_like(x)
        for k in range(self.num_paths):
            phi, phi_prime = self._activation_and_derivative(W1x_paths[k], self.path_activations[k])
            # Energy gradient: nabla_h E_k = W2_k^T phi(W1_k h) + W1_k^T [phi'(W1_k h) * W2_k h]
            term1 = phi @ W2_weights[k]          # (b, t, hidden_size)
            term2 = (phi_prime * W2x_paths[k]) @ W1_weights[k]  # (b, t, hidden_size)
            out = out + self.alphas[k] * (term1 + term2)

        if not torch.compiler.is_compiling():
            self._log_norms(out)
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """Compute total compositional energy: E = -sum_k alpha_k * phi_k(W1_k h)^T (W2_k h)."""
        W1x = self.W1(x)
        W2x = self.W2(x)
        W1x_paths = W1x.chunk(self.num_paths, dim=-1)
        W2x_paths = W2x.chunk(self.num_paths, dim=-1)

        energy = torch.zeros(x.shape[0], x.shape[1], device=x.device, dtype=x.dtype)
        for k in range(self.num_paths):
            phi, _ = self._activation_and_derivative(W1x_paths[k], self.path_activations[k])
            energy = energy - self.alphas[k] * (phi * W2x_paths[k]).sum(dim=-1)
        return energy

    def _log_norms(self, out: torch.Tensor) -> None:
        with torch.no_grad():
            w1_norm = self.W1.weight.norm().item()
            w2_norm = self.W2.weight.norm().item()
            w_total_norm = math.sqrt(w1_norm**2 + w2_norm**2)
            out_norm = out.norm(dim=-1).mean().item()

            metrics = {
                "W1_norm": w1_norm,
                "W2_norm": w2_norm,
                "W_total_norm": w_total_norm,
                "output_norm": out_norm,
            }
            # Per-path alpha values
            for k in range(self.num_paths):
                metrics[f"alpha_{self.path_activations[k]}"] = self.alphas[k].item()

            self._cached_metrics = metrics

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics


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


class Mixed_Energy_MLP(nn.Module):
    """Half Energy_MLP (∂E_FF/∂h) + half standard GELU MLP.

    Implements the full-block energy gradient: attention heads already compute ∂E_attn/∂h;
    this FFN adds ∂E_FF/∂h so the combined block output is the full energy gradient.

    Iso-param vs SwiGLU with intermediate I: use energy_size = standard_size = 0.75*I.
    E.g. I=1536 (d=768) → each=1152; I=2048 (d=1024) → each=1536.
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        energy_intermediate_size: int,
        standard_intermediate_size: int,
        init_method: str,
        activation_function: str,
        dropout: float,
        initializer_range: float,
        m_width: float,
        num_layers: int,
        add_bias: bool = False,
        layer_idx: int | None = None,
        **kwargs,  # absorb unused intermediate_size from base kwargs
    ) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self._cached_metrics = None

        std = _get_std_for_linear(initializer_range, init_method, m_width)

        # Energy FFN: two projections, output = ∂E_FF/∂h
        self.W1 = ParameterizedLinear(hidden_size, energy_intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, energy_intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        # Standard GELU MLP: W_in → act → W_out
        self.c_fc = ParameterizedLinear(hidden_size, standard_intermediate_size, bias=add_bias, std=std)
        self.c_proj = ParameterizedLinear(
            standard_intermediate_size, hidden_size, bias=add_bias, std=std / math.sqrt(2 * num_layers)
        )
        self.act = get_activation_function(activation_function)
        self.dropout = Dropout(dropout)
        mark_parameter_as_mup_learning_rate(self.c_fc.weight)
        mark_parameter_as_mup_learning_rate(self.c_proj.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Energy gradient: ∂/∂h [-phi(W1 h)^T (W2 h)] = W2^T phi(W1h) + W1^T [phi'(W1h) * W2h]
        W1x = self.W1(x)
        W2x = self.W2(x)
        phi = F.gelu(W1x)
        phi_prime_W2x = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5 * W2x
        energy_out = phi @ self.W2.weight + phi_prime_W2x @ self.W1.weight

        # Standard MLP
        standard_out = self.c_proj(self.dropout(self.act(self.c_fc(x))))

        out = energy_out + standard_out
        if not torch.compiler.is_compiling():
            self._cached_metrics = {"output_norm": out.norm(dim=-1).mean().item()}
        return out

    def get_metrics(self):
        return self._cached_metrics


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
