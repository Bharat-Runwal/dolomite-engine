# **************************************************
# Copyright (c) 2025, Mayank Mishra
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
from ..activations import get_activation_function, is_glu
from ..dropout import Dropout
from ..linear import ParameterizedLinear


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

        assert intermediate_size % n_experts == 0, (
            f"intermediate_size ({intermediate_size}) must be divisible by n_experts ({n_experts})"
        )

        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        self._routing_scale = self.expert_I ** -0.5  # 1/sqrt(expert_I): prevents routing energy growth → spikes
        self.temperature = temperature
        self.repulsion_coef = repulsion_coef
        self.n_repulsion_pairs = n_repulsion_pairs
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

        # Activation and its approximate derivative (GELU)
        phi = F.gelu(W1x)                                               # (..., n_experts, expert_I)
        phi_prime = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5

        # First gradient term: φ(W1ᵢh) @ W2ᵢᵀ   shape (..., n_experts, hidden)
        term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)

        # Routing energy: Eᵢ(h) = h · term1ᵢ / sqrt(expert_I)
        # Scaled like attention (1/sqrt(d_k)) to prevent ||E_i|| from growing as
        # O(||W||² · expert_I) during training, which causes routing spikes.
        # Without this scaling, E_i grows unboundedly → softmax saturates → hard
        # routing shift → loss spikes (observed at steps ~1000-4000).
        E = torch.einsum("...h,...eh->...e", x, term1) * self._routing_scale  # (..., n_experts)

        # Boltzmann routing weights: pᵢ = softmax_i(E / τ)
        p = F.softmax(E / self.temperature, dim=-1)                     # (..., n_experts)

        # Second gradient term needs W2(x); independent dropout for regularisation
        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.expert_I)
        term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2                                    # (..., n_experts, hidden)

        # Boltzmann-weighted sum: ∂E_moe/∂h = Σᵢ pᵢ · (term1ᵢ + term2ᵢ)
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


class TopK_Energy_MoE_MLP(nn.Module):
    """Top-K MoE with full-size Energy_MLP experts.

    Unlike BoltzmannMoE_Energy_MLP (iso-param: divides int by n_experts), each expert
    here gets the FULL intermediate_size — following Switch Transformer / Mixtral practice.
    A learned linear router selects top_k experts per token; outputs are softmax-normalised
    among the selected (differentiable top-K, same as Switch/Mixtral during training).

    load_balance_coef: auxiliary load-balancing loss (Switch Transformer §2.1) that
    penalises routing imbalance. L_lb = n_experts * Σ_i f_i * P_i where f_i is the
    fraction of tokens dispatched to expert i and P_i is the mean router probability for
    expert i. Without this, the router collapses to always selecting the same top_k experts.

    All n_experts compute their gradients in the forward pass; only top_k outputs are used.
    This wastes FLOPs on unselected experts — acceptable for an ablation prototype.
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,  # per-expert; NOT divided by n_experts
        n_experts: int,
        top_k: int,
        load_balance_coef: float,  # auxiliary load-balancing loss weight (e.g. 0.01)
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

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        # Fused expert weights: each (n_experts * intermediate_size, hidden)
        self.W1 = ParameterizedLinear(hidden_size, n_experts * intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, n_experts * intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        # Linear router — cheap (d → n_experts)
        self.router = nn.Linear(hidden_size, n_experts, bias=False)
        torch.nn.init.normal_(self.router.weight, std=0.01)

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]

        # Router: top-K selection
        logits = self.router(x)                                             # (..., n_experts)
        topk_logits, topk_indices = logits.topk(self.top_k, dim=-1)        # (..., top_k)
        topk_weights = F.softmax(topk_logits, dim=-1)                      # (..., top_k)

        # Expert weights (reshaped views — zero-copy)
        W1_e = self.W1.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.intermediate_size, self.hidden_size)

        # Compute all expert gradients (fused matmuls)
        W1x = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.intermediate_size)
        W2x = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.intermediate_size)
        phi        = F.gelu(W1x)
        phi_prime  = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5
        term1      = torch.einsum("...ei,eih->...eh", phi,             W2_e)
        term2      = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2                                        # (..., n_experts, hidden)

        # Auxiliary load-balancing loss (Switch Transformer §2.1): prevents routing collapse.
        # L_lb = n_experts * Σ_i f_i * P_i  (differentiable through P_i only)
        if self.training and self.load_balance_coef > 0:
            # f_i: fraction of tokens routed to expert i (non-differentiable indicator sum)
            # P_i: mean router softmax probability for expert i over the batch (differentiable)
            all_probs = F.softmax(logits, dim=-1)               # (..., n_experts) full probs
            tokens = all_probs.reshape(-1, self.n_experts)       # (T, n_experts)
            T = tokens.shape[0]
            # f_i: fraction of tokens where expert i is in top_k (indicator)
            one_hot = torch.zeros_like(tokens)
            one_hot.scatter_(-1, topk_indices.reshape(T, self.top_k), 1.0 / self.top_k)
            f = one_hot.mean(0)                                  # (n_experts,) non-diff
            P = tokens.mean(0)                                   # (n_experts,) differentiable
            lb_loss = self.n_experts * (f.detach() * P).sum()
            add_aux_loss(self.load_balance_coef * lb_loss)

        # Gather top_k and combine
        idx      = topk_indices.unsqueeze(-1).expand(*leading, self.top_k, self.hidden_size)
        selected = expert_grads.gather(dim=-2, index=idx)                   # (..., top_k, hidden)
        return (topk_weights.unsqueeze(-1) * selected).sum(-2)              # (..., hidden)


class SurrogateBoltzmannMoE_Energy_MLP(nn.Module):
    """BoltzmannMoE_Energy_MLP with a linear surrogate router for cheap inference.

    Iso-parameter with Energy_MLP (same as BoltzmannMoE): n_experts experts each with
    (intermediate_size // n_experts) neurons.

    Training:
      - Computes Boltzmann weights p_boltz from expert energies (full cost)
      - Computes surrogate weights p_surr from a learned linear layer (cheap: d → n_experts)
      - KL(p_surr ‖ p_boltz.detach()) added to aux loss with coefficient surrogate_coef

    Inference (use_surrogate=True, default on eval):
      - Skip energy routing; use p_surr only — O(d·n_experts) vs O(d·expert_I) per token
      - Expert gradients (term1 + term2) still computed for the output
    """

    _SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,    # total = n_experts * per_expert_I (iso-param)
        n_experts: int,
        temperature: float,
        repulsion_coef: float,
        n_repulsion_pairs: int,
        surrogate_coef: float,
        use_surrogate: bool,
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
        assert intermediate_size % n_experts == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        self.temperature = temperature
        self.repulsion_coef = repulsion_coef
        self.n_repulsion_pairs = n_repulsion_pairs
        self.surrogate_coef = surrogate_coef
        self.use_surrogate = use_surrogate
        self.layer_idx = layer_idx
        self._all_pairs: list[tuple[int, int]] = list(itertools.combinations(range(n_experts), 2))
        self._cached_metrics: dict[str, float] | None = None

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

        # Surrogate router (cheap linear)
        self.surrogate_router = nn.Linear(hidden_size, n_experts, bias=False)
        torch.nn.init.normal_(self.surrogate_router.weight, std=0.01)

        self.dropout = Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        leading = x.shape[:-1]

        W1_e = self.W1.weight.view(self.n_experts, self.expert_I, self.hidden_size)
        W2_e = self.W2.weight.view(self.n_experts, self.expert_I, self.hidden_size)
        W1x  = self.dropout(self.W1(x)).view(*leading, self.n_experts, self.expert_I)
        phi        = F.gelu(W1x)
        phi_prime  = torch.sigmoid(self._SIGMOID_SCALE * W1x) * 0.5
        term1      = torch.einsum("...ei,eih->...eh", phi,            W2_e)
        W2x        = self.dropout(self.W2(x)).view(*leading, self.n_experts, self.expert_I)
        term2      = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
        expert_grads = term1 + term2                                        # (..., n_experts, hidden)

        # Surrogate routing (always computed — needed for KL loss and cheap inference)
        p_surr = F.softmax(self.surrogate_router(x) / self.temperature, dim=-1)

        if not self.training and self.use_surrogate:
            # Cheap inference path: skip energy routing
            return torch.einsum("...e,...eh->...h", p_surr, expert_grads)

        # Full Boltzmann routing
        E = torch.einsum("...h,...eh->...e", x, term1)                     # (..., n_experts)
        p_boltz = F.softmax(E / self.temperature, dim=-1)
        out = torch.einsum("...e,...eh->...h", p_boltz, expert_grads)

        if self.training:
            if self.surrogate_coef > 0:
                kl = -(p_boltz.detach() * (p_surr + 1e-8).log()).sum(-1).mean()
                add_aux_loss(self.surrogate_coef * kl)
            if self.repulsion_coef > 0:
                eg      = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
                eg_norm = F.normalize(eg, dim=-1)
                k       = min(self.n_repulsion_pairs, len(self._all_pairs))
                pairs   = random.sample(self._all_pairs, k)
                i_idx   = [p[0] for p in pairs]
                j_idx   = [p[1] for p in pairs]
                cos_sim = (eg_norm[:, i_idx] * eg_norm[:, j_idx]).sum(-1).mean()
                add_aux_loss(self.repulsion_coef * cos_sim)

        if not torch.compiler.is_compiling():
            self._log_metrics(p_boltz, p_surr, out)
        return out

    def _log_metrics(self, p_boltz, p_surr, out):
        with torch.no_grad():
            p_flat = p_boltz.reshape(-1, self.n_experts)
            H = -(p_flat * (p_flat + 1e-8).log()).sum(-1)
            kl = -(p_boltz.detach().reshape(-1, self.n_experts) *
                   (p_surr.reshape(-1, self.n_experts) + 1e-8).log()).sum(-1).mean().item()
            dominant  = p_flat.argmax(-1)
            max_load  = dominant.bincount(minlength=self.n_experts).float().max().item() / p_flat.shape[0]
            self._cached_metrics = {
                "effective_n_experts": math.exp(H.mean().item()),
                "max_load": max_load,
                "kl_surr_boltz": kl,
                "output_norm": out.norm(dim=-1).mean().item(),
            }

    def get_metrics(self):
        return self._cached_metrics


def interleave_up_gate_tensor_for_mlp(up_weight: torch.Tensor, gate_weight: torch.Tensor) -> torch.Tensor:
    return torch.cat([up_weight, gate_weight])


def split_up_gate_tensor_for_mlp(c_fc_weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return c_fc_weight.chunk(2)
