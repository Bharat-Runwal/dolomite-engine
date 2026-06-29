# **************************************************
# Composable Energy-FF class hierarchy (2026-06-28 refactor)
# **************************************************
"""Plug-and-play feedforward-energy classes.

Design (replaces the per-variant duplication in mlp.py):

    FFEnergyBase                        — abstract: every FF energy module
                                          exposes
                                            forward(x)           -> [..., hidden]
                                                                    (descent gradient
                                                                     of E_FF w.r.t. h,
                                                                     consumed by EnergyBlock
                                                                     as `ffwd_out`)
                                            energy_per_token(x)  -> [..., ]
                                                                    (E_FF(h) for action /
                                                                     descent-loss aux)
                                          and the leaf-cache contract
                                            _capture_energy : bool
                                            _last_energy_per_token : Tensor | None
                                          that EnergyBlock relies on under FSDP-2.
                                          ``forward`` always sets the cache; outer
                                          callers don't have to call
                                          ``energy_per_token`` separately.

    W1W2FFEnergy(FFEnergyBase)          — E = -(gelu(W1 h) · W2 h)
                                          (unbounded below; legacy Energy_MLP form;
                                          phi/phi' selectable via gelu_grad_method).
    HopfieldFFEnergy(FFEnergyBase)      — E = (1/d_int) ||gelu(W h)||²
                                          (≥ 0; single shared W; legacy
                                          Hopfield_Energy_MLP form).

    BoltzmannMoEFFEnergy(FFEnergyBase)  — composable MoE wrapper. Takes a
                                          *list* of FFEnergyBase experts (any
                                          subclass; mix and match), routes them
                                          via softmax(-E_k/τ), aggregates with
                                          either Boltzmann free-energy
                                          ``E_total = -τ·LSE_k(-E_k/τ)`` (when the
                                          experts give E ≥ 0 — Hopfield) or
                                          ``log Σ_k exp(E_k/τ)`` (Boltzmann
                                          partition matching the validated W1W2-MoE
                                          form). Adds optional stochastic
                                          repulsion on expert outputs. The
                                          repulsion + τ + n_repulsion_pairs
                                          machinery that was missing from the
                                          standalone ``BoltzmannMoE_Hopfield_Energy_MLP``
                                          is back in by composition.

Factory helpers ``make_w1w2_experts`` and ``make_hopfield_experts`` build the
K expert list cheaply (fused projections, zero-copy views into shared weights)
so the MoE has the same parameter count and the same flop budget as the legacy
single-class implementations. Numerical equivalence with the legacy classes at
identical weights is part of the test suite (see
``projects/EGPT-RL/scripts/smoke_energy_ff_refactor_20260628.py``).
"""

from __future__ import annotations

import itertools
import math
import random
from typing import Iterable, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss import add_aux_loss
from ...parameter import mark_parameter_as_mup_learning_rate
from ..linear import ParameterizedLinear


_SIGMOID_SCALE: float = (2.0 / math.pi) ** 0.5


def _get_std_for_linear(initializer_range: float, init_method: str, m_width: float | None) -> float:
    std = initializer_range
    if init_method == "mup":
        std /= math.sqrt(m_width)
    elif init_method != "normal":
        raise ValueError(f"unexpected init_method ({init_method})")
    return std


def _gelu_and_grad(x: torch.Tensor, method: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``(phi, phi')`` for the requested GELU-derivative convention.

    Three branches — identical numerics to ``BoltzmannMoE_Energy_MLP._SIGMOID_SCALE``
    code path in mlp.py so that pre-2026-06 checkpoints reproduce bit-identically.
    """
    if method == "erf_exact":
        phi = F.gelu(x)
        inv_sqrt_2 = 0.7071067811865476
        inv_sqrt_2pi = 0.3989422804014327
        phi_prime = (0.5 * (1.0 + torch.erf(x * inv_sqrt_2))
                     + x * torch.exp(-0.5 * x * x) * inv_sqrt_2pi)
        return phi, phi_prime
    if method == "tanh_exact":
        t = torch.tanh(_SIGMOID_SCALE * x)
        phi = 0.5 * x * (1.0 + t)
        phi_prime = 0.5 * (1.0 + t) + 0.5 * _SIGMOID_SCALE * x * (1.0 - t * t)
        return phi, phi_prime
    # "sigmoid" (legacy default)
    phi = F.gelu(x)
    phi_prime = torch.sigmoid(_SIGMOID_SCALE * x) * 0.5
    return phi, phi_prime


# --------------------------------------------------------------------------- #
# Abstract base                                                               #
# --------------------------------------------------------------------------- #


class FFEnergyBase(nn.Module):
    """Abstract base for plug-and-play FF energy modules.

    Sub-classes must implement:

    * ``forward(x) -> Tensor``     — same shape as ``x`` along all leading dims,
                                     last dim = ``hidden_size``. This is the
                                     descent gradient ``∇_h E_FF`` that
                                     ``EnergyBlock`` consumes as ``ffwd_out``.
                                     The implementation MUST set
                                     ``self._last_energy_per_token`` to
                                     ``E_FF(h)`` (shape ``[..., ]``) when
                                     ``self.training and self._capture_energy``
                                     is True, and to ``None`` otherwise — this
                                     is the FSDP-2-safe capture contract used by
                                     ``EnergyBlock`` and ``mixins/dense/base.py``.

    * ``energy_per_token(x) -> Tensor`` — recompute ``E_FF(h)`` from scratch.
                                     Used by callers that already have gathered
                                     parameters (e.g. the standalone-PyTorch
                                     smoke loop) and want energy without doing
                                     the full descent-gradient forward.
    """

    hidden_size: int
    intermediate_size: int

    def __init__(self) -> None:
        super().__init__()
        self._capture_energy: bool = False
        self._last_energy_per_token: torch.Tensor | None = None
        self._cached_metrics: dict[str, float] | None = None

    def get_metrics(self) -> dict[str, float] | None:
        return self._cached_metrics

    # Default: no-op. Subclasses with expensive metrics override this.
    def _log_metrics(self, out: torch.Tensor) -> None:
        if torch.compiler.is_compiling():
            return
        with torch.no_grad():
            self._cached_metrics = {"output_norm": out.norm(dim=-1).mean().item()}


# --------------------------------------------------------------------------- #
# Concrete experts                                                            #
# --------------------------------------------------------------------------- #


class W1W2FFEnergy(FFEnergyBase):
    """E_FF(h) = (1/√d_int) · -gelu(W1 h)·(W2 h) — W1/W2 form with init-normalised energy.

    **Init-scale convention.** A ``1/√d_int`` prefactor is folded into the
    energy itself so ``E ~ O(1)`` at init regardless of ``intermediate_size``.
    For random W1, W2 (std ~ 1/√d_hidden), the dot product ``gelu(W1 h)·(W2 h)``
    sums d_int near-independent O(1) terms → magnitude ~√d_int; the prefactor
    cancels that. ``∇_h E`` is scaled by the same factor. This unifies the
    W1/W2 line with the Hopfield line (which uses 1/d_int because it sums
    d_int squares of O(1) values) so the BoltzmannMoEFFEnergy wrapper does
    NOT need its own routing scale and ``temperature=1`` is the natural
    default for both. Relative E_FF↔E_AT magnitude is set by the learnable
    ``scale_ff`` parameter, not by the operator definition.

    Relation to legacy: the legacy ``BoltzmannMoE_Energy_MLP`` applied
    ``1/√expert_I`` at the routing layer (``_routing_scale``); the legacy
    ``Energy_MLP`` had no scale. This class moves the factor into the
    energy. Old checkpoints (Energy_MLP / BoltzmannMoE_Energy_MLP) load
    unchanged via the legacy path.

    ``forward(x)`` returns ``∇_h E_FF``:
        ∇_h E_FF = (1/√d_int) · [ W2ᵀ gelu(W1 h) + W1ᵀ (φ'(W1 h) ⊙ (W2 h)) ]
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        init_method: str = "normal",
        initializer_range: float = 0.02,
        m_width: float | None = None,
        num_layers: int = 1,
        add_bias: bool = False,
        gelu_grad_method: str = "sigmoid",
        layer_idx: int | None = None,
        # absorb upstream kwargs (activation_function, dropout) without using them
        **_unused: object,
    ) -> None:
        super().__init__()
        assert gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)

    # --- public surface --------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W1x = self.W1(x)
        W2x = self.W2(x)
        phi, phi_prime = _gelu_and_grad(W1x, self.gelu_grad_method)

        # ∇_h E = (1/√d_int) · [ W2ᵀ phi(W1h) + W1ᵀ (phi'(W1h) ⊙ (W2h)) ]
        # The (1/√d_int) prefactor lives in the energy definition itself —
        # see class docstring for the init-scale rationale.
        inv_sqrt_d = self.intermediate_size ** -0.5
        term1 = phi @ self.W2.weight
        term2 = (phi_prime * W2x) @ self.W1.weight
        out = inv_sqrt_d * (term1 + term2)

        if self.training and self._capture_energy:
            self._last_energy_per_token = -inv_sqrt_d * (phi * W2x).sum(dim=-1)
        else:
            self._last_energy_per_token = None

        if not torch.compiler.is_compiling():
            self._log_norms(out)
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W1x = self.W1(x)
        W2x = self.W2(x)
        phi, _ = _gelu_and_grad(W1x, self.gelu_grad_method)
        inv_sqrt_d = self.intermediate_size ** -0.5
        return -inv_sqrt_d * (phi * W2x).sum(dim=-1)

    # --- private ---------------------------------------------------------- #

    def _log_norms(self, out: torch.Tensor) -> None:
        with torch.no_grad():
            w1_norm = self.W1.weight.norm().item()
            w2_norm = self.W2.weight.norm().item()
            self._cached_metrics = {
                "W1_norm": w1_norm,
                "W2_norm": w2_norm,
                "W_total_norm": math.sqrt(w1_norm ** 2 + w2_norm ** 2),
                "output_norm": out.norm(dim=-1).mean().item(),
            }


class HopfieldFFEnergy(FFEnergyBase):
    """E_FF(h) = (1/d_int) ||gelu(W h)||² — bounded below by 0 (legacy ``Hopfield_Energy_MLP``).

    ``forward(x)`` returns ``∇_h E_FF``:
        ∇_h E_FF = (2/d_int) Wᵀ (gelu(W h) ⊙ gelu'(W h))

    Single shared weight ``W`` of shape ``[intermediate, hidden]`` — half the
    params of ``W1W2FFEnergy``. To match params, double ``intermediate_size``
    at config time. MEAN form (1/d_int) is scale-invariant — without it the
    descent step grows with ``intermediate_size`` and training NaN's
    (validated empirically in EGPT-RL run 1714840).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        init_method: str = "normal",
        initializer_range: float = 0.02,
        m_width: float | None = None,
        num_layers: int = 1,
        add_bias: bool = False,
        gelu_grad_method: str = "sigmoid",
        layer_idx: int | None = None,
        **_unused: object,
    ) -> None:
        super().__init__()
        assert gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        self.layer_idx = layer_idx

        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        Wx = self.W(x)
        gelu_Wx, gelu_prime = _gelu_and_grad(Wx, self.gelu_grad_method)
        # NOTE: legacy ``Hopfield_Energy_MLP`` used the *sigmoid approx* gelu'
        # divided by 0.5 implicitly (it just took ``sigmoid(c·Wx)`` without the
        # 0.5 factor). To stay bit-equivalent with that class for
        # gelu_grad_method="hopfield_legacy_no_half" we'd branch; instead we
        # provide a faithful 0.5-scaled phi' that matches the W1W2 case AND
        # adjust the (2/d_int) constant to (4/d_int) so the gradient magnitude
        # is the same as legacy. See ``_HOPFIELD_LEGACY_FACTOR`` for context.
        inv_d = 1.0 / self.intermediate_size
        gated = gelu_Wx * gelu_prime  # phi' has its own 0.5 factor inside _gelu_and_grad
        out = (4.0 * inv_d) * (gated @ self.W.weight)

        if self.training and self._capture_energy:
            self._last_energy_per_token = (gelu_Wx ** 2).mean(dim=-1)
        else:
            self._last_energy_per_token = None

        if not torch.compiler.is_compiling():
            self._log_norms(out)
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        Wx = self.W(x)
        return (F.gelu(Wx) ** 2).mean(dim=-1)

    def _log_norms(self, out: torch.Tensor) -> None:
        with torch.no_grad():
            w_norm = self.W.weight.norm().item()
            self._cached_metrics = {
                "W_norm": w_norm,
                "W_total_norm": w_norm,
                "output_norm": out.norm(dim=-1).mean().item(),
            }


# --------------------------------------------------------------------------- #
# Composable Boltzmann MoE                                                    #
# --------------------------------------------------------------------------- #


class BoltzmannMoEFFEnergy(FFEnergyBase):
    """Composable Boltzmann-mixture wrapper over a list of expert FFEnergyBase modules.

    Per-expert energy ``E_k(h)`` and gradient ``∇_h E_k(h)`` are supplied by the
    expert class — pluggable between any subclass of ``FFEnergyBase``. The
    wrapper handles:

      * Boltzmann routing weights ``w_k = softmax(s_k / τ)`` where ``s_k`` is
        either ``-E_k`` (Hopfield, ``e_sign="neg"``) or ``+E_k`` (W1W2,
        ``e_sign="pos"``). The choice mirrors the convention each legacy class
        used for its softmax argument.
      * Total energy:
          ``e_sign="neg"`` (Hopfield):
              E_total = -τ · LSE_k(-E_k / τ)
          ``e_sign="pos"`` (W1W2, matches legacy ``BoltzmannMoE_Energy_MLP``):
              E_total = -τ · LSE_k( E_k / τ)
      * Stochastic repulsion (cosine-similarity penalty over sampled expert
        output pairs) — the feature that was missing from the
        ``BoltzmannMoE_Hopfield_Energy_MLP`` regression.
      * Optional sparse top-k truncation (matches legacy).

    Param + flop budget: identical to the legacy ``BoltzmannMoE_*_Energy_MLP``
    classes when constructed via the ``make_*_experts`` factories below, which
    share a single fused weight tensor across experts.
    """

    def __init__(
        self,
        experts: Sequence[FFEnergyBase],
        *,
        hidden_size: int,
        temperature: float = 1.0,
        repulsion_coef: float = 0.0,
        n_repulsion_pairs: int = 4,
        top_k: int | None = None,
        e_sign: str = "neg",
        layer_idx: int | None = None,
    ) -> None:
        super().__init__()
        assert len(experts) >= 2, "BoltzmannMoEFFEnergy requires at least 2 experts"
        assert e_sign in ("neg", "pos")
        assert temperature > 0
        self.experts = nn.ModuleList(experts)
        self.n_experts = len(experts)
        self.hidden_size = hidden_size
        self.intermediate_size = sum(e.intermediate_size for e in experts)
        self.temperature = float(temperature)
        self.repulsion_coef = float(repulsion_coef)
        self.n_repulsion_pairs = int(n_repulsion_pairs)
        self.top_k = top_k
        self.e_sign = e_sign
        self.layer_idx = layer_idx
        self._all_pairs: list[tuple[int, int]] = list(
            itertools.combinations(range(self.n_experts), 2)
        )

    # --- public surface --------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Collect each expert's descent gradient AND its per-token energy.
        # We do this with the cache-flag flipped on so the experts populate
        # ``_last_energy_per_token`` regardless of self.training — the wrapper
        # is the only thing that ever reads them, so the contract is local.
        expert_outs = []   # [(..., hidden), ...]  length n_experts
        expert_es = []     # [(..., ), ...]
        for expert in self.experts:
            prev = expert._capture_energy
            expert._capture_energy = True
            try:
                out_k = expert(x)
            finally:
                expert._capture_energy = prev
            # During eval, expert.forward() may have left
            # _last_energy_per_token = None (because it gates on self.training).
            # Fall back to a fresh recomputation.
            e_k = expert._last_energy_per_token
            if e_k is None:
                e_k = expert.energy_per_token(x)
            expert_outs.append(out_k)
            expert_es.append(e_k)

        # E_k stacked along a new "expert" dim: shape (..., n_experts)
        E_k = torch.stack(expert_es, dim=-1)
        # Logits for the Boltzmann softmax. ``e_sign="neg"`` (Hopfield) makes
        # softmax(-E_k/τ) favor the LOWEST-energy expert (consistent with
        # "lower energy = better"). ``e_sign="pos"`` matches the validated
        # W1W2-MoE class which computed softmax(+E_k/τ) on its (negative)
        # routing energy — same effect, different sign convention.
        logits = (-E_k if self.e_sign == "neg" else E_k) / self.temperature
        p = F.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < self.n_experts:
            _, topk_idx = logits.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p * mask  # sparse Boltzmann approx; sum < 1 intentionally

        # Aggregate gradients: ∇_h E_total = Σ_k w_k · ∇_h E_k.
        expert_grads = torch.stack(expert_outs, dim=-2)        # (..., n_experts, hidden)
        out = torch.einsum("...e,...eh->...h", p, expert_grads)

        # E_total per token: -τ · LSE_k(logits) — same sign convention as
        # the legacy classes.
        if self.training and self._capture_energy:
            self._last_energy_per_token = -self.temperature * torch.logsumexp(logits, dim=-1)
        else:
            self._last_energy_per_token = None

        if self.training and self.repulsion_coef > 0:
            self._add_repulsion_loss(expert_grads)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        e_list = [expert.energy_per_token(x) for expert in self.experts]
        E_k = torch.stack(e_list, dim=-1)
        logits = (-E_k if self.e_sign == "neg" else E_k) / self.temperature
        return -self.temperature * torch.logsumexp(logits, dim=-1)

    # --- private ---------------------------------------------------------- #

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        """Cosine-similarity repulsion on random expert output pairs."""
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)
        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, k)
        i_idx = [p[0] for p in sampled]
        j_idx = [p[1] for p in sampled]
        cos_sim = (eg_norm[:, i_idx, :] * eg_norm[:, j_idx, :]).sum(-1).mean()
        add_aux_loss(self.repulsion_coef * cos_sim)

    def _log_metrics(self, p: torch.Tensor, out: torch.Tensor) -> None:
        with torch.no_grad():
            p_flat = p.reshape(-1, self.n_experts)
            max_H = math.log(self.n_experts) if self.n_experts > 1 else 1.0
            per_token_H = -(p_flat * (p_flat + 1e-8).log()).sum(-1)
            mean_token_H = per_token_H.mean().item()
            effective_n = math.exp(mean_token_H)
            dominant = p_flat.argmax(-1)
            counts = dominant.bincount(minlength=self.n_experts).float()
            n_dominant = int((counts > 0).sum().item())
            max_load = (counts / p_flat.shape[0]).max().item()

            self._cached_metrics = {
                "effective_n_experts": effective_n,
                "n_dominant_experts": float(n_dominant),
                "max_expert_load": max_load,
                "mean_token_entropy_norm": mean_token_H / max_H,
                "output_norm": out.norm(dim=-1).mean().item(),
            }


# --------------------------------------------------------------------------- #
# Expert-list factories                                                       #
# --------------------------------------------------------------------------- #
#
# These produce a list of expert FFEnergyBase modules backed by a SHARED fused
# weight tensor (zero-copy chunked views), so a K-expert MoE has the same
# param/flop budget as the legacy single-fused-W1/W2 implementation.


class _W1W2Expert(FFEnergyBase):
    """View-backed W1W2 expert — shares fused W1/W2 with siblings via slices.

    Constructed only by ``make_w1w2_experts``. Do not instantiate directly —
    the slice setup is fragile (must keep a Python ref to the shared tensors
    so PyTorch sees the experts as parameter-less sub-modules with the master
    weight registered on the parent factory holder).
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        W1_slice: torch.Tensor,        # rows [start:end] of fused W1
        W2_slice: torch.Tensor,
        gelu_grad_method: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        # Stored as non-parameters; the fused parent holds the actual Parameters.
        # We reference via the parent through closure (held in ``self._get_W1``);
        # but stashing a direct view is fine for forward — slices are recomputed
        # each forward to follow Parameter updates.
        self._W1_slice = W1_slice
        self._W2_slice = W2_slice

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1/√expert_I prefactor — see W1W2FFEnergy class docstring. Folded
        # into the energy itself so the MoE wrapper doesn't need to know
        # about routing-scale conventions; τ=1 is the natural default.
        W1 = self._W1_slice()  # callable that returns the current view
        W2 = self._W2_slice()
        W1x = x @ W1.t()
        W2x = x @ W2.t()
        phi, phi_prime = _gelu_and_grad(W1x, self.gelu_grad_method)
        inv_sqrt_d = self.intermediate_size ** -0.5
        out = inv_sqrt_d * (phi @ W2 + (phi_prime * W2x) @ W1)

        if self._capture_energy:
            self._last_energy_per_token = -inv_sqrt_d * (phi * W2x).sum(dim=-1)
        else:
            self._last_energy_per_token = None
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W1 = self._W1_slice()
        W2 = self._W2_slice()
        W1x = x @ W1.t()
        W2x = x @ W2.t()
        phi, _ = _gelu_and_grad(W1x, self.gelu_grad_method)
        inv_sqrt_d = self.intermediate_size ** -0.5
        return -inv_sqrt_d * (phi * W2x).sum(dim=-1)


class _HopfieldExpert(FFEnergyBase):
    """View-backed Hopfield expert — shares fused W with siblings via slices."""

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        W_slice,  # callable
        gelu_grad_method: str,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        self._W_slice = W_slice

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._W_slice()
        Wx = x @ W.t()
        gelu_Wx, gelu_prime = _gelu_and_grad(Wx, self.gelu_grad_method)
        inv_d = 1.0 / self.intermediate_size
        gated = gelu_Wx * gelu_prime
        out = (4.0 * inv_d) * (gated @ W)

        if self._capture_energy:
            self._last_energy_per_token = (gelu_Wx ** 2).mean(dim=-1)
        else:
            self._last_energy_per_token = None
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W = self._W_slice()
        Wx = x @ W.t()
        return (F.gelu(Wx) ** 2).mean(dim=-1)


class _FusedW1W2Holder(nn.Module):
    """Holds the fused W1/W2 weights for a W1W2-MoE wrapper.

    Children are ``_W1W2Expert`` views, all sharing the parent's two
    ``ParameterizedLinear`` weights via row-slices. The holder itself does
    nothing in forward — the MoE wrapper calls each expert in turn.
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        init_method: str,
        initializer_range: float,
        m_width: float | None,
        add_bias: bool,
        gelu_grad_method: str,
    ) -> None:
        super().__init__()
        assert intermediate_size % n_experts == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W1 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        self.W2 = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W1.weight)
        mark_parameter_as_mup_learning_rate(self.W2.weight)
        self.gelu_grad_method = gelu_grad_method

    def make_experts(self) -> list[FFEnergyBase]:
        experts: list[FFEnergyBase] = []
        for k in range(self.n_experts):
            lo, hi = k * self.expert_I, (k + 1) * self.expert_I
            # Closures keep the experts pointing at the live (possibly updated
            # by FSDP-gather) Parameter rather than a stale view.
            W1_slice = (lambda lo=lo, hi=hi: self.W1.weight[lo:hi])
            W2_slice = (lambda lo=lo, hi=hi: self.W2.weight[lo:hi])
            experts.append(
                _W1W2Expert(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.expert_I,
                    W1_slice=W1_slice,
                    W2_slice=W2_slice,
                    gelu_grad_method=self.gelu_grad_method,
                )
            )
        return experts


class _FusedHopfieldHolder(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        n_experts: int,
        init_method: str,
        initializer_range: float,
        m_width: float | None,
        add_bias: bool,
        gelu_grad_method: str,
    ) -> None:
        super().__init__()
        assert intermediate_size % n_experts == 0
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.n_experts = n_experts
        self.expert_I = intermediate_size // n_experts
        std = _get_std_for_linear(initializer_range, init_method, m_width)
        self.W = ParameterizedLinear(hidden_size, intermediate_size, bias=add_bias, std=std)
        mark_parameter_as_mup_learning_rate(self.W.weight)
        self.gelu_grad_method = gelu_grad_method

    def make_experts(self) -> list[FFEnergyBase]:
        experts: list[FFEnergyBase] = []
        for k in range(self.n_experts):
            lo, hi = k * self.expert_I, (k + 1) * self.expert_I
            W_slice = (lambda lo=lo, hi=hi: self.W.weight[lo:hi])
            experts.append(
                _HopfieldExpert(
                    hidden_size=self.hidden_size,
                    intermediate_size=self.expert_I,
                    W_slice=W_slice,
                    gelu_grad_method=self.gelu_grad_method,
                )
            )
        return experts


class FusedMoEContainer(FFEnergyBase):
    """Container that holds a fused-weight expert pool + a BoltzmannMoEFFEnergy.

    This is what gets registered as ``self.ffwd`` on the energy block when
    the user picks ``EnergyFF_BoltzmannMoE`` — exposes the standard
    ``forward / energy_per_token`` interface but internally delegates to the
    composed MoE.
    """

    def __init__(
        self,
        *,
        expert_holder: nn.Module,
        moe: BoltzmannMoEFFEnergy,
    ) -> None:
        super().__init__()
        self.expert_holder = expert_holder
        self.moe = moe
        self.hidden_size = moe.hidden_size
        self.intermediate_size = moe.intermediate_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Propagate capture flag down to the MoE, which propagates to experts.
        self.moe._capture_energy = self._capture_energy
        out = self.moe(x)
        self._last_energy_per_token = self.moe._last_energy_per_token
        self._cached_metrics = self.moe.get_metrics()
        return out

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        return self.moe.energy_per_token(x)


def build_boltzmann_moe(
    *,
    expert_kind: str,     # "w1w2" or "hopfield"
    hidden_size: int,
    intermediate_size: int,
    n_experts: int,
    temperature: float = 1.0,
    repulsion_coef: float = 0.0,
    n_repulsion_pairs: int = 4,
    top_k: int | None = None,
    init_method: str = "normal",
    initializer_range: float = 0.02,
    m_width: float | None = None,
    add_bias: bool = False,
    gelu_grad_method: str = "sigmoid",
    layer_idx: int | None = None,
) -> FusedMoEContainer:
    """Factory: composable Boltzmann-MoE over W1W2 or Hopfield experts.

    Returns a FusedMoEContainer registering a single fused-weight holder + a
    BoltzmannMoEFFEnergy wrapper. The wrapper picks ``e_sign`` to match each
    base class's convention (W1W2 → +E_k logits; Hopfield → -E_k logits).
    """
    if expert_kind == "w1w2":
        holder = _FusedW1W2Holder(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            n_experts=n_experts, init_method=init_method,
            initializer_range=initializer_range, m_width=m_width,
            add_bias=add_bias, gelu_grad_method=gelu_grad_method,
        )
        e_sign = "pos"
    elif expert_kind == "hopfield":
        holder = _FusedHopfieldHolder(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            n_experts=n_experts, init_method=init_method,
            initializer_range=initializer_range, m_width=m_width,
            add_bias=add_bias, gelu_grad_method=gelu_grad_method,
        )
        e_sign = "neg"
    else:
        raise ValueError(f"unknown expert_kind ({expert_kind})")
    experts = holder.make_experts()
    moe = BoltzmannMoEFFEnergy(
        experts,
        hidden_size=hidden_size,
        temperature=temperature,
        repulsion_coef=repulsion_coef,
        n_repulsion_pairs=n_repulsion_pairs,
        top_k=top_k,
        e_sign=e_sign,
        layer_idx=layer_idx,
    )
    return FusedMoEContainer(expert_holder=holder, moe=moe)
