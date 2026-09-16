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



_HOPFIELD_GRAD_SCALES = ("mean", "inv_sqrt", "sqrt_consistent")


def _hopfield_grad_prefactor(intermediate_size: int, mode: str) -> float:
    """Prefactor on the Hopfield descent gradient ``W^T(gelu(Wh) . gelu'(Wh))``.

    ==========================================================================
    REVERT NOTE (2026-09-12) — IF A RUN DIVERGES, SET ``hopfield_grad_scale:
    "mean"`` IN THE CONFIG. That is the pre-2026-09-12 behaviour, bit-for-bit.
    ==========================================================================

    Why this knob exists. Measured on
    ``math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu``:
    ``||ffwd_out|| = 0.0050`` against ``||attn_out|| = 18.53``, i.e. the energy-FF
    branch supplied 0.04% of ``grad_E`` and deleting it entirely moved perplexity by
    +0.0003 over 87,997 tokens. The branch was inert.

    WHAT THE "mean" FORM ACTUALLY DOES TO SCALING. For W with iid entries of std
    sigma and ||h|| = sqrt(d):
        ||gated||        ~ sigma sqrt(I_e d)
        ||gated @ W||    ~ sigma^2 d sqrt(I_e)
    so the *descent step* scales as ``prefactor * sqrt(I_e)``:

        prefactor      step vs I_e        status
        2/I_e (mean)   ~ 1/sqrt(I_e)      DECAYS with width   <- current default
        1/sqrt(I_e)    ~ const            width-INVARIANT
        2   (sum)      ~ sqrt(I_e)        GROWS with width     <- this diverged

    So "mean" makes the ENERGY width-invariant (correct: E sums I_e positive
    squares, so 1/I_e is the right O(1) normalisation) but over-corrects the
    GRADIENT, which then decays as 1/sqrt(I_e).

    SAFETY MARGIN VS THE KNOWN DIVERGENCE. Run 1714840 went NaN at step 3710 using
    the SUM form (no 1/d_int at all) at d_int=8192, LR 7.5e-4 — see the
    ``Hopfield_Energy_MLP`` docstring in mlp.py:153 and the header of
    ``configs/multi_block_ablation/math_fet_hopfield_mean_*.yml``. At the MoE's
    per-expert width I_e=1024 the sum-form prefactor would be 4.0. Relative to that:
        "mean"            4/1024  = 0.0039   1024x below sum
        "inv_sqrt"        1/32    = 0.03125   128x below sum   (8x above mean)
        "sqrt_consistent" 4/32    = 0.125      32x below sum  (32x above mean)
    Both new options keep a large margin below the configuration that diverged, and
    both restore the width-invariance the original fix was reaching for. This is
    NOT a revert of that fix.

    WHAT THIS KNOB DOES NOT FIX. It does not touch the ENERGY, deliberately:
    ``E = (1/I_e)||gelu(Wh)||^2`` sums I_e positive terms, so 1/I_e is the correct
    O(1) scale and inflating it to 1/sqrt(I_e) would make E grow as sqrt(I_e) —
    the same direction as the router saturation that commit 16500e8 ("scale E_i by
    1/sqrt(expert_I)") was introduced to cure. The flat routing
    (effective_n_experts 7.999/8, E ~ 0.0126 against tau=1) is a SEPARATE defect and
    is fixed by ``routing_norm``, which is scale-free and therefore immune to the
    weight-norm drift that caused it (||W||_F fell 70.9 -> 20.5 over training).

    CONSISTENCY CAVEAT. With "inv_sqrt" or "sqrt_consistent" the returned vector is
    no longer exactly ``grad`` of the ``E`` that ``energy_per_token`` reports — it is
    that gradient times sqrt(I_e)/4 or sqrt(I_e) respectively. Harmless while
    ``energy_descent_loss_coef`` and ``energy_action_loss_coef`` are 0 (as in every
    current config), but it MATTERS if either is switched on, because those losses
    assume ffwd_out == grad_h E.

    MAGNITUDE EXPECTATION — READ BEFORE ASSUMING THIS IS SUFFICIENT. "inv_sqrt" is
    only 8x and "sqrt_consistent" only 32x above the current default. Applied to the
    measured checkpoint that lifts the FF share of grad_E from 0.04% to roughly 0.3%
    or 1.3% — i.e. to about the level of the (also weak) non-MoE hopfield_mean
    sibling at 1.57%, NOT to the 85.8% of the healthy w1w2 line. The remaining ~32x
    sits in the weight norms, which shrank under weight decay precisely BECAUSE the
    branch was inert. Whether a from-scratch run escapes that feedback loop with 8-32x
    more gradient signal is an empirical question — hence the smoke tests.
    """
    if mode == "mean":
        return 4.0 / intermediate_size          # PRE-2026-09-12 DEFAULT; revert here
    if mode == "inv_sqrt":
        return intermediate_size ** -0.5
    if mode == "sqrt_consistent":
        return 4.0 * intermediate_size ** -0.5
    raise ValueError(f"unknown hopfield_grad_scale ({mode})")


def _repulsion_penalty(cos: torch.Tensor, form: str) -> torch.Tensor:
    """Scalar penalty from pairwise expert-output cosines.

    "signed" is minimised at cos = -1 and therefore rewards ANTI-alignment, not
    diversity. Under near-uniform routing the anti-aligned expert gradients then
    cancel in sum_k p_k g_k and the whole FF branch collapses (measured: 271x
    smaller ||ffwd_out|| than the no-repulsion sibling). Prefer "squared".

    LAMBDA CALIBRATION: the existing repulsion_coef sweeps (B4/B5 found 0.1 good,
    the FET runs used 0.01) were tuned under "signed". At |cos| ~ 0.3 the squared
    form is ~3x weaker than signed/abs for the same lambda, since cos^2 vanishes
    quadratically near orthogonality. So "abs" is the drop-in replacement that
    preserves those lambda values, while "squared" is smoother but wants lambda
    scaled up correspondingly. Re-sweep lambda when switching.
    """
    if form == "squared":
        return (cos ** 2).mean()
    if form == "abs":
        return cos.abs().mean()
    if form == "hinge":
        return F.relu(cos).mean()
    if form == "signed":
        return cos.mean()          # LEGACY, mis-specified; see docstring
    raise ValueError(f"unknown repulsion_form ({form})")


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


# EMA momentum for the running Sinkhorn dual. 0.01 => ~100-step time constant, fast
# enough to converge inside any real run and inside a short calibration pass.
_SINKHORN_MU_MOMENTUM = 0.01


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
        hopfield_grad_scale: str = "mean",
        **_unused: object,
    ) -> None:
        super().__init__()
        assert gelu_grad_method in ("sigmoid", "tanh_exact", "erf_exact")
        assert hopfield_grad_scale in _HOPFIELD_GRAD_SCALES
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        self.hopfield_grad_scale = hopfield_grad_scale
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
        # PREFACTOR: was hardcoded (4.0 / intermediate_size) before 2026-09-12.
        # REVERT by setting hopfield_grad_scale="mean". See _hopfield_grad_prefactor.
        pref = _hopfield_grad_prefactor(self.intermediate_size, self.hopfield_grad_scale)
        gated = gelu_Wx * gelu_prime  # phi' has its own 0.5 factor inside _gelu_and_grad
        out = pref * (gated @ self.W.weight)

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

    # Hard bound on the balancing bias. The logits it is added to are z-scored and hence
    # O(1), so an unbounded bias silently takes over routing: the first version of this
    # update was unclamped, reached |bias| = 1482, and destabilised the one arm that
    # enabled balancing (loss 4.05 -> 5.18 with 77 upward jumps).
    _BIAS_MAX: float = 1.0

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
        repulsion_form: str = "squared",
        routing_norm: str = "none",
        renormalize_topk: bool = False,
        track_load: bool = True,
        balance_rate: float = 0.0,
        repulsion_interval: int = 1,
        repulsion_scale_comp: bool = True,
        fused_experts: bool = False,
        fused_spec: dict | None = None,
        proxy_rank: int = 0,
        proxy_loss_coef: float = 0.0,
        proxy_route: bool = False,
        proxy_kind: str = "quad",
        proxy_out_dim: int = 0,
        proxy_iters: int = 1,
        cos_probe_interval: int = 0,
        cos_probe_pairs: int = 8,
        repulsion_space: str = "output",
        sinkhorn_iters: int = 0,
        sinkhorn_persist_mu: bool = False,
        sinkhorn_mu_iters: int = 1,
        sparse_backproj: bool = False,
        sparse_forward: bool = False,
        sparse_candidates: int = 0,
        sparse_capacity_factor: float = 1.25,
        repulsion_subsample: int = 0,
        repulsion_tensor_idx: bool = False,
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
        assert repulsion_form in ("squared", "abs", "hinge", "signed")
        self.repulsion_form = repulsion_form
        assert routing_norm in ("none", "zscore", "sqrt_width")
        self.routing_norm = routing_norm
        self.renormalize_topk = bool(renormalize_topk)

        # ------------------------------------------------------------------ #
        # ROUTING-LOAD TRACKING and AUX-LOSS-FREE BALANCING                  #
        # ------------------------------------------------------------------ #
        # Two problems this addresses, found 2026-09-13 by auditing checkpoints:
        #
        # (a) WE WERE FLYING BLIND. effective_n_experts / max_expert_load are computed in
        #     _log_metrics behind `if not torch.compiler.is_compiling()`, because they use
        #     .item(). Every arm trains with torch_compile: true, so that branch is traced
        #     away and the metrics NEVER reached wandb. Collapse was therefore invisible
        #     during training and only found by an offline probe.
        # (b) REPULSION IS THE WRONG LEVER. It penalises expert-WEIGHT cosine similarity,
        #     and it works: measured cos_mean <= 0.021 with zero dead experts on every arm.
        #     Yet the pure single-block arms route 99% of inputs to ONE expert
        #     (effective_n_experts 1.38 of 16, against 4.64 for the hybrids at identical
        #     expert width). Diverse weights do not imply diverse routing.
        #
        # The buffers below are updated with pure tensor ops and no .item(), so they trace
        # cleanly under torch.compile and give load statistics every step at negligible
        # cost. The trainer reads and resets them outside the graph.
        #
        # `load_balance_bias` implements DeepSeek-V3-style AUX-LOSS-FREE balancing: a
        # per-expert additive bias on the routing logits, nudged (under no_grad, outside
        # autograd) toward whichever experts are under-loaded. It is NOT a loss term, adds
        # no gradient pathway and no learned gate, so the paper's "no load-balancing loss
        # and no gate parameters" claim survives it. DEFAULT OFF: enabling it changes the
        # routing of every existing checkpoint, so it must be opted into per config.
        self.track_load = bool(track_load)
        self.balance_rate = float(balance_rate)
        # ------------------------------------------------------------------ #
        # ENERGY-MAGNITUDE TRACKING (2026-09-15)                              #
        # ------------------------------------------------------------------ #
        # Added because correcting the routing sign creates a POSITIVE FEEDBACK
        # path that the inverted sign did not have: the block ASCENDS the energy
        # (out = +grad E), and the corrected router now selects the HIGHEST-energy
        # experts -- so a step toward the best-matching expert raises its overlap,
        # which raises its energy, which enlarges the next step. The inverted sign
        # was self-limiting here too, for the same reason it self-balanced.
        # `ffwd/output_norm` alone cannot distinguish "the branch finally
        # contributes" from "the branch is running away", so track the ENERGY scale
        # directly. Pure tensor ops, no .item(), so it traces under torch.compile.
        self.register_buffer("_E_abs_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_E_abs_max", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_E_n", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_load_sum", torch.zeros(self.n_experts), persistent=False)
        self.register_buffer("_ent_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        self.register_buffer("_tok_sum", torch.zeros((), dtype=torch.float32), persistent=False)
        # REGISTER ONLY WHEN BALANCING IS ON. A persistent buffer adds a key to the
        # state_dict, and registering it unconditionally broke resume for EVERY existing
        # energy-MoE checkpoint:
        #     RuntimeError: Missing key in checkpoint state_dict:
        #                   state.model.transformer.h.0.ffwd.moe.load_balance_bias
        # (hit on slope90k_1blk resuming from its step-2000 checkpoint). Arms with
        # balance_rate=0 must keep exactly their original state_dict shape. Balancing arms
        # do want it persistent, since the bias is part of the trained routing rule.
        if self.balance_rate > 0.0:
            self.register_buffer("load_balance_bias", torch.zeros(self.n_experts), persistent=True)
        else:
            self.load_balance_bias = None
        self.top_k = top_k
        self.e_sign = e_sign
        self.layer_idx = layer_idx
        self._all_pairs: list[tuple[int, int]] = list(
            itertools.combinations(range(self.n_experts), 2)
        )

        # ------------------------------------------------------------------ #
        # TENSOR-INDEXED REPULSION SAMPLING (2026-09-15)                      #
        # ------------------------------------------------------------------ #
        # WHY. `_sample_pairs` uses `random.sample`, and the resulting PYTHON LISTS are
        # used to index tensors. dynamo cannot trace Python's `random` ("Attempted to
        # call function marked as skipped") and it specialises on the list VALUES, so
        # every new draw is a new graph. Measured with dynamo.explain at K=8:
        #     looped, no repulsion : 1 graph,  0 breaks,  1 frame  / 8 calls
        #     looped + repulsion   : 4 graphs, 3 breaks, 17 frames / 8 calls
        #     fused,  no repulsion : 1 graph,  0 breaks,  1 frame
        #     fused + repulsion    : 3 graphs, 2 breaks, 17 frames
        # So the FUSED GEMM is compile-clean and REPULSION is what breaks the graph --
        # in the existing looped path too. Past dynamo's cache-size limit that region
        # falls back to eager, a standing performance loss, and it is the leading
        # suspect for why `fused_experts` wedged a 2-node job.
        #
        # FIX. Draw with torch RNG into a TENSOR of pair indices. Tensor indices are
        # DATA, not graph constants, so there is nothing to specialise on and nothing to
        # recompile. Bonus: torch RNG is restored by activation checkpointing where
        # Python's `random` is not, so the forward and its recompute finally draw the
        # SAME pairs -- fixing a pre-existing inconsistency where the repulsion gradient
        # was computed against different pairs than the forward loss.
        #
        # DEFAULT OFF: it changes which pairs are drawn, so existing runs stay
        # bit-identical. Enable per config alongside `fused_experts`.
        self.repulsion_tensor_idx = bool(repulsion_tensor_idx)
        if self.repulsion_tensor_idx:
            self.register_buffer(
                "_pairs_t",
                torch.tensor(self._all_pairs, dtype=torch.long).reshape(-1, 2),
                persistent=False,
            )

        # ------------------------------------------------------------------ #
        # INTERMITTENT REPULSION (2026-09-15)                                #
        # ------------------------------------------------------------------ #
        # Repulsion measured +65% of the MoE block in the isolated microbench
        # (19.29 -> 11.70 ms/call without it). Firing it on ~1 call in `interval`
        # and scaling the coefficient by `interval` keeps E[repulsion gradient]
        # unchanged while paying the cost 1/interval of the time.
        # interval == 1 -> prob 1.0, coef x1 -> byte-identical to the old path,
        # so no existing config or checkpoint changes.
        # NOTE: the Phase-B profiler put repulsion at only ~2-4% of the FULL step
        # (the 65% was one isolated block-call), so this is a safe near-free win,
        # NOT the headline lever. The headline lever is `fused_experts` below.
        # ------------------------------------------------------------------ #
        # REPULSION SPACE (2026-09-15)                                        #
        # ------------------------------------------------------------------ #
        # "output" (default, as trained): cosine between per-token expert OUTPUTS.
        #   Cost scales with the token count -- measured 4.06 / 7.59 / 14.91 ms per
        #   call at N = 2048 / 4096 / 8192 -- because it needs the (N, K, hidden)
        #   output stack, and in the fused path that stack must be back-projected
        #   specially since the fused combine never materialises it.
        # "weight": cosine between the expert WEIGHT blocks. Has NO token dimension
        #   at all, so it is O(K * I_e * d) and INDEPENDENT of N -- measured 2.21 /
        #   2.20 / 2.33 ms at those same three N, i.e. 3.5x cheaper at N=4096 and
        #   6.4x at N=8192. It also needs no expert outputs, so unlike output
        #   repulsion it survives a future true-sparse MoE kernel.
        #
        # This is the option the cost analysis points at: at ~20% of the step,
        # weight-space repulsion collects most of that saving while keeping the
        # regulariser at FULL strength EVERY step -- whereas intermittent output
        # repulsion buys speed by weakening it (measured: expert alignment rises
        # 2.7-4.4x at 1 pair/step or 2 pairs 1-in-4).
        #
        # ⚠ THE COEFFICIENT DOES NOT TRANSFER. Weight cosines are far smaller than
        # output cosines -- measured on 134M final checkpoints, weight mean|cos| is
        # ~0.004 against ~0.021 for outputs, and the gap widens at 400M (~0.10
        # output). So `repulsion_coef` must be re-swept for this space; reusing 0.1
        # would apply a much weaker effective pressure. Whether keeping WEIGHTS
        # apart also keeps OUTPUTS apart is exactly what `cos_probe_interval`
        # (which always measures output-space) is there to answer.
        assert repulsion_space in ("output", "weight")
        self.repulsion_space = repulsion_space

        # ------------------------------------------------------------------ #
        # SINKHORN BALANCING (2026-09-15) — the EXACT dual of the capacity     #
        # constraint, i.e. the chemical potential solved rather than          #
        # controlled. See ROUTING_SIGN_BUG_20260915.md.                       #
        # ------------------------------------------------------------------ #
        # `balance_rate` reaches balance with a PROPORTIONAL CONTROL rule on a
        # per-expert bias, and in the corrected-sign sweep it worked but sat PINNED
        # at the +-1.0 `_BIAS_MAX` clamp in every arm (T4, T5) — so it was
        # delivering its result while saturated, and wanted to push harder. Raising
        # the clamp is the obvious move and a bad one: the bound exists because an
        # earlier unclamped sign()-based version reached |bias| = 1482 and
        # destabilised an arm (loss 4.05 -> 5.18, 77 upward jumps).
        #
        # Sinkhorn removes the multiplier entirely. Writing the routing distribution
        # as p_k ∝ exp((E_k - mu_k)/tau), the mu_k that equalise expert load are the
        # dual variables of the constraint sum_tokens p_k = N/K, and they are the
        # fixed point of a log-domain iteration:
        #     mu <- mu + log(load(mu) * K),   load(mu) = mean_tokens softmax(L - mu)
        # A handful of iterations converges. There is NO clamp, NO gain to tune, and
        # the solution is exact rather than lagged.
        #
        # THREE PROPERTIES THAT MATTER HERE:
        #  * Solved under no_grad, so mu is a constant w.r.t. differentiation —
        #    which is correct for a Lagrange multiplier, and keeps the "no gradient
        #    pathway / no auxiliary loss" property that `balance_rate` has.
        #  * Deterministic (no RNG), so activation checkpointing recomputes the
        #    identical mu and cannot trip the metadata check that the fused
        #    repulsion did.
        #  * TRAIN-ONLY. The load is a property of the batch, so applying this at
        #    inference would make routing depend on batch composition. Gated on
        #    self.training, which is the standard Sinkhorn-router choice and does
        #    introduce a train/inference mismatch — see the doc.
        # APPROXIMATION: the load is the LOCAL (per-rank) batch load. The true
        # constraint is global across data-parallel ranks; doing it locally avoids a
        # collective, which is deliberate given the multi-node hang the fused path hit.
        assert sinkhorn_iters >= 0
        self.sinkhorn_iters = int(sinkhorn_iters)
        assert not (self.sinkhorn_iters > 0 and balance_rate > 0.0), (
            "sinkhorn_iters and balance_rate are two solutions to the SAME constraint "
            "(exact dual vs proportional control); enabling both double-counts it"
        )
        if self.sinkhorn_iters > 0:
            self.register_buffer("_sink_mu_absmax", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
        # ---- RUNNING mu, so balancing survives into eval (2026-09-15) ----------------
        # mu is a BATCH statistic solved under no_grad and, until now, applied only when
        # self.training. So a model trained with mu-tilted routing was EVALUATED with
        # mu = 0: a train/test shift in the router itself. Exactly the BatchNorm
        # situation, and the fix is the same -- keep a running estimate and use it at eval.
        #
        # MEASURED COST of not doing this (two pure-energy isoP arms, identical but for the
        # balancer): the clamped `load_balance_bias`, which is a persistent buffer with NO
        # self.training gate and therefore DOES survive to eval, scored Avg11 41.49; the
        # Sinkhorn twin, whose mu was dropped, scored 36.16. A 5.33pp gap from eval-time
        # treatment alone. Hybrids barely notice (1 MoE block of 7, six GPT layers immune);
        # pure-energy stacks are all-MoE so it compounds every iteration.
        #
        # OPT-IN, and deliberately so. The checkpoint loader is STRICT: adding a persistent
        # buffer to an arm that already has checkpoints fails its resume with
        #   "Missing key in checkpoint state_dict: ...ffwd.moe.load_balance_bias"
        # which is documented above and already cost slope90k_1blk a resume. Four sinkhorn
        # arms were mid-training when this landed, so the default keeps their state_dict
        # shape byte-identical.
        self.sinkhorn_persist_mu = bool(sinkhorn_persist_mu)
        assert not (self.sinkhorn_persist_mu and self.sinkhorn_iters == 0), (
            "sinkhorn_persist_mu has nothing to persist with sinkhorn_iters=0"
        )
        # PER-ITERATION mu. A shared (recurrent) block is called N times per forward and
        # solves a DIFFERENT dual each time -- measured on the pure isoP arm
        # (mu_per_iteration_20260915.py): iteration 0 reaches |mu|max 2.24 while later ones sit
        # near 0.9-1.2, per-expert spread ACROSS iterations averages 1.75 and reaches 3.10, and
        # individual experts flip sign between iterations (expert 1: -1.27 at iter 0, +1.39 at
        # iter 1). A single averaged buffer therefore destroys the signal twice: it washes out
        # the magnitude (|mean(mu)|max 0.66 vs 2.24) and cancels the sign-flipping components.
        # That is why the first, single-buffer version of this fix recovered almost nothing
        # (PPL 177.09 -> 175.16) -- the test was doomed by the averaging, not by the hypothesis.
        # So the buffer is (n_iter, K) and eval CYCLES through it.
        #
        # The cycle length is taken from the buffer shape, so the block needs no knowledge of
        # `layer_iterations` and layer.py is untouched.
        #
        # SCOPE: calibration and eval only, both of which run under no_grad with a clean call
        # order. A training-time per-iteration EMA is NOT supported, because activation
        # checkpointing replays the forward during backward and would desynchronise the counter.
        # ---- TRUE SPARSITY, back-projection only (2026-09-16) ----------------------
        # `top_k` has always been a post-hoc MASK: all K experts' forward AND back
        # projections are computed, then multiplied by a p that is zero for K-k of them. The
        # FORWARD projection cannot be skipped by an exact router -- it needs all K energies
        # to decide, which is the 1/2(1+k/K) floor ("cannot beat 2x"). The BACK projection
        # can: by the time it runs, p is known.
        #
        # Capacity-based dispatch: gather each expert's assigned tokens into a fixed
        # (K, C, I_e) buffer, one bmm against W viewed as (K, I_e, hidden), scatter-add back.
        # Fixed shapes, so it is torch.compile-safe -- unlike a per-expert Python loop, which
        # was measured at 0.59x, i.e. SLOWER than the dense mask it replaces.
        #
        # C = ceil(capacity_factor * T * k / K). EXACT while no expert exceeds C; on overflow
        # the surplus (token, expert) pairs are DROPPED, which changes the function. Sinkhorn
        # makes that unlikely -- measured max_share 0.041-0.077 against 1/K = 0.031-0.0625 --
        # and `_sparse_overflow` counts it so silence is not mistaken for exactness.
        #
        # ---- TRUE SPARSITY, the other half: `sparse_forward` (2026-09-16) ----------
        # The forward projection can only be skipped if something OTHER than the expert
        # matmuls decides where each token goes -- hence `sparse_forward` REQUIRES the rank-r
        # proxy router. With it, both GEMMs shrink from K to k experts:
        #
        #   dense   2*K*d*I_e            per token
        #   sparse  2*k*d*I_e + K*d*r    per token          (r ~ 8, so the proxy is ~0.1%)
        #
        # i.e. ~K/k, against the 1/2*(1+k/K) floor that an exact router cannot beat. It also
        # cuts the ACTIVATION footprint by the same factor -- the (K, C, I_e) intermediates are
        # cf*k/K of the dense (T, K, I_e) -- which is what limits I_e in the pure design.
        #
        # WHAT IS EXACT AND WHAT IS NOT. Given the selection, every weight is computed from the
        # EXACT energy of the experts actually evaluated: the Hopfield energy is a by-product of
        # the forward projection, so exactness costs nothing. The approximations are, in order
        # of size:
        #   1. WHICH k experts get picked (the proxy's job). Measured agreement 0.94 top-1 /
        #      0.90 top-2 at r=8. This is the real approximation and it is the one to report.
        #   2. With renormalize_topk=False the softmax denominator runs over all K logits,
        #      K-k of which we deliberately never computed; those terms are filled from the
        #      proxy. They are by construction the SMALL terms. With renormalize_topk=True
        #      there is no such term and the weights are exact given the selection.
        #   3. routing_norm="zscore" needs per-token moments over all K energies; the proxy's
        #      are used. The mean cancels in the softmax, so only the std survives, as a mild
        #      per-token rescaling of tau.
        # None of these can be waved at: `test_sparse_forward_20260916.py` pins 1 by forcing
        # the exact selection (whereupon the path must reproduce the dense output bit for bit)
        # and then measures 2 and 3 separately.
        #
        # TRAINING RECIPE. The proxy is distilled against the exact all-K routing distribution,
        # which the sparse path does not compute -- so a sparse arm cannot train from scratch.
        # Run a dense phase with proxy_rank>0 and proxy_loss_coef>0 (free: off the main path),
        # then switch on sparse_forward. For an already-trained dense checkpoint, fit the proxy
        # post-hoc from cached (x, E_k) pairs, the same way sinkhorn mu is recalibrated.
        # ---- OVER-SELECT then RE-RANK (2026-09-16) ---------------------------------
        # MEASURED. Ablation on pure_hop_T12_sink, wikitext bits/byte against a dense 1.0996:
        #   proxy SELECTS, exact energies weight  ->  1.1161  (+0.0165)  <- selection is FINE
        #   sparse_forward, renormalize_topk      ->  1.9177
        #   sparse_forward, as trained            ->  3.4346  (+2.335)
        # So 0.88 top-2 agreement costs 1.5%, and the damage is in the SOFTMAX DENOMINATOR: with
        # renormalize_topk=False the sum runs over all K logits, and filling the K-k missing ones
        # from the proxy is worth -1.52 nats. The proxy ranks well and sums badly.
        #
        # `sparse_candidates` = p >= top_k: the proxy NOMINATES p experts, their exact energies
        # (free -- a by-product of the forward projection we now do for them) RE-RANK to the final
        # top-k, and the denominator uses p exact terms instead of k. Both errors shrink together,
        # because a softmax denominator is dominated by its largest terms and those are exactly
        # the ones we now compute exactly.
        #
        # Cost: the forward projection covers p experts, so the ceiling moves from 2K/(2k*cf) to
        # 2K/((p+k)*cf) -- 6.16x at p=2, ~4.3x at p=4, ~3.2x at p=6, all far above the 1.73x that
        # an EXACT router can never beat. p = n_experts is the correctness self-test: every energy
        # is then computed exactly and the path must reproduce the dense output.
        self.sparse_backproj = bool(sparse_backproj)
        self.sparse_forward = bool(sparse_forward)
        self.sparse_candidates = int(sparse_candidates)
        self.sparse_capacity_factor = float(sparse_capacity_factor)
        if self.sparse_backproj or self.sparse_forward:
            assert self.top_k is not None and self.top_k < len(experts), (
                "sparse paths need top_k < n_experts; with dense routing there is nothing to skip"
            )
            self.register_buffer("_sparse_overflow", torch.zeros((), dtype=torch.long), persistent=False)
        if self.sparse_forward:
            assert not self.sparse_backproj, (
                "sparse_forward already sparsifies the back projection; sparse_backproj is redundant"
            )
            assert fused_experts, "sparse_forward is built on the fused-weight view; set fused_experts: true"
            p_cand = self.sparse_candidates or int(top_k)
            assert int(top_k) <= p_cand <= len(experts), (
                f"sparse_candidates must satisfy top_k <= p <= n_experts, got {p_cand}"
            )
            self.sparse_candidates = p_cand
            assert int(proxy_rank) > 0 or p_cand == len(experts), (
                "sparse_forward cannot skip the forward projection without a cheap selector: "
                "set proxy_rank > 0. (p = n_experts needs no proxy, and is the self-test.)"
            )
            if float(repulsion_coef) > 0.0 and repulsion_space != "weight":
                assert int(repulsion_subsample) > 0, (
                    "sparse_forward cannot do full output-space repulsion (it never computes all K "
                    "expert outputs). Set repulsion_subsample > 0 to evaluate it on m tokens "
                    "instead -- ~1.2% of the step at m=64 -- or repulsion_space: weight, which a "
                    "sweep showed does NOT control output alignment."
                )
            # NOTE: repulsion and the cosine probe are checked in _forward_sparse, not here.
            # Both fire only under self.training, and the main use of sparse_forward is EVALUATING
            # an already-trained checkpoint whose config says repulsion_space: output. Asserting
            # at construction would make every such checkpoint unloadable for a reason that
            # cannot arise at eval.
        self.sinkhorn_mu_iters = max(1, int(sinkhorn_mu_iters))
        if self.sinkhorn_persist_mu:
            self.register_buffer("sinkhorn_mu",
                                 torch.zeros(self.sinkhorn_mu_iters, len(experts), dtype=torch.float32),
                                 persistent=True)
            self.register_buffer("sinkhorn_mu_count", torch.zeros(self.sinkhorn_mu_iters, dtype=torch.float32),
                                 persistent=True)
            self.register_buffer("_mu_call", torch.zeros((), dtype=torch.long), persistent=False)
        else:
            self.sinkhorn_mu = None
            self.sinkhorn_mu_count = None
        assert repulsion_interval >= 1
        self.repulsion_interval = int(repulsion_interval)
        self.repulsion_scale_comp = bool(repulsion_scale_comp)

        # ------------------------------------------------------------------ #
        # FUSED EXPERT PATH (2026-09-15) -- EXACT, the real speedup           #
        # ------------------------------------------------------------------ #
        # The `_Fused*Holder` classes already keep ONE weight tensor and hand each
        # expert a CONTIGUOUS ROW-SLICE of it. So the per-expert loop
        #     out = sum_k p_k * pref * (gated_k @ W_k)
        # is algebraically one GEMM against the fused weight:
        #     out = ((pref * p) (x) gated_all) @ W_fused
        # because W_fused is the vertical stack of the W_k and gated_all the
        # horizontal concat of the gated_k. Same for the forward projection.
        # This replaces 2*K slice-GEMMs (64 at K=32) with 2 GEMMs, which is what
        # the profiler's 79.5k kernels/step and GPU-busy 1.71s << 6.75s wall says
        # is the actual cost. The arithmetic is IDENTICAL -- this is not an
        # approximation, so the loss curve must match to numerical tolerance.
        self.fused_experts = bool(fused_experts)
        self._fused_spec = fused_spec
        if self.fused_experts:
            assert fused_spec is not None, "fused_experts=True needs fused_spec"
            widths = {e.intermediate_size for e in experts}
            assert len(widths) == 1, f"fused path needs equal expert widths, got {widths}"
            assert fused_spec["kind"] == "hopfield", (
                "fused_experts is implemented for expert_kind='hopfield' only "
                f"(got {fused_spec['kind']!r}); the w1w2 line still uses the loop"
            )
            self._expert_I = experts[0].intermediate_size

        # ------------------------------------------------------------------ #
        # LEARNABLE RANK-r PROXY ROUTER (2026-09-15)                          #
        # ------------------------------------------------------------------ #
        # Offline study (HANDOFF 7.6): the naive spectral proxy ||W_k x||^2 FAILS
        # (~0% top-1 agreement -- gelu(z)^2 is not pointwise proportional to z^2).
        # What works is the exact energy restricted to a rank-r subspace: r=8 gave
        # 0.94 top-1 / 0.90 top-2 at 98.3K MACs against the exact router's 12.58M,
        # i.e. ~128x cheaper. That study needed an SVD of a TRAINED W plus a head
        # fitted on cached (x, E_k) pairs, so it cannot route a run from scratch.
        #
        # This is the trainable version: a learned per-expert projection V_k plus a
        # diagonal-quadratic head, distilled ONLINE against the exact routing
        # distribution via an aux loss. It costs d*r*K per token (~3% of the block
        # at r=8) and -- crucially -- it does NOT touch the main forward path unless
        # `proxy_route` is set, so it cannot degrade training. Its measured top-k
        # agreement is logged every step, so we learn whether r is sufficient
        # BEFORE trusting it for inference.
        # ---- proxy FORM (2026-09-16) ------------------------------------------------
        # MEASURED, and it is why the first fit failed. The 0.94 top-1 / 0.90 top-2 in
        # HANDOFF 7.6 was for the EXACT energy restricted to a rank-r subspace. The
        # "diagonal-quadratic head on the r coefficients" below it was a RECOMMENDATION,
        # never a measurement, and the 0.90 was then quoted as though it applied to the
        # head. Fitted post-hoc on pure_hop_T12_sink the head reaches only 0.348 top-2 at
        # r=8 against a chance floor of k/K = 0.125, and 0.461 even at r=32. A diagonal
        # quadratic in `a` simply cannot represent mean(gelu(.)^2).
        #
        #   "quad"      E_hat_k = sum_j q_kj a_kj^2 + l_kj a_kj + b_k        (as before)
        #   "subspace"  E_hat_k = mean_j gelu((B_k a_k)_j)^2 * s_k + b_k     (7.6's form)
        #
        # `subspace` keeps the true nonlinearity: with a_k = V_k^T x and B_k = W_k V_k
        # precomputed, W_k P_k x = B_k a_k exactly, where P_k projects onto span(V_k). So the
        # only error is the rank truncation, not the function class.
        #
        # COST, and the trap in it. B_k is (I_e, r), so the naive version produces an
        # (T, K, I_e) tensor -- the same elementwise work and the same activation footprint as
        # the DENSE path, destroying two of the three savings and leaving only the GEMM.
        # `proxy_out_dim` = m fixes that: E_k is a MEAN over I_e coordinates, so a fixed
        # subsample of m rows of B_k is an UNBIASED estimator of it with variance ~1/m. Cost
        # falls to K*r*(d+m) MACs and K*m elementwise. At K=16, r=8, d=768, I_e=4480:
        #   m=0 (all)  672K MAC, 71,680 elementwise  -- elementwise NO BETTER than dense
        #   m=512      164K MAC,  8,192 elementwise  -- 0.15% of dense MACs, 11% of its elementwise
        # `proxy_scale`/`proxy_bias` absorb the residual bias of the estimator.
        self.proxy_rank = int(proxy_rank)
        self.proxy_loss_coef = float(proxy_loss_coef)
        self.proxy_route = bool(proxy_route)
        assert proxy_kind in ("quad", "subspace")
        self.proxy_kind = proxy_kind
        self.proxy_out_dim = int(proxy_out_dim)
        # PER-ITERATION HEADS. A shared recurrent block applies the SAME proxy at every
        # iteration, and the hidden-state distribution differs between them: measured on
        # pure_hop_T12_sink (12 iterations), agreement runs 0.080 at cycle position 2 -- BELOW
        # the 0.125 chance floor, i.e. anti-correlated -- to 0.592 at position 11, a 7.4x
        # spread. Exactly the structure that defeated a single shared sinkhorn mu buffer
        # (see the mu note above). proxy_iters > 1 gives one head per iteration, cycled by call
        # index.
        #
        # SCOPE: eval / inference / calibration ONLY, and asserted below. Indexing by a call
        # counter is UNSOUND in training under activation checkpointing, which replays the
        # forward during backward (`gradient_checkpointing_method: block` is set on the 400M
        # arm): the replay would advance the counter and apply a different head than the
        # forward did, so the gradients would not match the function. Training with
        # per-iteration heads needs the block to be told its iteration index, which it is not.
        self.proxy_iters = max(1, int(proxy_iters))
        # The proxy is only exercised by the FUSED path -- `_forward_looped` never calls
        # `_proxy_step`. Without this assert, `proxy_rank: 16` + `proxy_loss_coef: 0.01` on a
        # config that leaves fused_experts at its default False is a SILENT NO-OP: the run trains,
        # logs no proxy_topk_agree, costs nothing extra and teaches nothing. Caught while writing
        # the online-distillation test, whose base arm has fused_experts off. Pre-flight check 7.
        assert not (self.proxy_rank > 0 and not self.fused_experts), (
            "proxy_rank > 0 requires fused_experts: true -- the looped expert path never runs the "
            "proxy, so the proxy would silently never be trained or measured"
        )
        if self.proxy_rank > 0:
            r = self.proxy_rank
            n_it = self.proxy_iters
            # keep the (K, ...) shape when there is one head, so existing proxy checkpoints
            # load unchanged; only the per-iteration case gains a leading axis.
            pre = () if n_it == 1 else (n_it,)
            # DEDICATED GENERATOR, not the global RNG. torch.randn here would consume
            # the global stream and shift the initialisation of every parameter created
            # AFTER this block -- so merely enabling the proxy gave the whole model a
            # different init. That is what made arm D's lm_loss diverge 4.4x the
            # A-vs-A' noise floor while arm C, which shares every other knob, stayed
            # inside it: a different starting point, not a different computation.
            # (The gradient-clipping explanation I first reached for was refuted --
            # grad_norm never touched the 1.0 threshold on either arm.)
            _g = torch.Generator().manual_seed(0xB01742 + (layer_idx or 0))
            self.proxy_V = nn.Parameter(
                torch.randn(*pre, self.n_experts, hidden_size, r, generator=_g)
                / (hidden_size ** 0.5)
            )
            if self.proxy_kind == "quad":
                # Diagonal quadratic + linear head on the r coefficients. Kept for the A/B
                # against "subspace"; measured inadequate on its own (see above).
                self.proxy_quad = nn.Parameter(torch.ones(*pre, self.n_experts, r) / r)
                self.proxy_lin = nn.Parameter(torch.zeros(*pre, self.n_experts, r))
                self.proxy_bias = nn.Parameter(torch.zeros(*pre, self.n_experts))
                self.proxy_B = self.proxy_scale = None
            else:
                m = self.proxy_out_dim or experts[0].intermediate_size
                self._proxy_m = m
                self.proxy_B = nn.Parameter(
                    torch.randn(*pre, self.n_experts, m, r, generator=_g) / (r ** 0.5))
                self.proxy_scale = nn.Parameter(torch.ones(*pre, self.n_experts))
                self.proxy_bias = nn.Parameter(torch.zeros(*pre, self.n_experts))
                self.proxy_quad = self.proxy_lin = None
            if self.proxy_iters > 1:
                self.register_buffer("_proxy_call", torch.zeros((), dtype=torch.long),
                                     persistent=False)
            self.register_buffer("_proxy_agree_sum", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("_proxy_agree_n", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
        else:
            self.proxy_V = self.proxy_B = None
            self.proxy_quad = self.proxy_lin = self.proxy_scale = self.proxy_bias = None

        # ------------------------------------------------------------------ #
        # EXPERT-ALIGNMENT PROBE (2026-09-15)                                 #
        # ------------------------------------------------------------------ #
        # Measures mean|cos| between expert OUTPUTS *independently of the
        # repulsion loss*, under no_grad, on a fixed number of sampled pairs.
        #
        # Why it is needed: the only alignment signal we had was the repulsion aux
        # loss itself, which is coef * mean|cos| -- so it is unreadable on an arm
        # with repulsion_coef=0, which is exactly the control that tells us how
        # much of the observed mean|cos| ~= 0.10 plateau repulsion is actually
        # buying. It also makes arms with different coef or n_repulsion_pairs
        # directly comparable on alignment, which dividing the aux loss does not
        # (different coef, and the mean is over a different number of pairs).
        #
        # no_grad and forward-only, so it saves no tensors and cannot perturb
        # activation checkpointing. Runs on 1 call in cos_probe_interval; 0 = off,
        # which is the default, so this costs nothing unless asked for.
        # ---- SUBSAMPLED OUTPUT-SPACE REPULSION (2026-09-16) -------------------------
        # MEASURED NEED. Output-space repulsion needs all K expert outputs, which sparse_forward
        # does not compute, so the obvious substitute was repulsion_space="weight". A 1500-step
        # sweep says that does not work: at coef 0.7 and 2.0 the OUTPUT alignment RISES
        # 0.29 -> 0.44 -> 0.46 across steps 500/1000/1400 -- which is the signature HANDOFF 11.8
        # records for NO REPULSION AT ALL ("sits at 0.43 and RISES to 0.51; it never decays").
        # Weight cosines are 5-25x smaller than output cosines and evidently do not control them.
        #
        # So keep genuine OUTPUT-space repulsion and make it affordable by evaluating it on m
        # tokens instead of all T. m << T is what makes this cheap: at m=64, K=16, I_e=4480,
        # d=768 the extra forward projection is 3.5 GMAC against the sparse mixture's ~293 GMAC
        # at T=16384, i.e. ~1.2%.
        #
        # Also a WIN FOR DENSE ARMS: HANDOFF 11.8 measured output-space repulsion at 17-21% of the
        # optimizer step, with a benefit that is "steeply front-loaded". Subsampling should buy
        # most of that benefit for a fraction of the cost.
        #
        # DETERMINISTIC STRIDE, not a random draw. Activation checkpointing replays the forward
        # during backward; a fresh sample would make the replayed loss differ from the original.
        # `_add_repulsion_loss_fused` already carries a CheckpointError scar from exactly this
        # class of bug (random.sample changed a tensor SHAPE between forward and recompute), so
        # nothing here may depend on RNG. Different batches cover different tokens anyway.
        self.repulsion_subsample = int(repulsion_subsample)
        self.cos_probe_interval = int(cos_probe_interval)
        self.cos_probe_pairs = int(cos_probe_pairs)
        if self.cos_probe_interval > 0:
            self.register_buffer("_cos_sum", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("_cos_n", torch.zeros((), dtype=torch.float32),
                                 persistent=False)

    # --- public surface --------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.sparse_forward:
            return self._forward_sparse(x)
        if self.fused_experts:
            return self._forward_fused(x)
        return self._forward_looped(x)

    def _forward_looped(self, x: torch.Tensor) -> torch.Tensor:
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
        p, logits = self._route(E_k)
        self._track_load(p)
        self._track_energy(E_k)


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
            if self.repulsion_space == "weight":
                self._add_repulsion_loss_weight()
            else:
                self._add_repulsion_loss(expert_grads)

        if self.training and self._cos_probe_fires():
            self._probe_expert_cos(expert_grads)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    # --- shared routing (used by BOTH the looped and fused paths, so they can
    # --- never drift apart) ------------------------------------------------- #

    def _route(self, E_k: torch.Tensor, proxy_E: torch.Tensor | None = None):
        """Energies -> (p, logits). Verbatim the pre-2026-09-15 routing block, except that
        `proxy_E` (when `proxy_route` is on) moves the SELECTION onto the cheap router while
        the WEIGHTS stay exact.

        That split is the whole point. Selection only has to get the top-k SET right, and a
        rank-r proxy does that at 0.94 top-1 / 0.90 top-2 for ~1/128 of the router's MACs
        (HANDOFF 7.6). Weights have to be numerically right, and here they cost nothing extra
        because the exact energies are a by-product of the forward projection we already did.
        In the DENSE path this buys no FLOPs -- it is the A/B that isolates the quality cost of
        proxy selection from the dispatch machinery of `_forward_sparse`, where the same split
        is what makes skipping the forward projection possible at all.
        """
        logits = self._logits_raw(E_k)
        mu = self._mu_for(logits)               # exactly one dual solve / one _mu_call tick
        if mu is not None:
            logits = logits - mu.to(logits.dtype)
        p = F.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < self.n_experts:
            if proxy_E is None:
                sel_logits = logits
            else:
                sel_logits = self._logits_raw(proxy_E, moments=self._zscore_moments(proxy_E))
                if mu is not None:
                    sel_logits = sel_logits - mu.to(sel_logits.dtype)
            _, topk_idx = sel_logits.topk(self.top_k, dim=-1)
            mask = torch.zeros_like(p, dtype=torch.bool)
            mask.scatter_(-1, topk_idx, True)
            p = p * mask
            # NOTE ON COMPARABILITY. Leaving sum(p) < 1 was a deliberate choice (it
            # avoids abrupt weight redistribution at routing boundaries), but the
            # baselines we compare against do NOT do this: TopK_Energy_MoE_MLP takes
            # softmax over only the top-k logits, and the Switch-style MoE class uses
            # normalized_topk, so both give sum(p) = 1 exactly. Empirically our masked
            # form gives sum(p) ~ 0.45 at K=16, k=2, i.e. the mixture output is scaled
            # down by roughly 2x relative to the baselines. scale_ff can absorb that, but
            # it is a real asymmetry in the comparison, so expose the matched option.
            if self.renormalize_topk:
                p = p / p.sum(-1, keepdim=True).clamp_min(1e-9)
        return p, logits

    def _track_energy(self, E_k: torch.Tensor) -> None:
        """Accumulate the per-expert energy SCALE. See the __init__ note: with the
        corrected sign the energy sits on a positive-feedback path, so a growing
        `energy_abs_mean` across training is the divergence warning that
        `output_norm` cannot give on its own."""
        if not self.track_load:
            return
        with torch.no_grad():
            e = E_k.detach().float()
            self._E_abs_sum += e.abs().mean()
            self._E_abs_max.copy_(torch.maximum(self._E_abs_max, e.abs().max()))
            self._E_n += 1.0

    def _track_load(self, p: torch.Tensor) -> None:
        # Traceable load accounting: tensor ops only, no .item(), no host sync, so this
        # survives torch.compile where _log_metrics does not.
        if self.track_load and self.training:
            with torch.no_grad():
                pf = p.detach().reshape(-1, self.n_experts).float()
                self._load_sum += pf.sum(0)
                self._ent_sum += -(pf * (pf + 1e-9).log()).sum(-1).sum()
                self._tok_sum += pf.shape[0]
                if self.balance_rate > 0.0 and self.load_balance_bias is not None:
                    # BOUNDED, PROPORTIONAL update. The first version used
                    #     bias += rate * sign(target - share)
                    # which has TWO defects and destabilised training on the one arm that
                    # enabled it (pure_hop_isoP_bal: 77 upward loss jumps > 0.15, loss rising
                    # 4.05 -> 5.18, while two sibling arms with balance_rate=0 descended
                    # smoothly to 3.51 with zero jumps).
                    #   1. sign() never shrinks as load approaches balance, so there is no
                    #      equilibrium -- the bias drifts at a constant +-rate forever. It
                    #      reached |bias| = 1482 against z-scored logits of unit scale
                    #      divided by tau=0.35, so routing was decided almost entirely by the
                    #      bias and oscillated.
                    #   2. It fired once per MICROBATCH, so with gradient_accumulation_steps
                    #      = 16 the bias moved 16x per optimiser step.
                    # Now: proportional to the actual imbalance (so it has a fixed point at
                    # uniform load), divided by the microbatch count, and hard-clamped to
                    # BIAS_MAX -- the logits it is added to are z-scored, hence O(1), so a
                    # bias beyond ~1 can only overwhelm the energy it is meant to nudge.
                    share = pf.mean(0)
                    target = 1.0 / self.n_experts
                    upd = self.balance_rate * (target - share) * self.n_experts
                    self.load_balance_bias += upd.to(self.load_balance_bias.dtype)
                    self.load_balance_bias.clamp_(-self._BIAS_MAX, self._BIAS_MAX)


    # --- fused-GEMM path ---------------------------------------------------- #

    def _dispatch_plan(self, idx: torch.Tensor, T: int):
        """(token, expert) pairs -> slots in a fixed (K, C, *) buffer.

        Shared by the back-projection-only and fully-sparse paths so the two cannot drift
        apart. Returns the kept pairs' (expert, token, slot) ids, their positions `sp` in the
        flat (T*k) selection layout, the capacity C, and a (T, k) bool of what survived.

        C = ceil(capacity_factor * T * k / K), i.e. sized for balanced load. Surplus pairs are
        DROPPED, which changes the function, so `_sparse_overflow` counts them -- silence must
        not be mistaken for exactness. Fixed shapes throughout, hence torch.compile-safe.
        """
        K = self.n_experts
        k = idx.shape[-1]
        dev = idx.device
        C = max(1, int(math.ceil(self.sparse_capacity_factor * T * k / K)))
        flat_e = idx.reshape(-1)
        flat_t = torch.arange(T, device=dev).unsqueeze(1).expand(T, k).reshape(-1)
        # stable sort groups pairs by expert; rank within the group is the slot. `order` is
        # also the map from sorted position back to flat (T*k) position, which is what lets the
        # caller scatter per-pair results into the (T, k) layout.
        order = torch.argsort(flat_e, stable=True)
        se, st = flat_e[order], flat_t[order]
        counts = torch.zeros(K, dtype=torch.long, device=dev).scatter_add_(
            0, se, torch.ones_like(se))
        starts = torch.cat([torch.zeros(1, dtype=torch.long, device=dev),
                            counts.cumsum(0)[:-1]])
        slot = torch.arange(se.shape[0], device=dev) - starts[se]
        keep = slot < C
        if self.track_load and hasattr(self, "_sparse_overflow"):
            with torch.no_grad():
                self._sparse_overflow += (~keep).sum()
        kept_flat = torch.zeros(T * k, dtype=torch.bool, device=dev)
        kept_flat[order] = keep
        # MASK, NEVER FILTER. Returning se[keep] would make every index tensor's SIZE depend on
        # how many pairs overflow, i.e. on the data. That is not torch.compile-safe -- and worse,
        # under FSDP each rank sees different data and so compiles a DIFFERENT graph. Measured:
        # the first distributed run (sprobe_T12_p2, 4 GPUs) produced 0 steps in 8 minutes, 6
        # recompile frames and 527 KB of `spmd_check` rank-divergence warnings
        # (rank 0: aten.select.int vs rank 3: aten.select_scatter.default on the same node). It is
        # the same failure class HANDOFF records for fused_experts wedging at 2 nodes, and a
        # SINGLE-GPU benchmark cannot see it: there is no other rank to diverge from, and
        # recompilation merely costs time. So the earlier 4.69x was measured on code that cannot
        # run distributed.
        #
        # Instead: keep every tensor at the static size T*k, and send overflow pairs to a dedicated
        # DUMP ROW at index C, which the caller allocates (C+1 rows) and discards. Their weight is
        # zeroed via `kept`, so they contribute nothing -- identical arithmetic, static shapes.
        slot_safe = torch.where(keep, slot, torch.full_like(slot, C))
        # `keep` is in SORTED order (aligned with se/st/slot_safe); `kept_flat.view(T, k)` is in the
        # token-major selection layout. Both are returned because the two callers need different
        # ones, and re-deriving either from the other is where an ordering bug would hide.
        return se, st, slot_safe, order, C, kept_flat.view(T, k), keep

    def _sparse_backproj(self, gated: torch.Tensor, p: torch.Tensor,
                         W: torch.Tensor, pref: float) -> torch.Tensor:
        """Back-projection over only the top-k experts per token.

        gated (..., K, I_e), p (..., K) with K-k zeros, W (K*I_e, hidden).
        Returns (..., hidden), equal to the dense `(gated * p) @ W` up to fp associativity
        whenever no expert exceeds capacity.
        """
        K, I_e = self.n_experts, self._expert_I
        H = self.hidden_size
        lead = gated.shape[:-2]
        g = gated.reshape(-1, K, I_e)
        pf = p.reshape(-1, K)
        T = g.shape[0]

        # (token, expert) pairs to evaluate. Taking topk of p (not of the logits) keeps this
        # consistent with whatever mask _route applied, including renormalize_topk.
        idx = pf.topk(int(self.top_k), dim=-1).indices          # (T, k)
        se_k, st_k, slot_k, _, C, _, keep = self._dispatch_plan(idx, T)
        buf = g.new_zeros(K, C + 1, I_e)          # +1 = dump row for overflow, discarded
        buf[se_k, slot_k] = g[st_k, se_k]                      # gather
        y = torch.bmm(buf, W.view(K, I_e, H))                  # ONE batched GEMM
        # `keep` is already in se/st order, so it multiplies elementwise: overflow pairs read the
        # dump row and are then zeroed, contributing nothing.
        contrib = y[se_k, slot_k] * (pref * pf[st_k, se_k] * keep).unsqueeze(-1)
        out = g.new_zeros(T, H).index_add_(0, st_k, contrib)    # scatter-add
        return out.reshape(*lead, H)

    def _subsampled_gated(self, xf: torch.Tensor, W: torch.Tensor, spec: dict, m: int):
        """`gated` over ALL K experts for a deterministic stride of m tokens.

        This is what lets the sparse path keep output-space repulsion and the alignment probe:
        both need every expert's output, but only for enough tokens to estimate a cosine.
        Cost m*d*K*I_e, and m is ~64 against T of 16384.
        """
        T = xf.shape[0]
        m = max(1, min(int(m), T))
        step = max(1, T // m)
        idx = torch.arange(0, m * step, step, device=xf.device)[:m]     # no RNG: replay-safe
        z = xf[idx] @ W.t()
        g, gp = _gelu_and_grad(z, spec["gelu_grad_method"])
        return (g * gp).view(m, self.n_experts, self._expert_I)

    def _fused_W(self) -> torch.Tensor:
        return self._fused_spec["weight_fn"]()

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """EXACT equivalent of _forward_looped for equal-width Hopfield experts.

        Replaces 2*K slice-GEMMs with 2 GEMMs against the fused weight. See the
        `fused_experts` note in __init__ for why this is algebraically identical.
        """
        spec = self._fused_spec
        W = self._fused_W()                      # (K*I_e, hidden)
        K, I_e = self.n_experts, self._expert_I
        pref = _hopfield_grad_prefactor(I_e, spec["hopfield_grad_scale"])

        Wx = x @ W.t()                           # ONE forward GEMM, all experts
        gelu_Wx, gelu_prime = _gelu_and_grad(Wx, spec["gelu_grad_method"])

        lead = Wx.shape[:-1]
        g = gelu_Wx.view(*lead, K, I_e)
        E_k = (g * g).mean(dim=-1)               # (..., K) -- same as per-expert mean

        # `proxy_route` moves SELECTION to the cheap router while weights stay exact. In the
        # dense path this saves nothing -- it is the controlled A/B for the quality cost of
        # proxy selection, with the dispatch machinery of _forward_sparse held out.
        proxy_E = self._proxy_energies(x) if (self.proxy_route and self.proxy_rank > 0) else None
        p, logits = self._route(E_k, proxy_E=proxy_E)
        self._track_load(p)
        self._track_energy(E_k)

        if self.proxy_rank > 0:
            self._proxy_step(x, E_k, logits, E_hat=proxy_E)

        gated = (gelu_Wx * gelu_prime).view(*lead, K, I_e)
        if self.sparse_backproj:
            # skips the (1 - k/K) of the back GEMM that the dense mask multiplies by zero
            out = self._sparse_backproj(gated, p, W, pref)
        else:
            gw = (gated * (pref * p).unsqueeze(-1)).reshape(*lead, K * I_e)
            out = gw @ W                         # ONE backward-projection GEMM

        if self.training and self._capture_energy:
            self._last_energy_per_token = -self.temperature * torch.logsumexp(logits, dim=-1)
        else:
            self._last_energy_per_token = None

        if self.training and self.repulsion_coef > 0:
            if self.repulsion_space == "weight":
                self._add_repulsion_loss_weight()
            elif self.repulsion_subsample > 0:
                # same estimator the sparse path uses, so a dense and a sparse arm with the same
                # repulsion_subsample are directly comparable on this term
                x2 = x.reshape(-1, self.hidden_size)
                self._add_repulsion_loss_fused(
                    self._subsampled_gated(x2, W, spec, self.repulsion_subsample), W, pref)
            else:
                self._add_repulsion_loss_fused(gated, W, pref)

        if self.training and self._cos_probe_fires():
            self._probe_expert_cos(gated, W=W, pref=pref)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    def _forward_sparse(self, x: torch.Tensor, sel_idx: torch.Tensor | None = None,
                        exact_denominator: torch.Tensor | None = None) -> torch.Tensor:
        """GENUINELY sparse: only the top-k experts' forward AND back projections are computed.

        See the `sparse_forward` note in __init__ for the cost model and for exactly which
        three things are approximated. `sel_idx` (a (T, k) expert index) and
        `exact_denominator` (all-K logits) override the proxy and are used ONLY by the test,
        which needs to hold the selection fixed to check the arithmetic underneath it.
        """
        spec = self._fused_spec
        W = self._fused_W()                              # (K*I_e, hidden)
        K, I_e = self.n_experts, self._expert_I
        H = self.hidden_size
        pref = _hopfield_grad_prefactor(I_e, spec["hopfield_grad_scale"])
        Wv = W.view(K, I_e, H)
        lead = x.shape[:-1]
        xf = x.reshape(-1, H)
        T = xf.shape[0]
        k = int(self.top_k)

        # ---- 1. SELECT with the cheap router: K*d*r MACs and not one expert matmul ------
        E_prox = self._proxy_energies(x).reshape(-1, K)
        mom = self._zscore_moments(E_prox)
        s_prox = self._logits_raw(E_prox, moments=mom)
        # The dual is solved on the PROXY logits because they are the only all-K quantity here
        # -- which is also the right place for it: mu exists to steer the SELECTION, and
        # selection is made on these logits.
        mu = self._mu_for(s_prox)
        if mu is not None:
            s_prox = s_prox - mu.to(s_prox.dtype)
        p_cand = self.sparse_candidates or k
        if sel_idx is None:
            # p = K needs no proxy at all: every expert is a candidate, so the re-rank below is
            # over the exact energies of all of them and the whole path becomes exact.
            sel_idx = (torch.arange(K, device=xf.device).expand(T, K) if p_cand == K
                       else s_prox.topk(p_cand, dim=-1).indices)
        else:
            sel_idx = sel_idx.reshape(T, -1)
            p_cand = sel_idx.shape[-1]

        # ---- 2. capacity-dispatch the TOKENS (dense path dispatches nothing) -----------
        se_k, st_k, slot_k, sp_k, C, kept, _ = self._dispatch_plan(sel_idx, T)
        xbuf = xf.new_zeros(K, C + 1, H)         # +1 = dump row; overflow tokens collide there
                                                 # harmlessly because `kept` zeroes their weight
        xbuf[se_k, slot_k] = xf[st_k]

        # ---- 3. SPARSE FORWARD PROJECTION: k/K of the dense GEMM ----------------------
        z = torch.bmm(xbuf, Wv.transpose(1, 2))          # (K, C, I_e)
        gz, gzp = _gelu_and_grad(z, spec["gelu_grad_method"])

        # ---- 4. EXACT energies -- free, they fall out of the projection we just did ----
        E_pair = (gz * gz).mean(-1)                      # (K, C)
        # p_cand, NOT k: with over-selection there are p candidate slots per token, and sizing
        # this with k silently broke every p > k configuration (caught by the p = K self-test).
        E_tk = E_prox.new_zeros(T * p_cand)
        E_tk[sp_k] = E_pair[se_k, slot_k].to(E_tk.dtype)
        E_tk = E_tk.view(T, p_cand)
        # zscore moments: exact when every expert is a candidate, else the proxy's
        mom_w = self._zscore_moments(E_tk) if p_cand == K else mom
        lg = self._logits_raw(E_tk, moments=mom_w, expert_idx=sel_idx)
        if mu is not None:
            lg = lg - mu.to(lg.dtype)[sel_idx]
        # overflow-dropped pairs were never evaluated, so they must carry zero weight rather
        # than the weight of a zero energy
        neg = torch.finfo(lg.dtype).min
        lg = lg.masked_fill(~kept, neg)

        # ---- 4b. RE-RANK: the exact energies pick the final top-k out of the p candidates ----
        # This is the point of over-selecting. The proxy only has to get the true top-k INTO its
        # top-p, which is a far weaker requirement than getting the top-k exactly right.
        if p_cand > k:
            win = lg.topk(k, dim=-1).indices                      # (T, k) positions within p
            wmask = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, win, True)
        else:
            wmask = kept

        # Weights. Shared max-subtraction over BOTH the exact selected logits and the proxy's,
        # since the non-renormalised denominator mixes the two.
        ref = s_prox if exact_denominator is None else exact_denominator
        mx = torch.maximum(lg.max(-1, keepdim=True).values, ref.max(-1, keepdim=True).values)
        num = torch.exp(lg - mx) * wmask                           # winners only carry weight
        # DENOMINATOR. All p candidates contribute EXACTLY -- including the p-k that lost the
        # re-rank, which is why over-selecting fixes the denominator as well as the selection.
        Zcand = (torch.exp(lg - mx) * kept).sum(-1, keepdim=True)
        if p_cand == K:
            Zrest = torch.zeros_like(Zcand)                        # nothing left out: exact
        else:
            Zrest = (torch.exp(ref - mx).sum(-1, keepdim=True)
                     - torch.exp(ref.gather(-1, sel_idx) - mx).sum(-1, keepdim=True)).clamp_min(0)
        Zall = (Zcand + Zrest).clamp_min(1e-30)
        Zwin = num.sum(-1, keepdim=True).clamp_min(1e-30)
        w = num / (Zwin if self.renormalize_topk else Zall)

        # ---- 5. SPARSE BACK PROJECTION ------------------------------------------------
        y = torch.bmm(gz * gzp, Wv)                      # (K, C, hidden)
        wp = w.reshape(-1)[sp_k]
        contrib = y[se_k, slot_k] * (pref * wp).unsqueeze(-1)
        out = xf.new_zeros(T, H).index_add_(0, st_k, contrib.to(xf.dtype)).reshape(*lead, H)

        # ---- 6. metrics and side channels --------------------------------------------
        # A (T, K) p is rebuilt for the load metrics: it costs T*K, not T*K*I_e, so the whole
        # point of the sparse path survives it and every existing routing metric keeps working.
        p_full = w.new_zeros(T, K).scatter_(-1, sel_idx, w)
        self._track_load(p_full)
        # NOTE: this is the mean over SELECTED experts, i.e. the high-energy tail, where the
        # dense path averages over all K. Comparable across sparse arms, not against dense ones.
        self._track_energy(E_tk)

        if self.training and self._capture_energy:
            # -tau*logsumexp over all K, exact on the k evaluated and proxy-completed elsewhere
            self._last_energy_per_token = (-self.temperature * (mx + torch.log(Zall))).squeeze(-1)
        else:
            self._last_energy_per_token = None

        # Repulsion and the alignment probe both need all K expert outputs, which this path
        # deliberately does not compute -- so build them for a small token subsample. Sharing one
        # `gated_sub` means the probe measures exactly what the repulsion acts on, and it is the
        # only way to SEE alignment in a sparse arm (which is the metric that motivated all this).
        need_rep = self.training and self.repulsion_coef > 0 and self.repulsion_space != "weight"
        need_probe = self.training and self._cos_probe_fires()
        if need_rep or need_probe:
            gated_sub = self._subsampled_gated(xf, W, spec, self.repulsion_subsample or 64)
            if need_rep:
                self._add_repulsion_loss_fused(gated_sub, W, pref)
            if need_probe:
                self._probe_expert_cos(gated_sub, W=W, pref=pref)
        elif self.training and self.repulsion_coef > 0:
            self._add_repulsion_loss_weight()

        if not torch.compiler.is_compiling():
            self._log_metrics(p_full, out)

        return out

    def _add_repulsion_loss_fused(self, gated: torch.Tensor, W: torch.Tensor,
                                  pref: float) -> None:
        """Repulsion without materialising all K expert outputs.

        Only the experts appearing in the sampled pairs are needed (<=2*n_pairs of
        K, so <=8 of 32), so we back-project just those via one bmm instead of all
        K. Identical VALUE to the looped version, a fraction of the cost -- and on
        a non-firing intermittent step it costs nothing at all.
        """
        if not self._repulsion_fires():
            return
        i_idx, j_idx = self._sample_pairs()
        K, I_e = self.n_experts, self._expert_I
        Wv = W.view(K, I_e, self.hidden_size)
        g = gated.reshape(-1, K, I_e)

        # SHAPES MUST NOT DEPEND ON THE RANDOM DRAW. The first version gathered the
        # UNIQUE expert set of the sampled pairs, whose SIZE varies (7 or 8 of 32
        # for n_pairs=4). Activation checkpointing re-runs this forward during
        # backward, `random.sample` drew different pairs, the size changed, and the
        # recompute check failed with
        #     saved [7,1280,1536] vs recomputed [8,1280,1536] -> CheckpointError
        # Indexing by i_idx/j_idx directly makes every tensor here (n_pairs, ...),
        # constant regardless of the draw -- which is exactly why the pre-existing
        # looped path never hit this (its cos is always (N, n_pairs)). Cost is
        # 2*n_pairs=8 expert back-projections instead of <=8 unique: same work,
        # shape-stable.
        if torch.is_tensor(i_idx):
            gi, gj = g.index_select(1, i_idx), g.index_select(1, j_idx)
            Wi, Wj = Wv.index_select(0, i_idx), Wv.index_select(0, j_idx)
        else:
            gi, gj, Wi, Wj = g[:, i_idx, :], g[:, j_idx, :], Wv[i_idx], Wv[j_idx]
        ei = pref * torch.einsum("npi,pih->nph", gi, Wi)
        ej = pref * torch.einsum("npi,pih->nph", gj, Wj)
        cos = (F.normalize(ei, dim=-1) * F.normalize(ej, dim=-1)).sum(-1)
        add_aux_loss(self._repulsion_coef_now() * _repulsion_penalty(cos, self.repulsion_form))

    def _cos_probe_fires(self) -> bool:
        if self.cos_probe_interval <= 0:
            return False
        if self.cos_probe_interval == 1:
            return True
        return bool(torch.randint(self.cos_probe_interval, (1,), device="cpu").item() == 0)

    @torch.no_grad()
    def _probe_expert_cos(self, eg_or_gated, W=None, pref=1.0) -> None:
        """Accumulate mean|cos| between expert outputs. Pure measurement.

        `eg_or_gated` is the per-expert output stack (..., K, hidden) in the looped
        path, or the gated intermediates (..., K, I_e) in the fused path -- in the
        latter case W/pref are given and the selected experts are back-projected
        here, for cos_probe_pairs pairs only.
        """
        n_pairs = min(self.cos_probe_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, n_pairs)
        i_idx = [p[0] for p in sampled]
        j_idx = [p[1] for p in sampled]
        if W is None:
            eg = eg_or_gated.reshape(-1, self.n_experts, self.hidden_size)
            ei, ej = eg[:, i_idx, :], eg[:, j_idx, :]  # probe is no_grad; lists are fine
        else:
            K, I_e = self.n_experts, self._expert_I
            Wv = W.view(K, I_e, self.hidden_size)
            g = eg_or_gated.reshape(-1, K, I_e)
            ei = pref * torch.einsum("npi,pih->nph", g[:, i_idx, :], Wv[i_idx])
            ej = pref * torch.einsum("npi,pih->nph", g[:, j_idx, :], Wv[j_idx])
        cos = (F.normalize(ei, dim=-1) * F.normalize(ej, dim=-1)).sum(-1)
        self._cos_sum += cos.abs().mean().float()
        self._cos_n += 1.0

    # --- learnable rank-r proxy router -------------------------------------- #

    def _proxy_energies(self, x: torch.Tensor) -> torch.Tensor:
        """Cheap approximate per-expert energies from a rank-r projection.

        Cost d*r per expert (K*d*r total) against the exact router's K*d*I_e --
        at r=8, I_e=1280 that is 160x fewer MACs for the routing decision.
        """
        # x is DETACHED: the proxy is a passive observer that learns to predict the
        # exact router's decision from the hidden state. Without the detach, the
        # distillation KL back-propagates into the backbone and reshapes
        # representations to be cheaply-routable -- which may well be desirable, but
        # it changes what the model computes and so could move quality. Keeping the
        # proxy strictly off the main gradient path is what makes "enabling it cannot
        # degrade training" true. Shaping is a separate experiment, not a default.
        if self.proxy_iters == 1:
            V, B, q, l, sc, b = (self.proxy_V, self.proxy_B, self.proxy_quad,
                                 self.proxy_lin, self.proxy_scale, self.proxy_bias)
        else:
            assert not self.training, (
                "proxy_iters > 1 is eval/calibration only: indexing the head by a call counter "
                "is unsound in training under activation checkpointing, which replays the "
                "forward during backward. Train with proxy_iters: 1 and refine per-iteration "
                "heads post-hoc."
            )
            g = int(self._proxy_call.item()) % self.proxy_iters
            self._proxy_call += 1
            pick = lambda t: None if t is None else t[g]          # noqa: E731
            V, B, q, l, sc, b = map(pick, (self.proxy_V, self.proxy_B, self.proxy_quad,
                                           self.proxy_lin, self.proxy_scale, self.proxy_bias))
        a = torch.einsum("...h,khr->...kr", x.detach(), V.to(x.dtype))
        if self.proxy_kind == "quad":
            return ((q.to(x.dtype) * a * a).sum(-1) + (l.to(x.dtype) * a).sum(-1)
                    + b.to(x.dtype))
        # subspace: the EXACT energy form, on the rank-r projection of x. F.gelu matches
        # _gelu_and_grad's phi for every method, so this is the same nonlinearity the true
        # energy uses -- the only error is the truncation (and the m-subsample, if any).
        z = torch.einsum("...kr,kmr->...km", a, B.to(x.dtype))
        gz = F.gelu(z)
        return (gz * gz).mean(-1) * sc.to(x.dtype) + b.to(x.dtype)

    def _proxy_step(self, x: torch.Tensor, E_k: torch.Tensor,
                    logits: torch.Tensor, E_hat: torch.Tensor | None = None) -> None:
        """Distil the proxy against the exact routing distribution, and MEASURE its
        top-k agreement. Off the main forward path: this adds an aux loss and a
        metric, and changes nothing the model computes (unless proxy_route).

        The measurement matters more than the loss. HANDOFF 7.9 records that
        `torch.isin` silently fakes top-k agreement, so agreement is computed here
        as a genuine per-row set overlap: |top-k(proxy) INTERSECT top-k(exact)| / k.
        """
        if E_hat is None:
            E_hat = self._proxy_energies(x)
        # _logits_raw, NOT _logits: the dual has already been solved once this call and
        # _logits would tick _mu_call a second time, desynchronising the per-iteration buffer
        # that eval cycles through.
        logits_hat = self._logits_raw(E_hat, moments=self._zscore_moments(E_hat))

        if self.training and self.proxy_loss_coef > 0:
            # PER-TOKEN normalisation. F.kl_div(reduction="batchmean") divides by dim 0
            # ONLY, so on a (batch, seq, K) tensor it sums over seq*K and divides by
            # batch -- inflating the loss by the sequence length. Measured: aux_loss
            # 29.47 against an lm_loss of 7.52 at seq=4096, i.e. the proxy term was ~4x
            # the language model. Flattening to (tokens, K) first makes batchmean a true
            # per-token mean, so the term is O(proxy_loss_coef).
            #
            # This mattered for more than cosmetics. The proxy input is detached, so the
            # proxy cannot reshape the backbone through the graph -- but the trainer
            # clips on the GLOBAL grad norm (gradient_clipping: 1), and the proxy's
            # parameters are in model.parameters(). An inflated proxy gradient therefore
            # raises the global norm and scales the BACKBONE gradients down. Detaching
            # blocks the graph path; it does not decouple the optimizer.
            flat_hat = logits_hat.reshape(-1, self.n_experts)
            flat_tgt = F.softmax(logits.detach().reshape(-1, self.n_experts), dim=-1)
            add_aux_loss(
                self.proxy_loss_coef
                * F.kl_div(F.log_softmax(flat_hat, dim=-1), flat_tgt, reduction="batchmean")
            )

        if self.track_load and self.training:
            with torch.no_grad():
                k = self.top_k if self.top_k is not None else 1
                k = min(k, self.n_experts)
                a = logits.detach().reshape(-1, self.n_experts).topk(k, dim=-1).indices
                b = logits_hat.detach().reshape(-1, self.n_experts).topk(k, dim=-1).indices
                hit = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1)
                self._proxy_agree_sum += hit.sum() / k
                self._proxy_agree_n += hit.shape[0]

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        e_list = [expert.energy_per_token(x) for expert in self.experts]
        E_k = torch.stack(e_list, dim=-1)
        logits = self._logits(E_k)
        return -self.temperature * torch.logsumexp(logits, dim=-1)

    # --- private ---------------------------------------------------------- #

    def _logits(self, E_k: torch.Tensor) -> torch.Tensor:
        """Routing logits from per-expert energies.

        WHY THIS EXISTS. In this class the routing logit and the energy are the SAME
        object, so a single normalisation has to serve two incompatible jobs: keeping
        the descent gradient stable, and keeping softmax(E/tau) informative.

        The Hopfield form uses a MEAN, E = (1/d_int)||gelu(Wh)||^2, which is required
        for gradient stability (without it the descent step grows with d_int and
        training NaNs). But a mean over d_int units puts E ~ 1e-2, which against
        tau=1 is 100x too small to separate experts: measured on
        math_fet_boltz_hopfield_rep, E_mean = 0.0134 with per-token spread across
        experts of 0.0129, giving effective_n_experts = 7.999 / 8 -- routing is
        EXACTLY uniform, and top-k is then a pure loss rather than a trade.

        The w1w2 line escaped this because it carries a 1/sqrt(expert_I) routing
        scale, leaving E_mean = 0.109 with spread 0.437 -- commensurate with tau=1,
        giving effective_n_experts 3.55/4 and genuinely informative routing.

          "none"        logits = +-E/tau                     (as-trained; legacy)
          "sqrt_width"  logits = +-E*sqrt(expert_I)/tau      (matches the w1w2 line)
          "zscore"      logits = +-(E - mean_k)/std_k/tau    (scale-free: immune to
                        the energy magnitude drifting as weight decay shrinks W,
                        which is what happened here over training)
        """
        logits = self._logits_raw(E_k)
        mu = self._mu_for(logits)
        return logits if mu is None else logits - mu.to(logits.dtype)

    def _logits_raw(self, E_k: torch.Tensor, *, moments: tuple | None = None,
                    expert_idx: torch.Tensor | None = None) -> torch.Tensor:
        """The logit map WITHOUT the Sinkhorn dual: sign, normalisation, bias, temperature.

        Split out of `_logits` for the sparse forward path, which must apply the SAME map
        twice per call -- once to the proxy energies over all K (to choose experts) and once
        to the exact energies of the chosen ones (to weight them) -- while solving the dual
        only ONCE. Calling `_logits` twice would double-count `_mu_call` and, in training,
        run the dual solve on the proxy and then again on a (T, k) slice where it is
        meaningless.

        `moments` supplies (mean, std) for `routing_norm="zscore"` instead of computing them
        over the last axis. The sparse path has only k of the K exact energies, so it cannot
        form the exact moments; it passes the PROXY's. This is a smaller approximation than
        it looks: a per-token additive constant cancels exactly between numerator and
        denominator of the softmax, so the mean drops out entirely and only the std survives,
        acting as a mild per-token rescaling of tau.

        `expert_idx` gathers the per-expert `load_balance_bias` into a (T, k) selection
        layout instead of broadcasting over a full (T, K) one.
        """
        s = -E_k if self.e_sign == "neg" else E_k
        if self.routing_norm == "sqrt_width":
            s = s * (self.experts[0].intermediate_size ** 0.5)
        elif self.routing_norm == "zscore":
            if moments is None:
                m, sd = s.mean(-1, keepdim=True), s.std(-1, keepdim=True).clamp_min(1e-12)
            else:
                m, sd = moments
            s = (s - m) / sd
        # Bias is added AFTER normalisation: zscore would otherwise rescale it away, and
        # it is defined in the same (post-norm) space the temperature divides.
        if self.balance_rate > 0.0 and self.load_balance_bias is not None:
            # Cast to the logits' dtype: the buffer is fp32 while s is bf16 under mixed
            # precision, and the implicit promotion failed during fake-tensor tracing.
            b = self.load_balance_bias.to(s.dtype)
            s = s + (b if expert_idx is None else b[expert_idx])
        return s / self.temperature

    def _zscore_moments(self, E_k: torch.Tensor) -> tuple | None:
        """The (mean, std) `_logits_raw` would compute for these energies, or None if the
        active `routing_norm` does not use them. Mirrors `_logits_raw` exactly, including the
        sign flip that precedes the standardisation."""
        if self.routing_norm != "zscore":
            return None
        s = -E_k if self.e_sign == "neg" else E_k
        return s.mean(-1, keepdim=True), s.std(-1, keepdim=True).clamp_min(1e-12)

    def _mu_for(self, logits: torch.Tensor) -> torch.Tensor | None:
        """The Sinkhorn dual to subtract from these logits, or None when there is none.

        Carries ALL the `_mu_call` bookkeeping, so it must be called exactly once per forward
        call of this block -- that counter is what lets eval cycle through the per-iteration
        buffer in the same order training filled it.
        """
        if self.sinkhorn_iters <= 0:
            return None
        if self.training:
            mu = self._solve_sinkhorn_mu(logits)
            if self.sinkhorn_persist_mu:
                self._update_running_mu(mu)
            return mu
        if self.sinkhorn_persist_mu and float(self.sinkhorn_mu_count.sum()) > 0:
            # EVAL: apply the dual for THIS iteration of the shared block, cycling with the
            # call index. Without this the router is evaluated untilted while it was trained
            # tilted, and the error compounds with iteration count -- measured WikiPPL among
            # pure+sinkhorn arms: 4 iterations 103.78, 8 iterations 177.09 and 316.21,
            # against ~41 for every hybrid (1 MoE block of 7) and 61.65 for the pure arm
            # whose clamped bias DOES survive to eval.
            k = int(self._mu_call.item()) % self.sinkhorn_mu_iters
            self._mu_call += 1
            if float(self.sinkhorn_mu_count[k]) > 0:
                return self.sinkhorn_mu[k]
        return None

    @torch.no_grad()
    def _update_running_mu(self, mu: torch.Tensor) -> None:
        """Running estimate of the per-batch dual, for use at eval.

        CUMULATIVE AVERAGE that decays into an EMA: weight = max(momentum, 1/count). For the
        first ~1/momentum batches this is an exact running mean (low variance, which is what a
        short calibration pass needs); afterwards it is a fixed-momentum EMA (tracks drift as
        mu falls over training -- measured 3.44 -> 1.97 at 134M). A plain EMA would have made
        a 64-batch calibration essentially a ONE-batch estimate, since after seeding, each
        later batch would move it only 1%."""
        k = int(self._mu_call.item()) % self.sinkhorn_mu_iters
        cnt = float(self.sinkhorn_mu_count[k]) + 1.0
        w = max(_SINKHORN_MU_MOMENTUM, 1.0 / cnt)
        self.sinkhorn_mu[k].mul_(1.0 - w).add_(mu.float(), alpha=w)
        self.sinkhorn_mu_count[k] += 1.0
        self._mu_call += 1

    @torch.no_grad()
    def _solve_sinkhorn_mu(self, logits: torch.Tensor) -> torch.Tensor:
        """Dual variables (chemical potentials) that equalise expert load.

        Fixed point of the log-domain Sinkhorn iteration
            mu <- mu + log(load(mu) * K),   load(mu) = mean_tokens softmax(logits - mu)
        Returned in LOGIT units (the tau division has already been applied), so the
        caller subtracts it directly. fp32 throughout for numerical headroom under bf16.
        """
        L = logits.detach().reshape(-1, self.n_experts).float()
        mu = torch.zeros(self.n_experts, device=L.device, dtype=L.dtype)
        for _ in range(self.sinkhorn_iters):
            load = F.softmax(L - mu, dim=-1).mean(0).clamp_min(1e-9)
            mu = mu + torch.log(load * self.n_experts)
        if self.track_load:
            self._sink_mu_absmax.copy_(mu.abs().max())
        return mu

    def _add_repulsion_loss_weight(self) -> None:
        """Repulsion on expert WEIGHT blocks. No token dimension, so O(1) in N."""
        if not self._repulsion_fires():
            return
        W = self._fused_W() if self.fused_experts else None
        if W is None:                      # looped path: stack the expert slices
            W = torch.cat([e._W_slice() for e in self.experts], dim=0)
        Wv = W.reshape(self.n_experts, -1)
        i_idx, j_idx = self._sample_pairs()
        if torch.is_tensor(i_idx):
            wi, wj = Wv.index_select(0, i_idx), Wv.index_select(0, j_idx)
        else:
            wi, wj = Wv[i_idx], Wv[j_idx]
        a = F.normalize(wi.float(), dim=-1)
        b = F.normalize(wj.float(), dim=-1)
        cos = (a * b).sum(-1)
        add_aux_loss(self._repulsion_coef_now()
                     * _repulsion_penalty(cos, self.repulsion_form).to(W.dtype))

    def _repulsion_fires(self) -> bool:
        """Stochastic 1-in-`interval` gate. Bernoulli rather than a step counter
        because this path is already stochastic (`random.sample` of pairs) and the
        block is called 48x per optimizer step (6 recurrence x 8 grad-accum), so a
        counter would need extra bookkeeping to mean '1 in N OPTIMIZER steps' and
        could desync from grad-accum. Equivalent in expectation, needs no state.

        USES TORCH CPU RNG, NOT `random`. Activation checkpointing re-runs the
        forward during backward and restores torch's RNG state around that
        recompute (preserve_rng_state=True) -- but NOT Python's `random`. With
        `random.random()` the gate could fire in the forward and not in the
        recompute, which changes the set of saved tensors and trips
        CheckpointError. The CPU generator keeps this off the GPU, so no sync.
        """
        if self.repulsion_interval <= 1:
            return True
        return bool(torch.randint(self.repulsion_interval, (1,), device="cpu").item() == 0)

    def _repulsion_coef_now(self) -> float:
        if self.repulsion_scale_comp and self.repulsion_interval > 1:
            return self.repulsion_coef * self.repulsion_interval
        return self.repulsion_coef

    def _sample_pairs(self):
        """Return (i_idx, j_idx). Python lists by default (bit-identical to every
        trained run); TENSORS when `repulsion_tensor_idx` is set, which keeps the
        indices out of the dynamo graph as constants. See the __init__ note."""
        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        if self.repulsion_tensor_idx:
            sel = torch.randint(self._pairs_t.shape[0], (k,), device=self._pairs_t.device)
            ij = self._pairs_t.index_select(0, sel)
            return ij[:, 0], ij[:, 1]
        sampled = random.sample(self._all_pairs, k)
        return [p[0] for p in sampled], [p[1] for p in sampled]

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        """Repulsion on random expert output pairs. See ``repulsion_form``."""
        if not self._repulsion_fires():
            return
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)
        i_idx, j_idx = self._sample_pairs()
        cos = (eg_norm.index_select(1, i_idx) * eg_norm.index_select(1, j_idx)).sum(-1) \
            if torch.is_tensor(i_idx) else \
            (eg_norm[:, i_idx, :] * eg_norm[:, j_idx, :]).sum(-1)
        add_aux_loss(self._repulsion_coef_now() * _repulsion_penalty(cos, self.repulsion_form))

    def pop_load_metrics(self) -> dict[str, float] | None:
        """Read and reset the traced load buffers. Call from the trainer, never in-graph.

        Unlike _log_metrics this works under torch.compile, because the accumulation is
        pure tensor arithmetic and only the final read (here) touches .item().
        """
        if not self.track_load:
            return None
        with torch.no_grad():
            n = self._tok_sum.clamp_min(1.0)
            share = (self._load_sum / n.clamp_min(1e-9))
            share = share / share.sum().clamp_min(1e-9)
            eff = torch.exp(-(share * (share + 1e-9).log()).sum())
            m = {
                "load_effective_n_experts": eff.item(),
                "load_max_share": share.max().item(),
                "load_min_share": share.min().item(),
                "load_mean_token_entropy": (self._ent_sum / n).item(),
                "load_tokens_seen": n.item(),
            }
            if self.balance_rate > 0.0 and self.load_balance_bias is not None:
                m["load_bias_absmax"] = self.load_balance_bias.abs().max().item()
            # Proxy-router fidelity. This is the number that decides whether the
            # cheap router is usable: it is the fraction of the exact router's
            # top-k set that the rank-r proxy also picks. Offline study got 0.94
            # top-1 at r=8; if this stays well below that, raise proxy_rank.
            if self.proxy_rank > 0 and self._proxy_agree_n > 0:
                m["proxy_topk_agree"] = (
                    self._proxy_agree_sum / self._proxy_agree_n.clamp_min(1.0)
                ).item()
                self._proxy_agree_sum.zero_(); self._proxy_agree_n.zero_()
            if self._E_n > 0:
                # energy_abs_mean GROWING across training = the positive-feedback
                # runaway the corrected sign makes possible. Watch its trend, not
                # its level.
                m["energy_abs_mean"] = (self._E_abs_sum / self._E_n.clamp_min(1.0)).item()
                m["energy_abs_max"] = self._E_abs_max.item()
                self._E_abs_sum.zero_(); self._E_abs_max.zero_(); self._E_n.zero_()
            m["repulsion_interval"] = float(self.repulsion_interval)
            m["repulsion_n_pairs"] = float(self.n_repulsion_pairs)
            m["repulsion_space_is_weight"] = float(self.repulsion_space == "weight")
            if self.sinkhorn_iters > 0:
                # Compare against load_bias_absmax, which pinned at the 1.0 clamp.
                m["sinkhorn_mu_absmax"] = self._sink_mu_absmax.item()
                m["sinkhorn_iters"] = float(self.sinkhorn_iters)
                if self.sinkhorn_persist_mu:
                    m["sinkhorn_mu_running_absmax"] = float(self.sinkhorn_mu.abs().max())
                    m["sinkhorn_mu_count"] = float(self.sinkhorn_mu_count.sum())
                    m["sinkhorn_mu_iters"] = float(self.sinkhorn_mu_iters)
            if self.cos_probe_interval > 0 and self._cos_n > 0:
                m["expert_cos_abs_mean"] = (self._cos_sum / self._cos_n.clamp_min(1.0)).item()
                self._cos_sum.zero_(); self._cos_n.zero_()
            self._load_sum.zero_(); self._ent_sum.zero_(); self._tok_sum.zero_()
        return m

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
        hopfield_grad_scale: str = "mean",
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.gelu_grad_method = gelu_grad_method
        assert hopfield_grad_scale in _HOPFIELD_GRAD_SCALES
        self.hopfield_grad_scale = hopfield_grad_scale
        self._W_slice = W_slice

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self._W_slice()
        Wx = x @ W.t()
        gelu_Wx, gelu_prime = _gelu_and_grad(Wx, self.gelu_grad_method)
        # PREFACTOR: was hardcoded (4.0 / intermediate_size) before 2026-09-12.
        # REVERT by setting hopfield_grad_scale="mean". See _hopfield_grad_prefactor.
        pref = _hopfield_grad_prefactor(self.intermediate_size, self.hopfield_grad_scale)
        gated = gelu_Wx * gelu_prime
        out = pref * (gated @ W)

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
        hopfield_grad_scale: str = "mean",
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
        self.hopfield_grad_scale = hopfield_grad_scale

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
                    hopfield_grad_scale=self.hopfield_grad_scale,
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

    def pop_load_metrics(self):
        # Mirror the inner MoE's traced load stats so the trainer finds them on the
        # container too (train_utils skips the inner '.moe' to avoid double-logging).
        return self.moe.pop_load_metrics()

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
    repulsion_form: str = "squared",
    routing_norm: str = "none",
    renormalize_topk: bool = False,
    track_load: bool = True,
    balance_rate: float = 0.0,
    repulsion_interval: int = 1,
    repulsion_scale_comp: bool = True,
    fused_experts: bool = False,
    proxy_rank: int = 0,
    proxy_loss_coef: float = 0.0,
    proxy_route: bool = False,
    proxy_kind: str = "quad",
    proxy_out_dim: int = 0,
    proxy_iters: int = 1,
    cos_probe_interval: int = 0,
    cos_probe_pairs: int = 8,
    repulsion_space: str = "output",
    e_sign_override: str | None = None,
    sinkhorn_iters: int = 0,
    sinkhorn_persist_mu: bool = False,
    sinkhorn_mu_iters: int = 1,
    sparse_backproj: bool = False,
    sparse_forward: bool = False,
    sparse_candidates: int = 0,
    sparse_capacity_factor: float = 1.25,
    repulsion_subsample: int = 0,
    repulsion_tensor_idx: bool = False,
    init_method: str = "normal",
    initializer_range: float = 0.02,
    m_width: float | None = None,
    add_bias: bool = False,
    gelu_grad_method: str = "sigmoid",
    hopfield_grad_scale: str = "mean",
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
            hopfield_grad_scale=hopfield_grad_scale,
        )
        e_sign = "neg"
    else:
        raise ValueError(f"unknown expert_kind ({expert_kind})")
    # ------------------------------------------------------------------------ #
    # ROUTING SIGN. `e_sign` decides whether softmax favours the SMALLEST stored
    # energy ("neg" -> logits = -E) or the LARGEST ("pos" -> logits = +E).
    #
    # The kind-based defaults below are what every existing checkpoint trained
    # with, and they are RETAINED as the default so nothing changes implicitly.
    # But they appear to be INVERTED -- see ROUTING_SIGN_BUG_20260915.md:
    #   _HopfieldExpert stores E = +mean(gelu(Wx)^2), which GROWS with overlap,
    #   and "neg" then selects the SMALLEST -> the WORST-matching experts
    #   (measured: 0.62x the average expert's overlap, exactly the lowest set).
    #   _W1W2Expert stores E = -overlap and uses "pos", which also selects the
    #   worst match -- whereas the legacy class it was meant to reproduce does
    #   E = +overlap with softmax(+E/tau), i.e. the BEST match.
    # Set e_sign_override="pos" on a hopfield MoE to route on HIGH overlap.
    if e_sign_override is not None:
        assert e_sign_override in ("neg", "pos")
        e_sign = e_sign_override
    experts = holder.make_experts()
    # The fused path needs the single shared weight tensor + the scalars the
    # per-expert forward would have applied. `weight_fn` is a closure so FSDP
    # re-gathers are picked up (same reason the expert W_slice closures exist).
    if expert_kind == "hopfield":
        fused_spec = {
            "kind": "hopfield",
            "weight_fn": (lambda: holder.W.weight),
            "gelu_grad_method": gelu_grad_method,
            "hopfield_grad_scale": hopfield_grad_scale,
        }
    else:
        fused_spec = {"kind": expert_kind}
    moe = BoltzmannMoEFFEnergy(
        experts,
        hidden_size=hidden_size,
        temperature=temperature,
        repulsion_coef=repulsion_coef,
        n_repulsion_pairs=n_repulsion_pairs,
        top_k=top_k,
        e_sign=e_sign,
        layer_idx=layer_idx,
        repulsion_form=repulsion_form,
        routing_norm=routing_norm,
        renormalize_topk=renormalize_topk,
        track_load=track_load,
        balance_rate=balance_rate,
        repulsion_interval=repulsion_interval,
        repulsion_scale_comp=repulsion_scale_comp,
        fused_experts=fused_experts,
        fused_spec=fused_spec,
        proxy_rank=proxy_rank,
        proxy_loss_coef=proxy_loss_coef,
        proxy_route=proxy_route,
        proxy_kind=proxy_kind,
        proxy_out_dim=proxy_out_dim,
        proxy_iters=proxy_iters,
        cos_probe_interval=cos_probe_interval,
        cos_probe_pairs=cos_probe_pairs,
        repulsion_space=repulsion_space,
        sinkhorn_iters=sinkhorn_iters,
        sinkhorn_persist_mu=sinkhorn_persist_mu,
        sinkhorn_mu_iters=sinkhorn_mu_iters,
        sparse_backproj=sparse_backproj,
        sparse_forward=sparse_forward,
        sparse_candidates=sparse_candidates,
        repulsion_subsample=repulsion_subsample,
        sparse_capacity_factor=sparse_capacity_factor,
        repulsion_tensor_idx=repulsion_tensor_idx,
    )
    return FusedMoEContainer(expert_holder=holder, moe=moe)
