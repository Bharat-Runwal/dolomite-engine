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
        cos_probe_interval: int = 0,
        cos_probe_pairs: int = 8,
        repulsion_space: str = "output",
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
        self.proxy_rank = int(proxy_rank)
        self.proxy_loss_coef = float(proxy_loss_coef)
        self.proxy_route = bool(proxy_route)
        if self.proxy_rank > 0:
            r = self.proxy_rank
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
                torch.randn(self.n_experts, hidden_size, r, generator=_g)
                / (hidden_size ** 0.5)
            )
            # Diagonal quadratic + linear head on the r coefficients: the true
            # Hopfield energy is quadratic in W_k x, so a quadratic form in the
            # projection is the right inductive bias (and is 2r+1 params/expert).
            self.proxy_quad = nn.Parameter(torch.ones(self.n_experts, r) / r)
            self.proxy_lin = nn.Parameter(torch.zeros(self.n_experts, r))
            self.proxy_bias = nn.Parameter(torch.zeros(self.n_experts))
            self.register_buffer("_proxy_agree_sum", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("_proxy_agree_n", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
        else:
            self.proxy_V = None

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
        self.cos_probe_interval = int(cos_probe_interval)
        self.cos_probe_pairs = int(cos_probe_pairs)
        if self.cos_probe_interval > 0:
            self.register_buffer("_cos_sum", torch.zeros((), dtype=torch.float32),
                                 persistent=False)
            self.register_buffer("_cos_n", torch.zeros((), dtype=torch.float32),
                                 persistent=False)

    # --- public surface --------------------------------------------------- #

    def forward(self, x: torch.Tensor) -> torch.Tensor:
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

    def _route(self, E_k: torch.Tensor):
        """Energies -> (p, logits). Verbatim the pre-2026-09-15 routing block."""
        logits = self._logits(E_k)
        p = F.softmax(logits, dim=-1)
        if self.top_k is not None and self.top_k < self.n_experts:
            _, topk_idx = logits.topk(self.top_k, dim=-1)
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

        p, logits = self._route(E_k)
        self._track_load(p)

        if self.proxy_rank > 0:
            self._proxy_step(x, E_k, logits)

        gated = (gelu_Wx * gelu_prime).view(*lead, K, I_e)
        gw = (gated * (pref * p).unsqueeze(-1)).reshape(*lead, K * I_e)
        out = gw @ W                             # ONE backward-projection GEMM

        if self.training and self._capture_energy:
            self._last_energy_per_token = -self.temperature * torch.logsumexp(logits, dim=-1)
        else:
            self._last_energy_per_token = None

        if self.training and self.repulsion_coef > 0:
            if self.repulsion_space == "weight":
                self._add_repulsion_loss_weight()
            else:
                self._add_repulsion_loss_fused(gated, W, pref)

        if self.training and self._cos_probe_fires():
            self._probe_expert_cos(gated, W=W, pref=pref)

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

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
        ei = pref * torch.einsum("npi,pih->nph", g[:, i_idx, :], Wv[i_idx])
        ej = pref * torch.einsum("npi,pih->nph", g[:, j_idx, :], Wv[j_idx])
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
            ei, ej = eg[:, i_idx, :], eg[:, j_idx, :]
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
        a = torch.einsum("...h,khr->...kr", x.detach(), self.proxy_V.to(x.dtype))
        q = self.proxy_quad.to(x.dtype)
        l = self.proxy_lin.to(x.dtype)
        return (q * a * a).sum(-1) + (l * a).sum(-1) + self.proxy_bias.to(x.dtype)

    def _proxy_step(self, x: torch.Tensor, E_k: torch.Tensor,
                    logits: torch.Tensor) -> None:
        """Distil the proxy against the exact routing distribution, and MEASURE its
        top-k agreement. Off the main forward path: this adds an aux loss and a
        metric, and changes nothing the model computes (unless proxy_route).

        The measurement matters more than the loss. HANDOFF 7.9 records that
        `torch.isin` silently fakes top-k agreement, so agreement is computed here
        as a genuine per-row set overlap: |top-k(proxy) INTERSECT top-k(exact)| / k.
        """
        E_hat = self._proxy_energies(x)
        logits_hat = self._logits(E_hat)

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
        s = -E_k if self.e_sign == "neg" else E_k
        if self.routing_norm == "sqrt_width":
            s = s * (self.experts[0].intermediate_size ** 0.5)
        elif self.routing_norm == "zscore":
            s = (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp_min(1e-12)
        # Bias is added AFTER normalisation: zscore would otherwise rescale it away, and
        # it is defined in the same (post-norm) space the temperature divides.
        if self.balance_rate > 0.0 and self.load_balance_bias is not None:
            # Cast to the logits' dtype: the buffer is fp32 while s is bf16 under mixed
            # precision, and the implicit promotion failed during fake-tensor tracing.
            s = s + self.load_balance_bias.to(s.dtype)
        return s / self.temperature

    def _add_repulsion_loss_weight(self) -> None:
        """Repulsion on expert WEIGHT blocks. No token dimension, so O(1) in N."""
        if not self._repulsion_fires():
            return
        W = self._fused_W() if self.fused_experts else None
        if W is None:                      # looped path: stack the expert slices
            W = torch.cat([e._W_slice() for e in self.experts], dim=0)
        Wv = W.reshape(self.n_experts, -1)
        i_idx, j_idx = self._sample_pairs()
        a = F.normalize(Wv[i_idx].float(), dim=-1)
        b = F.normalize(Wv[j_idx].float(), dim=-1)
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
        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, k)
        return [p[0] for p in sampled], [p[1] for p in sampled]

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        """Repulsion on random expert output pairs. See ``repulsion_form``."""
        if not self._repulsion_fires():
            return
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)
        i_idx, j_idx = self._sample_pairs()
        cos = (eg_norm[:, i_idx, :] * eg_norm[:, j_idx, :]).sum(-1)
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
            m["repulsion_interval"] = float(self.repulsion_interval)
            m["repulsion_n_pairs"] = float(self.n_repulsion_pairs)
            m["repulsion_space_is_weight"] = float(self.repulsion_space == "weight")
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
    cos_probe_interval: int = 0,
    cos_probe_pairs: int = 8,
    repulsion_space: str = "output",
    e_sign_override: str | None = None,
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
        cos_probe_interval=cos_probe_interval,
        cos_probe_pairs=cos_probe_pairs,
        repulsion_space=repulsion_space,
    )
    return FusedMoEContainer(expert_holder=holder, moe=moe)
