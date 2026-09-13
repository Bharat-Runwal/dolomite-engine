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
        # persistent: the bias is part of the trained routing rule and must survive a resume
        self.register_buffer("load_balance_bias", torch.zeros(self.n_experts), persistent=True)
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

        # Traceable load accounting: tensor ops only, no .item(), no host sync, so this
        # survives torch.compile where _log_metrics does not.
        if self.track_load and self.training:
            with torch.no_grad():
                pf = p.detach().reshape(-1, self.n_experts).float()
                self._load_sum += pf.sum(0)
                self._ent_sum += -(pf * (pf + 1e-9).log()).sum(-1).sum()
                self._tok_sum += pf.shape[0]
                if self.balance_rate > 0.0:
                    # Under-loaded experts get a positive nudge, over-loaded a negative one.
                    share = pf.mean(0)
                    target = share.new_full((), 1.0 / self.n_experts)
                    self.load_balance_bias += self.balance_rate * torch.sign(target - share)

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
        if self.balance_rate > 0.0:
            s = s + self.load_balance_bias
        return s / self.temperature

    def _add_repulsion_loss(self, expert_grads: torch.Tensor) -> None:
        """Repulsion on random expert output pairs. See ``repulsion_form``."""
        eg = expert_grads.reshape(-1, self.n_experts, self.hidden_size)
        eg_norm = F.normalize(eg, dim=-1)
        k = min(self.n_repulsion_pairs, len(self._all_pairs))
        sampled = random.sample(self._all_pairs, k)
        i_idx = [p[0] for p in sampled]
        j_idx = [p[1] for p in sampled]
        cos = (eg_norm[:, i_idx, :] * eg_norm[:, j_idx, :]).sum(-1)
        add_aux_loss(self.repulsion_coef * _repulsion_penalty(cos, self.repulsion_form))

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
            if self.balance_rate > 0.0:
                m["load_bias_absmax"] = self.load_balance_bias.abs().max().item()
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
        repulsion_form=repulsion_form,
        routing_norm=routing_norm,
        renormalize_topk=renormalize_topk,
        track_load=track_load,
        balance_rate=balance_rate,
    )
    return FusedMoEContainer(expert_holder=holder, moe=moe)
