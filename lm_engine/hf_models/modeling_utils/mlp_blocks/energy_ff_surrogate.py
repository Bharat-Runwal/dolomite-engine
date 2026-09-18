# **************************************************
# KL-distilled surrogate router for the COMPOSABLE energy-FF family -- 2026-09-18
# **************************************************
"""Port of the legacy ``SurrogateBoltzmannMoE_Energy_MLP`` onto ``BoltzmannMoEFFEnergy``.

The legacy class (``mlp.py:994``, config ``mlp.py:483``) bolted a learned linear router onto
the legacy w1w2 Boltzmann MoE: compute ``p_boltz`` from the expert energies, compute ``p_surr``
from a ``d -> K`` linear layer, add a distillation loss between them, and at eval with
``use_surrogate=True`` route on the head alone. It predates EVERY routing feature the
composable class has (Sinkhorn dual, top-k, renormalisation, routing_norm, balance bias,
fused/sparse paths), so a literal transcription would silently disagree with the model it is
meant to imitate. This module ports the IDEA and wires it through the composable machinery.

NOTHING IN ``energy_ff.py`` IS MODIFIED -- six long training runs import it. Everything here is
a subclass override or an import.

===============================================================================================
WHAT IS ACTUALLY REUSED: ZERO COPIED BLOCKS
===============================================================================================

The sibling ``energy_ff_w1w2_sparse.py`` had to copy ~40 lines of ``_forward_sparse`` because
the weighting block is inline with no seam. This module needs NO copy, because the seam it
needs already exists: every dense forward path funnels its energies through ``_route``.

    forward()        stash the head's pseudo-energies, then call the BASE forward
    _route()         (a) training: add the distillation loss
                     (b) surrogate eval: hand the BASE `_route` the HEAD's energies
                         instead of the exact ones, so mu / top_k / renormalize_topk /
                         routing_norm / balance-bias are applied by the SAME code

So ``_forward_looped`` and ``_forward_fused`` are used verbatim, for both expert kinds, and
there is no drift guard to maintain.

===============================================================================================
(a) WHY A DISTILLED HEAD IS EXPERT-KIND-AGNOSTIC, AND WHERE THAT STOPS BEING TRUE
===============================================================================================

The head reads ``x`` and writes K numbers. It never touches an expert weight, never assumes a
functional form for the energy, and never estimates it -- it REGRESSES the routing
distribution. So ``expert_kind="hopfield"`` and ``expert_kind="w1w2"`` are handled by the same
code with no change, and so would any future ``FFEnergyBase`` expert.

CONTRAST WITH THE RANK-r SUBSPACE PROXY, WHICH IS *NOT* KIND-AGNOSTIC. That proxy evaluates
the energy's own algebraic form on a rank-r projection and subsamples m of the I_e inner
units. For Hopfield, ``E = mean_j gelu(z_j)^2`` is a mean of NONNEGATIVE terms, so an m-row
subsample has relative error ``~1/sqrt(m)``. For w1w2, ``E = -I_e^-0.5 * sum_j phi(u_j) v_j``
sums SIGNED terms that cancel, so the same estimator has relative error ``~sqrt(I_e/m)`` --
measured cancellation ratio **0.0254 for w1w2 against 1.000 for hopfield**
(``energy_ff_w1w2_sparse.py`` docstring section 8, TEST 8 of
``experiments/boltzmann-moe/scripts/test_sparse_w1w2_20260918.py``). At I_e=2240, m=512 that is
~210% error, i.e. noise, which forces ``m = I_e`` and destroys the proxy's cost advantage.

WHERE THE AGNOSTICISM STOPS:
  * It is agnostic to the energy's FORM. It is NOT agnostic to the energy's LEARNABILITY from
    ``x`` -- a head with too little capacity is wrong for every kind, and how much capacity is
    enough is an empirical question per architecture (unmeasured here; needs a GPU).
  * It gives RANKING, not VALUES. With ``renormalize_topk: true`` the softmax runs over the
    selected logits only, so a monotone-correct head suffices. With ``renormalize_topk:
    false`` the denominator runs over all K and the head's MAGNITUDE errors enter the output
    scale directly. Prefer ``renormalize_topk: true`` with a surrogate.
  * It cannot supply the exact energies that ``_track_energy``, ``energy_per_token`` and the
    block's energy-descent probes consume. Those all keep using the exact path here (see
    ``_route``: we substitute the ROUTING distribution, not the energy).

===============================================================================================
(b) HEAD CAPACITY AND ITS PER-TOKEN MAC COST -- INDEPENDENT OF I_e
===============================================================================================

    surrogate_kind="linear"   x -> K                 MACs/token = d*K
    surrogate_kind="mlp"      x -> h -> K            MACs/token = d*h + h*K

NEITHER CONTAINS I_e. That is the whole structural point against the subspace proxy, whose
cost is ``K*d*r + K*m*r`` with ``m <= I_e`` and ``m = I_e`` forced for w1w2 -- i.e. ``K*I_e*r``
in the term that dominates. Worked numbers at the 400M hybrid shape (d=1024, K=32,
I_total=187872 so I_e=5871, r=16):

    exact Hopfield router      K*I_e*d = I_total*d       192,380,928 MACs/token
    subspace proxy, m=512      K*d*r + K*m*r                 786,432
    subspace proxy, m=I_e      K*d*r + K*I_e*r             3,530,240
    surrogate, linear          d*K                            32,768    (5871x under exact)
    surrogate, mlp h=256       d*h + h*K                     270,336

and the head's parameter count equals its MAC count in both cases (no weight reuse).

BE HONEST ABOUT WHAT THAT BUYS TODAY. In the FUSED DENSE Hopfield path the exact energy is a
by-product: ``E_k = (gelu_Wx**2).mean(-1)`` is a T*K*I_e ELEMENTWISE reduction over an
activation the expert gradient already materialised, not a T*K*I_e*d GEMM. So switching a
dense arm's eval onto the head saves that reduction and nothing else -- the forward projection
still runs for all K because the expert OUTPUTS need it. The FLOP win requires skipping
experts, which is ``sparse_forward``, which is refused here (see (d)). What the head is for
right now:
  1. a DIAGNOSTIC: how much of the energy router's decision is decodable from h at what
     capacity -- measured by ``surrogate_topk_agree``, directly comparable to the proxy's
     ``proxy_topk_agree`` because both use the same set-overlap estimator;
  2. the TRAINING-TIME distillation that has to happen before a head can ever replace the
     proxy in the sparse path;
  3. a selector whose cost does not grow with I_e, which is the one property the subspace
     proxy structurally cannot have.

===============================================================================================
(c) THE DESIGN DECISION: THE KL TARGET IS THE **PRE-mu** (PRE-SINKHORN) DISTRIBUTION
===============================================================================================

``_route`` in the base class does, in order:

    logits = _logits_raw(E_k)      # sign flip, routing_norm, balance bias, /temperature
    mu     = _mu_for(logits)       # Sinkhorn dual (chemical potential), or None
    logits = logits - mu
    p      = softmax(logits)       # then top_k mask, then optional renormalisation

DECISION: the head is trained against ``softmax(_logits_raw(E_k))`` -- the distribution
BEFORE ``mu`` is subtracted -- and ``mu`` is applied to the head's own logits at use time,
because the head is used THROUGH ``_route``. Trained pre-mu, used pre-mu-then-tilted: the
head sees the same space in training and in eval, and the model is tilted in both.

FIVE REASONS, in decreasing order of how badly the alternative fails:

 1. **mu IS NOT A FUNCTION OF THE TOKEN.** It is the dual variable of a BATCH-MARGINAL
    constraint: ``mu <- mu + log(K * mean_tokens softmax(logits - mu))``. A per-token head
    cannot represent a per-batch constant except by absorbing its running average, so
    distilling the post-mu target asks the head to fit something its input does not contain.
    The residual lands in the head's bias as an average, which is wrong for every batch.

 2. **A RECURRENT BLOCK HAS N DIFFERENT mu AND ONE HEAD.** ``sinkhorn_mu_iters`` equals the
    block's ``layer_iterations`` entry, and ``_mu_for`` cycles ``sinkhorn_mu[_mu_call % N]``
    so that eval replays the per-iteration duals in the order training filled them. The live
    arms set N = 4 and N = 6. One head is called N times per forward; a head that had
    absorbed "mu" could only have absorbed the average of N genuinely different vectors.
    Applying the correct per-iteration mu on top is the only way to get all N right, and it
    is FREE because the base class already does it.

 3. **DISTILLING POST-mu AND THEN APPLYING mu COUNTS IT TWICE.** This is not hypothetical --
    it is what the existing rank-r proxy does in the dense path, and it is worth reporting
    rather than reproducing:
        ``_proxy_step`` (energy_ff.py:1788) sets ``flat_tgt = softmax(logits.detach())`` where
        ``logits`` is the POST-mu value returned by ``_route``, while its prediction
        ``logits_hat = _logits_raw(E_hat)`` is PRE-mu. So the proxy is trained toward
        ``raw_exact - mu``. Then ``_route`` selects with ``sel_logits = _logits_raw(proxy_E) -
        mu``, i.e. ``~ raw_exact - 2*mu``.
    ``sinkhorn_mu_absmax`` is measured at **3.44** in logit units at 134M, i.e. an e^3.44 ~ 31x
    tilt; doubling it is not a rounding error. (The SPARSE path is self-consistent -- it
    distils post-mu against post-mu ``s_prox`` -- so the inconsistency is dense-path-only, and
    it also means a ``sparse_start_step`` run changes the proxy's target convention at the
    dense->sparse handover. Reported, not fixed: energy_ff.py is frozen.)

 4. **THE FAILURE THE PRE-mu CHOICE COULD HAVE CAUSED IS CLOSED BY CONSTRUCTION.** "The head
    learns the untilted distribution while the model is trained tilted" happens if you train
    pre-mu and then USE the head's output as the routing logits directly. We never do: the
    surrogate path calls ``super()._route(E_surr)``, which subtracts mu. Training space and
    use space match.

 5. **THE BALANCE BIAS IS HANDLED BY THE SAME ARGUMENT, IN THE OPPOSITE DIRECTION.**
    ``load_balance_bias`` (``balance_rate > 0``) is also a per-expert, non-per-token additive
    term -- but unlike mu it is applied INSIDE ``_logits_raw``, which we apply to BOTH the
    exact energies and the head's pseudo-energies. So it appears symmetrically on both sides
    of the KL and is applied exactly once at use. No special handling; nothing to decide.
    (``sinkhorn_iters > 0`` and ``balance_rate > 0`` are mutually exclusive in the base class.)

CONSEQUENCE THAT MUST BE ASSERTED, NOT ASSUMED: at eval the surrogate path's ``_mu_for``
cannot solve the dual -- solving it needs the all-K exact logits, which is exactly what the
head exists to avoid. It can only READ the persisted running dual. So
``sinkhorn_iters > 0`` with ``use_surrogate`` REQUIRES ``sinkhorn_persist_mu: true`` AND a
calibration pass that filled ``sinkhorn_mu_count``; otherwise ``_mu_for`` returns None and
the surrogate routes UNTILTED, which is the +1.587 nats (8 iterations) / +1.827 nats
(12 iterations) failure already recorded for untilted eval. Both are checked below --
the config one at construction, the buffer one at first surrogate eval.

WHAT IS *NOT* DISTILLED, DELIBERATELY: the top-k MASK and the renormalisation. Those are
functions of the logits, so the head inherits them from ``_route`` for free and the KL stays a
comparison of two full-K distributions. Distilling the post-mask distribution would put
hard zeros in the target and make the loss blow up on any token the head ranks differently.

===============================================================================================
(d) NO COMBINATION WITH sparse_forward YET
===============================================================================================

Refused at construction. ``_forward_sparse`` does not call ``_route``; it inlines
``_logits_raw`` twice and solves the dual on the PROXY logits, because the proxy is the only
all-K quantity it has. Substituting the head there means deciding whether the head REPLACES
the proxy (then ``proxy_rank`` machinery, the SVD warm start and the re-rank denominator all
need rethinking) or merely ADDS to it (then two cheap routers disagree and nothing says which
wins). That is a separate change, and it has a hard prerequisite: a dense phase, because the
head is distilled against the exact all-K routing distribution, which the sparse path never
computes.

===============================================================================================
FIVE DIFFERENCES FROM THE LEGACY CLASS, ALL DELIBERATE
===============================================================================================

 1. **THE LEGACY "KL" IS A CROSS-ENTROPY.** ``mlp.py`` computes
        ``kl = -(p_boltz.detach() * (p_surr + 1e-8).log()).sum(-1).mean()``
    which is ``KL(p_boltz || p_surr) + H(p_boltz)``. The gradient is identical (the entropy is
    a constant once p_boltz is detached), so no trained checkpoint is affected -- but the
    VALUE has an irreducible floor of H(p_boltz) and can never reach 0. It was also logged as
    ``kl_surr_boltz`` and is readable as a distillation gap only after subtracting an entropy
    nobody recorded. Here the loss is a TRUE KL via ``F.kl_div(log_softmax(pred), tgt)``, so
    0 means "the head reproduces the router" (TEST 1).
 2. **DIRECTION.** Both the legacy class and ``_proxy_step`` use FORWARD KL with the router as
    target (mass-covering: the head must put mass wherever the router does). Kept as the
    default so the surrogate and the proxy are directly comparable. ``surrogate_kl_direction
    = "reverse"`` is available for the mode-seeking variant; it is UNMEASURED.
 3. **INPUT DETACH.** The legacy head read the LIVE ``x``, so its KL reshaped the backbone.
    ``_proxy_energies`` detaches, and its comment gives the reason: detaching is what makes
    "enabling this cannot degrade training" a true statement. Default here is ``True``
    (detach); ``surrogate_detach_input: false`` recovers the legacy behaviour as an explicit
    experiment. NOTE, from the same comment: detaching blocks the GRAPH but not the OPTIMISER
    -- the head's parameters are in ``model.parameters()`` and ``gradient_clipping: 1`` is a
    GLOBAL norm, so an inflated head gradient still scales the backbone's gradients down.
    Hence the per-token normalisation in (4).
 4. **PER-TOKEN NORMALISATION.** ``F.kl_div(reduction="batchmean")`` divides by dim 0 only, so
    on a ``(batch, seq, K)`` tensor it sums over ``seq*K`` and divides by ``batch``, inflating
    the loss by the sequence length -- measured aux_loss 29.47 against lm_loss 7.52 at
    seq=4096. Flattened to ``(tokens, K)`` first here, as ``_proxy_step`` does.
 5. **NO SEPARATE temperature.** The legacy head divided its own logits by
    ``self.temperature``. Here ``_logits_raw`` does that for both sides, so the head and the
    router share one temperature by construction instead of by coincidence.
"""

from __future__ import annotations

import logging
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss import add_aux_loss
from .energy_ff import (
    BoltzmannMoEFFEnergy,
    FFEnergyBase,
    FusedMoEContainer,
    _FusedHopfieldHolder,
    _FusedW1W2Holder,
)


logger = logging.getLogger(__name__)

_SURROGATE_KINDS = ("linear", "mlp")
_KL_DIRECTIONS = ("forward", "reverse")


def surrogate_head_macs(hidden_size: int, n_experts: int, kind: str,
                        hidden_dim: int = 0) -> int:
    """Per-token MACs of the head. See section (b): NO ``I_e`` appears in either branch."""
    assert kind in _SURROGATE_KINDS, kind
    if kind == "linear":
        return hidden_size * n_experts
    return hidden_size * hidden_dim + hidden_dim * n_experts


class _SurrogateRouterMixin:
    """The distilled-head behaviour, as a mixin so it composes with either MoE base class.

    A MIXIN AND NOT A SUBCLASS, for one concrete reason: the frozen base class asserts
    ``fused_spec["kind"] == "hopfield"`` whenever ``fused_experts=True``, so a single subclass
    of ``BoltzmannMoEFFEnergy`` could only ever be fused for Hopfield. Mixing into
    ``BoltzmannMoEW1W2Sparse`` (the sibling module, which already carries the shim for that
    assert) gives the fused w1w2 line the same head with no duplicated code -- which is the
    demonstration that the head is kind-agnostic, rather than a claim about it.

    Cooperative ``__init__``: consumes its own kwargs and forwards the rest up the MRO.
    """

    def __init__(
        self,
        *args,
        surrogate_coef: float = 0.0,
        use_surrogate: bool = False,
        surrogate_kind: str = "linear",
        surrogate_hidden: int = 0,
        surrogate_kl_direction: str = "forward",
        surrogate_detach_input: bool = True,
        surrogate_init_std: float = 0.01,
        surrogate_track_agree: bool = True,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        assert surrogate_kind in _SURROGATE_KINDS, surrogate_kind
        assert surrogate_kl_direction in _KL_DIRECTIONS, surrogate_kl_direction
        assert surrogate_coef >= 0.0
        if surrogate_kind == "mlp":
            assert surrogate_hidden > 0, "surrogate_kind='mlp' needs surrogate_hidden > 0"

        # ---- (d) no sparse combination ------------------------------------------------ #
        assert not self.sparse_forward, (
            "the surrogate head is NOT wired into _forward_sparse. That path does not call "
            "_route (it inlines _logits_raw twice and solves the Sinkhorn dual on the PROXY "
            "logits, because the proxy is its only all-K quantity), so the head would be "
            "computed and then ignored -- a silent no-op, not an error. Deciding whether the "
            "head REPLACES the proxy there (proxy_rank / proxy_init: svd / the re-rank "
            "denominator all change) or merely ADDS to it is a separate change. Train the "
            "head on a DENSE arm first: it is distilled against the exact all-K routing "
            "distribution, which the sparse path never computes."
        )
        assert not self.sparse_backproj, (
            "sparse_backproj is untested with the surrogate head; it is also redundant with "
            "sparse_forward (1.18-1.35x against 4.69x)."
        )

        # ---- (c) the Sinkhorn prerequisite -------------------------------------------- #
        # At eval the head cannot solve the dual -- that needs the all-K exact logits, which
        # is the thing it exists to avoid -- so it can only READ the persisted running mu.
        if use_surrogate and self.sinkhorn_iters > 0:
            assert self.sinkhorn_persist_mu, (
                "use_surrogate with sinkhorn_iters > 0 REQUIRES sinkhorn_persist_mu: true. "
                "Without it _mu_for returns None at eval and the surrogate routes UNTILTED "
                "while the model was trained tilted -- the measured cost of untilted eval is "
                "+1.587 nats at 8 iterations and +1.827 at 12. Also run a calibration pass so "
                "sinkhorn_mu_count is non-zero; that is checked at the first surrogate eval."
            )

        self.surrogate_coef = float(surrogate_coef)
        self.use_surrogate = bool(use_surrogate)
        self.surrogate_kind = surrogate_kind
        self.surrogate_hidden = int(surrogate_hidden)
        self.surrogate_kl_direction = surrogate_kl_direction
        self.surrogate_detach_input = bool(surrogate_detach_input)
        self.surrogate_track_agree = bool(surrogate_track_agree)

        # ---- the head ----------------------------------------------------------------- #
        # DEDICATED GENERATOR. The base class's own comment records that drawing on the
        # global stream shifts the init of every parameter created afterwards and moved an
        # arm's lm_loss by 4.4x the noise floor. The head is built last here, so the risk is
        # to anything built after this block -- and to bitwise A/B against a no-surrogate
        # twin, which TEST 4 requires. A private generator makes that A/B exact.
        g = torch.Generator().manual_seed(0x5C0FFEE ^ (self.layer_idx or 0))
        H, K = self.hidden_size, self.n_experts
        if surrogate_kind == "linear":
            self.surrogate_W1 = nn.Parameter(
                torch.randn(K, H, generator=g) * surrogate_init_std)
            self.surrogate_b1 = nn.Parameter(torch.zeros(K))
            self.surrogate_W2 = self.surrogate_b2 = None
        else:
            h = surrogate_hidden
            self.surrogate_W1 = nn.Parameter(
                torch.randn(h, H, generator=g) * surrogate_init_std)
            self.surrogate_b1 = nn.Parameter(torch.zeros(h))
            self.surrogate_W2 = nn.Parameter(
                torch.randn(K, h, generator=g) * surrogate_init_std)
            self.surrogate_b2 = nn.Parameter(torch.zeros(K))

        # Traced accumulators, same discipline as the base class's load buffers: pure tensor
        # arithmetic, no `.item()`, so they survive torch.compile where `_log_metrics` does
        # not. Read (and reset) in `pop_load_metrics`.
        self.register_buffer("_surr_kl_sum", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_surr_kl_n", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_surr_agree_sum", torch.zeros((), dtype=torch.float32),
                             persistent=False)
        self.register_buffer("_surr_agree_n", torch.zeros((), dtype=torch.float32),
                             persistent=False)

        # Stash for the head's output between `forward` and `_route`. Holding a tensor on the
        # module for the duration of one forward is the convention this file already uses for
        # `_last_energy_per_token` and the `_capture_energy` flag, so it is not a new kind of
        # state. It is cleared in a `finally`.
        self._surr_E: torch.Tensor | None = None
        self._surr_warned_mu = False

    # ------------------------------------------------------------------------------- #
    # the head                                                                         #
    # ------------------------------------------------------------------------------- #

    def surrogate_energies(self, x: torch.Tensor) -> torch.Tensor:
        """PSEUDO-ENERGIES, shape ``(..., K)``, in the same units as the expert energies.

        RETURNING ENERGIES RATHER THAN LOGITS IS THE REUSE DECISION. Everything downstream --
        the ``e_sign`` flip, ``routing_norm`` (none / zscore / sqrt_width), the balance bias,
        the temperature division, the Sinkhorn dual, the top-k mask and
        ``renormalize_topk`` -- lives in ``_logits_raw`` and ``_route``, which take energies.
        Hand them energies and all eight behaviours come along with zero duplicated code and
        no possibility of drifting from the exact path. Hand them logits instead and every one
        of those has to be re-implemented here.

        The head is free to learn whatever sign ``e_sign`` demands; with
        ``routing_norm="zscore"`` its output scale is also irrelevant, since the z-score is
        taken over the K axis of ITS OWN output (exactly as the proxy's is -- see
        ``_zscore_moments``, and note this means the two sides of the KL are each standardised
        by their own moments, which is what makes a scale-free head trainable at all).
        """
        h = x.detach() if self.surrogate_detach_input else x
        z = F.linear(h, self.surrogate_W1.to(h.dtype), self.surrogate_b1.to(h.dtype))
        if self.surrogate_kind == "linear":
            return z
        return F.linear(F.gelu(z), self.surrogate_W2.to(h.dtype),
                        self.surrogate_b2.to(h.dtype))

    def surrogate_macs_per_token(self) -> int:
        return surrogate_head_macs(self.hidden_size, self.n_experts, self.surrogate_kind,
                                   self.surrogate_hidden)

    # ------------------------------------------------------------------------------- #
    # the two overrides                                                                #
    # ------------------------------------------------------------------------------- #

    @property
    def _surrogate_routes_now(self) -> bool:
        """Eval-time replacement of the router. Same gate as the legacy class:
        ``not self.training and self.use_surrogate``. Never active in training -- the head has
        to be distilled against the exact router, so the exact router must run."""
        return (not self.training) and self.use_surrogate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        need = self._surrogate_routes_now or (self.training and self.surrogate_coef > 0) \
            or (self.surrogate_track_agree and self.training)
        if not need:
            # Nothing to add: fall straight through, so a `surrogate_coef: 0` /
            # `use_surrogate: false` instance is EXACTLY the base class (TEST 4).
            return super().forward(x)
        self._surr_E = self.surrogate_energies(x)
        try:
            return super().forward(x)
        finally:
            self._surr_E = None

    def _route(self, E_k: torch.Tensor, proxy_E: torch.Tensor | None = None):
        """The single seam. Called exactly once per forward by ``_forward_looped`` and
        ``_forward_fused`` (``_forward_sparse`` does not call it at all -- refused in
        ``__init__``), which is also why the Sinkhorn bookkeeping in ``_mu_for`` stays
        correct: we call ``super()._route`` exactly once, so ``_mu_call`` ticks once.
        """
        E_surr = self._surr_E

        if E_surr is not None and self.training:
            # PRE-mu on BOTH sides. `_logits_raw` does NOT touch `_mu_call`, which is why it
            # is the right function here -- `_logits` would tick the counter and desynchronise
            # the per-iteration dual buffer that eval cycles through.
            self._add_surrogate_loss(self._logits_raw(E_k), self._logits_raw(E_surr))

        if E_surr is not None and self._surrogate_routes_now:
            self._check_persisted_mu()
            # Route on the HEAD. `super()._route` applies mu, top_k and renormalize_topk, so
            # the surrogate and exact paths differ in ONE tensor and nothing else.
            # `proxy_E` is passed through untouched: if an arm also has `proxy_route: true`,
            # the proxy still owns SELECTION while the head owns the WEIGHTS. That
            # combination is legal but unmeasured; it is not the default anywhere.
            return super()._route(E_surr, proxy_E=proxy_E)

        return super()._route(E_k, proxy_E=proxy_E)

    # ------------------------------------------------------------------------------- #
    # loss + metrics                                                                   #
    # ------------------------------------------------------------------------------- #

    def _add_surrogate_loss(self, logits_exact: torch.Tensor,
                            logits_surr: torch.Tensor) -> None:
        """Distillation between the two PRE-mu routing distributions. See section (c)."""
        K = self.n_experts
        flat_exact = logits_exact.reshape(-1, K)
        flat_surr = logits_surr.reshape(-1, K)

        if self.surrogate_coef > 0:
            if self.surrogate_kl_direction == "forward":
                # KL(router || head): mass-covering, matches the legacy gradient and
                # `_proxy_step`, so `surrogate_topk_agree` and `proxy_topk_agree` are
                # comparable numbers.
                pred_log = F.log_softmax(flat_surr, dim=-1)
                tgt = F.softmax(flat_exact.detach(), dim=-1)
            else:
                pred_log = F.log_softmax(flat_exact.detach(), dim=-1)
                tgt = F.softmax(flat_surr, dim=-1)
            # batchmean on a FLATTENED (tokens, K) is a true per-token mean; on an
            # unflattened (batch, seq, K) it would divide seq*K terms by batch. See
            # difference (4) in the module docstring.
            kl = F.kl_div(pred_log, tgt, reduction="batchmean")
            add_aux_loss(self.surrogate_coef * kl)
            if self.track_load:
                with torch.no_grad():
                    self._surr_kl_sum += kl.detach().float()
                    self._surr_kl_n += 1.0

        if self.surrogate_track_agree and self.track_load:
            with torch.no_grad():
                # The SAME set-overlap estimator `_proxy_step` uses, deliberately: HANDOFF 7.9
                # records that `torch.isin` silently fakes agreement, so this is a genuine
                # per-row |intersection| / k.
                k = self.top_k if self.top_k is not None else 1
                k = min(int(k), K)
                a = flat_exact.detach().topk(k, dim=-1).indices
                b = flat_surr.detach().topk(k, dim=-1).indices
                hit = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1)
                self._surr_agree_sum += hit.sum() / k
                self._surr_agree_n += hit.shape[0]

    def _check_persisted_mu(self) -> None:
        """Warn ONCE if the surrogate is about to route untilted. Host-sync + Python logging,
        so it is behind the compile guard -- the config-level half of this check is an assert
        in ``__init__``, which is where it belongs; this catches the case where the config is
        right but no calibration pass ever ran."""
        if self._surr_warned_mu or torch.compiler.is_compiling():
            return
        if self.sinkhorn_iters > 0 and self.sinkhorn_persist_mu:
            if self.sinkhorn_mu_count is None or float(self.sinkhorn_mu_count.sum()) == 0.0:
                self._surr_warned_mu = True
                logger.warning(
                    "surrogate eval on layer %s: sinkhorn_persist_mu is set but "
                    "sinkhorn_mu_count is all zero, so _mu_for will return None and routing "
                    "will be UNTILTED. Run a calibration pass in TRAIN mode first.",
                    self.layer_idx,
                )

    def pop_load_metrics(self):
        """Base metrics plus the head's. Additive by construction: we call the base first and
        only add keys, so every existing wandb series is unchanged."""
        m = super().pop_load_metrics()
        if m is None:
            return None
        with torch.no_grad():
            if float(self._surr_kl_n) > 0:
                m["surrogate_kl"] = (self._surr_kl_sum / self._surr_kl_n.clamp_min(1.0)).item()
                self._surr_kl_sum.zero_(); self._surr_kl_n.zero_()
            if float(self._surr_agree_n) > 0:
                # Directly comparable to `proxy_topk_agree`: same estimator, same k.
                m["surrogate_topk_agree"] = (
                    self._surr_agree_sum / self._surr_agree_n.clamp_min(1.0)).item()
                self._surr_agree_sum.zero_(); self._surr_agree_n.zero_()
            m["surrogate_macs_per_token"] = float(self.surrogate_macs_per_token())
            m["surrogate_active"] = float(self._surrogate_routes_now)
        return m

    # ------------------------------------------------------------------------------- #
    # active-parameter accounting (TASK 2 lives in energy_ff_paramcount.py; this is the  #
    # one fact only this class knows)                                                   #
    # ------------------------------------------------------------------------------- #

    def surrogate_parameter_numel(self) -> int:
        """The head is ALWAYS active -- every token pays for it, like the proxy router."""
        n = self.surrogate_W1.numel() + self.surrogate_b1.numel()
        if self.surrogate_W2 is not None:
            n += self.surrogate_W2.numel() + self.surrogate_b2.numel()
        return n


class SurrogateBoltzmannMoEFFEnergy(_SurrogateRouterMixin, BoltzmannMoEFFEnergy):
    """``BoltzmannMoEFFEnergy`` + the distilled head. Hopfield (fused or looped) and w1w2
    (looped only -- the frozen base rejects ``fused_experts`` for w1w2; use
    ``SurrogateBoltzmannMoEW1W2`` for the fused w1w2 line)."""


try:  # optional: the sibling module is a separate, independently-landed change
    from .energy_ff_w1w2_sparse import BoltzmannMoEW1W2Sparse

    class SurrogateBoltzmannMoEW1W2(_SurrogateRouterMixin, BoltzmannMoEW1W2Sparse):
        """The same head on the FUSED w1w2 line, with ``sparse_forward`` refused by the mixin.

        This class is the point of section (a): the head's code is byte-identical across the
        two energy forms, while the rank-r subspace proxy needed a whole separate module and
        still loses its cost advantage for w1w2.
        """

    _HAVE_W1W2 = True
except ImportError:  # pragma: no cover
    SurrogateBoltzmannMoEW1W2 = None
    _HAVE_W1W2 = False


def build_surrogate_boltzmann_moe(
    *,
    expert_kind: str,
    hidden_size: int,
    intermediate_size: int,
    n_experts: int,
    init_method: str = "normal",
    initializer_range: float = 0.02,
    m_width: float | None = None,
    add_bias: bool = False,
    gelu_grad_method: str = "sigmoid",
    hopfield_grad_scale: str = "mean",
    e_sign_override: str | None = None,
    fused_experts: bool = False,
    layer_idx: int | None = None,
    **moe_kwargs,
) -> FusedMoEContainer:
    """Factory mirroring ``build_boltzmann_moe`` and returning the same ``FusedMoEContainer``.

    Reuses ``_FusedHopfieldHolder`` / ``_FusedW1W2Holder`` and ``FusedMoEContainer`` by
    import, so the parameter layout and the checkpoint keys of the expert pool are unchanged --
    a surrogate arm and its no-surrogate twin differ only by the ``moe.surrogate_*`` keys.

    ROUTING SIGN. The kind-based defaults in the frozen builder are the INVERTED ones (see
    ``ROUTING_SIGN_BUG_20260915.md``). They are kept here for exact comparability with
    published arms, so pass ``e_sign_override`` explicitly on anything new:
    ``"pos"`` for hopfield, ``"neg"`` for composable w1w2 (CLAUDE.md pre-flight check 8 -- a
    blanket ``"pos"`` is a silent no-op on w1w2).
    """
    if expert_kind == "hopfield":
        holder = _FusedHopfieldHolder(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            n_experts=n_experts, init_method=init_method,
            initializer_range=initializer_range, m_width=m_width,
            add_bias=add_bias, gelu_grad_method=gelu_grad_method,
            hopfield_grad_scale=hopfield_grad_scale,
        )
        e_sign = "neg"
        fused_spec = {
            "kind": "hopfield",
            "weight_fn": (lambda: holder.W.weight),
            "gelu_grad_method": gelu_grad_method,
            "hopfield_grad_scale": hopfield_grad_scale,
        } if fused_experts else None
        cls, extra = SurrogateBoltzmannMoEFFEnergy, {}
    elif expert_kind == "w1w2":
        holder = _FusedW1W2Holder(
            hidden_size=hidden_size, intermediate_size=intermediate_size,
            n_experts=n_experts, init_method=init_method,
            initializer_range=initializer_range, m_width=m_width,
            add_bias=add_bias, gelu_grad_method=gelu_grad_method,
        )
        e_sign = "pos"
        if fused_experts:
            assert _HAVE_W1W2, (
                "fused w1w2 needs energy_ff_w1w2_sparse.py (the frozen base class asserts "
                "fused_spec['kind'] == 'hopfield'); use fused_experts=False for the looped path"
            )
            cls = SurrogateBoltzmannMoEW1W2
            # That class builds its own fused_spec (with the kind shim) from these closures.
            extra = {"w1_fn": (lambda: holder.W1.weight),
                     "w2_fn": (lambda: holder.W2.weight),
                     "gelu_grad_method": gelu_grad_method}
            fused_spec = None
        else:
            cls, extra, fused_spec = SurrogateBoltzmannMoEFFEnergy, {}, None
    else:
        raise ValueError(f"unknown expert_kind ({expert_kind})")

    if e_sign_override is not None:
        assert e_sign_override in ("neg", "pos")
        e_sign = e_sign_override

    kw = dict(moe_kwargs)
    if cls is SurrogateBoltzmannMoEFFEnergy:
        kw.update(fused_experts=bool(fused_experts), fused_spec=fused_spec, e_sign=e_sign)
    else:
        # BoltzmannMoEW1W2Sparse forces fused_experts=True and builds its own spec.
        kw.update(e_sign=e_sign)
    moe = cls(holder.make_experts(), hidden_size=hidden_size, layer_idx=layer_idx,
              **extra, **kw)
    return FusedMoEContainer(expert_holder=holder, moe=moe)


__all__ = [
    "SurrogateBoltzmannMoEFFEnergy",
    "SurrogateBoltzmannMoEW1W2",
    "_SurrogateRouterMixin",
    "build_surrogate_boltzmann_moe",
    "surrogate_head_macs",
]
