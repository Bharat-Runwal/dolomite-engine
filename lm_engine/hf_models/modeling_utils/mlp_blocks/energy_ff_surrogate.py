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

BE HONEST ABOUT WHERE THAT BUYS ANYTHING. In the FUSED DENSE Hopfield path the exact energy is
a by-product: ``E_k = (gelu_Wx**2).mean(-1)`` is a T*K*I_e ELEMENTWISE reduction over an
activation the expert gradient already materialised, not a T*K*I_e*d GEMM. So switching a
DENSE arm's eval onto the head saves that reduction and nothing else -- the forward projection
still runs for all K because the expert OUTPUTS need it. **The FLOP win requires skipping
experts, i.e. ``sparse_forward``, which section (d) now wires up** (``surrogate_replaces_proxy``).
In the dense path the head remains:
  1. a DIAGNOSTIC: how much of the energy router's decision is decodable from h at what
     capacity -- measured by ``surrogate_topk_agree``, directly comparable to the proxy's
     ``proxy_topk_agree`` because both use the same set-overlap estimator;
  2. the TRAINING-TIME distillation that has to happen before a head can ever replace the
     proxy in the sparse path;
  3. a selector whose cost does not grow with I_e, which is the one property the subspace
     proxy structurally cannot have -- and which is what it is FOR in the sparse path.

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
(d) SPARSE SELECTION: THE HEAD **REPLACES** THE RANK-r PROXY  (2026-09-19)
===============================================================================================

``surrogate_replaces_proxy: true`` makes the head the module's ONE cheap all-K router, by
overriding ``_proxy_energies``. Every consumer of a cheap all-K energy estimate then reads the
head, with NO copied block and no new arithmetic -- ``_forward_sparse`` is inherited verbatim:

    _forward_sparse  step 1   sel_idx = topk_p(_logits_raw(head) - mu)   <- NOMINATION
    _forward_sparse  step 1   mom     = _zscore_moments(head)            <- zscore moments
    _forward_sparse  step 4b  lg      = EXACT energies of the p          <- RE-RANK to k
    _forward_sparse  step 4b  ref     = _logits_raw(head) - mu           <- denominator tail
    _forward_sparse  step 5b  KL(softmax_p(lg) || softmax_p(head[sel]))  <- DISTILLATION
    _forward_fused            proxy_E = head   (iff ``proxy_route``)     <- dense A/B
    _proxy_step               KL over ALL K, PRE-mu                      <- dense distillation

THE SEMANTICS ARE EXACTLY "CHEAP HEAD NOMINATES p, EXACT ENERGY RE-RANKS TO k", AND IT IS NOT
A SWITCH GATE. The head's output never becomes a routing weight. ``lg`` -- the EXACT Hopfield /
w1w2 energies of the p nominated experts, free because they are a by-product of the forward
projection those p experts needed anyway -- picks the final top-k at step 4b AND sets every
weight. Set ``sparse_candidates`` (= p) > ``top_k`` (= k) and the head's only job is to get the
true top-k INSIDE its top-p, which is a far weaker requirement than getting it right.
``use_surrogate`` (head supplies the WEIGHTS too) is therefore REFUSED under this mode -- that
combination is the Switch gate this design exists to avoid. The dense analogue of "head
selects, exact energies weight" is the base class's own ``proxy_route: true``, which now reads
the head.

WHY **REPLACES** AND NOT **ADDS** -- a correctness argument, not a preference. The
non-renormalised denominator completes the softmax with

    Zrest = sum_allK exp(ref) - sum_cand exp(ref[sel_idx])

which removes exactly the candidates' terms from an all-K sum. That is meaningful ONLY if
``ref`` is THE SAME vector whose top-p produced ``sel_idx``. Two disagreeing cheap routers make
``Zrest`` the difference of two unrelated estimates -- and ``.clamp_min(0)`` hides the sign
error, so it fails silently. The same argument applies to ``mom``: the z-score moments must be
the moments of the vector that was ranked. One cheap router, or none.

-----------------------------------------------------------------------------------------------
(d.1) WHAT THE HEAD IS DISTILLED AGAINST IN THE SPARSE REGIME -- the crux
-----------------------------------------------------------------------------------------------
In the dense path the target is the full-K ``softmax(_logits_raw(E_k))`` (section (c): PRE-mu on
both sides). ``_forward_sparse`` never computes the K-p unevaluated energies, so THAT TARGET
DOES NOT EXIST THERE. DECISION: **candidate-restricted KL** -- step 5b of the base
``_forward_sparse``, reused verbatim, gated by ``proxy_loss_coef``:

    tgt  = softmax_p( lg.detach() )        lg     = _logits_raw(E_exact[sel]) - mu[sel]
    pred = log_softmax_p( s_head[sel] )    s_head = _logits_raw(head)         - mu

Three reasons this is the right target and not a compromise:

 1. **IT IS STILL A PRE-mu CORRESPONDENCE, so there is NO convention change at the
    dense->sparse handover.** ``mu`` is per-expert and the candidate set is per-token, so
    ``mu[sel_idx]`` is a per-(token, slot) shift that appears IDENTICALLY on both sides of the
    restricted softmax; it cancels from the KL gradient. What the head learns is the pre-mu
    ranking -- the same function section (c)'s dense KL teaches it. This is the one place the
    head is strictly BETTER OFF than the rank-r proxy: the proxy's dense target is POST-mu
    under ``proxy_mu_convention: "legacy"`` and pre-mu inside ``_forward_sparse``, so a
    ``sparse_start_step`` run RE-TARGETS the proxy at exactly the moment it stops being
    trainable from all-K energies (``energy_ff.py:1841``, and mu was measured at 3.44 logit
    units, i.e. an e^3.44 ~ 31x tilt). Hence this mode ASSERTS
    ``proxy_mu_convention: "pre_mu"``: with it, both phases teach the same function.
 2. **THE RESTRICTION IS TO THE SET THAT MATTERS.** A softmax denominator is dominated by its
    largest terms and the p candidates are, by construction, the head's own largest. The
    experts dropped from the KL are the ones whose routing weight would have been negligible.
    (Note this is exactly why over-selecting fixes the denominator as well as the selection.)
 3. **THERE IS NO CHEAPER HONEST ALTERNATIVE.** Forming the all-K target inside the sparse path
    means running the all-K forward projection -- the thing being skipped.

THE FAILURE MODE OF (d.1) AND ITS FIX. Candidate-restricted distillation is SELF-REINFORCING:
the head only ever sees exact energies for experts it nominated, so an expert it wrongly ranks
last is never corrected. ``sparse_explore > 0`` is the fix and it is already in the base --
token t additionally evaluates experts ``(t+1 .. t+n_exp) mod K``, so over a batch of T >> K
tokens every expert is covered by 1/K of tokens every step, deterministically (no RNG in the
graph). **A sparse surrogate arm with ``sparse_explore: 0`` trains a router that cannot
discover its own mistakes.** Treat ``sparse_explore >= 1`` as part of this mode.

-----------------------------------------------------------------------------------------------
(d.2) IS A DENSE WARM-UP (``sparse_start_step > 0``) *REQUIRED*?
-----------------------------------------------------------------------------------------------
**Not by construction. YES in practice -- treat it as required for any arm whose number will be
quoted.** The distinction matters, so state both halves:

  * NOT logically required. With ``sparse_explore > 0`` the candidate-restricted KL has a
    corrective signal on every expert from step 0, so the head IS trainable from scratch inside
    the sparse path. ``sparse_start_step: 0`` will not deadlock and will not silently freeze the
    head -- that is what happens with ``sparse_explore: 0``.
  * Required in practice, for a reason about the MODEL rather than the head. At step 0 the head
    is random (``surrogate_init_std: 0.01``), so the p experts whose projections are computed
    are an arbitrary subset, and the model trains against arbitrary routing for as long as the
    head takes to become useful. That regime is MEASURED, not hypothetical: the base class's
    note records expert-output alignment 0.698 (against 0.24-0.46 for dense arms) for a sparse
    arm whose proxy never trained -- the experts never specialise at all.
  * The dense phase is also where the head gets its STRONGEST signal, and gets it for free:
    ``_forward_fused`` computes all K exact energies as a by-product of a projection it needs
    anyway, so ``_proxy_step``'s all-K KL costs only the head's own ``d*K`` MACs.
  * The dense phase must run until ``proxy_topk_agree`` has PLATEAUED, not merely risen. A head
    frozen at its phase-1 quality decays against a moving target (the dual drifts 3.44 -> 1.97
    over training), which is why distillation continues in the sparse phase.

RECOMMENDED RECIPE: dense until agreement plateaus (``sparse_start_step``), then sparse with
``sparse_explore >= 1`` and ``proxy_loss_coef > 0`` throughout, ``renormalize_topk: true``.

-----------------------------------------------------------------------------------------------
(d.3) WHICH COEFFICIENT TRAINS THE HEAD IN WHICH PHASE -- A REAL TRAP
-----------------------------------------------------------------------------------------------
``_forward_sparse`` DOES NOT CALL ``_route``, so ``_add_surrogate_loss`` -- and therefore
``surrogate_coef`` -- is a **silent no-op for the entire sparse phase**. Under this mode the
head is trained by ``proxy_loss_coef`` in BOTH phases (dense: ``_proxy_step``, all K; sparse:
step 5b, p candidates). That is also why ``proxy_rank`` is forced > 0 below: in the frozen base
it is the gate on both of those blocks. ``surrogate_coef`` is redundant here -- in the dense
phase it computes the identical KL a second time -- and is warned about at construction.

Metric names follow the same split. ``proxy_topk_agree`` IS the head's agreement under this
mode: all-K in the dense phase, candidate-restricted in the sparse phase, exactly as for the
rank-r proxy -- so the published proxy numbers are the comparison baseline. It is re-exported
as ``surrogate_sel_agree`` so a wandb panel cannot mistake one series for the other, and
``surrogate_selects`` / ``surrogate_sparse_active`` record which regime produced it.

-----------------------------------------------------------------------------------------------
(d.4) ``renormalize_topk: true`` IS EFFECTIVELY MANDATORY, FOR TWO INDEPENDENT REASONS
-----------------------------------------------------------------------------------------------
 1. QUALITY, measured (base ``__init__`` note): on pure_hop_T12_sink, wikitext bits/byte
    1.0996 dense -> 1.1161 with proxy selection and exact weights -> 3.4346 with the
    proxy-completed denominator. The denominator, not the selection, is where sparse arms lose.
 2. GRADIENT HYGIENE, structural. With ``renormalize_topk: true`` the head reaches the output
    ONLY through the discrete ``topk``, so there is NO gradient path from ``lm_loss`` into the
    head and "the head cannot change what the model computes except through its selection" is
    literally true. With ``false``, ``ref`` enters ``Zall`` enters ``w``, so the LM loss trains
    the head directly -- a second, unmeasured objective fighting the KL. (``x`` is detached
    either way, so the BACKBONE is never reshaped by the head.)
    ONE RESIDUAL PATH SURVIVES, INHERITED FROM THE PROXY AND NOT NEW HERE: with
    ``routing_norm: "zscore"`` and p < K the per-token std in ``mom`` comes from the head and
    does not cancel in the restricted softmax, so it acts as a per-token rescaling of tau that
    the LM loss can push on. It is a single scalar per token. ``routing_norm: "none"`` or
    ``"sqrt_width"`` removes it entirely.
Asserted; ``surrogate_sparse_allow_proxy_denominator: true`` is the deliberate escape hatch
(it is what the p = K exactness self-test needs).

-----------------------------------------------------------------------------------------------
(d.5) THE RANK-r PROXY'S PARAMETERS UNDER THIS MODE
-----------------------------------------------------------------------------------------------
The frozen base asserts ``proxy_rank > 0`` whenever ``sparse_forward`` is set, and gates both
distillation blocks on it, so it is forced to 1 when a config does not give it. Its tensors are
then NEVER READ (the override returns before touching them) and would otherwise sit in
``model.parameters()`` receiving no gradient for the whole run -- decayed by AdamW, sharded by
FSDP, and reported in the parameter count. ``surrogate_free_proxy: true`` (the default)
converts them to non-persistent BUFFERS immediately after construction, so they leave the
optimizer and FSDP's parameter groups entirely. ``proxy_init: "svd"`` is neutralised for the
same reason -- the head is not a factorisation of W, so there is nothing to warm-start -- and
is warned about rather than silently ignored.

-----------------------------------------------------------------------------------------------
(d.6) W1W2 SPARSE
-----------------------------------------------------------------------------------------------
``SurrogateBoltzmannMoEW1W2`` inherits ``BoltzmannMoEW1W2Sparse._forward_sparse``, whose step 1
is line-for-line the Hopfield one and calls the same ``self._proxy_energies``. So the override
lands there with ZERO additional code, and ``surrogate_replaces_proxy`` works on the w1w2 line
as soon as that class is registered in ``get_mlp_block`` (it is not yet). Two things change,
both in the head's FAVOUR: the w1w2 subspace proxy is forced to ``m = I_e`` (its m-row
subsample has relative error ~sqrt(I_e/m), section (a)), at which point the proxy costs MORE
per token than the sparse mixture it is meant to make cheap -- while the head's cost has no
``I_e`` in it at all; and the head needs no common-row SVD heuristic. The sibling's
``proxy_V2``/``proxy_B2`` are freed by the same code path.

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
        surrogate_replaces_proxy: bool = False,
        surrogate_free_proxy: bool = True,
        surrogate_sparse_allow_proxy_denominator: bool = False,
        **kwargs,
    ) -> None:
        # ---- (d.5) `proxy_rank` is a pure GATE once the head supplies the cheap energies ---- #
        # The frozen base asserts `proxy_rank > 0` for `sparse_forward` and keys BOTH
        # distillation blocks (`_forward_fused` -> `_proxy_step`, and `_forward_sparse` step 5b)
        # on it, so it has to be truthy -- but nothing reads its VALUE once `_proxy_energies` is
        # overridden. Forced to the minimum here so a config need not carry a number that no
        # longer means a rank, and so the allocation it forces is as small as possible before
        # `surrogate_free_proxy` releases it. MUST happen before `super().__init__`, which is
        # where the assert lives.
        _rank_autoset = False
        if surrogate_replaces_proxy and int(kwargs.get("proxy_rank", 0) or 0) <= 0:
            kwargs = dict(kwargs)
            kwargs["proxy_rank"] = 1
            _rank_autoset = True
        super().__init__(*args, **kwargs)
        self._surr_proxy_rank_autoset = _rank_autoset
        self.surrogate_replaces_proxy = bool(surrogate_replaces_proxy)
        self.surrogate_sparse_allow_proxy_denominator = bool(
            surrogate_sparse_allow_proxy_denominator)

        assert surrogate_kind in _SURROGATE_KINDS, surrogate_kind
        assert surrogate_kl_direction in _KL_DIRECTIONS, surrogate_kl_direction
        assert surrogate_coef >= 0.0
        if surrogate_kind == "mlp":
            assert surrogate_hidden > 0, "surrogate_kind='mlp' needs surrogate_hidden > 0"

        # ---- (d) sparse selection -------------------------------------------------------- #
        # `_forward_sparse` does NOT call `_route`: it inlines `_logits_raw` twice and solves the
        # Sinkhorn dual on whatever all-K cheap energies it has. So the head reaches it ONLY by
        # being that cheap router -- `surrogate_replaces_proxy` overrides `_proxy_energies`.
        # Without the flag the old refusal stands, because the head would be computed and then
        # ignored: a silent no-op, not an error.
        assert self.surrogate_replaces_proxy or not self.sparse_forward, (
            "sparse_forward with the surrogate head requires surrogate_replaces_proxy: true. "
            "That path never calls _route, so without the flag the head is computed and then "
            "IGNORED -- a silent no-op. With it, the head becomes the module's single cheap "
            "all-K router (_proxy_energies) and supplies the NOMINATION, the zscore moments and "
            "the denominator tail, while the exact energies of the nominated p still re-rank to "
            "the final top-k and set every weight. See section (d) of this module's docstring."
        )
        assert not self.sparse_backproj, (
            "sparse_backproj is untested with the surrogate head; it is also redundant with "
            "sparse_forward (1.18-1.35x against 4.69x)."
        )

        if self.surrogate_replaces_proxy:
            # (d) the head supplies WEIGHTS nowhere. `use_surrogate` routes `_route` on the
            # head's own pseudo-energies, i.e. the head would set the mixture weights -- which
            # is precisely the Switch-style gate this design exists not to be. The dense
            # analogue of "head selects, exact energies weight" is the base's `proxy_route`,
            # which now reads the head.
            assert not use_surrogate, (
                "surrogate_replaces_proxy and use_surrogate are mutually exclusive. "
                "use_surrogate hands the head's pseudo-energies to _route, so the head would "
                "set the mixture WEIGHTS -- a Switch-style gate. For 'head selects, exact "
                "energy weights' in the dense path set proxy_route: true (it reads the head "
                "under this mode); in the sparse path that split is what _forward_sparse does "
                "by construction."
            )
            # (d.1) ONE convention in both phases. With "legacy" the dense `_proxy_step` teaches
            # the POST-mu logits while `_forward_sparse` step 5b and `_route`'s proxy selection
            # are pre-mu, so the head would be RE-TARGETED at the dense->sparse handover and mu
            # (measured 3.44 logit units) counted twice on the dense selection path.
            assert self.proxy_mu_convention == "pre_mu", (
                "surrogate_replaces_proxy requires proxy_mu_convention: pre_mu. With 'legacy' "
                "the dense _proxy_step target is POST-mu while _forward_sparse step 5b and "
                "_route's proxy selection are PRE-mu: the head gets re-targeted at the "
                "dense->sparse handover and mu is counted twice on the dense selection path "
                "(mu was measured at 3.44 in logit units, an e^3.44 ~ 31x tilt)."
            )
            # The head has no per-iteration variants; `proxy_iters > 1` indexes `proxy_V` by a
            # call counter, which the override never reads -- it would be a silent no-op.
            assert self.proxy_iters == 1, (
                "proxy_iters > 1 has no meaning for the surrogate head (it indexes the rank-r "
                "proxy's per-iteration tensors, which the override never reads). Use 1."
            )
            if self.sparse_forward:
                p_cand = self.sparse_candidates or int(self.top_k)
                # (d.4) the proxy-completed denominator is where sparse arms lose (-1.52 nats
                # measured), and it is also the only thing that gives lm_loss a gradient path
                # into the head.
                assert (self.renormalize_topk or p_cand == self.n_experts
                        or self.surrogate_sparse_allow_proxy_denominator), (
                    "sparse_forward + surrogate_replaces_proxy with renormalize_topk: false "
                    "completes the softmax denominator from the HEAD over the K-p experts it "
                    "never evaluated. Measured cost of the proxy-completed denominator: "
                    "wikitext bits/byte 1.1161 -> 3.4346. It also makes lm_loss train the head "
                    "directly, fighting the KL. Set renormalize_topk: true, or "
                    "surrogate_sparse_allow_proxy_denominator: true to accept it deliberately."
                )
                if self.sparse_explore <= 0 and self.sparse_candidates < self.n_experts:
                    # (d.1) candidate-restricted distillation with no exploration is
                    # self-reinforcing: the head never sees an exact energy for an expert it
                    # ranked out, so it cannot discover the mistake.
                    logger.warning(
                        "layer %s: surrogate_replaces_proxy + sparse_forward with "
                        "sparse_explore=0. The candidate-restricted KL then only ever sees "
                        "experts the head already nominated, so it is SELF-REINFORCING and the "
                        "head cannot discover an expert it wrongly ranked out. Set "
                        "sparse_explore >= 1.", self.layer_idx)
            if self.proxy_loss_coef <= 0.0:
                # The head is trained by proxy_loss_coef in BOTH phases under this mode (d.3).
                logger.warning(
                    "layer %s: surrogate_replaces_proxy with proxy_loss_coef = 0. The head is "
                    "distilled by proxy_loss_coef in BOTH phases under this mode "
                    "(surrogate_coef is a NO-OP in the sparse phase because _forward_sparse "
                    "never calls _route), so the head will stay at its random init and "
                    "selection will be ARBITRARY. Fine for eval of an already-trained "
                    "checkpoint; a bug for training.", self.layer_idx)
            if surrogate_coef > 0.0:
                logger.warning(
                    "layer %s: surrogate_coef > 0 is REDUNDANT under "
                    "surrogate_replaces_proxy: in the dense phase _proxy_step already applies "
                    "the same pre-mu all-K KL to the same head, and in the sparse phase "
                    "surrogate_coef is a no-op. The two simply add.", self.layer_idx)
            if self.proxy_init == "svd":
                logger.warning(
                    "layer %s: proxy_init: svd is neutralised under surrogate_replaces_proxy -- "
                    "the head is not a low-rank factorisation of W, so there is nothing to warm "
                    "start. Distillation is the only way it learns.", self.layer_idx)

        # ---- (c) the Sinkhorn prerequisite -------------------------------------------- #
        # At eval the head cannot solve the dual -- that needs the all-K exact logits, which
        # is the thing it exists to avoid -- so it can only READ the persisted running mu.
        # `surrogate_replaces_proxy` needs it for the SAME reason: at eval `_mu_for` is called on
        # the HEAD's all-K logits and cannot solve the dual (training solves it fine -- the head
        # gives all K), so it can only read the persisted running dual. This is CLAUDE.md
        # pre-flight rule 9 for every sparse arm, restated for the head.
        if (use_surrogate or surrogate_replaces_proxy) and self.sinkhorn_iters > 0:
            assert self.sinkhorn_persist_mu, (
                "use_surrogate / surrogate_replaces_proxy with sinkhorn_iters > 0 REQUIRES "
                "sinkhorn_persist_mu: true. "
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

        # ---- (d.5) release the now-unread rank-r proxy tensors -------------------------- #
        # Built LAST, after the head, so nothing created here can shift another parameter's
        # init. Converting to buffers rather than deleting keeps `_svd_refit_proxy` and any
        # future base-class read from raising AttributeError on a None.
        self._surr_proxy_freed = False
        if self.surrogate_replaces_proxy and surrogate_free_proxy:
            self._free_proxy_parameters()

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

    def _free_proxy_parameters(self) -> None:
        """Turn the unread rank-r proxy tensors into non-persistent buffers. See (d.5).

        They are forced into existence by the frozen base's `proxy_rank > 0` assert and are never
        read once `_proxy_energies` is overridden, so as PARAMETERS they would be decayed by
        AdamW, sharded by FSDP and counted in the model size while receiving no gradient for the
        entire run. `persistent=False` also keeps them out of the state_dict, so a surrogate
        sparse arm's checkpoint carries only `moe.surrogate_*` beyond its no-surrogate twin.
        """
        names = ("proxy_V", "proxy_B", "proxy_quad", "proxy_lin", "proxy_scale", "proxy_bias",
                 # the w1w2 sibling's second basis, so SurrogateBoltzmannMoEW1W2 needs no
                 # override of this method
                 "proxy_V2", "proxy_B2")
        freed = []
        for name in names:
            prm = self._parameters.pop(name, None)
            if prm is None:
                continue
            # `register_buffer` refuses a name that still resolves as an attribute, and popping
            # from `_parameters` is what makes it stop resolving.
            self.register_buffer(name, prm.detach().clone(), persistent=False)
            freed.append(name)
        self._surr_proxy_freed = bool(freed)
        if freed:
            logger.info("layer %s: surrogate_replaces_proxy -- released unread rank-r proxy "
                        "tensors %s from model.parameters()", self.layer_idx, freed)

    # ------------------------------------------------------------------------------- #
    # (d) THE SPARSE SEAM: the head IS the module's cheap all-K router                  #
    # ------------------------------------------------------------------------------- #

    def _proxy_energies(self, x: torch.Tensor) -> torch.Tensor:
        """The head's pseudo-energies wherever the base wants cheap all-K energies.

        THIS ONE OVERRIDE IS THE WHOLE SPARSE WIRING. `_forward_sparse` is inherited verbatim,
        so the nomination (`topk_p`), the zscore moments, the Sinkhorn dual, the exact-energy
        RE-RANK to k, the weights, the denominator and the candidate-restricted distillation all
        come from the frozen, tested code -- there is no copied block here and no drift guard to
        maintain. `_forward_fused` (`proxy_route`) and `_proxy_step` read it too, which is what
        makes the dense warm-up train the SAME head the sparse phase then selects with.

        Returns the STASH when `forward` made one, so the head runs exactly once per forward
        even though two call sites want it; recomputes otherwise, which is what lets a test call
        `_forward_sparse` directly.
        """
        if not self.surrogate_replaces_proxy:
            return super()._proxy_energies(x)
        E = self._surr_E
        if E is None or E.shape[:-1] != x.shape[:-1]:
            E = self.surrogate_energies(x)
        return E

    def _svd_refit_proxy(self) -> None:
        """No-op under this mode: the head is not a low-rank factorisation of W, so an SVD of W
        says nothing about its parameters. Marked done so the base never retries."""
        if self.surrogate_replaces_proxy:
            self._svd_done = True
            return
        super()._svd_refit_proxy()

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
            or (self.surrogate_track_agree and self.training) \
            or self.surrogate_replaces_proxy
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
            if self.surrogate_replaces_proxy:
                # (d.3) `proxy_topk_agree` IS the head's agreement under this mode -- all-K in
                # the dense phase, candidate-restricted in the sparse one, exactly as for the
                # rank-r proxy. Re-exported so a panel cannot confuse the two, with the flag
                # that says which regime produced it.
                m["surrogate_selects"] = 1.0
                m["surrogate_sparse_active"] = float(getattr(self, "_sparse_active", False))
                if "proxy_topk_agree" in m:
                    m["surrogate_sel_agree"] = m["proxy_topk_agree"]
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
