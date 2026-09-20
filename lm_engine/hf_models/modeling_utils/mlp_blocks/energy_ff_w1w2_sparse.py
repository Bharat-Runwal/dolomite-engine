# **************************************************
# Sparse Boltzmann-MoE for W1W2 (bilinear) energy experts — 2026-09-18
# **************************************************
"""True-sparsity path for ``expert_kind="w1w2"``, the second expert type.

``energy_ff.py`` implements ``sparse_forward`` (skip the forward AND back projection of the
K-k experts a token was not routed to) for **Hopfield experts only**; this module adds the
w1w2 line as a subclass. Everything here is either a subclass override or an import.

UPDATE 2026-09-19 — REACHABLE FROM YAML. Two things changed on that date:
  * ``energy_ff.py``'s ``assert fused_spec["kind"] == "hopfield"`` now accepts ``"w1w2"`` too,
    AND additionally requires that the instance actually overrode ``_forward_fused`` (which
    only this class and its surrogate subclass do). So the ``_KIND_SHIM`` that used to pass the
    literal ``"hopfield"`` with the real kind hidden under ``kind_real`` is GONE.
  * ``get_mlp_block`` routes ``mlp_type: EnergyFF_BoltzmannMoE`` with ``expert_kind: w1w2`` and
    ``fused_experts: true`` to ``build_boltzmann_moe_w1w2_sparse`` below. Before that, this
    module was dead code from a config's point of view.
Nothing else in ``energy_ff.py`` is modified: six long training runs import it.

===============================================================================================
IS THE EXTENSION MECHANICAL? MOSTLY YES, WITH ONE STEP THAT GENUINELY FAILS.
===============================================================================================

The hypothesis under test was "we just need to do the low-rank thing for two matrices instead
of one". Step by step against the 11 steps of the Hopfield path:

  MECHANICAL (algebra carries over verbatim, only the arity changes 1 -> 2):

  1. FUSED WEIGHT VIEW. ``_FusedW1W2Holder`` already keeps two ``(K*I_e, hidden)`` tensors and
     hands each expert a contiguous row-slice, exactly as ``_FusedHopfieldHolder`` does for one.
     Two ``.view(K, I_e, H)`` instead of one. Reused by import; no new holder class.
  2. CAPACITY DISPATCH. ``_dispatch_plan`` maps (token, expert) pairs to slots in a fixed
     ``(K, C, *)`` buffer and never looks at a weight. Reused verbatim.
  3. ONE GATHER, TWO PROJECTIONS. ``xbuf`` is gathered once; ``bmm`` against ``W1v`` and
     against ``W2v``. The token dispatch is shared, so the gather/scatter cost does NOT double.
  4. THE ENERGY IS STILL FREE. ``E = -I_e^-0.5 * (phi(W1x) . (W2x))`` needs both projections —
     and both are needed for the descent gradient anyway, so the energy of an evaluated expert
     still costs zero extra, which is the property the whole design rests on.
  5. BACK PROJECTION. ``phi @ W2 + (phi' * W2x) @ W1`` — two bmms against the same two views,
     same ``(se, slot)`` indices, one ``index_add_``.
  6. WEIGHTS / DENOMINATOR / RE-RANK / mu / zscore moments. All of it acts on ``(T, p)``
     energies and never on a weight, so ``_logits_raw``, ``_zscore_moments``, ``_mu_for``,
     ``_route``, ``_track_load``, ``_track_energy``, ``_proxy_step``, ``pop_load_metrics`` are
     reused unchanged. Sign handling included: w1w2 stores ``E = -overlap``, so the CORRECTED
     routing sign is ``e_sign="neg"`` (CLAUDE.md pre-flight check 8) — but that is a config
     value flowing through ``_logits_raw``, not a structural difference.
  7. COST MODEL. Dense ``4*K*d*I_e`` per token against sparse ``4*cf*k*d*I_e + proxy``: the
     ratio ``K/(cf*k)`` is IDENTICAL to Hopfield's. w1w2 pays 2x the FLOPs at equal ``I_e``
     but sparsity buys exactly the same factor.

  NOT MECHANICAL — THE ONE THAT FAILS:

  8. ``proxy_out_dim`` (the m-row subsample) IS STATISTICALLY VALID FOR HOPFIELD AND INVALID
     FOR W1W2, and this is not a detail: it is the knob that makes the proxy cheap enough for
     the sparse path to be worth running at all.

     Hopfield: ``E = mean_j gelu(z_j)^2`` — a mean of I_e NONNEGATIVE terms. The mean is
     O(term), so an m-row subsample has relative error ``(sigma/mu)/sqrt(m)`` with
     ``sigma/mu = O(1)``: ~4% at m=512. energy_ff.py's comment is right, for Hopfield.

     w1w2: ``E = -I_e^-0.5 * sum_j phi(u_j) v_j`` — a sum of I_e terms of RANDOM SIGN, because
     ``v = W2 x`` is sign-indeterminate. The sum is O(sqrt(I_e) * term), not O(I_e * term);
     that cancellation is exactly why this energy carries ``1/sqrt(I_e)`` where Hopfield
     carries ``1/I_e``. An m-subsample estimator ``(I_e/m) * sum_sub`` then has error
     ``~ sigma*I_e/sqrt(m)`` against a signal ``~ sigma*sqrt(I_e)``:

         relative error  ~  sqrt(I_e / m)        (w1w2)
         relative error  ~  1 / sqrt(m)          (Hopfield)

     At I_e=2240, m=512 that is ~2.1, i.e. 210% — the "estimate" is noise. MEASURED on CPU in
     ``experiments/boltzmann-moe/scripts/test_sparse_w1w2_20260918.py`` (TEST 8), which prints
     both curves side by side.

     CONSEQUENCE. For w1w2 the faithful subspace proxy is forced to ``m = I_e``, and at m=I_e
     the proxy does ``K*I_e`` elementwise work per token — which is MORE than the sparse
     mixture's own ``cf*k*I_e`` (6.4x more at K=16, k=2, cf=1.25). The proxy would then cost
     more than the thing it is there to make cheap. So the Hopfield recipe does not transfer,
     and ``proxy_out_dim < I_e`` here is a MODEL-CAPACITY choice that MUST be distilled, not an
     unbiased estimator you can read off an SVD. Both facts are asserted/logged below rather
     than left to be rediscovered.

  9. THE SVD WARM START HAS A CORRECTNESS TRAP THAT THE HOPFIELD CODE WALKS INTO IF COPIED.
     ``_svd_refit_proxy`` picks the m most energetic rows of ``U S`` PER MATRIX. The w1w2
     energy pairs coordinate j of ``W1 x`` with coordinate j of ``W2 x``. Choosing the top-m
     rows of W1 and, independently, the top-m rows of W2 pairs MISMATCHED coordinates and
     computes a quantity with no relation to the energy — silently, with no shape error. The
     row subset must be COMMON to both. Implemented that way below (scored by
     ``||B1_row|| * ||B2_row||``, a heuristic, flagged as such).

 10. OUTPUT-SPACE REPULSION CANNOT REUSE ``_add_repulsion_loss_fused``. That helper exploits
     ``expert_out = pref * (gated @ W)`` to back-project only the sampled experts from a single
     ``gated``. A w1w2 expert output is a SUM OF TWO back-projections against two different
     matrices, so there is no single ``(gated, W)`` pair. Handled by building the expert-OUTPUT
     stack for a token subsample and feeding the LOOPED helpers ``_add_repulsion_loss`` /
     ``_probe_expert_cos(W=None)``, which already take ``(..., K, hidden)``. Reuse, not a copy.

 11. WEIGHT-SPACE REPULSION needs a definition that does not exist for one matrix: the per-
     expert "weight block" is now the pair. Defined here as the concatenation
     ``[W1_k ; W2_k]`` flattened, then handed to the base helper.

===============================================================================================
WHAT IS COPIED RATHER THAN IMPORTED, AND WHY
===============================================================================================

Exactly one block: the ~40 lines of ``_forward_sparse`` that turn ``(T, p)`` exact energies
into weights (duplicate-candidate suppression, re-rank, shared max-subtraction, the
proxy-completed denominator). It is expert-kind-agnostic and SHOULD be shared, but it is
INLINE in ``BoltzmannMoEFFEnergy._forward_sparse`` with no seam to call, and energy_ff.py is
frozen while eight runs are live. It is marked ``### COPIED BLOCK`` below.

DRIFT WARNING: if that block changes in energy_ff.py it must change here too. TEST 9 of the
CPU test hashes ``inspect.getsource(BoltzmannMoEFFEnergy._forward_sparse)`` and fails loudly
when it moves. When energy_ff.py is next editable, extract the block as
``_sparse_weights(lg, kept, ref, sel_idx, k, p_cand)`` and have both paths call it.
"""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...loss import add_aux_loss
from .energy_ff import (
    BoltzmannMoEFFEnergy,
    FFEnergyBase,
    FusedMoEContainer,
    _FusedW1W2Holder,
    _gelu_and_grad,
)


class BoltzmannMoEW1W2Sparse(BoltzmannMoEFFEnergy):
    """``BoltzmannMoEFFEnergy`` with the fused + sparse paths implemented for w1w2 experts.

    Overrides exactly the four methods that touch a weight tensor
    (``_forward_fused``, ``_forward_sparse``, ``_proxy_energies``, ``_svd_refit_proxy``)
    plus the two repulsion helpers whose algebra assumes a single matrix. Everything
    else — routing, the Sinkhorn dual, load tracking, the proxy distillation loss and its
    agreement metric, the dense->sparse gate, the looped fallback — is inherited.

    Extra constructor arguments beyond the base class:
        w1_fn, w2_fn        closures returning the live fused ``(K*I_e, hidden)`` weights
                            (closures, not tensors, so FSDP re-gathers are picked up —
                            same reason the expert ``_W_slice`` closures exist)
        gelu_grad_method    phi/phi' convention, as for the experts
        proxy_kind          "subspace" (default, faithful) or "bilinear" (see _proxy_energies)
    """

    def __init__(
        self,
        experts: Sequence[FFEnergyBase],
        *,
        hidden_size: int,
        w1_fn,
        w2_fn,
        gelu_grad_method: str = "sigmoid",
        proxy_kind: str = "subspace",
        proxy_rank: int = 0,
        proxy_out_dim: int = 0,
        layer_idx: int | None = None,
        **kw,
    ) -> None:
        assert proxy_kind in ("subspace", "bilinear"), proxy_kind
        # The base allocates proxy parameters for us; ask it for the "subspace" shapes in both
        # cases so `proxy_V` / `proxy_B` / `proxy_scale` / `proxy_bias` and the
        # `_proxy_agree_*` / `_proxy_call` buffers all exist. For the bilinear head `proxy_B` is
        # unused, so request m=1 and it costs (K, 1, r).
        base_out_dim = proxy_out_dim if proxy_kind == "subspace" else 1
        super().__init__(
            experts,
            hidden_size=hidden_size,
            layer_idx=layer_idx,
            proxy_rank=proxy_rank,
            proxy_kind="subspace",
            proxy_out_dim=base_out_dim,
            fused_experts=True,
            fused_spec={
                # The TRUE kind. Until 2026-09-19 this had to be the literal "hopfield" (a
                # `_KIND_SHIM`) because the base class compared it with `==`; that assert now
                # accepts "w1w2" and additionally checks that `_forward_fused` really is
                # overridden, which this class does -- so the shim is gone and the spec no
                # longer lies about the expert type.
                "kind": "w1w2",
                "weight_fn": w1_fn,          # base `_fused_W()` -> W1; `_fused_W2()` added here
                "weight_fn_2": w2_fn,
                "gelu_grad_method": gelu_grad_method,
                # `hopfield_grad_scale` deliberately ABSENT: a KeyError is the desired outcome
                # if an unoverridden Hopfield code path is ever reached with this spec.
            },
            **kw,
        )
        # `sparse_backproj` would need `_sparse_backproj`, which assumes `expert_out = gated @ W`
        # for ONE W. It is redundant with `sparse_forward` and measured at 1.18-1.35x against
        # 4.69x (HANDOFF 12.9), so it is not worth a second implementation.
        assert not self.sparse_backproj, (
            "sparse_backproj is not implemented for w1w2 experts (and is redundant with "
            "sparse_forward); use sparse_forward"
        )
        self.proxy_kind = proxy_kind
        # restore the CONFIGURED value: the base was handed 1 for the bilinear head so that its
        # unused `proxy_B` costs (K, 1, r), and a serialised `proxy_out_dim: 1` would be a lie.
        self.proxy_out_dim = int(proxy_out_dim)
        self.gelu_grad_method = gelu_grad_method

        if self.proxy_rank > 0:
            r = self.proxy_rank
            pre = () if self.proxy_iters == 1 else (self.proxy_iters,)
            # DEDICATED GENERATOR, and a DIFFERENT seed from the base's `0xB01742 + layer_idx`.
            # Both reasons matter: (a) torch.randn on the global stream would shift the init of
            # every parameter created after this block, which the base class's comment records
            # as having moved an arm's lm_loss by 4.4x the noise floor; (b) reusing the base's
            # seed would make V2 a bit-identical copy of V1, i.e. one basis pretending to be two.
            _g = torch.Generator().manual_seed((0xB01742 ^ 0x57EFA2) + (layer_idx or 0))
            self.proxy_V2 = nn.Parameter(
                torch.randn(*pre, self.n_experts, hidden_size, r, generator=_g)
                / (hidden_size ** 0.5)
            )
            if proxy_kind == "subspace":
                m = self._proxy_m
                self.proxy_B2 = nn.Parameter(
                    torch.randn(*pre, self.n_experts, m, r, generator=_g) / (r ** 0.5))
                # LOUD, because this is the non-mechanical step (module docstring §8). m < I_e is
                # a function-class choice for w1w2, NOT the unbiased estimator it is for
                # Hopfield, so the SVD warm start is a heuristic there and distillation is
                # mandatory rather than optional.
                self._proxy_subsample_is_unbiased = (m >= self._expert_I)
                if not self._proxy_subsample_is_unbiased:
                    # SAY IT AT CONSTRUCTION. The Hopfield path treats m < I_e as a free,
                    # unbiased variance reduction; for this energy it is not (docstring §8), and
                    # a silent m=512 is how a router gets trained on noise.
                    import logging
                    logging.getLogger(__name__).warning(
                        "w1w2 proxy: proxy_out_dim=%d < expert_I=%d. For the BILINEAR energy an "
                        "m-row subsample is NOT an unbiased estimator (rel err ~ sqrt(I_e/m) = "
                        "%.2f, vs 1/sqrt(m) = %.3f for Hopfield): the head is a LEARNED "
                        "low-dimensional form and the SVD warm start is only a heuristic, so "
                        "proxy_loss_coef > 0 (distillation) is mandatory.",
                        m, self._expert_I, (self._expert_I / max(m, 1)) ** 0.5, m ** -0.5)
            else:
                self.proxy_B2 = None
                self._proxy_subsample_is_unbiased = True   # no I_e axis at all
        else:
            self.proxy_V2 = self.proxy_B2 = None
            self._proxy_subsample_is_unbiased = True

    # --- fused weight views -------------------------------------------------- #

    def _fused_W2(self) -> torch.Tensor:
        return self._fused_spec["weight_fn_2"]()

    def _fused_views(self):
        """(W1, W2, W1v, W2v) — flat ``(K*I_e, H)`` and expert-major ``(K, I_e, H)``."""
        K, I_e, H = self.n_experts, self._expert_I, self.hidden_size
        W1, W2 = self._fused_W(), self._fused_W2()
        return W1, W2, W1.view(K, I_e, H), W2.view(K, I_e, H)

    @property
    def _inv_sqrt_I(self) -> float:
        """The ``1/sqrt(expert_I)`` folded into the w1w2 energy itself (see ``W1W2FFEnergy``)."""
        return self._expert_I ** -0.5

    # --- dense fused path (also the DENSE PHASE of a sparse_start_step run) ---- #

    def _forward_fused(self, x: torch.Tensor) -> torch.Tensor:
        """EXACT equivalent of ``_forward_looped`` for equal-width w1w2 experts.

        Replaces 4*K slice-GEMMs with 4 GEMMs against the two fused weights. Algebraically
        identical for the same reason the Hopfield version is: ``W1``/``W2`` are the vertical
        stacks of the ``W1_k``/``W2_k``, so
            sum_k p_k * (phi_k @ W2_k)  ==  ((p (x) phi) flattened) @ W2
        and likewise for the second gradient term.

        This path is REQUIRED even for a run that only ever wants sparsity: `_proxy_step`
        distils the proxy against the exact all-K routing distribution, which only the dense
        path computes (the base class asserts this for Hopfield for the same reason).
        """
        W1, W2, W1v, W2v = self._fused_views()
        K, I_e = self.n_experts, self._expert_I
        inv = self._inv_sqrt_I
        method = self._fused_spec["gelu_grad_method"]

        z1 = x @ W1.t()                                  # forward GEMM 1 of 2
        z2 = x @ W2.t()                                  # forward GEMM 2 of 2
        phi, phi_prime = _gelu_and_grad(z1, method)

        lead = z1.shape[:-1]
        phi_v = phi.view(*lead, K, I_e)
        z2_v = z2.view(*lead, K, I_e)
        g2_v = (phi_prime * z2).view(*lead, K, I_e)      # phi'(W1x) * (W2x)

        # E_k = -(1/sqrt(I_e)) * phi(W1_k x) . (W2_k x)  -- identical to `_W1W2Expert`
        E_k = -inv * (phi_v * z2_v).sum(dim=-1)

        proxy_E = self._proxy_energies(x) if (self.proxy_route and self.proxy_rank > 0) else None
        p, logits = self._route(E_k, proxy_E=proxy_E)
        self._track_load(p)
        self._track_energy(E_k)

        if self.proxy_rank > 0:
            self._proxy_step(x, E_k, logits, E_hat=proxy_E)

        pw = p.unsqueeze(-1)
        out = inv * ((phi_v * pw).reshape(*lead, K * I_e) @ W2
                     + (g2_v * pw).reshape(*lead, K * I_e) @ W1)

        if self.training and self._capture_energy:
            self._last_energy_per_token = -self.temperature * torch.logsumexp(logits, dim=-1)
        else:
            self._last_energy_per_token = None

        if self.training and self.repulsion_coef > 0:
            if self.repulsion_space == "weight":
                self._add_repulsion_loss_weight(self._weight_blocks(K, W1, W2))
            elif self.repulsion_subsample > 0:
                self._add_repulsion_loss(self._subsampled_expert_out(
                    x.reshape(-1, self.hidden_size), self.repulsion_subsample,
                    W1, W2, W1v, W2v))
            else:
                self._add_repulsion_loss(self._expert_out_stack(phi_v, g2_v, W1v, W2v))

        if self.training and self._cos_probe_fires():
            with torch.no_grad():
                self._probe_expert_cos(self._subsampled_expert_out(
                    x.reshape(-1, self.hidden_size), self.repulsion_subsample or 64,
                    W1, W2, W1v, W2v))

        if not torch.compiler.is_compiling():
            self._log_metrics(p, out)

        return out

    # --- genuinely sparse path ------------------------------------------------ #

    def _forward_sparse(self, x: torch.Tensor, sel_idx: torch.Tensor | None = None,
                        exact_denominator: torch.Tensor | None = None) -> torch.Tensor:
        """Only the p candidate experts' forward AND back projections are computed.

        Mirrors ``BoltzmannMoEFFEnergy._forward_sparse`` step for step; the two differences are
        that steps 3 and 5 each do TWO bmms instead of one, and that step 4's energy is the
        bilinear contraction instead of a squared norm. The token dispatch of step 2 is SHARED
        between the two projections, so the gather/scatter cost does not double.

        ``sel_idx`` / ``exact_denominator`` override the proxy and are used ONLY by the test,
        which must hold the selection fixed to check the arithmetic underneath it.
        """
        spec = self._fused_spec
        W1, W2, W1v, W2v = self._fused_views()
        K, I_e, H = self.n_experts, self._expert_I, self.hidden_size
        inv = self._inv_sqrt_I
        method = spec["gelu_grad_method"]
        lead = x.shape[:-1]
        xf = x.reshape(-1, H)
        T = xf.shape[0]
        k = int(self.top_k)

        # ---- 1. SELECT with the cheap router: no expert matmul at all -------------------
        E_prox = self._proxy_energies(x).reshape(-1, K)
        mom = self._zscore_moments(E_prox)
        s_prox = self._logits_raw(E_prox, moments=mom)
        # The dual is solved on the PROXY logits: they are the only all-K quantity here, and
        # mu exists to steer SELECTION, which is made on these logits.
        mu = self._mu_for(s_prox)
        if mu is not None:
            s_prox = s_prox - mu.to(s_prox.dtype)
        p_cand = self.sparse_candidates or k
        n_exp = self.sparse_explore if self.training else 0
        if sel_idx is None and n_exp > 0 and p_cand + n_exp <= K:
            # Deterministic rotation, NO RNG -- the base class records a 14-minute hang with
            # zero optimiser steps when this was a torch.randint inside the compiled graph.
            tt = torch.arange(T, device=xf.device).unsqueeze(1)
            jj = torch.arange(1, n_exp + 1, device=xf.device).unsqueeze(0)
            top = s_prox.topk(p_cand, dim=-1).indices
            sel_idx = torch.cat([top, (tt + jj) % K], dim=-1)
            p_cand = p_cand + n_exp
        elif sel_idx is None:
            sel_idx = (torch.arange(K, device=xf.device).expand(T, K) if p_cand == K
                       else s_prox.topk(p_cand, dim=-1).indices)
        else:
            sel_idx = sel_idx.reshape(T, -1)
            p_cand = sel_idx.shape[-1]

        # ---- 2. capacity-dispatch the TOKENS (shared by both projections) ---------------
        se_k, st_k, slot_k, sp_k, C, kept, _ = self._dispatch_plan(sel_idx, T)
        xbuf = xf.new_zeros(K, C + 1, H)      # +1 = dump row for overflow; `kept` zeroes it
        xbuf[se_k, slot_k] = xf[st_k]

        # ---- 3. SPARSE FORWARD PROJECTIONS: two of them, each cf*k/K of the dense GEMM --
        z1 = torch.bmm(xbuf, W1v.transpose(1, 2))        # (K, C+1, I_e)
        z2 = torch.bmm(xbuf, W2v.transpose(1, 2))
        phi, phi_prime = _gelu_and_grad(z1, method)

        # ---- 4. EXACT energies -- free, they fall out of the projections just done ------
        # NOTE vs Hopfield: `.sum(-1)` with the 1/sqrt(I_e) prefactor, not `.mean(-1)`. The
        # difference is not cosmetic -- it is the cancellation that breaks the m-subsample
        # proxy (module docstring §8).
        E_pair = -inv * (phi * z2).sum(-1)               # (K, C+1)
        E_tk = E_prox.new_zeros(T * p_cand)
        E_tk[sp_k] = E_pair[se_k, slot_k].to(E_tk.dtype)
        E_tk = E_tk.view(T, p_cand)

        # ###################### COPIED BLOCK -- START ###################################
        # Verbatim from BoltzmannMoEFFEnergy._forward_sparse (energy_ff.py steps 4->5). It is
        # expert-kind-agnostic and belongs in a shared helper, but it is inline there and
        # energy_ff.py is frozen (live runs). See the module docstring's DRIFT WARNING; TEST 9
        # of test_sparse_w1w2_20260918.py hashes the original and fails if it moves.
        mom_w = self._zscore_moments(E_tk) if p_cand == K else mom
        lg = self._logits_raw(E_tk, moments=mom_w, expert_idx=sel_idx)
        if mu is not None:
            lg = lg - mu.to(lg.dtype)[sel_idx]
        neg = torch.finfo(lg.dtype).min
        lg = lg.masked_fill(~kept, neg)
        if p_cand > k:
            eq = sel_idx.unsqueeze(-1) == sel_idx.unsqueeze(-2)
            kept = kept & ~eq.tril(-1).any(-1)
        if p_cand > k:
            win = lg.topk(k, dim=-1).indices
            wmask = torch.zeros_like(lg, dtype=torch.bool).scatter_(-1, win, True)
        else:
            wmask = kept
        ref = s_prox if exact_denominator is None else exact_denominator
        mx = torch.maximum(lg.max(-1, keepdim=True).values, ref.max(-1, keepdim=True).values)
        num = torch.exp(lg - mx) * wmask
        Zcand = (torch.exp(lg - mx) * kept).sum(-1, keepdim=True)
        if p_cand == K:
            Zrest = torch.zeros_like(Zcand)
        else:
            Zrest = (torch.exp(ref - mx).sum(-1, keepdim=True)
                     - torch.exp(ref.gather(-1, sel_idx) - mx).sum(-1, keepdim=True)).clamp_min(0)
        Zall = (Zcand + Zrest).clamp_min(1e-30)
        Zwin = num.sum(-1, keepdim=True).clamp_min(1e-30)
        w = num / (Zwin if self.renormalize_topk else Zall)
        # ###################### COPIED BLOCK -- END #####################################

        # ---- 5. SPARSE BACK PROJECTIONS: two bmms, one scatter-add ---------------------
        # grad_x E_k = (1/sqrt(I_e)) * [ W2_k^T phi(W1_k x) + W1_k^T (phi'(W1_k x) * W2_k x) ]
        y = torch.bmm(phi, W2v) + torch.bmm(phi_prime * z2, W1v)     # (K, C+1, hidden)
        wp = w.reshape(-1)[sp_k]
        contrib = y[se_k, slot_k] * (inv * wp).unsqueeze(-1)
        out = xf.new_zeros(T, H).index_add_(0, st_k, contrib.to(xf.dtype)).reshape(*lead, H)

        # ---- 5b. TRAIN THE PROXY on the candidates whose exact energies we just computed -
        if self.training and self.proxy_loss_coef > 0 and self.proxy_rank > 0:
            tgt = F.softmax(lg.detach().masked_fill(~kept, torch.finfo(lg.dtype).min), dim=-1)
            pred = s_prox.gather(-1, sel_idx).masked_fill(~kept, torch.finfo(s_prox.dtype).min)
            add_aux_loss(self.proxy_loss_coef
                         * F.kl_div(F.log_softmax(pred, dim=-1), tgt, reduction="batchmean"))
            if self.track_load:
                with torch.no_grad():
                    kk = min(k, p_cand)
                    a = lg.detach().topk(kk, dim=-1).indices
                    b = s_prox.gather(-1, sel_idx).detach().topk(kk, dim=-1).indices
                    hit = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1)
                    self._proxy_agree_sum += hit.sum() / kk
                    self._proxy_agree_n += hit.shape[0]

        # ---- 6. metrics and side channels ---------------------------------------------
        p_full = w.new_zeros(T, K).scatter_(-1, sel_idx, w)
        self._track_load(p_full)
        self._track_energy(E_tk)

        if self.training and self._capture_energy:
            self._last_energy_per_token = (-self.temperature * (mx + torch.log(Zall))).squeeze(-1)
        else:
            self._last_energy_per_token = None

        need_rep = self.training and self.repulsion_coef > 0 and self.repulsion_space != "weight"
        need_probe = self.training and self._cos_probe_fires()
        if need_rep or need_probe:
            # w1w2 expert outputs are a SUM of two back-projections, so there is no single
            # (gated, W) pair for `_add_repulsion_loss_fused`. Build the output stack for a
            # token subsample instead and use the LOOPED helpers, which take (..., K, hidden).
            eg = self._subsampled_expert_out(xf, self.repulsion_subsample or 64,
                                             W1, W2, W1v, W2v)
            if need_rep:
                self._add_repulsion_loss(eg)
            if need_probe:
                self._probe_expert_cos(eg)
        elif self.training and self.repulsion_coef > 0:
            self._add_repulsion_loss_weight(self._weight_blocks(K, W1, W2))

        if not torch.compiler.is_compiling():
            self._log_metrics(p_full, out)

        return out

    # --- expert-output helpers (w1w2 needs two back-projections) -------------- #

    # PASS THE VIEWS IN. `self._fused_W()` resolves the closure `lambda: holder.W1.weight`,
    # which under FSDP is a SHARDED parameter, so calling it again here is a SECOND gather of a
    # tensor the forward has already materialised -- an all-gather issued from inside a dynamo
    # graph, which HANDOFF 12.27 isolated as the long-standing 2-node wedge. Every helper below
    # therefore takes W1/W2/W1v/W2v from its caller.
    def _expert_out_stack(self, phi_v: torch.Tensor, g2_v: torch.Tensor,
                          W1v: torch.Tensor, W2v: torch.Tensor) -> torch.Tensor:
        """All K expert outputs, ``(..., K, hidden)``, from the dense fused intermediates."""
        return self._inv_sqrt_I * (torch.einsum("...ki,kih->...kh", phi_v, W2v)
                                   + torch.einsum("...ki,kih->...kh", g2_v, W1v))

    @staticmethod
    def _weight_blocks(K: int, W1: torch.Tensor, W2: torch.Tensor) -> torch.Tensor:
        """Per-expert weight block for weight-space repulsion: ``[W1_k ; W2_k]`` flattened."""
        return torch.cat([W1.reshape(K, -1), W2.reshape(K, -1)], dim=1)

    def _subsampled_expert_out(self, xf: torch.Tensor, m: int, W1: torch.Tensor,
                               W2: torch.Tensor, W1v: torch.Tensor,
                               W2v: torch.Tensor) -> torch.Tensor:
        """All K expert OUTPUTS for a deterministic stride of m tokens -> (m, K, hidden).

        The w1w2 counterpart of ``_subsampled_gated``, which returns ``gated`` because a
        Hopfield output is ``gated @ W``; here the output needs both projections, so the
        back-projection is done inside. DETERMINISTIC STRIDE, no RNG: activation checkpointing
        replays the forward during backward and a fresh draw would change the saved tensors
        (``_add_repulsion_loss_fused`` carries a CheckpointError scar from exactly that).
        """
        K, I_e = self.n_experts, self._expert_I
        T = xf.shape[0]
        m = max(1, min(int(m), T))
        step = max(1, T // m)
        idx = torch.arange(0, m * step, step, device=xf.device)[:m]
        xs = xf[idx]
        z1 = xs @ W1.t()
        z2 = xs @ W2.t()
        phi, phi_prime = _gelu_and_grad(z1, self._fused_spec["gelu_grad_method"])
        return self._expert_out_stack(phi.view(m, K, I_e), (phi_prime * z2).view(m, K, I_e),
                                     W1v, W2v)

    def _add_repulsion_loss_weight(self, W: torch.Tensor | None = None) -> None:
        """Weight-space repulsion on the PAIR: expert k's block is ``[W1_k ; W2_k]`` flattened.

        There is no one-matrix definition to inherit. Concatenating is the choice that keeps
        the quantity a single cosine per pair and treats the two matrices symmetrically; the
        alternative (mean of two separate cosines) would let one matrix's alignment hide in the
        other's. UNMEASURED at scale -- and note the base class's own warning that the
        `repulsion_coef` calibrated for output space does NOT transfer to weight space.
        """
        if W is None:
            # Fallback for the LOOPED path only (it has no fused W to hand). The fused and
            # sparse paths always pass theirs -- see the note above `_expert_out_stack`.
            W1, W2, _, _ = self._fused_views()
            W = self._weight_blocks(self.n_experts, W1, W2)
        super()._add_repulsion_loss_weight(W)

    # --- rank-r proxy router for a BILINEAR energy ---------------------------- #

    def _proxy_energies(self, x: torch.Tensor) -> torch.Tensor:
        """Cheap approximate per-expert energies for ``E = -I_e^-0.5 * phi(W1 x) . (W2 x)``.

        THE CORRECT ANALOGUE OF THE HOPFIELD SUBSPACE PROXY, and where it stops being one.

        Hopfield's ``proxy_kind="subspace"`` computes ``mean_j gelu((B_k a_k)_j)^2 * s_k + b_k``
        with ``a_k = V_k^T x`` and ``B_k = W_k V_k``, so that ``B_k a_k = W_k P_k x`` EXACTLY
        for ``P_k = V_k V_k^T``: the only error is the rank truncation, not the function class.

        WHERE THE ANALOGY HOLDS. The same construction works here, once per matrix:
            a1 = V1_k^T x,  u = B1_k a1 = W1_k P1 x        B1_k = W1_k V1_k
            a2 = V2_k^T x,  v = B2_k a2 = W2_k P2 x        B2_k = W2_k V2_k
            E_hat_k = -sqrt(I_e) * mean_j( phi(u_j) * v_j ) * s_k + b_k
        The ``-sqrt(I_e) * mean`` is the same number as ``-1/sqrt(I_e) * sum`` when m = I_e, so
        at ``r = d``, ``m = I_e`` and orthonormal V this reproduces the exact energy BIT FOR BIT
        (TEST 10 of the CPU test measures ~1e-16). So the function class is right and, exactly
        as for Hopfield, truncation is the only error -- PROVIDED m = I_e.

        WHERE IT DOES NOT HOLD -- and this is the one genuinely non-mechanical step:

        (a) TWO SUBSPACES, NOT ONE. There is no single SVD of a pair. Separate bases are used
            (2*K*d*r MACs instead of K*d*r, still ~0.1% of the mixture) because that is what
            admits an exact per-matrix SVD warm start. A single shared basis would have to span
            the union of the two row spaces, i.e. ~r/2 of effective rank each, and its
            "principled" init (right singular vectors of the stacked [W1;W2]) optimises
            Frobenius error of the stack, which is not the error of the PRODUCT.

        (b) THE m-ROW SUBSAMPLE IS NOT AN UNBIASED ESTIMATOR HERE. Hopfield's energy is a MEAN
            of NONNEGATIVE terms, so m rows estimate it with relative error ~1/sqrt(m). This
            energy is a SUM OF SIGNED terms -- ``v = W2 x`` has no sign constraint -- so it is
            O(sqrt(I_e)*term), not O(I_e*term), and the subsample's relative error is
            ~sqrt(I_e/m): 210% at I_e=2240, m=512. MEASURED, TEST 8. Consequence: with m = I_e
            the proxy costs K*I_e elementwise per token, which EXCEEDS the sparse mixture's own
            cf*k*I_e (6.4x at K=16, k=2, cf=1.25) and defeats the purpose; with m < I_e the head
            is a LEARNED low-dimensional bilinear form that must be distilled, and its SVD warm
            start is a heuristic rather than an estimator. `_proxy_subsample_is_unbiased` records
            which regime this instance is in.

        (c) A FORM WITH NO HOPFIELD ANALOGUE EXISTS: ``proxy_kind="bilinear"``. Because
            ``gelu(u) ~ u * sigmoid(1.702 u)``, freezing the gate to its mean makes the energy a
            pure QUADRATIC FORM ``-I_e^-0.5 * gbar * x^T (W1^T W2) x``, whose rank-r
            factorisation ``sum_i lambda_i (v_i . x)^2`` needs 2*d*r MACs and r elementwise --
            NO I_e axis at all, which is exactly what (b) says the subspace form cannot give us.
            Hopfield has no such form: its linearisation is ``||W x||^2``, MEASURED at ~0% top-1
            agreement (HANDOFF 7.6) because gelu(z)^2 is not pointwise proportional to z^2.
            ⚠ THE SAME FAILURE MODE APPLIES HERE IN KIND: dropping the gate keeps the terms
            where ``W1 x < 0``, which the true energy suppresses. It is milder (the gate is a
            smooth 0->1 factor rather than a hard rectifier of a squared quantity) but it is NOT
            measured, and HANDOFF 12.11's negative result for the quad head is the prior. Do not
            report a `bilinear` speedup without a top-k agreement number from
            `calibrate_proxy_router_20260916.py`-style fitting on a TRAINED checkpoint.

        Cost, per token, at K=16, r=8, d=768, I_e=2240 (the mixture is 4*K*d*I_e = 110 MMAC):
            subspace, m=I_e :  2*K*r*(d + I_e) = 0.77 MMAC,  K*I_e = 35,840 elementwise  <- (b)
            subspace, m=512 :  2*K*r*(d + 512) = 0.33 MMAC,  K*512 =  8,192 elementwise
            bilinear        :  2*K*d*r         = 0.20 MMAC,  K*r   =    128 elementwise
        """
        xd = x.detach()      # passive observer; see the base class's note on why this matters
        if self.proxy_iters == 1:
            V1, V2, B1, B2 = self.proxy_V, self.proxy_V2, self.proxy_B, self.proxy_B2
            sc, b = self.proxy_scale, self.proxy_bias
        else:
            assert not self.training, (
                "proxy_iters > 1 is eval/calibration only: indexing the head by a call counter "
                "is unsound under activation checkpointing, which replays the forward during "
                "backward"
            )
            g = int(self._proxy_call.item()) % self.proxy_iters
            self._proxy_call += 1
            pick = lambda t: None if t is None else t[g]          # noqa: E731
            V1, V2, B1, B2, sc, b = map(pick, (self.proxy_V, self.proxy_V2, self.proxy_B,
                                               self.proxy_B2, self.proxy_scale, self.proxy_bias))

        a1 = torch.einsum("...h,khr->...kr", xd, V1.to(x.dtype))
        a2 = torch.einsum("...h,khr->...kr", xd, V2.to(x.dtype))
        if self.proxy_kind == "bilinear":
            # -I_e^-0.5 * x^T (A_k B_k^T) x ; the mean gate is absorbed into proxy_scale
            return -self._inv_sqrt_I * (a1 * a2).sum(-1) * sc.to(x.dtype) + b.to(x.dtype)
        u = torch.einsum("...kr,kmr->...km", a1, B1.to(x.dtype))
        v = torch.einsum("...kr,kmr->...km", a2, B2.to(x.dtype))
        # F.gelu is exactly `phi` for every `gelu_grad_method` branch of `_gelu_and_grad`
        # (sigmoid and erf_exact both use F.gelu; tanh_exact differs by <1e-3 pointwise).
        return -(self._expert_I ** 0.5) * (F.gelu(u) * v).mean(-1) * sc.to(x.dtype) + b.to(x.dtype)

    def _svd_refit_proxy(self) -> None:
        """Warm-start the proxy from the trained weights. Called once at the dense->sparse flip.

        subspace: per-matrix rank-r SVD, ``W = U S Vh`` -> ``V1_k = Vh[:r].T``,
        ``B1_k = (U S)[:, :r]``, and the same for W2. Then ``B1 a1 = W1 P1 x`` exactly.

        ⚠ THE ROW SUBSET MUST BE COMMON TO BOTH MATRICES. The energy pairs coordinate j of
        ``W1 x`` with coordinate j of ``W2 x``. The Hopfield version picks the m most energetic
        rows of ``U S``; doing that INDEPENDENTLY for W1 and W2 pairs mismatched coordinates and
        computes a quantity unrelated to the energy -- silently, with no shape error. A single
        row set is chosen here, scored by ``||B1_row|| * ||B2_row||``.
        That score is a HEURISTIC, not an energy-preserving subsample: the summands cancel, so
        the largest ones do not carry most of the sum (module docstring §8). `proxy_scale` is
        left at 1 and distillation is expected to do the rest.

        bilinear: eigendecomposition of the SYMMETRIC PART of ``M_k = W1_k^T W2_k`` (only the
        symmetric part contributes to ``x^T M x``), top-r by |eigenvalue|, split as
        ``A_i = v_i sqrt|l_i|``, ``B_i = sign(l_i) v_i sqrt|l_i|``, so
        ``(A^T x).(B^T x) = sum_i l_i (v_i.x)^2 = x^T M_r x``. `proxy_scale` starts at 0.5, the
        mean of the gelu gate.
        """
        if getattr(self, "_svd_done", False) or self.proxy_rank <= 0:
            return
        if self.proxy_V is None or self.proxy_V2 is None:
            return
        if self.proxy_iters > 1:
            # proxy_V is (n_iter, K, d, r) here, so `.data[k]` would index the ITERATION axis.
            # The base class has the same latent bug; refuse loudly instead of writing garbage.
            import logging
            logging.getLogger(__name__).warning(
                "w1w2 proxy SVD refit skipped: proxy_iters=%d (per-iteration heads must be "
                "fitted offline, e.g. by fit_subspace_proxy_20260916.py)", self.proxy_iters)
            self._svd_done = True
            return
        try:
            with torch.no_grad():
                K, I_e, H = self.n_experts, self._expert_I, self.hidden_size
                mats = []
                for fn in (self._fused_W, self._fused_W2):
                    M = fn()
                    if hasattr(M, "full_tensor"):        # sharded DTensor under FSDP
                        M = M.full_tensor()
                    # fp32 for the factorisation (bf16 has no headroom for an SVD), but do NOT
                    # DOWNcast a float64 weight: the base class's unconditional `.float()` caps
                    # the warm start's fidelity at ~1e-6, which hides the fact that this head
                    # reproduces the exact energy at full rank (TEST 9 measures exactly that).
                    dt = torch.float64 if M.dtype == torch.float64 else torch.float32
                    mats.append(M.view(K, I_e, H).to(dt))
                A1, A2 = mats
                r = self.proxy_V.shape[-1]
                if self.proxy_kind == "bilinear":
                    for k in range(K):
                        Msym = 0.5 * (A1[k].T @ A2[k] + A2[k].T @ A1[k])
                        lam, vec = torch.linalg.eigh(Msym)
                        pick = lam.abs().topk(min(r, lam.numel())).indices
                        lam, vec = lam[pick], vec[:, pick]
                        sq = lam.abs().sqrt()
                        self.proxy_V.data[k].zero_()
                        self.proxy_V2.data[k].zero_()
                        self.proxy_V.data[k][:, : sq.numel()].copy_((vec * sq).to(self.proxy_V.dtype))
                        self.proxy_V2.data[k][:, : sq.numel()].copy_(
                            (vec * sq * lam.sign()).to(self.proxy_V2.dtype))
                        self.proxy_scale.data[k] = 0.5
                        self.proxy_bias.data[k] = 0.0
                else:
                    m = self.proxy_B.shape[-2]
                    for k in range(K):
                        U1, S1, Vh1 = torch.linalg.svd(A1[k], full_matrices=False)
                        U2, S2, Vh2 = torch.linalg.svd(A2[k], full_matrices=False)
                        self.proxy_V.data[k].copy_(Vh1[:r].T.to(self.proxy_V.dtype))
                        self.proxy_V2.data[k].copy_(Vh2[:r].T.to(self.proxy_V2.dtype))
                        B1, B2 = U1[:, :r] * S1[:r], U2[:, :r] * S2[:r]      # (I_e, r) each
                        if m >= I_e:
                            rows = torch.arange(I_e, device=B1.device)
                        else:
                            # COMMON row subset -- see the docstring's warning
                            score = B1.norm(dim=-1) * B2.norm(dim=-1)
                            rows = score.topk(m).indices
                        self.proxy_B.data[k].zero_()
                        self.proxy_B2.data[k].zero_()
                        self.proxy_B.data[k][: rows.numel()].copy_(B1[rows].to(self.proxy_B.dtype))
                        self.proxy_B2.data[k][: rows.numel()].copy_(B2[rows].to(self.proxy_B2.dtype))
                        # the estimator constant is `-sqrt(I_e) * mean_over_m`, which is the
                        # exact energy only at m = I_e; with m < I_e the subsample is biased and
                        # `scale` is the only thing that can absorb it, so it must be distilled.
                        self.proxy_scale.data[k] = 1.0
                        self.proxy_bias.data[k] = 0.0
            self._svd_done = True
        except Exception as e:                        # never let a warm start kill a long run
            import logging
            logging.getLogger(__name__).warning("w1w2 proxy SVD refit skipped: %r", e)
            self._svd_done = True


def build_boltzmann_moe_w1w2_sparse(
    *,
    hidden_size: int,
    intermediate_size: int,
    n_experts: int,
    init_method: str = "normal",
    initializer_range: float = 0.02,
    m_width: float | None = None,
    add_bias: bool = False,
    gelu_grad_method: str = "sigmoid",
    e_sign_override: str | None = None,
    # CONSUMED, NOT FORWARDED (2026-09-19). `BoltzmannMoEW1W2Sparse.__init__` passes
    # `fused_experts=True` to its super() itself -- this whole module IS the fused w1w2 path --
    # so letting the caller's `fused_experts` fall through `**moe_kwargs` raises
    # "got multiple values for keyword argument 'fused_experts'". It is accepted here (and
    # asserted) rather than ignored, because `get_mlp_block` forwards a config's every field
    # by name and a config that says `fused_experts: true` must not crash the build.
    fused_experts: bool = True,
    layer_idx: int | None = None,
    **moe_kwargs,
) -> FusedMoEContainer:
    """Factory mirroring ``build_boltzmann_moe(expert_kind="w1w2", ...)`` with fused+sparse.

    Reuses ``_FusedW1W2Holder`` (the expert pool, unchanged) and ``FusedMoEContainer``
    (the ``self.ffwd`` wrapper, unchanged) by import; only the MoE wrapper differs.

    ROUTING SIGN. ``_W1W2Expert`` stores ``E = -overlap``, so ``e_sign="pos"`` -- the default
    the frozen builder uses -- selects the LOWEST-overlap experts. That is the documented
    inversion (``ROUTING_SIGN_BUG_20260915.md``; CLAUDE.md pre-flight check 8: "composable
    w1w2 needs `neg`"). There are no existing w1w2-sparse checkpoints to preserve, so the
    DEFAULT HERE IS THE CORRECTED SIGN, ``"neg"``. Pass ``e_sign_override="pos"`` to reproduce
    the frozen builder's (inverted) behaviour for an A/B against a published arm.
    """
    # Neither `_W1W2Expert` nor the fused/sparse paths below read a bias (they all use
    # `x @ W.t()`), so a True here would be a SILENT no-op that makes the parameter count
    # disagree with the function. The Hopfield path has the same hazard; refuse it explicitly.
    assert not add_bias, "add_bias is not read by the fused/sparse w1w2 paths; keep it False"
    assert fused_experts, (
        "build_boltzmann_moe_w1w2_sparse IS the fused w1w2 path; fused_experts=False would be a "
        "silent lie. For the looped w1w2 path call build_boltzmann_moe(expert_kind='w1w2')."
    )
    holder = _FusedW1W2Holder(
        hidden_size=hidden_size, intermediate_size=intermediate_size,
        n_experts=n_experts, init_method=init_method,
        initializer_range=initializer_range, m_width=m_width,
        add_bias=add_bias, gelu_grad_method=gelu_grad_method,
    )
    e_sign = e_sign_override if e_sign_override is not None else "neg"
    assert e_sign in ("neg", "pos")
    moe = BoltzmannMoEW1W2Sparse(
        holder.make_experts(),
        hidden_size=hidden_size,
        w1_fn=(lambda: holder.W1.weight),
        w2_fn=(lambda: holder.W2.weight),
        gelu_grad_method=gelu_grad_method,
        e_sign=e_sign,
        layer_idx=layer_idx,
        **moe_kwargs,
    )
    return FusedMoEContainer(expert_holder=holder, moe=moe)


__all__ = ["BoltzmannMoEW1W2Sparse", "build_boltzmann_moe_w1w2_sparse"]
