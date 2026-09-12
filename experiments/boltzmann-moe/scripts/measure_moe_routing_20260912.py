"""Phase 0: measure the REAL routing behaviour of the Boltzmann-MoE-Hopfield FET block.

Everything downstream (cheap proxy router, top-k, iteration reuse) hinges on facts
this script establishes on real activations rather than synthetic input:

  A. Routing sharpness per iteration — effective_n_experts, top-k mass, expert load.
     If p is ~uniform, top-k is a pure quality loss, not a FLOPs trade, and the MoE
     is algebraically a dense Hopfield FF.
  B. Iteration stability — the energy block runs 6x. If p at iteration t matches
     p at iteration 0, we can route ONCE and reuse: a 6x router cut, free.
  C. Sharpening — what tau (or z-scored logits) would be needed to make routing
     usable for top-k, given the measured E_k scale.
  D. Cheap-router feasibility on real x — rank-r spectral proxy of E_k, and the
     "exact energy restricted to the rank-r subspace" variant, scored by top-k
     selection agreement (what actually matters) not energy MSE.
  E. Caches (x, E_k) pairs so the proxy router can be fitted with a frozen backbone.
  F. Per-task routing profiles — does routing specialize across the 27 BBH tasks?

Real activations come from the BBH few-shot prompts already stored alongside the
checkpoint (fully offline, same distribution as the reported BBH eval).

Usage (1 GPU is plenty):
  python experiments/boltzmann-moe/scripts/measure_moe_routing_20260912.py \
      --n_prompts 540 --max_len 1024 --cache_per_iter 8000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_SIGMOID_SCALE_LOCAL = (2.0 / math.pi) ** 0.5

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401  (registers the `energy` model type)
from transformers import AutoModelForCausalLM, AutoTokenizer

RESULTS = REPO / "experiments/energy-inference/results/multi-block-ablation"
DEFAULT_RUN = "math_fet_boltz_hopfield_rep_8gpt_1egpt6x_d1536_int8k_K8_lra32_itd3_lr1p5e3_33b_16gpu"
OUT = REPO / "experiments/boltzmann-moe/results/router_analysis"
RUN, CKPT = DEFAULT_RUN, RESULTS / DEFAULT_RUN / "unsharded"


# --------------------------------------------------------------------------- #
# instrumentation                                                             #
# --------------------------------------------------------------------------- #


def _row_overlap(ref: torch.Tensor, cand: torch.Tensor, k: int, largest: bool = True) -> float:
    """Fraction of each row's top-k of ``cand`` that lands in the SAME row's top-k of ``ref``.

    Must be done row-wise: ``torch.isin`` flattens its test set, which silently
    compares against every row at once and reports ~1.0 for anything.
    """
    a = ref.topk(k, -1, largest=largest).indices
    b = cand.topk(k, -1, largest=largest).indices
    mask = torch.zeros_like(ref, dtype=torch.bool).scatter_(-1, a, True)
    return mask.gather(-1, b).float().mean().item()


class _LegacyExpert:
    """One expert of a legacy ``BoltzmannMoE_Energy_MLP``, exposing the composable API.

    Legacy routing energy (mlp.py, ``BoltzmannMoE_Energy_MLP.forward``):
        term1_k = phi(W1_k x) @ W2_k        phi = F.gelu  (gelu_grad_method="sigmoid")
        E_k     = (x . term1_k) * _routing_scale,   _routing_scale = 1/sqrt(expert_I)
    and it routes with ``softmax(+E_k/tau)`` -> e_sign="pos".
    """

    def __init__(self, parent, k: int) -> None:
        self.parent, self.k = parent, k
        self.hidden_size = parent.hidden_size
        self.intermediate_size = parent.expert_I
        lo, hi = k * parent.expert_I, (k + 1) * parent.expert_I
        self._lo, self._hi = lo, hi

    def _w(self):
        p = self.parent
        return p.W1.weight[self._lo:self._hi], p.W2.weight[self._lo:self._hi]

    def stacked_weight(self) -> torch.Tensor:
        W1, W2 = self._w()
        return torch.cat([W1, W2], dim=0)          # rows spanning what E_k can see

    def energy_per_token(self, x: torch.Tensor) -> torch.Tensor:
        W1, W2 = self._w()
        phi = F.gelu(x @ W1.t())
        term1 = phi @ W2                            # [..., hidden]
        scale = self.parent.expert_I ** -0.5
        return (x * term1).sum(-1) * scale

    def grad_per_token(self, x: torch.Tensor) -> torch.Tensor:
        """expert_grads = term1 + term2 (legacy class applies no scale here)."""
        W1, W2 = self._w()
        W1x, W2x = x @ W1.t(), x @ W2.t()
        phi = F.gelu(W1x)
        phi_prime = torch.sigmoid(_SIGMOID_SCALE_LOCAL * W1x) * 0.5
        return phi @ W2 + (phi_prime * W2x) @ W1


class LegacyMoEAdapter:
    """Wraps a legacy ``BoltzmannMoE_Energy_MLP`` in the composable MoE interface
    the Recorder/reporters expect, so h1_* checkpoints can be measured unchanged."""

    def __init__(self, legacy) -> None:
        self._legacy = legacy
        self.n_experts = legacy.n_experts
        self.hidden_size = legacy.hidden_size
        self.temperature = legacy.temperature
        self.e_sign = "pos"
        self.experts = [_LegacyExpert(legacy, k) for k in range(legacy.n_experts)]

    @property
    def top_k(self):
        return self._legacy.top_k

    @top_k.setter
    def top_k(self, v):
        self._legacy.top_k = v


def expert_stacked_weight(expert) -> torch.Tensor:
    """The weight rows whose row-space determines this expert's energy.

    Hopfield: W (the energy is a function of W h).
    W1W2:     [W1; W2] (the energy needs both W1 h and W2 h).
    A rank-r router that only sees ``V_r^T x`` can at best rank experts as well as
    the exact energy evaluated at the projection ``x_hat = V_r V_r^T x``.
    """
    if hasattr(expert, "stacked_weight"):
        return expert.stacked_weight()
    if hasattr(expert, "_W_slice"):                 # _HopfieldExpert
        return expert._W_slice()
    if hasattr(expert, "_W1_slice"):                # _W1W2Expert
        return torch.cat([expert._W1_slice(), expert._W2_slice()], dim=0)
    if hasattr(expert, "W") and hasattr(expert.W, "weight"):     # HopfieldFFEnergy
        return expert.W.weight
    if hasattr(expert, "W1"):                                    # W1W2FFEnergy
        return torch.cat([expert.W1.weight, expert.W2.weight], dim=0)
    raise TypeError(f"cannot locate weights for {type(expert).__name__}")


class Recorder:
    """Captures (x, E_k, p, ||out||) at every call of the MoE, tagged by iteration."""

    def __init__(self, moe, cache_per_iter: int, n_iter: int, block=None) -> None:
        self.moe = moe
        self.block = block
        self.scale_ff = float(block.scale_ff.item()) if block is not None else 1.0
        self.K = moe.n_experts if moe is not None else 1
        self.tau = moe.temperature if moe is not None else 1.0
        self.e_sign = moe.e_sign if moe is not None else "neg"
        self.n_iter = n_iter
        self.cache_per_iter = cache_per_iter
        self.call = 0
        self.task = None
        # accumulators, per iteration
        self.stats = [defaultdict(float) for _ in range(n_iter)]
        self.counts = [0 for _ in range(n_iter)]
        self.load = [torch.zeros(self.K) for _ in range(n_iter)]
        self.E_sum = [torch.zeros(self.K) for _ in range(n_iter)]
        self.E_sq = [torch.zeros(self.K) for _ in range(n_iter)]
        self.p_by_task: dict[str, torch.Tensor] = defaultdict(lambda: torch.zeros(self.K))
        self.n_by_task: dict[str, int] = defaultdict(int)
        self.cache_x: list[list[torch.Tensor]] = [[] for _ in range(n_iter)]
        self.cache_E: list[list[torch.Tensor]] = [[] for _ in range(n_iter)]
        self.cached: list[int] = [0 for _ in range(n_iter)]
        # per-forward record of p, for iteration-stability comparison
        self._p_this_forward: list[torch.Tensor] = []
        self.stability = defaultdict(lambda: defaultdict(float))
        self.stability_n = defaultdict(int)
        self._x = None

        # Hook the MoE when present; otherwise hook the plain FF so branch
        # magnitudes are still measured for the non-MoE Hopfield baseline.
        target = moe if isinstance(moe, nn.Module) else block.ffwd
        target.register_forward_pre_hook(self._pre)
        target.register_forward_hook(self._post)

        # Branch-magnitude accounting: is the energy-FF branch actually doing work?
        # grad_E = attn_out + scale_ff * ffwd_out, so compare ||scale_ff*ffwd|| to
        # ||attn_out||, and both to the residual stream ||h|| they perturb.
        self.branch = defaultdict(float)
        self.branch_n = 0
        if block is not None:
            block.attn.register_forward_hook(self._attn_hook)
            block.register_forward_pre_hook(self._block_hook)

    @torch.no_grad()
    def _block_hook(self, mod, args):
        h = (args[0] if args else None)
        if h is None:
            return
        h = h.detach().float().reshape(-1, h.shape[-1])
        self.branch["h_norm"] += h.norm(dim=-1).sum().item()
        self.branch["h_n"] += h.shape[0]

    @torch.no_grad()
    def _attn_hook(self, mod, args, out):
        a = out[0] if isinstance(out, tuple) else out
        a = a.detach().float().reshape(-1, a.shape[-1])
        self.branch["attn_norm"] += a.norm(dim=-1).sum().item()
        self.branch_n += a.shape[0]

    def reset_forward(self) -> None:
        self.call = 0
        self._p_this_forward = []

    def _pre(self, mod, args):
        self._x = args[0].detach()

    @torch.no_grad()
    def _post(self, mod, args, out):
        it = min(self.call, self.n_iter - 1)
        self.call += 1
        x = self._x.reshape(-1, mod.hidden_size)                    # [N, d]
        N0 = x.shape[0]
        self.branch["ffwd_norm"] += out.detach().float().reshape(-1, mod.hidden_size).norm(dim=-1).sum().item()
        self.branch["ffwd_n"] += N0
        if self.moe is None:
            return
        # exact per-expert energy, via the model's own code path
        E = torch.stack([e.energy_per_token(x) for e in self.moe.experts], dim=-1).float()  # [N,K]
        logits = (-E if self.e_sign == "neg" else E) / self.tau
        p = F.softmax(logits, dim=-1)
        N = x.shape[0]

        H = -(p * (p + 1e-9).log()).sum(-1)
        s = self.stats[it]
        s["H"] += H.sum().item()
        s["top1"] += p.max(-1).values.sum().item()
        s["top2"] += p.topk(2, -1).values.sum(-1).sum().item()
        s["top4"] += p.topk(4, -1).values.sum(-1).sum().item()
        s["out_norm"] += out.detach().float().norm(dim=-1).reshape(-1).sum().item()
        # spread of E across experts, per token, relative to |E|
        s["E_std_over_mean"] += (E.std(-1) / E.abs().mean(-1).clamp_min(1e-12)).sum().item()
        s["E_range"] += (E.max(-1).values - E.min(-1).values).sum().item()
        self.counts[it] += N
        sel = logits.argmax(-1)
        self.load[it] += sel.bincount(minlength=self.K).float().cpu()
        self.E_sum[it] += E.sum(0).cpu()
        self.E_sq[it] += (E ** 2).sum(0).cpu()

        if self.task is not None:
            self.p_by_task[self.task] += p.sum(0).cpu()
            self.n_by_task[self.task] += N

        # iteration stability vs iteration 0 of this same forward
        self._p_this_forward.append(p)
        if it > 0:
            p0 = self._p_this_forward[0]
            if p0.shape == p.shape:
                a1 = (p0.argmax(-1) == p.argmax(-1)).float().mean().item()
                ov = _row_overlap(p0, p, 2)
                tv = 0.5 * (p0 - p).abs().sum(-1).mean().item()
                st = self.stability[it]
                st["argmax_agree"] += a1 * N
                st["top2_overlap"] += ov * N
                st["total_variation"] += tv * N
                self.stability_n[it] += N

        # cache a subsample for the frozen-backbone router fit
        want = self.cache_per_iter - self.cached[it]
        if want > 0:
            take = min(want, N)
            idx = torch.randperm(N, device=x.device)[:take]
            self.cache_x[it].append(x[idx].to(torch.float16).cpu())
            self.cache_E[it].append(E[idx].to(torch.float32).cpu())
            self.cached[it] += take


# --------------------------------------------------------------------------- #
# reporting                                                                   #
# --------------------------------------------------------------------------- #


def report_routing(rec: Recorder) -> dict:
    K = rec.K
    print("\n" + "=" * 78)
    print("A. ROUTING SHARPNESS PER ITERATION  (exact energies, real activations)")
    print("=" * 78)
    print(f"  {'iter':>4s} {'tokens':>9s} {'eff_n':>7s} {'top1':>7s} {'top2':>7s} {'top4':>7s} "
          f"{'E_mean':>10s} {'E_range':>10s} {'|out|':>8s}")
    rows = []
    for it in range(rec.n_iter):
        n = rec.counts[it]
        if n == 0:
            continue
        s = rec.stats[it]
        eff = math.exp(s["H"] / n)
        E_mean = (rec.E_sum[it].sum() / (n * K)).item()
        row = dict(iter=it, tokens=n, eff_n_experts=eff,
                   top1=s["top1"] / n, top2=s["top2"] / n, top4=s["top4"] / n,
                   E_mean=E_mean, E_range=s["E_range"] / n, out_norm=s["out_norm"] / n,
                   argmax_load=(rec.load[it] / n).tolist())
        rows.append(row)
        print(f"  {it:4d} {n:9d} {eff:7.3f} {row['top1']:7.4f} {row['top2']:7.4f} "
              f"{row['top4']:7.4f} {E_mean:10.5f} {row['E_range']:10.5f} {row['out_norm']:8.3f}")
    print(f"  (uniform reference: eff_n={K}, top1={1/K:.4f}, top2={2/K:.4f}, top4={4/K:.4f})")
    print("\n  argmax load per expert:")
    for r in rows:
        print(f"    iter {r['iter']}: " + " ".join(f"{v:.3f}" for v in r["argmax_load"]))

    print("\n" + "=" * 78)
    print("B. ITERATION STABILITY  (p at iter t vs iter 0 -> can we route once and reuse?)")
    print("=" * 78)
    print(f"  {'iter':>4s} {'argmax agree':>13s} {'top2 overlap':>13s} {'total var':>11s}")
    stab = []
    for it in sorted(rec.stability):
        n = rec.stability_n[it]
        st = rec.stability[it]
        d = dict(iter=it, argmax_agree=st["argmax_agree"] / n,
                 top2_overlap=st["top2_overlap"] / n,
                 total_variation=st["total_variation"] / n)
        stab.append(d)
        print(f"  {it:4d} {d['argmax_agree']:13.4f} {d['top2_overlap']:13.4f} "
              f"{d['total_variation']:11.4f}")
    return {"per_iter": rows, "stability": stab}


def report_branches(rec: Recorder) -> dict:
    """Does the energy-FF branch contribute anything? grad_E = attn_out + scale_ff*ffwd_out."""
    print("\n" + "=" * 78)
    print("A2. BRANCH MAGNITUDES  (is the MoE FF branch load-bearing at all?)")
    print("=" * 78)
    b = rec.branch
    if not b.get("ffwd_n"):
        return {}
    ffwd = b["ffwd_norm"] / b["ffwd_n"]
    attn = b["attn_norm"] / rec.branch_n if rec.branch_n else float("nan")
    h = b["h_norm"] / b["h_n"] if b.get("h_n") else float("nan")
    scaled = rec.scale_ff * ffwd
    print(f"  mean ||h||            (residual stream into block) : {h:10.4f}")
    print(f"  mean ||attn_out||                                  : {attn:10.4f}")
    print(f"  mean ||ffwd_out||     (MoE energy gradient)         : {ffwd:10.4f}")
    print(f"  scale_ff                                           : {rec.scale_ff:10.4f}")
    print(f"  mean ||scale_ff * ffwd_out||                        : {scaled:10.4f}")
    print(f"  ---> FF share of grad_E magnitude  : {scaled/(scaled+attn)*100:6.2f}%")
    print(f"  ---> FF perturbation relative to h : {scaled/h*100:6.4f}%")
    return {"h_norm": h, "attn_norm": attn, "ffwd_norm": ffwd,
            "scale_ff": rec.scale_ff, "scaled_ffwd_norm": scaled,
            "ff_share_of_gradE": scaled / (scaled + attn) if attn == attn else None}


def report_expert_geometry(moe, rec: Recorder, max_tokens: int = 4096) -> dict:
    """Why is the FF branch alive or dead? Measure expert-output GEOMETRY.

    The repulsion loss (legacy "signed" form) is minimised at cos = -1, so it drives
    expert gradients ANTI-aligned. Anti-aligned vectors summed with near-uniform
    weights cancel. This section separates the two ingredients:

      mean pairwise cos(g_i, g_j)   how anti-aligned the experts actually are
      cancellation ratio            ||sum_k p_k g_k|| / sum_k p_k ||g_k||
                                    1.0 = no cancellation, 0.0 = total cancellation

    A dead branch needs BOTH anti-alignment AND flat routing; either alone is
    survivable, which is the prediction to check against h1 (repulsion 0.1, ten
    times stronger than the FET run, yet a healthy branch).
    """
    print("\n" + "=" * 78)
    print("A3. EXPERT-OUTPUT GEOMETRY  (anti-alignment x flat routing = cancellation)")
    print("=" * 78)
    X = torch.cat([t for lst in rec.cache_x for t in lst])[:max_tokens]
    E = torch.cat([t for lst in rec.cache_E for t in lst])[:max_tokens].float()
    K = E.shape[-1]
    experts = moe.experts
    w0 = expert_stacked_weight(experts[0])
    Xd = X.to(device=w0.device, dtype=w0.dtype)

    G = []
    for e in experts:
        if hasattr(e, "grad_per_token"):
            G.append(e.grad_per_token(Xd).float().cpu())
        else:
            prev = getattr(e, "_capture_energy", False)
            e._capture_energy = False
            G.append(e(Xd).float().cpu())
            e._capture_energy = prev
    G = torch.stack(G, dim=1)                                    # [N, K, d]

    Gn = F.normalize(G, dim=-1)
    C = torch.einsum("nkd,njd->nkj", Gn, Gn)                     # [N, K, K]
    off = ~torch.eye(K, dtype=torch.bool)
    cos_off = C[:, off]
    logits = (-E if rec.e_sign == "neg" else E) / rec.tau
    p = F.softmax(logits, dim=-1)
    num = torch.einsum("nk,nkd->nd", p, G).norm(dim=-1)
    den = (p * G.norm(dim=-1)).sum(-1)
    cancel = (num / den.clamp_min(1e-12))

    print(f"  tokens {G.shape[0]}   K={K}")
    print(f"  mean pairwise cos(g_i, g_j)        : {cos_off.mean().item():+.4f}"
          f"   (min {cos_off.min().item():+.3f}, max {cos_off.max().item():+.3f})")
    print(f"  geometric floor for K vectors      : {-1.0/(K-1):+.4f}"
          f"   (mean cos cannot go below this)")
    print(f"  mean ||g_k||                       : {G.norm(dim=-1).mean().item():.5f}")
    print(f"  cancellation ratio ||sum p g||/sum p||g||: {cancel.mean().item():.5f}"
          f"   (1 = none, 0 = total)")
    print(f"  => surviving fraction of expert magnitude: "
          f"{100*cancel.mean().item():.2f}%")
    return {"mean_pairwise_cos": cos_off.mean().item(),
            "geometric_floor": -1.0 / (K - 1),
            "mean_expert_grad_norm": G.norm(dim=-1).mean().item(),
            "cancellation_ratio": cancel.mean().item()}


def report_sharpening(rec: Recorder) -> dict:
    """What would routing look like under a rescaled / z-scored logit?"""
    print("\n" + "=" * 78)
    print("C. SHARPENING  (frozen weights; what tau / normalization makes top-k viable?)")
    print("=" * 78)
    X = torch.cat([t for lst in rec.cache_x for t in lst]).float()
    E = torch.cat([t for lst in rec.cache_E for t in lst]).float()
    K = E.shape[-1]
    sgn = -1.0 if rec.e_sign == "neg" else 1.0
    per_tok_std = (sgn * E).std(-1)
    print(f"  cached tokens: {E.shape[0]}   E mean {E.mean():.6f}  "
          f"per-token std across experts {per_tok_std.mean():.6f}")
    out = []
    schemes = [("tau=1.0 (as trained)", lambda z: z / 1.0)]
    for m in (0.1, 0.03, 0.01, 0.003, 0.001):
        schemes.append((f"tau={m}", lambda z, m=m: z / m))
    schemes.append(("z-score per token", lambda z: (z - z.mean(-1, keepdim=True))
                    / z.std(-1, keepdim=True).clamp_min(1e-12)))
    for name, fn in schemes:
        p = F.softmax(fn(sgn * E), dim=-1)
        H = -(p * (p + 1e-9).log()).sum(-1)
        d = dict(scheme=name, eff_n=math.exp(H.mean().item()),
                 top1=p.max(-1).values.mean().item(),
                 top2=p.topk(2, -1).values.sum(-1).mean().item())
        out.append(d)
        print(f"  {name:22s} eff_n={d['eff_n']:6.3f}  top1={d['top1']:.4f}  top2={d['top2']:.4f}")
    print(f"  (uniform reference: eff_n={K}, top1={1/K:.4f}, top2={2/K:.4f})")
    return {"schemes": out, "E_mean": E.mean().item(),
            "E_per_token_std": per_tok_std.mean().item()}


def report_proxy(rec: Recorder, moe, ranks: list[int]) -> dict:
    """Cheap-router ceiling on REAL activations, scored by top-k selection agreement.

    Question: if the router only sees ``r`` linear features of x, how well can it
    possibly rank the experts?  Upper bound = the EXACT energy evaluated at the
    rank-r projection ``x_hat = V_r V_r^T x``.  Any implementable head on ``V_r^T x``
    (polynomial, tiny MLP) is bounded by this, and this is kind-agnostic: it works
    for Hopfield (energy sees W h) and for W1W2 (energy sees both W1 h and W2 h).

    Two subspace choices:
      per-expert  V_r^(k) from SVD of expert k's stacked weights   -> K*d*r MACs
      shared      one V_r from SVD of all experts stacked together -> d*r   MACs
    Shared is what an implementation would actually use.
    """
    print("\n" + "=" * 78)
    print("D. CHEAP-ROUTER CEILING on real x  (exact energy at a rank-r projection of x)")
    print("=" * 78)
    X = torch.cat([t for lst in rec.cache_x for t in lst]).float()
    E = torch.cat([t for lst in rec.cache_E for t in lst]).float()
    K, d = E.shape[-1], X.shape[-1]
    largest = (rec.e_sign != "neg")     # "neg" routes to the LOWEST energy
    experts = moe.experts

    Ws = [expert_stacked_weight(e).detach().float().cpu() for e in experts]
    exact_router_macs = sum(w.shape[0] for w in Ws) * d
    # per-expert right-singular bases
    V_per = [torch.linalg.svd(w, full_matrices=False)[2] for w in Ws]
    # one shared basis over all experts' rows
    V_shared = torch.linalg.svd(torch.cat(Ws, 0), full_matrices=False)[2]

    # The cached activations live on CPU while the model is on GPU. Score on the
    # experts' own device so `energy_per_token` sees matching device+dtype, then
    # bring the (tiny) [N,K] energies back to CPU.
    w0 = expert_stacked_weight(experts[0])
    wdev, wdtype = w0.device, w0.dtype

    def energies_at(Xh_fn) -> torch.Tensor:
        cols = []
        for k in range(K):
            xh = Xh_fn(k).to(device=wdev, dtype=wdtype)
            cols.append(experts[k].energy_per_token(xh).float().cpu())
            del xh
        return torch.stack(cols, dim=-1)

    print(f"  {'subspace':12s} {'r':>4s} {'top1':>7s} {'top2':>7s} {'top4':>7s} "
          f"{'router MACs':>12s} {'vs exact':>9s}")
    rows = []
    for r in ranks:
        # shared subspace: one projection reused by every expert
        Vs = V_shared[:r]
        Xh = X @ Vs.T @ Vs
        Ep = energies_at(lambda k, Xh=Xh: Xh)
        m = d * r
        rows.append(dict(subspace="shared", r=r, macs=m,
                         top1=_row_overlap(E, Ep, 1, largest),
                         top2=_row_overlap(E, Ep, 2, largest),
                         top4=_row_overlap(E, Ep, 4, largest)))
        # per-expert subspaces
        # build per-expert projections lazily -- K copies of X would be large
        Ep2 = energies_at(lambda k, r=r: X @ V_per[k][:r].T @ V_per[k][:r])
        rows.append(dict(subspace="per-expert", r=r, macs=K * d * r,
                         top1=_row_overlap(E, Ep2, 1, largest),
                         top2=_row_overlap(E, Ep2, 2, largest),
                         top4=_row_overlap(E, Ep2, 4, largest)))
    for x in sorted(rows, key=lambda z: (z["subspace"], z["r"])):
        print(f"  {x['subspace']:12s} {x['r']:4d} {x['top1']:7.4f} {x['top2']:7.4f} "
              f"{x['top4']:7.4f} {x['macs']/1e3:11.1f}K {x['macs']/exact_router_macs:8.4f}x")
    print(f"  exact router costs {exact_router_macs/1e6:.2f}M MACs/token/iteration")
    return {"rows": rows, "exact_router_macs": exact_router_macs}


def report_tasks(rec: Recorder) -> dict:
    print("\n" + "=" * 78)
    print("F. PER-TASK ROUTING PROFILES  (does routing specialize by BBH task?)")
    print("=" * 78)
    tasks = sorted(rec.p_by_task)
    if not tasks:
        return {}
    P = torch.stack([rec.p_by_task[t] / rec.n_by_task[t] for t in tasks])   # [T,K]
    print(f"  {'task':38s} " + " ".join(f"e{j}" for j in range(P.shape[1])))
    for t, row in zip(tasks, P):
        print(f"  {t:38s} " + " ".join(f"{v:.3f}" for v in row.tolist()))
    spread = P.std(0)
    print(f"\n  across-task std of mean routing weight, per expert: "
          + " ".join(f"{v:.4f}" for v in spread.tolist()))
    print(f"  max across-task swing in any expert: {(P.max(0).values - P.min(0).values).max():.4f}")
    return {"tasks": tasks, "profiles": P.tolist(), "across_task_std": spread.tolist()}


# --------------------------------------------------------------------------- #
# main                                                                        #
# --------------------------------------------------------------------------- #


@torch.no_grad()
def nll_over_prompts(model, tok, prompts, max_len: int, device: str) -> tuple[float, int]:
    """Mean next-token NLL over the prompt set (same tokens every time -> paired A/B)."""
    tot, n = 0.0, 0
    for _, text in prompts:
        ids = tok(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids.to(device)
        if ids.shape[1] < 2:
            continue
        logits = model(input_ids=ids).logits.float()
        loss = F.cross_entropy(logits[0, :-1], ids[0, 1:], reduction="sum")
        tot += loss.item()
        n += ids.shape[1] - 1
    return tot / n, n


def report_ablation(model, tok, block, prompts, max_len: int, device: str) -> dict:
    """The decisive test: does zeroing / uniformizing the MoE branch change the loss?

    ``grad_E = attn_out + scale_ff * ffwd_out``, so replacing ``ffwd_out`` with 0
    deletes the entire energy-FF branch (25% of this model's inference FLOPs).
    Replacing routing with exactly-uniform weights turns the MoE into a plain dense
    Hopfield FF of the same total width -- algebraically identical when p == 1/K.
    """
    print("\n" + "=" * 78)
    print("G. BRANCH ABLATION  (paired NLL on identical tokens)")
    print("=" * 78)
    ffwd = block.ffwd
    real_forward = ffwd.forward
    moe = getattr(ffwd, "moe", None)
    if moe is None and hasattr(ffwd, "n_experts"):
        moe = ffwd            # legacy BoltzmannMoE_Energy_MLP

    base, ntok = nll_over_prompts(model, tok, prompts, max_len, device)
    print(f"  baseline (as trained)                 NLL {base:.6f}  ppl {math.exp(base):9.4f}"
          f"   ({ntok} tokens)")

    ffwd.forward = lambda x: torch.zeros_like(x)
    zero, _ = nll_over_prompts(model, tok, prompts, max_len, device)
    ffwd.forward = real_forward
    print(f"  MoE branch DELETED (ffwd_out = 0)     NLL {zero:.6f}  ppl {math.exp(zero):9.4f}"
          f"   d_ppl {math.exp(zero)-math.exp(base):+8.4f}")

    res = {"n_tokens": ntok, "nll_baseline": base, "nll_ffwd_zero": zero,
           "ppl_baseline": math.exp(base), "ppl_ffwd_zero": math.exp(zero)}
    if moe is None:
        print("  (no MoE on this block -- skipping routing variants)")
        return res

    if not hasattr(moe, "temperature"):
        print("  (learned router: no temperature/top_k energy variants)")
        return res
    orig_tau = moe.temperature
    moe.temperature = 1e9          # softmax -> exactly uniform == dense Hopfield FF
    unif, _ = nll_over_prompts(model, tok, prompts, max_len, device)
    moe.temperature = orig_tau
    print(f"  routing forced UNIFORM (dense Hopf.)  NLL {unif:.6f}  ppl {math.exp(unif):9.4f}"
          f"   d_ppl {math.exp(unif)-math.exp(base):+8.4f}")

    orig_k = moe.top_k
    topk = {}
    for k in (4, 2, 1):
        moe.top_k = k
        v, _ = nll_over_prompts(model, tok, prompts, max_len, device)
        topk[k] = math.exp(v)
        print(f"  top-{k} (truncated, no renorm)          NLL {v:.6f}  ppl {math.exp(v):9.4f}"
              f"   d_ppl {math.exp(v)-math.exp(base):+8.4f}")
    moe.top_k = orig_k

    res.update({"nll_uniform_routing": unif, "ppl_uniform_routing": math.exp(unif),
                "ppl_topk": topk})
    return res


def load_prompts(n_prompts: int) -> list[tuple[str, str]]:
    # Prefer this checkpoint's own BBH samples; otherwise reuse any available set.
    # Using ONE shared prompt set across checkpoints is better for comparison anyway.
    dirs = sorted(CKPT.glob("bbh_samples_*"))
    if not dirs:
        dirs = sorted(RESULTS.glob("*/unsharded/bbh_samples_*"))
    if not dirs:
        raise SystemExit("no bbh_samples_* directory found to draw real prompts from")
    sample_dir = dirs[0]
    files = sorted(sample_dir.glob("samples_bbh_fewshot_*.jsonl"))
    per_file = max(1, n_prompts // max(1, len(files)))
    out = []
    for f in files:
        task = f.name.split("samples_bbh_fewshot_")[1].rsplit("_2026", 1)[0]
        with f.open() as fh:
            for i, line in enumerate(fh):
                if i >= per_file:
                    break
                d = json.loads(line)
                args = d["arguments"]
                key = next(iter(args))
                prompt = args[key]["arg_0"] if isinstance(args[key], dict) else args[key][0]
                out.append((task, prompt))
    return out[:n_prompts]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=DEFAULT_RUN,
                    help="run directory name under results/multi-block-ablation")
    ap.add_argument("--n_prompts", type=int, default=540)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--cache_per_iter", type=int, default=8000)
    ap.add_argument("--ranks", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32])
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--block_idx", type=int, default=None,
                    help="which energy block to instrument (default: last)")
    ap.add_argument("--ablate", action="store_true",
                    help="paired-NLL ablation of the MoE branch (section G)")
    ap.add_argument("--n_ablate", type=int, default=120,
                    help="prompts used for the ablation NLL (5 passes over them)")
    args = ap.parse_args()
    global RUN, CKPT
    RUN = Path(args.run).name if "/" in args.run else args.run
    CKPT = (Path(args.run) if "/" in args.run else RESULTS / args.run) / "unsharded"
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"loading {CKPT}")
    tok = AutoTokenizer.from_pretrained(CKPT)
    model = AutoModelForCausalLM.from_pretrained(
        CKPT, torch_dtype=torch.bfloat16, trust_remote_code=False
    ).to(args.device).eval()

    egpt = [i for i, b in enumerate(model.config.sequence_mixer_blocks)
            if b.sequence_mixer_type == "energy_attention"]
    assert egpt, "no energy_attention block found"
    if len(egpt) > 1:
        print(f"  {len(egpt)} energy blocks {egpt}; instrumenting the LAST "
              f"(override with --block_idx)")
    bi = args.block_idx if args.block_idx is not None else egpt[-1]
    block = model.transformer.h[bi]
    moe = getattr(block.ffwd, "moe", None)
    if moe is None and type(block.ffwd).__name__ == "BoltzmannMoE_Energy_MLP":
        moe = LegacyMoEAdapter(block.ffwd)   # h1_boltz_* checkpoints
    elif moe is None and hasattr(block.ffwd, "n_experts"):
        # e.g. TopK_Energy_MoE_MLP: a LEARNED router, no per-expert energy to
        # measure. Branch magnitudes + ablation still apply.
        print(f"  {type(block.ffwd).__name__}: learned router, "
              f"skipping energy-routing sections")
    n_iter = model.config.layer_iterations[bi]
    print(f"energy block idx={bi}, iterations={n_iter}, "
          f"scale_ff={block.scale_ff.item():.4f}, ffwd={type(block.ffwd).__name__}")
    if moe is not None:
        print(f"  MoE: K={moe.n_experts}, tau={moe.temperature}, e_sign={moe.e_sign}, "
              f"top_k={moe.top_k}")

    rec = Recorder(moe, args.cache_per_iter, n_iter, block=block)
    prompts = load_prompts(args.n_prompts)
    print(f"{len(prompts)} real BBH prompts over "
          f"{len(set(t for t, _ in prompts))} tasks; max_len={args.max_len}")

    with torch.no_grad():
        for i, (task, text) in enumerate(prompts):
            ids = tok(text, return_tensors="pt", truncation=True,
                      max_length=args.max_len).input_ids.to(args.device)
            rec.task = task
            rec.reset_forward()
            model(input_ids=ids)
            if (i + 1) % 100 == 0:
                print(f"  {i+1}/{len(prompts)} prompts", flush=True)

    res = {"run": RUN, "n_prompts": len(prompts)}
    if moe is not None:
        res.update(report_routing(rec))
    res["branches"] = report_branches(rec)
    if moe is not None:
        res["geometry"] = report_expert_geometry(moe, rec)
        res["sharpening"] = report_sharpening(rec)
        res["proxy"] = report_proxy(rec, moe, args.ranks)
        res["tasks"] = report_tasks(rec)
    if args.ablate:
        res["ablation"] = report_ablation(
            model, tok, block, prompts[:args.n_ablate], args.max_len, args.device)

    tag = RUN            # basename; args.run may be a path
    (OUT / f"routing_measure__{tag}.json").write_text(json.dumps(res, indent=1))
    print(f"\nwrote {OUT}/routing_measure__{tag}.json")
    if moe is not None:
        torch.save(
            {"x": torch.cat([t for lst in rec.cache_x for t in lst]),
             "E": torch.cat([t for lst in rec.cache_E for t in lst]),
             "iter": torch.cat([torch.full((sum(t.shape[0] for t in lst),), it)
                                for it, lst in enumerate(rec.cache_x)]),
             "K": moe.n_experts, "tau": moe.temperature, "e_sign": moe.e_sign},
            OUT / f"router_fit_cache__{tag}.pt",
        )
        print(f"wrote {OUT}/router_fit_cache__{tag}.pt  (frozen-backbone router-fit dataset)")


if __name__ == "__main__":
    main()
