"""Fit a cheap rank-r proxy router against the exact Boltzmann energies, frozen backbone.

THE ICLR CLAIM THIS SUPPORTS
----------------------------
Boltzmann routing needs no learned router and no load-balance loss, but the exact
energy router must evaluate ``W_k h`` for EVERY expert, so top-k can only skip the
second matmul (0.5 + 0.5k/K of dense). If a cheap proxy can reproduce the routing
decisions, top-k skips BOTH matmuls for rejected experts (k/K of dense), which the
throughput benchmark measures at 19.0x on the FFN at K=128.

Measured ceiling for the proxy (exact energy at a rank-r projection of x, real
activations): Hopfield-kind 96.7% top-1 agreement at r=16 for 1.6% of the exact
router's MACs. For the w1w2 kind the same estimator plateaus near 0.5 and is
NON-MONOTONE in r, which is not a credible ceiling curve -- so for w1w2 we do not
project x, we FIT a head directly on (x, E_k). That is what this script does.

Router: one shared basis V (d x r) plus a small MLP head with K outputs.
  cost = d*r + r*h + h*K  MACs/token   vs   d*I_tot for the exact router
  at d=768, r=32, h=64, K=4: 24.6K + 2.0K + 0.3K = 27K vs 6.29M  ->  0.43%

Trained with KL against the exact routing distribution (what matters is the
distribution, not the energy values) plus an optional ranking term.

Then evaluated END TO END on the real model with paired NLL on identical tokens:
  exact router (baseline) | proxy router | proxy + top-k | exact + top-k

Usage:
  python experiments/boltzmann-moe/scripts/fit_proxy_router_20260912.py \
      --run experiments/boltzmann-moe/results/h1_boltz_moe_fullsize_d768 \
      --rank 32 --hidden 64 --steps 4000
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

OUT = REPO / "experiments/boltzmann-moe/results/router_analysis"
sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_moe_routing_20260912 import (  # noqa: E402
    LegacyMoEAdapter, expert_stacked_weight, _row_overlap, load_prompts,
    nll_over_prompts,
)


class ProxyRouter(nn.Module):
    """V^T x  ->  MLP  ->  K energy estimates. Deliberately tiny."""

    def __init__(self, d: int, K: int, rank: int, hidden: int) -> None:
        super().__init__()
        self.V = nn.Linear(d, rank, bias=False)
        self.head = nn.Sequential(nn.Linear(rank, hidden), nn.GELU(),
                                  nn.Linear(hidden, K))
        self.d, self.K, self.rank, self.hidden = d, K, rank, hidden

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.V(x))

    def macs_per_token(self) -> int:
        return self.d * self.rank + self.rank * self.hidden + self.hidden * self.K

    @torch.no_grad()
    def init_from_svd(self, Ws: list[torch.Tensor]) -> None:
        """Warm-start V with the top right-singular directions of the expert stack."""
        W = torch.cat([w.float() for w in Ws], 0)
        V = torch.linalg.svd(W, full_matrices=False)[2][: self.rank]
        self.V.weight.copy_(V.to(self.V.weight.dtype))


def fit(cache: dict, rank: int, hidden: int, steps: int, tau: float,
        Ws: list[torch.Tensor], device: str, lr: float = 3e-3) -> tuple[ProxyRouter, dict]:
    X = cache["x"].float()
    E = cache["E"].float()
    sgn = -1.0 if cache["e_sign"] == "neg" else 1.0
    N, K = E.shape
    d = X.shape[-1]
    # held-out split so we report generalisation, not fit
    g = torch.Generator().manual_seed(0)
    perm = torch.randperm(N, generator=g)
    ntr = int(0.9 * N)
    tr, va = perm[:ntr], perm[ntr:]
    Xtr, Etr, Xva, Eva = X[tr].to(device), E[tr].to(device), X[va].to(device), E[va].to(device)

    router = ProxyRouter(d, K, rank, hidden).to(device)
    router.init_from_svd(Ws)
    opt = torch.optim.AdamW(router.parameters(), lr=lr, weight_decay=0.0)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    # target routing distribution from the EXACT energies
    Ptr = F.softmax(sgn * Etr / tau, dim=-1)
    bs = min(4096, Xtr.shape[0])
    best = (1e9, None)
    for it in range(steps):
        idx = torch.randint(0, Xtr.shape[0], (bs,), device=device)
        logits = sgn * router(Xtr[idx]) / tau
        loss = F.kl_div(F.log_softmax(logits, -1), Ptr[idx], reduction="batchmean")
        opt.zero_grad(); loss.backward(); opt.step(); sched.step()
        if (it + 1) % max(1, steps // 10) == 0:
            with torch.no_grad():
                Ehat = router(Xva)
                Pva = F.softmax(sgn * Eva / tau, -1)
                vkl = F.kl_div(F.log_softmax(sgn * Ehat / tau, -1), Pva,
                               reduction="batchmean").item()
                a1 = _row_overlap(Eva.cpu(), Ehat.cpu(), 1, largest=(sgn > 0))
                a2 = _row_overlap(Eva.cpu(), Ehat.cpu(), 2, largest=(sgn > 0))
            print(f"    step {it+1:5d}  train_kl {loss.item():.5f}  val_kl {vkl:.5f}  "
                  f"top1 {a1:.4f}  top2 {a2:.4f}", flush=True)
            if vkl < best[0]:
                best = (vkl, {k: v.detach().clone() for k, v in router.state_dict().items()})
    if best[1] is not None:
        router.load_state_dict(best[1])
    with torch.no_grad():
        Ehat = router(Xva)
        stats = {
            "val_kl": best[0],
            "top1_agree": _row_overlap(Eva.cpu(), Ehat.cpu(), 1, largest=(sgn > 0)),
            "top2_agree": _row_overlap(Eva.cpu(), Ehat.cpu(), 2, largest=(sgn > 0)),
            "top4_agree": _row_overlap(Eva.cpu(), Ehat.cpu(), min(4, K), largest=(sgn > 0)),
            "router_macs": router.macs_per_token(),
        }
    return router, stats


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--steps", type=int, default=4000)
    ap.add_argument("--n_eval", type=int, default=250)
    ap.add_argument("--max_len", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    tag = Path(args.run).name
    ckpt = (Path(args.run) if "/" in args.run else None)
    ckpt = (ckpt or (REPO / args.run)) / "unsharded"
    cache_p = OUT / f"router_fit_cache__{tag}.pt"
    assert cache_p.exists(), f"no cached (x,E) for {tag}; run measure_moe_routing first"
    cache = torch.load(cache_p, weights_only=False)
    print(f"=== {tag} ===\ncache: {cache['x'].shape[0]} tokens, K={cache['K']}, "
          f"tau={cache['tau']}, e_sign={cache['e_sign']}")

    tok = AutoTokenizer.from_pretrained(ckpt)
    model = AutoModelForCausalLM.from_pretrained(
        ckpt, torch_dtype=torch.bfloat16).to(args.device).eval()
    egpt = [i for i, b in enumerate(model.config.sequence_mixer_blocks)
            if b.sequence_mixer_type == "energy_attention"]
    block = model.transformer.h[egpt[-1]]
    moe = getattr(block.ffwd, "moe", None)
    if moe is None and type(block.ffwd).__name__ == "BoltzmannMoE_Energy_MLP":
        moe = LegacyMoEAdapter(block.ffwd)
    Ws = [expert_stacked_weight(e).detach().cpu() for e in moe.experts]

    print(f"\nfitting proxy router (rank={args.rank}, hidden={args.hidden}) ...")
    router, stats = fit(cache, args.rank, args.hidden, args.steps,
                        float(cache["tau"]), Ws, args.device)
    exact_macs = sum(w.shape[0] for w in Ws) * cache["x"].shape[-1]
    print(f"\n  held-out: KL {stats['val_kl']:.5f}  top1 {stats['top1_agree']:.4f}  "
          f"top2 {stats['top2_agree']:.4f}")
    print(f"  router cost {stats['router_macs']/1e3:.1f}K vs exact "
          f"{exact_macs/1e6:.2f}M MACs/token = {100*stats['router_macs']/exact_macs:.3f}%")

    # ---- end-to-end paired NLL: does the proxy preserve the model? ----
    print("\npaired NLL on identical tokens (real model, frozen backbone):")
    prompts = load_prompts(args.n_eval)
    legacy = block.ffwd
    is_legacy = not hasattr(block.ffwd, "moe")
    router_bf = router.to(torch.bfloat16).eval()

    def set_proxy(on: bool):
        """Route with the proxy instead of the exact energies."""
        legacy._proxy = router_bf if on else None

    # monkeypatch the routing energy source
    if is_legacy:
        import types
        orig_forward = legacy.forward

        def patched(self, x):
            if getattr(self, "_proxy", None) is None:
                return orig_forward(x)
            leading = x.shape[:-1]
            W1_e = self.W1.weight.view(self.n_experts, self.expert_I, self.hidden_size)
            W2_e = self.W2.weight.view(self.n_experts, self.expert_I, self.hidden_size)
            W1x = self.W1(x).view(*leading, self.n_experts, self.expert_I)
            phi = F.gelu(W1x)
            phi_p = torch.sigmoid((2.0 / math.pi) ** 0.5 * W1x) * 0.5
            term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
            W2x = self.W2(x).view(*leading, self.n_experts, self.expert_I)
            term2 = torch.einsum("...ei,eih->...eh", phi_p * W2x, W1_e)
            grads = term1 + term2
            E = self._proxy(x)                       # <-- PROXY, not x.term1
            p = F.softmax(E / self.temperature, dim=-1)
            if self.top_k is not None and self.top_k < self.n_experts:
                idx = E.topk(self.top_k, -1).indices
                m = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, True)
                p = p * m
            return torch.einsum("...e,...eh->...h", p, grads)

        legacy.forward = types.MethodType(patched, legacy)

    else:
        # COMPOSABLE class (EnergyFF_BoltzmannMoE). Without this branch the proxy was
        # never actually wired in and every d_ppl came out as exactly +0.0000 -- a
        # no-op masquerading as a perfect result. Patch the MoE wrapper's forward to
        # take its routing logits from the proxy instead of the exact energies.
        import types as _t
        _moe = block.ffwd.moe
        _orig_moe_fwd = _moe.forward

        def moe_patched(self, x):
            if getattr(legacy, "_proxy", None) is None:
                return _orig_moe_fwd(x)
            outs = []
            for e in self.experts:
                prev = e._capture_energy
                e._capture_energy = False
                try:
                    outs.append(e(x))
                finally:
                    e._capture_energy = prev
            grads = torch.stack(outs, dim=-2)                 # [..., K, hidden]
            E = legacy._proxy(x)                              # <-- PROXY energies
            logits = (-E if self.e_sign == "neg" else E) / self.temperature
            p_ = F.softmax(logits, dim=-1)
            if self.top_k is not None and self.top_k < self.n_experts:
                idx = logits.topk(self.top_k, dim=-1).indices
                m = torch.zeros_like(p_, dtype=torch.bool).scatter_(-1, idx, True)
                p_ = p_ * m
            self._last_energy_per_token = None
            return torch.einsum("...e,...eh->...h", p_.to(grads.dtype), grads)

        _moe.forward = _t.MethodType(moe_patched, _moe)

        # top_k lives on the inner wrapper for this class, not on ffwd
        class _TopKShim:
            def __init__(self, m): self._m = m
            def __setattr__(self, k, v):
                if k == "top_k": self._m.top_k = v
                else: object.__setattr__(self, k, v)
        _shim = _TopKShim(_moe)

    res = {"run": tag, "fit": stats, "exact_router_macs": exact_macs}
    rows = []
    for label, proxy_on, k in [("exact router (baseline)", False, None),
                               ("PROXY router", True, None),
                               ("exact router + top-2", False, 2),
                               ("PROXY router + top-2", True, 2),
                               ("PROXY router + top-1", True, 1)]:
        set_proxy(proxy_on)
        if is_legacy: legacy.top_k = k
        else: block.ffwd.moe.top_k = k
        nll, ntok = nll_over_prompts(model, tok, prompts, args.max_len, args.device)
        rows.append(dict(variant=label, nll=nll, ppl=math.exp(nll)))
        base = rows[0]["ppl"]
        print(f"  {label:26s} NLL {nll:.6f}  ppl {math.exp(nll):8.4f}  "
              f"d_ppl {math.exp(nll)-base:+7.4f}   ({ntok} tok)")
    set_proxy(False)
    if is_legacy: legacy.top_k = None
    else: block.ffwd.moe.top_k = None
    res["eval"] = rows

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"proxy_router__{tag}.json").write_text(json.dumps(res, indent=1))
    torch.save(router.state_dict(), OUT / f"proxy_router__{tag}.pt")
    print(f"\nwrote {OUT}/proxy_router__{tag}.json")


if __name__ == "__main__":
    main()
