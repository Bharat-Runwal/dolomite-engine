#!/usr/bin/env python3
"""Is the _logits -> _logits_raw + _mu_for refactor behaviour-IDENTICAL to what HEAD did?

This matters more than it looks. The refactor sits on the routing path of EVERY arm, and a
preempted job re-imports the module from disk when LSF requeues it -- so a semantic change here
would silently alter a run that is 38k steps in. The Sinkhorn dual is the delicate part: it has
side effects (the running-mu EMA and the `_mu_call` counter that eval cycles the per-iteration
buffer with, HANDOFF 12.1), so "same returned logits" is not sufficient -- the BUFFERS have to
match too, call for call.

The pre-refactor `_logits` body is pasted verbatim from `git show HEAD:...energy_ff.py` and run
as a free function against the same module instance, so this compares the two implementations
rather than two checkouts.
"""
import copy

import torch
import torch.nn.functional as F

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe


def head_logits(self, E_k):
    """VERBATIM from HEAD, pre-refactor."""
    s = -E_k if self.e_sign == "neg" else E_k
    if self.routing_norm == "sqrt_width":
        s = s * (self.experts[0].intermediate_size ** 0.5)
    elif self.routing_norm == "zscore":
        s = (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp_min(1e-12)
    if self.balance_rate > 0.0 and self.load_balance_bias is not None:
        s = s + self.load_balance_bias.to(s.dtype)
    logits = s / self.temperature
    if self.sinkhorn_iters > 0:
        if self.training:
            mu = self._solve_sinkhorn_mu(logits)
            if self.sinkhorn_persist_mu:
                self._update_running_mu(mu)
            logits = logits - mu.to(logits.dtype)
        elif self.sinkhorn_persist_mu and float(self.sinkhorn_mu_count.sum()) > 0:
            k = int(self._mu_call.item()) % self.sinkhorn_mu_iters
            if float(self.sinkhorn_mu_count[k]) > 0:
                logits = logits - self.sinkhorn_mu[k].to(logits.dtype)
            self._mu_call += 1
    return logits


def mk(seed=0, **kw):
    torch.manual_seed(seed)
    base = dict(expert_kind="hopfield", hidden_size=64, intermediate_size=256, n_experts=8,
                temperature=1.0, top_k=2, e_sign_override="pos", add_bias=False,
                initializer_range=0.05, m_width=1.0, routing_norm="zscore",
                sinkhorn_iters=3, sinkhorn_persist_mu=True, sinkhorn_mu_iters=3)
    base.update(kw)
    c = build_boltzmann_moe(**base)
    c.double()
    return c.moe


def buffers(m):
    return {k: v.clone() for k, v in m.named_buffers() if "sinkhorn" in k or "mu_call" in k}


CASES = {
    "t90k_pure_T12-like (zscore, sinkhorn 3, mu_iters 12)": dict(sinkhorn_mu_iters=12),
    "hybrid-like (zscore, sinkhorn 3, mu_iters 6)": dict(sinkhorn_mu_iters=6),
    "no sinkhorn, clamped bias": dict(sinkhorn_iters=0, sinkhorn_persist_mu=False,
                                      balance_rate=0.01),
    "routing_norm=none, no sinkhorn": dict(routing_norm="none", sinkhorn_iters=0,
                                           sinkhorn_persist_mu=False),
    "sqrt_width, sinkhorn 3": dict(routing_norm="sqrt_width"),
}

print(__doc__.strip().split("\n")[0])
print("=" * 84)
ok_all = True
for label, kw in CASES.items():
    torch.manual_seed(7)
    Es = [torch.randn(2, 16, 8, dtype=torch.float64).abs() * 0.3 for _ in range(14)]
    worst_l, worst_b, mismatches = 0.0, 0.0, []
    for train in (True, False):
        a, b = mk(**kw), mk(**kw)
        a.train(train)
        b.train(train)
        # a: HEAD implementation.   b: refactored.
        for i, E in enumerate(Es):
            la = head_logits(a, E)
            lb = b._logits(E)
            d = (la - lb).abs().max().item()
            worst_l = max(worst_l, d)
            ba, bb = buffers(a), buffers(b)
            for k in ba:
                db = (ba[k].double() - bb[k].double()).abs().max().item()
                worst_b = max(worst_b, db)
                if db > 0:
                    mismatches.append(f"{k}@call{i}/{'train' if train else 'eval'}")
    ok = worst_l < 1e-15 and worst_b == 0.0
    ok_all &= ok
    print(f"  {label:52s} logits {worst_l:.2e}  buffers {worst_b:.2e}  "
          f"{'IDENTICAL' if ok else 'DIFFERS ' + ','.join(mismatches[:3])}")

print("\n  Also: _mu_call must advance exactly once per _logits call in train mode with persist.")
m = mk(sinkhorn_mu_iters=4)
m.train()
for E in [torch.randn(2, 8, 8, dtype=torch.float64).abs() for _ in range(9)]:
    m._logits(E)
n = int(m._mu_call)
print(f"    9 calls -> _mu_call = {n}   {'OK' if n == 9 else 'WRONG'}")
ok_all &= n == 9

# The sparse path must ALSO tick it exactly once, though it applies the logit map twice.
ms = mk(sinkhorn_mu_iters=4, fused_experts=True, sparse_forward=True, proxy_rank=8,
        renormalize_topk=True)
ms.train()
x = torch.randn(2, 8, 64, dtype=torch.float64)
for _ in range(5):
    ms._forward_sparse(x)
ns = int(ms._mu_call)
print(f"    5 sparse forwards -> _mu_call = {ns}   {'OK' if ns == 5 else 'WRONG (double-count)'}")
ok_all &= ns == 5

print(f"\nREFACTOR SAFE: {ok_all}")
