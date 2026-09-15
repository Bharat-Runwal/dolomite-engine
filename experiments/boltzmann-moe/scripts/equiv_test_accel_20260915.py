#!/usr/bin/env python3
"""Numerical-equivalence harness. Run with PYTHONPATH pointing at a checkout;
prints a JSON fingerprint of forward/backward so two checkouts can be compared."""
import json, os, sys, random
import torch
import torch.nn.functional as F

from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe
from lm_engine.hf_models import loss as L

MODE = sys.argv[1] if len(sys.argv) > 1 else "looped"
torch.use_deterministic_algorithms(True)

def build(fused, rep_interval=1, proxy_rank=0):
    torch.manual_seed(1234); random.seed(0)
    kw = dict(
        expert_kind="hopfield", hidden_size=32, intermediate_size=8 * 16,
        n_experts=8, temperature=0.35, top_k=2, repulsion_coef=0.1,
        n_repulsion_pairs=4, repulsion_form="abs", routing_norm="zscore",
        hopfield_grad_scale="sqrt_consistent", gelu_grad_method="sigmoid",
        init_method="normal", initializer_range=0.02, m_width=None,
        add_bias=False, track_load=True,
    )
    if MODE != "base":  # new kwargs only exist on the accel branch
        kw.update(fused_experts=fused, repulsion_interval=rep_interval,
                  repulsion_scale_comp=True, proxy_rank=proxy_rank,
                  proxy_loss_coef=(0.01 if proxy_rank else 0.0))
    return build_boltzmann_moe(**kw)

def run(fused, rep_interval=1, proxy_rank=0):
    m = build(fused, rep_interval, proxy_rank).double()
    m.train()
    torch.manual_seed(7)
    x = torch.randn(4, 6, 32, dtype=torch.double, requires_grad=True)
    L.clear_aux_loss()
    random.seed(42)                      # pin repulsion pair sampling
    m._capture_energy = True
    out = m(x)
    aux = L.get_aux_loss()
    tot = out.square().sum() + (aux if torch.is_tensor(aux) else torch.tensor(0.0))
    tot.backward()
    W = m.expert_holder.W.weight
    res = {
        "out_sum": float(out.sum()), "out_absmax": float(out.abs().max()),
        "out_sq": float(out.square().sum()),
        "energy_sum": float(m._last_energy_per_token.sum()),
        "aux": float(aux) if torch.is_tensor(aux) else float(aux),
        "gx_sum": float(x.grad.sum()), "gx_norm": float(x.grad.norm()),
        "gW_norm": float(W.grad.norm()),
    }
    lm = m.pop_load_metrics()
    for k in ("load_effective_n_experts", "load_max_share"):
        res[k] = round(lm[k], 12)
    if proxy_rank:
        res["proxy_topk_agree"] = round(lm.get("proxy_topk_agree", -1), 12)
    return res

if MODE == "base":
    print(json.dumps({"looped": run(False)}, indent=2))
else:
    print(json.dumps({
        "looped": run(False),
        "fused":  run(True),
        "fused_rep10": run(True, rep_interval=10),
        "fused_proxy4": run(True, proxy_rank=4),
    }, indent=2))
