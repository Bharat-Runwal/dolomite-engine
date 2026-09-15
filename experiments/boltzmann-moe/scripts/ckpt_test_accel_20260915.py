#!/usr/bin/env python3
"""Activation-checkpointing regression test for the fused Boltzmann-MoE path.

WHY THIS EXISTS. The first fused implementation gathered the UNIQUE expert set of
the randomly sampled repulsion pairs, so a tensor's SIZE depended on the draw.
Production trains with gradient_checkpointing_method=block, which re-runs the
forward during backward; the redraw changed the shape and every arm died with
    CheckpointError: saved [7,1280,1536] vs recomputed [8,1280,1536]
That cost a full 4-arm cluster round-trip to discover. This test reproduces the
condition on CPU in seconds. Run it before any cluster submission that touches
the MoE forward.
"""
import sys, random, torch
from torch.utils.checkpoint import checkpoint
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-accel")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe
from lm_engine.hf_models import loss as L

def build(fused, interval, proxy=0):
    torch.manual_seed(0)
    return build_boltzmann_moe(
        expert_kind="hopfield", hidden_size=32, intermediate_size=8 * 16, n_experts=8,
        temperature=0.35, top_k=2, repulsion_coef=0.1, n_repulsion_pairs=4,
        repulsion_form="abs", routing_norm="zscore",
        hopfield_grad_scale="sqrt_consistent", gelu_grad_method="sigmoid",
        init_method="normal", initializer_range=0.02, m_width=None, add_bias=False,
        track_load=True, fused_experts=fused, repulsion_interval=interval,
        repulsion_scale_comp=True, proxy_rank=proxy,
        proxy_loss_coef=(0.01 if proxy else 0.0),
    )

fails = []
for fused in (False, True):
    for interval in (1, 10):
        for proxy in (0, 4):
            m = build(fused, interval, proxy); m.train(); m._capture_energy = True
            ok, err = True, ""
            # many iters so the 1-in-10 gate fires on some and not others, and so
            # both gate branches get exercised under recompute
            for it in range(25):
                try:
                    L.clear_aux_loss()
                    random.seed(it)
                    x = torch.randn(2, 5, 32, requires_grad=True)
                    out = checkpoint(m, x, use_reentrant=False)
                    aux = L.get_aux_loss()
                    loss = out.square().sum() + (aux if torch.is_tensor(aux) else 0.0)
                    loss.backward()
                except Exception as e:
                    ok, err = False, f"iter {it}: {type(e).__name__}: {str(e)[:160]}"
                    break
            tag = f"fused={int(fused)} interval={interval:<2d} proxy={proxy}"
            print(("  PASS  " if ok else "  FAIL  ") + tag + ("" if ok else f"\n          {err}"))
            if not ok:
                fails.append(tag)

print()
if fails:
    print(f"FAILED under activation checkpointing: {fails}"); sys.exit(1)
print("all combinations survive activation checkpointing (non-reentrant, recompute-checked)")
