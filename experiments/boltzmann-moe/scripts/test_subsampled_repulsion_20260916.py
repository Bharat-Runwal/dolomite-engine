#!/usr/bin/env python3
"""Does subsampled OUTPUT-space repulsion work, and is it safe under activation checkpointing?

WHY IT EXISTS. sparse_forward never computes all K expert outputs, so it cannot do output-space
repulsion. The obvious substitute, repulsion_space="weight", was MEASURED not to work: a
1500-step sweep at coef 0.7 and 2.0 showed OUTPUT alignment RISING 0.29 -> 0.44 -> 0.46 across
steps 500/1000/1400, which is the signature HANDOFF 11.8 records for NO repulsion at all. So
instead we keep real output-space repulsion and evaluate it on m tokens over all K experts --
~1.2% of a sparse step at m=64.

A NOTE ON TEST 2, because I got this wrong first time. Comparing two consecutive forwards does
NOT test replay safety: `_sample_pairs` draws expert pairs randomly BY DESIGN, so consecutive
calls legitimately differ. Activation checkpointing restores the RNG state before recomputing
(torch.utils.checkpoint, preserve_rng_state=True), so the correct emulation restores the state
too. The token subsample itself must be RNG-free, and that is tested separately.
"""
import torch

from lm_engine.hf_models.modeling_utils.mlp_blocks import energy_ff as EF
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe


def mk(**kw):
    torch.manual_seed(0)
    base = dict(expert_kind="hopfield", hidden_size=64, intermediate_size=256, n_experts=8,
                temperature=1.0, top_k=2, e_sign_override="pos", add_bias=False,
                initializer_range=0.05, m_width=1.0, fused_experts=True, repulsion_coef=0.1,
                repulsion_form="abs", n_repulsion_pairs=3, repulsion_tensor_idx=True)
    base.update(kw)
    return build_boltzmann_moe(**base)


x = torch.randn(4, 32, 64)
box: list = []
EF.add_aux_loss = lambda t: box.append(t)

print(__doc__.strip().split("\n")[0])
print("=" * 78)

print("\nTEST 1 -- the guard: output-space repulsion under sparse_forward needs a subsample")
try:
    mk(sparse_forward=True, proxy_rank=8, renormalize_topk=True, repulsion_space="output")
    print("  NOT REJECTED -- bug")
except AssertionError as e:
    print(f"  rejected: {str(e)[:66]}")
c = mk(sparse_forward=True, proxy_rank=8, renormalize_topk=True, repulsion_space="output",
       repulsion_subsample=16)
c.moe.train()
box.clear()
out = c.moe._forward_sparse(x)
print(f"  with repulsion_subsample=16: runs, out {tuple(out.shape)}, aux terms {len(box)}, "
      f"finite {bool(box) and bool(torch.isfinite(box[0]).all())}")

print("\nTEST 2 -- the TOKEN subsample must be RNG-free (deterministic stride)")
m = c.moe
W, spec = m._fused_W(), m._fused_spec
xf = x.reshape(-1, 64)
same = torch.equal(m._subsampled_gated(xf, W, spec, 16), m._subsampled_gated(xf, W, spec, 16))
print(f"  two calls bit-identical: {same}   {'OK' if same else 'BUG -- would break checkpointing'}")

print("\nTEST 3 -- replay safety, emulated the way checkpointing actually behaves")
st = torch.get_rng_state()
box.clear(); m._forward_sparse(x); v1 = box[0].item()
torch.set_rng_state(st)
box.clear(); m._forward_sparse(x); v2 = box[0].item()
print(f"  RNG restored: {v1:.10f} vs {v2:.10f}  {'STABLE' if v1 == v2 else 'DIFFERS -- bug'}")
box.clear(); m._forward_sparse(x); v3 = box[0].item()
print(f"  RNG NOT restored: {v3:.10f} -- differs by design (random expert pairs), not a bug")

print("\nTEST 4 -- the alignment probe now works under sparse_forward, so a sparse arm can be")
print("         MEASURED on the metric this change exists to protect")
c = mk(sparse_forward=True, proxy_rank=8, renormalize_topk=True, repulsion_space="output",
       repulsion_subsample=16, cos_probe_interval=1, repulsion_coef=0.0)
c.moe.train()
c.moe._forward_sparse(x)
met = c.moe.pop_load_metrics() or {}
cos = {k: round(v, 4) for k, v in met.items() if "cos" in k}
print(f"  {cos}   {'OK' if cos else 'MISSING -- flying blind on alignment'}")

print("\nTEST 5 -- dense path honours it too (11.8: full output-space repulsion is 17-21% of the")
print("         optimizer step, and its benefit is steeply front-loaded)")
for ms in (0, 8, 32):
    c = mk(repulsion_space="output", repulsion_subsample=ms)
    c.moe.train()
    torch.manual_seed(7)                      # hold the PAIR draw fixed to compare estimators
    box.clear()
    c.moe._forward_fused(x)
    tag = f"all {xf.shape[0]}" if ms == 0 else f"m={ms}"
    print(f"  repulsion_subsample={ms:2d} ({tag:>8s} tokens) -> aux = {box[0].item():.6f}")
print("  (same pair draw, so these differ only by the token estimator -- all the same order)")
