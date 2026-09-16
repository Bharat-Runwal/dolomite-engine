"""Does torch.compile actually NOTICE the sparse_start_step flip?

The dangerous failure is NOT a wedge -- it is dynamo baking _sparse_active=False into the graph,
so the flag flips, no error is raised, and the run silently stays DENSE for its whole life. That
is the same shape as the three silent bugs in 12.13. Testable on CPU.
"""
import sys, torch
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
import torch._dynamo as dynamo
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe

D, K, IE, T = 64, 8, 32, 96
def build(**kw):
    torch.manual_seed(1234)
    return build_boltzmann_moe(expert_kind="hopfield", hidden_size=D, intermediate_size=IE*K,
        n_experts=K, top_k=2, routing_norm="zscore", temperature=1.0, e_sign_override="pos",
        fused_experts=True, renormalize_topk=True, proxy_rank=4, proxy_kind="subspace",
        proxy_out_dim=16, sparse_candidates=2, sparse_explore=2, hopfield_grad_scale="mean", **kw)

dynamo.reset()
c = build(sparse_forward=True, sparse_start_step=5)
paths = []
_fs, _fu = c.moe._forward_sparse, c.moe._forward_fused
c.moe._forward_sparse = lambda t: (paths.append("sparse"), _fs(t))[1]
c.moe._forward_fused  = lambda t: (paths.append("fused"),  _fu(t))[1]

compiled = torch.compile(c, dynamic=False)
x = torch.randn(1, T, D)

c.moe.set_training_step(1)
with torch.no_grad(): o1 = compiled(x)
n1 = len(paths); p1 = list(paths)

c.moe.set_training_step(9)          # <-- the flip
with torch.no_grad(): o2 = compiled(x)
p2 = paths[n1:]

print(f"  before flip, path(s) taken: {sorted(set(p1))}")
print(f"  after  flip, path(s) taken: {sorted(set(p2))}")
same = torch.equal(o1, o2)
print(f"  outputs identical? {same}")
ok = ("fused" in set(p1)) and ("sparse" in set(p2)) and not same
print()
print("VERDICT:", "OK -- compile re-traced and the sparse path really runs after the flip"
      if ok else "FAIL -- the flip did NOT change the executed path under torch.compile")
sys.exit(0 if ok else 1)
