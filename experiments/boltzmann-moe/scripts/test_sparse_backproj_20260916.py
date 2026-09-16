#!/usr/bin/env python3
"""Is the capacity-based sparse back-projection EXACT against the dense mask it replaces?

The dense path computes all K experts' back-projections and multiplies K-k of them by zero.
The sparse path evaluates only the k selected ones via capacity dispatch. They must agree to
float tolerance whenever no expert exceeds capacity, and the overflow counter must be the only
thing that changes when one does.
"""
import torch, math
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe

def mk(K, top_k, cf, sparse, H=64, I=256, seed=0):
    torch.manual_seed(seed)
    c = build_boltzmann_moe(expert_kind="hopfield", hidden_size=H, intermediate_size=I,
                            n_experts=K, temperature=1.0, top_k=top_k, e_sign_override="pos",
                            add_bias=False, initializer_range=0.05, m_width=1.0,
                            fused_experts=True, sparse_backproj=sparse,
                            sparse_capacity_factor=cf)
    c.double(); c.eval()
    return c

print("TEST 1 -- sparse vs dense, generous capacity (no overflow expected)")
K, k = 8, 2
x = None
outs = {}
for sparse, cf in ((False, 1.25), (True, 4.0)):
    c = mk(K, k, cf, sparse, seed=0)
    torch.manual_seed(123)
    if x is None: x = torch.randn(2, 12, 64, dtype=torch.float64)
    with torch.no_grad(): outs[sparse] = c(x).clone()
    if sparse: ov = int(c.moe._sparse_overflow) if hasattr(c.moe,"_sparse_overflow") else -1
d = (outs[True]-outs[False]).abs().max().item()
rel = d / outs[False].abs().max().item()
print(f"  max|dense - sparse| = {d:.3e}   relative = {rel:.3e}")
print(f"  overflow pairs = {ov}")
print(f"  EXACT: {rel < 1e-12 and ov == 0}")

print("\nTEST 2 -- tight capacity forces overflow, and it is COUNTED not silent")
c = mk(K, k, 0.15, True, seed=0)          # capacity far below balanced need
torch.manual_seed(123)
with torch.no_grad(): o = c(x).clone()
ov2 = int(c.moe._sparse_overflow)
dd = (o-outs[False]).abs().max().item()
print(f"  overflow pairs = {ov2}  (nonzero expected)")
print(f"  max|dense - sparse| = {dd:.3e}  (nonzero expected: dropped pairs change the function)")
print(f"  behaves as documented: {ov2 > 0 and dd > 0}")

print("\nTEST 3 -- FLOP saving: back-GEMM work, dense vs sparse")
for K_, k_ in ((8,2),(16,2),(32,2),(32,1)):
    T, I_e, H = 4096, 256, 768
    dense = T*K_*I_e*H
    C = math.ceil(1.25*T*k_/K_)
    sparse = K_*C*I_e*H
    print(f"  K={K_:2d} k={k_}  dense={dense/1e9:6.2f} GFLOP  sparse={sparse/1e9:6.2f}  "
          f"saving={100*(1-sparse/dense):5.1f}%  ({dense/sparse:.2f}x)")

print("\nTEST 4 -- guard: sparse_backproj with dense routing must be rejected")
try:
    mk(8, None, 1.25, True); print("  NO ASSERT -- bug")
except AssertionError as e:
    print(f"  correctly rejected: {str(e)[:60]}")
