"""CPU test for sparse_start_step: the dense->sparse switch inside ONE training run.

Checks, in order:
  1. the knob RESOLVES onto .moe (pre-flight 7: get_mlp_block forwards kwargs explicitly, so a
     field can parse onto the args object and never reach the builder)
  2. the gate starts dense and flips exactly AT sparse_start_step, not before or after
  3. the dense phase dispatches to _forward_fused and the sparse phase to _forward_sparse
  4. DEFAULTS ARE UNCHANGED: sparse_start_step=0 is sparse from step 0; sparse_forward=False is
     dense forever and set_training_step is a no-op
  5. the dense phase of a sparse_start_step run is BIT-IDENTICAL to a plain dense arm, and the
     sparse phase is bit-identical to a plain sparse arm -- i.e. the gate adds no numerical change
"""
import sys, torch
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe

torch.manual_seed(0)
D, K, IE, T = 64, 8, 32, 96          # tiny; p = 2 + 2 = 4 of 8
BASE = dict(expert_kind="hopfield", hidden_size=D, intermediate_size=IE * K, n_experts=K,
            top_k=2, routing_norm="zscore", temperature=1.0, e_sign_override="pos",
            fused_experts=True, renormalize_topk=True, proxy_rank=4, proxy_kind="subspace",
            proxy_out_dim=16, sparse_candidates=2, sparse_explore=2, hopfield_grad_scale="mean")

def build(**kw):
    torch.manual_seed(1234)                       # identical init across variants
    return build_boltzmann_moe(**{**BASE, **kw})

fails = []
def check(name, ok, extra=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f"  {extra}" if extra else ""))
    if not ok: fails.append(name)

print("1. knob resolves onto .moe")
c = build(sparse_forward=True, sparse_start_step=5)
check("sparse_start_step reached .moe", getattr(c.moe, "sparse_start_step", None) == 5,
      f"got {getattr(c.moe,'sparse_start_step',None)!r}")
check("gate starts INACTIVE (dense first)", c.moe._sparse_active is False)

print("2. the gate flips exactly at sparse_start_step")
seen = {}
for st in [1, 2, 3, 4, 5, 6, 20]:
    c.moe.set_training_step(st); seen[st] = c.moe._sparse_active
check("dense for steps < 5", all(seen[s] is False for s in [1, 2, 3, 4]), str({k: seen[k] for k in [1,2,3,4]}))
check("sparse from step >= 5", all(seen[s] is True for s in [5, 6, 20]), str({k: seen[k] for k in [5,6,20]}))
c.moe.set_training_step(3)
check("gate is a pure function of step (can go back)", c.moe._sparse_active is False)

print("3. dispatch actually changes path")
x = torch.randn(1, T, D, dtype=torch.float32)
calls = []
c2 = build(sparse_forward=True, sparse_start_step=5)
for nm in ("_forward_fused", "_forward_sparse"):
    orig = getattr(c2.moe, nm)
    setattr(c2.moe, nm, (lambda o=orig, n=nm: (lambda t: (calls.append(n), o(t))[1]))())
c2.moe.set_training_step(1); c2.moe(x)
c2.moe.set_training_step(9); c2.moe(x)
check("dense step -> _forward_fused, sparse step -> _forward_sparse",
      calls == ["_forward_fused", "_forward_sparse"], str(calls))

print("4. defaults unchanged")
c3 = build(sparse_forward=True)                      # sparse_start_step defaults to 0
check("sparse_start_step=0 -> active immediately", c3.moe._sparse_active is True)
c3.moe.set_training_step(1)
check("...and set_training_step does not deactivate it", c3.moe._sparse_active is True)
c4 = build(sparse_forward=False)
check("sparse_forward=False -> never active", c4.moe._sparse_active is False)
c4.moe.set_training_step(10_000)
check("...and stays inactive after set_training_step", c4.moe._sparse_active is False)

print("5. the gate introduces NO numerical change (float64, bit-exact)")
xd = torch.randn(1, T, D, dtype=torch.float64)
def out(cfg, step=None):
    c = build(**cfg).double()
    if step is not None: c.moe.set_training_step(step)
    with torch.no_grad(): return c(xd)
a = out(dict(sparse_forward=False))                                  # plain dense
b = out(dict(sparse_forward=True, sparse_start_step=100), step=3)     # gated, in dense phase
d1 = (a - b).abs().max().item()
check("dense phase == plain dense arm", d1 == 0.0, f"max|diff| = {d1:.3e}")
e = out(dict(sparse_forward=True))                                   # plain sparse (from step 0)
f = out(dict(sparse_forward=True, sparse_start_step=5), step=9)       # gated, in sparse phase
d2 = (e - f).abs().max().item()
check("sparse phase == plain sparse arm", d2 == 0.0, f"max|diff| = {d2:.3e}")

print()
print("ALL PASS" if not fails else f"FAILURES: {fails}")
sys.exit(1 if fails else 0)
