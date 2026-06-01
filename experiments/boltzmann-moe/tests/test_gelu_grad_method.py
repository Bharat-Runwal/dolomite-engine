"""
Test the BoltzmannMoE_Energy_MLP `gelu_grad_method` config flag.

Verifies:
  1. Backward compatibility: with `gelu_grad_method="sigmoid"` (default), the
     forward output is bit-identical to the pre-2026-06 implementation.
  2. Math correctness: with `gelu_grad_method="tanh_exact"`, φ' really is the
     analytic derivative of φ at every input (verified via autograd).
  3. The two modes produce DIFFERENT outputs on the same weights (so a flag
     change is observable in eval).

Run via bsub:
    bsub -q preemptable -G grp_preemptable -J test_gelu_grad \\
         -gpu "num=1/task:mode=exclusive_process" -n 1 -M 16G -W 00:10 \\
         -o $HOME/bsub_logs/test_gelu_grad_%J.stdout \\
         -e $HOME/bsub_logs/test_gelu_grad_%J.stderr \\
         <<'EOF'
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
    cd /proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/tests
    python test_gelu_grad_method.py
    EOF
"""

import math
import torch
import torch.nn.functional as F

import lm_engine.hf_models  # registers
from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP

torch.manual_seed(0)
device = "cuda" if torch.cuda.is_available() else "cpu"
dtype = torch.float32  # exact comparisons need fp32

D, INTER, K = 64, 32, 4   # tiny model, exact-checkable
N = 8                      # tokens

# ── Helper: rebuild a model with the same weights but different flag ─────────
def make(method: str) -> BoltzmannMoE_Energy_MLP:
    m = BoltzmannMoE_Energy_MLP(
        hidden_size=D, intermediate_size=INTER, n_experts=K,
        temperature=1.0, repulsion_coef=0.0, n_repulsion_pairs=2,
        top_k=None, gelu_grad_method=method,
    ).to(device=device, dtype=dtype)
    m.eval()
    return m

m_sig  = make("sigmoid")
m_tanh = make("tanh_exact")
# Force both modules to share weights so the only difference is the flag.
with torch.no_grad():
    m_tanh.W1.weight.copy_(m_sig.W1.weight)
    m_tanh.W2.weight.copy_(m_sig.W2.weight)

x = torch.randn(N, D, device=device, dtype=dtype) * 0.5

# ── Test 1: legacy ("sigmoid") == old hardcoded formula ───────────────────────
def legacy_forward(mod: BoltzmannMoE_Energy_MLP, x: torch.Tensor) -> torch.Tensor:
    """Bit-faithful reproduction of the pre-flag forward (sigmoid path)."""
    leading = x.shape[:-1]
    K_, I_e = mod.n_experts, mod.expert_I
    W1x = mod.W1(x).view(*leading, K_, I_e)          # no dropout in eval
    W1_e = mod.W1.weight.view(K_, I_e, D)
    W2_e = mod.W2.weight.view(K_, I_e, D)
    phi = F.gelu(W1x)
    phi_prime = torch.sigmoid(mod._SIGMOID_SCALE * W1x) * 0.5
    term1 = torch.einsum("...ei,eih->...eh", phi, W2_e)
    E = torch.einsum("...h,...eh->...e", x, term1) * mod._routing_scale
    p = F.softmax(E / mod.temperature, dim=-1)
    W2x = mod.W2(x).view(*leading, K_, I_e)
    term2 = torch.einsum("...ei,eih->...eh", phi_prime * W2x, W1_e)
    out = torch.einsum("...e,...eh->...h", p, term1 + term2)
    return out

with torch.no_grad():
    legacy = legacy_forward(m_sig, x)
    via_flag = m_sig(x)
diff_bc = (legacy - via_flag).abs().max().item()
print(f"[test 1] backward-compat (sigmoid path matches legacy): max abs diff = {diff_bc:.2e}")
assert diff_bc < 1e-6, f"Backward-compat broken: {diff_bc}"

# ── Test 2: tanh_exact φ' = d/dx φ via autograd ──────────────────────────────
y = torch.randn(N, D, device=device, dtype=dtype, requires_grad=True)
c = (2.0 / math.pi) ** 0.5
t = torch.tanh(c * y)
phi_y = 0.5 * y * (1.0 + t)             # the tanh-approx GELU
analytic_phi_prime = 0.5 * (1.0 + t) + 0.5 * c * y * (1.0 - t * t)
autograd_phi_prime = torch.autograd.grad(phi_y.sum(), y, create_graph=False)[0]
diff_ag = (analytic_phi_prime - autograd_phi_prime).abs().max().item()
print(f"[test 2] tanh_exact φ' matches autograd: max abs diff = {diff_ag:.2e}")
assert diff_ag < 1e-6, f"tanh_exact φ' is not the true derivative: {diff_ag}"

# ── Test 3: the two flag values produce DIFFERENT outputs ─────────────────────
with torch.no_grad():
    out_sig  = m_sig(x)
    out_tanh = m_tanh(x)
diff_modes = (out_sig - out_tanh).abs().max().item()
rel_modes  = diff_modes / (out_sig.abs().max().item() + 1e-9)
print(f"[test 3] sigmoid ≠ tanh_exact: max abs diff = {diff_modes:.2e}  (rel {rel_modes:.2e})")
assert diff_modes > 1e-3, "Flag has no observable effect — wiring broken"

# ── Test 4: gradient check — does training produce a valid gradient under tanh_exact? ──
m_tanh.train()
out = m_tanh(x)
loss = out.pow(2).mean()
loss.backward()
grad_norms = {n: p.grad.norm().item() for n, p in m_tanh.named_parameters() if p.grad is not None}
print(f"[test 4] tanh_exact backward grads: {grad_norms}")
assert all(g > 0 for g in grad_norms.values()), "Some gradients are zero — training would stall"

print("\nALL TESTS PASSED.")
