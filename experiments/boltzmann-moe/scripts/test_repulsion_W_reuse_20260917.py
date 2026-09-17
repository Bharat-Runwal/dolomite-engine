"""Does passing the already-gathered W into _add_repulsion_loss_weight change the MATH?

It must not. The fix (HANDOFF 12.27) exists to remove a SECOND read of the FSDP-sharded
holder.W.weight from inside the compiled region -- that second all-gather is what deadlocks at
2 nodes. Reusing the caller's W is meant to be a pure plumbing change.

_sample_pairs() is stochastic, so both variants are run under an identical reseed.
"""
import sys, torch
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
import lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff as EF
from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff import build_boltzmann_moe

D, K, IE = 64, 8, 32
def build():
    torch.manual_seed(1234)
    return build_boltzmann_moe(
        expert_kind="hopfield", hidden_size=D, intermediate_size=IE*K, n_experts=K, top_k=2,
        routing_norm="zscore", temperature=1.0, e_sign_override="pos", fused_experts=True,
        renormalize_topk=True, repulsion_coef=2.0, repulsion_space="weight",
        repulsion_form="abs", n_repulsion_pairs=4, repulsion_tensor_idx=True,
        hopfield_grad_scale="mean")

captured = []
orig_add = EF.add_aux_loss
EF.add_aux_loss = lambda v: captured.append(v.detach().clone() if torch.is_tensor(v) else v)

c = build().double(); c.moe.train()
W = c.moe._fused_W()

torch.manual_seed(77); captured.clear()
c.moe._add_repulsion_loss_weight()          # old behaviour: re-reads via _fused_W()
a = captured[0] if captured else None

torch.manual_seed(77); captured.clear()
c.moe._add_repulsion_loss_weight(W)         # new behaviour: reuse the caller's W
b = captured[0] if captured else None

EF.add_aux_loss = orig_add
print(f"  no-arg  aux = {a}")
print(f"  with-W  aux = {b}")
ok = (a is not None and b is not None and torch.equal(a, b))
print(f"  bit-identical: {ok}   |diff| = {(a-b).abs().item() if ok is not None and a is not None and b is not None else 'n/a'}")

# and the same object identity: _fused_W() must be returning the very tensor the forward uses
print(f"  _fused_W() is the closure's tensor: {W is c.moe._fused_W()}")
print("\n" + ("PASS -- the fix is pure plumbing, math unchanged" if ok else "FAIL -- the fix changed the math"))
sys.exit(0 if ok else 1)
