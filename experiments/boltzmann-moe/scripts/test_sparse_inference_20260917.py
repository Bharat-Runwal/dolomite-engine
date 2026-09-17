"""Dense vs sparse INFERENCE on ONE trained checkpoint -- measures the proxy's prediction error.

The sparse dispatch is already known exact given the selection (4e-16, test_sparse_forward_20260916).
So the only thing left to measure is how much the rank-r proxy's CHOICE of experts costs in loss.
Three paths, identical weights and identical batch:
  (1) DENSE          -- _sparse_active=False, all K experts
  (2) SPARSE p=K     -- proxy nominates all K, so the selection is EXACT; must match (1)
  (3) SPARSE p=top_k -- what actually runs at inference (exploration is off when not training)
gap = loss(3) - loss(1) is the number we care about.

NOTE: a checkpoint saved with sparse_start_step > 0 loads with _sparse_active FALSE, so eval
silently takes the dense path. This script sets the flag explicitly.
"""
import sys, torch, json
from transformers import AutoModelForCausalLM, AutoTokenizer

# model_type 'energy' is a CUSTOM architecture; register_model_classes() is NOT invoked on
# import, so from_pretrained fails with KeyError: 'energy' without this. (Cost one job.)
from lm_engine.hf_models import register_model_classes
register_model_classes()

ckpt = sys.argv[1]
tok_path = sys.argv[2] if len(sys.argv) > 2 else "/proj/datasets/tokenizers/granite-4.0-tiktoken"
dev = "cuda"
torch.manual_seed(0)

model = AutoModelForCausalLM.from_pretrained(ckpt, torch_dtype=torch.bfloat16, trust_remote_code=True).to(dev).eval()
tok = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)

# a fixed, real-text batch (web-like prose, in-distribution enough for a routing comparison)
text = (open("README.md").read() + open("lm_engine/pretrain.py").read()) * 2
T = 1024
ids = tok(text, return_tensors="pt").input_ids[:, : 8 * T].to(dev)
x = ids.reshape(-1, T)[:8]
# HF causal LM shifts labels INTERNALLY, so labels must be the UNSHIFTED input_ids.
# Passing a pre-shifted y double-shifts and scores token t+2 from context t, which drives the
# loss to ~ln(vocab) = 11.52 and makes any comparison meaningless. (Cost one job.)
y = x
print(f"batch {tuple(x.shape)}  ({x.numel()} tokens)  labels = input_ids (HF shifts internally)")

moes = [m for m in model.modules() if hasattr(m, "sparse_candidates") and hasattr(m, "_sparse_active")]
print(f"found {len(moes)} Boltzmann MoE modules; K={moes[0].n_experts} top_k={moes[0].top_k} "
      f"saved sparse_candidates={moes[0].sparse_candidates} _sparse_active={moes[0]._sparse_active}")
K = moes[0].n_experts
orig = [(m._sparse_active, m.sparse_candidates) for m in moes]

@torch.no_grad()
def loss_of(active, cand, label):
    for m in moes:
        m._sparse_active = active
        if cand is not None:
            m.sparse_candidates = cand
    tot, n = 0.0, 0
    for i in range(x.shape[0]):
        out = model(input_ids=x[i : i + 1], labels=y[i : i + 1])
        tot += out.loss.float().item(); n += 1
    l = tot / n
    print(f"  {label:34s} loss {l:.6f}   ppl {torch.tensor(l).exp().item():10.3f}")
    return l

print("\nthree paths, same weights, same batch:")
l_dense  = loss_of(False, None, "(1) DENSE (all K)")
l_exact  = loss_of(True,  K,    f"(2) SPARSE, p=K={K} (exact select)")
l_proxy  = loss_of(True,  orig[0][1], f"(3) SPARSE, p={orig[0][1]} (what inference runs)")
for m, (a, c) in zip(moes, orig): m._sparse_active, m.sparse_candidates = a, c

print(f"\n  (2)-(1) = {l_exact-l_dense:+.6f} nats   <- dispatch check, should be ~0")
print(f"  (3)-(1) = {l_proxy-l_dense:+.6f} nats   <- THE PROXY'S PREDICTION ERROR at inference")
print(f"  ppl ratio (3)/(1) = {torch.tensor(l_proxy-l_dense).exp().item():.4f}x")
