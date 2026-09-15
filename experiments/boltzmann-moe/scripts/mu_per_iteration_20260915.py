#!/usr/bin/env python3
"""Does mu differ ACROSS ITERATIONS of a shared (recurrent) MoE block?

Why it matters. calibrate_sinkhorn_mu_20260915.py keeps ONE running mu, but a pure-energy
stack calls the same block 8x per forward (layer_iterations [8]), solving a separate dual each
time. If those per-iteration duals differ a lot, a single averaged mu cannot reproduce training
behaviour at eval -- which would mean my recalibration test was doomed by construction and the
train/eval mu mismatch is NOT yet ruled out as the cause of the pure+sinkhorn PPL blowup.
If instead they are all similar, the averaged mu was a fair test and the mismatch really is
refuted.
"""
import argparse, torch
import lm_engine.hf_models  # noqa: F401  -- registers model_type "energy"
from transformers import AutoModelForCausalLM
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset

ap = argparse.ArgumentParser()
ap.add_argument("--ckpt", required=True)
ap.add_argument("--data_prefix", default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
ap.add_argument("--seqlen", type=int, default=4096)
ap.add_argument("--batches", type=int, default=3)
a = ap.parse_args()

m = AutoModelForCausalLM.from_pretrained(a.ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16)
dev = "cuda" if torch.cuda.is_available() else "cpu"
m.to(dev).train()
moes = [x for x in m.modules() if hasattr(x, "_solve_sinkhorn_mu") and getattr(x, "sinkhorn_iters", 0) > 0]
assert len(moes) == 1, f"expected exactly one sinkhorn MoE, found {len(moes)}"
moe = moes[0]
for attr, v in (("repulsion_coef", 0.0), ("cos_probe_interval", 0), ("proxy_loss_coef", 0.0)):
    if hasattr(moe, attr): setattr(moe, attr, v)

rec = []
orig = moe._solve_sinkhorn_mu
def spy(logits):
    mu = orig(logits); rec.append(mu.detach().float().cpu().clone()); return mu
moe._solve_sinkhorn_mu = spy

ds = MMapIndexedDataset(a.data_prefix); n = len(ds); start = int(n*0.995)
buf, seqs, i = [], [], start
while len(seqs) < a.batches and i < n:
    buf.extend(ds[i].tolist()); i += 1
    while len(buf) >= a.seqlen and len(seqs) < a.batches:
        seqs.append(buf[:a.seqlen]); buf = buf[a.seqlen:]

for b in range(a.batches):
    rec.clear()
    x = torch.tensor([seqs[b]], dtype=torch.long, device=dev)
    with torch.no_grad(): m(input_ids=x)
    M = torch.stack(rec)                       # (iterations, K)
    print(f"\nbatch {b}: captured {M.shape[0]} mu solves (one per block iteration), K={M.shape[1]}")
    for j in range(M.shape[0]):
        print(f"   iter {j}: |mu|max={M[j].abs().max():.4f}  mu[:6]={M[j][:6].numpy().round(3)}")
    mean = M.mean(0)
    dev_from_mean = (M - mean).abs()
    print(f"  ACROSS ITERATIONS: per-expert spread (max-min) mean={((M.max(0).values-M.min(0).values)).mean():.4f} "
          f"max={((M.max(0).values-M.min(0).values)).max():.4f}")
    print(f"  |mu_iter - mean(mu)| : mean={dev_from_mean.mean():.4f}  max={dev_from_mean.max():.4f}")
    print(f"  |mean(mu)| max       : {mean.abs().max():.4f}")
    print(f"  => an averaged buffer is {'a POOR substitute (duals differ across iterations)' if dev_from_mean.max() > 0.25*max(mean.abs().max(),1e-9) else 'a FAIR substitute'}")
