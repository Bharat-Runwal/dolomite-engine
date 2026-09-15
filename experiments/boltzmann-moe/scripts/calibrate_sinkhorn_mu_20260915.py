#!/usr/bin/env python3
"""Post-hoc calibration of the running Sinkhorn dual mu for an ALREADY-TRAINED checkpoint.

WHY. mu is a batch statistic solved under no_grad and, before 2026-09-15, applied only when
self.training. So a model trained with mu-tilted routing was EVALUATED with mu = 0 -- a
train/test shift in the router. Measured cost on two otherwise-identical pure-energy isoP
arms: the clamped `load_balance_bias` (a persistent buffer with no self.training gate, so it
DOES reach eval) scored Avg11 41.49, while its Sinkhorn twin, whose mu was dropped, scored
36.16.

This recovers the existing checkpoints WITHOUT retraining, exactly as one recalibrates
BatchNorm running statistics: run a few hundred forward batches with Sinkhorn active,
accumulate the EMA into the `sinkhorn_mu` buffer, and save a new checkpoint directory whose
config sets `sinkhorn_persist_mu: true` so eval applies it.

DATA. Calibration uses the TRAINING corpus, not the eval sets: mu is part of the trained
routing rule, so estimating it on wikitext or the Avg11 tasks would be tuning the router on
the test distribution. Documents are taken from the TAIL of the megatron index, which is the
validation side of the run's own 99.5/0.5/0 split.

The original unsharded/ directory is never modified; output goes to a sibling directory.

KNOWN LIMITATION for RECURRENT / pure-energy stacks. The MoE block is shared across
iterations (layer_iterations [8] for a pure arm, [1,1,1,1,1,1,6] for a hybrid), so a single
forward pass solves a DISTINCT mu at each iteration -- the hidden states differ -- while there
is only ONE `sinkhorn_mu` buffer. The running mean therefore collapses those per-iteration
duals into one compromise value, and eval applies that same value at every iteration. Strictly
better than mu = 0, but NOT what training did. Watch for it in the `count` line: 64 batches on
an 8-iteration arm reports count=512, i.e. 8 solves per batch.
If recovery is only partial, this is the first thing to suspect, and the fix would be a
per-iteration buffer (the block would need to know its iteration index, which it currently
does not).
"""
import argparse, json, shutil
from pathlib import Path
import torch

# REQUIRED before any AutoModel call: importing this runs register_model_classes(), which is
# what puts model_type "energy" into HF's AutoConfig/AutoModelForCausalLM registries. Without
# it from_pretrained dies with `KeyError: 'energy'` ->
# "Transformers does not recognize this architecture". eval_harness.py imports it for the
# same reason.
import lm_engine.hf_models  # noqa: F401

def load_docs(prefix, n_seq, seqlen, tail_frac=0.005):
    from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset
    ds = MMapIndexedDataset(str(prefix))
    n = len(ds)
    start = int(n * (1.0 - tail_frac))          # held-out tail
    buf, seqs = [], []
    i = start
    while len(seqs) < n_seq and i < n:
        buf.extend(ds[i].tolist()); i += 1
        while len(buf) >= seqlen and len(seqs) < n_seq:
            seqs.append(buf[:seqlen]); buf = buf[seqlen:]
    return torch.tensor(seqs, dtype=torch.long), i - start

def _probe_iterations(ckpt, a):
    """Count mu solves in ONE forward = how many times the shared block is applied."""
    from transformers import AutoModelForCausalLM
    m = AutoModelForCausalLM.from_pretrained(ckpt, trust_remote_code=True, torch_dtype=torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"; m.to(dev).train()
    moe = [x for x in m.modules() if getattr(x, "sinkhorn_iters", 0) > 0][0]
    for attr, v in (("repulsion_coef", 0.0), ("cos_probe_interval", 0), ("proxy_loss_coef", 0.0)):
        if hasattr(moe, attr): setattr(moe, attr, v)
    n = {"c": 0}; orig = moe._solve_sinkhorn_mu
    def spy(l): n["c"] += 1; return orig(l)
    moe._solve_sinkhorn_mu = spy
    seqs, _ = load_docs(a.data_prefix, 1, a.seqlen)
    with torch.no_grad(): m(input_ids=seqs[:1].to(dev))
    del m
    if dev == "cuda": torch.cuda.empty_cache()
    assert n["c"] >= 1, "no mu solves observed"
    return n["c"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True, help="results/... dir containing unsharded/")
    ap.add_argument("--data_prefix", default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
    ap.add_argument("--batches", type=int, default=64)
    ap.add_argument("--batch_size", type=int, default=2)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--out_name", default="unsharded_mucal")
    a = ap.parse_args()

    src = Path(a.run_dir) / "unsharded"
    dst = Path(a.run_dir) / a.out_name
    assert src.is_dir(), f"missing {src}"
    if dst.exists(): shutil.rmtree(dst)
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("harness_results*"))

    cfgp = dst / "config.json"; cfg = json.loads(cfgp.read_text())
    blocks = cfg.get("mlp_blocks") or cfg.get("mlp_block_args") or []
    moe_blocks = [b for b in blocks if isinstance(b, dict) and "Boltzmann" in str(b.get("mlp_type", ""))]
    assert moe_blocks, "no EnergyFF_BoltzmannMoE block found in config.json"
    for b in moe_blocks:
        assert int(b.get("sinkhorn_iters", 0)) > 0, "this checkpoint was not trained with sinkhorn"

    # STEP 0: how many times is the block applied per forward? The dual differs per
    # iteration (measured spread 1.75 mean / 3.10 max, with sign flips), so one shared buffer
    # cannot represent it -- that is why the first version of this script recovered nothing.
    # Probe by counting mu solves in a single forward with the buffer still OFF.
    n_iter = _probe_iterations(dst, a)
    print(f"probed {n_iter} mu solve(s) per forward -> sizing the buffer (n_iter, K)")
    for b in moe_blocks:
        b["sinkhorn_persist_mu"] = True
        b["sinkhorn_mu_iters"] = n_iter
    cfgp.write_text(json.dumps(cfg, indent=2))
    print(f"enabled sinkhorn_persist_mu (mu_iters={n_iter}) on {len(moe_blocks)} block(s)")

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained(dst, trust_remote_code=True,
                                                 torch_dtype=torch.bfloat16)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(dev)

    moes = [m for m in model.modules() if hasattr(m, "sinkhorn_mu") and getattr(m, "sinkhorn_mu", None) is not None]
    assert moes, "no module exposes a sinkhorn_mu buffer -- is sinkhorn_persist_mu wired?"
    print(f"found {len(moes)} MoE block(s) with a running-mu buffer; device={dev}")

    # silence the training-only extras so train() mode costs nothing but the mu solve
    for m in moes:
        for attr, val in (("repulsion_coef", 0.0), ("cos_probe_interval", 0), ("proxy_loss_coef", 0.0)):
            if hasattr(m, attr): setattr(m, attr, val)

    seqs, ndoc = load_docs(a.data_prefix, a.batches * a.batch_size, a.seqlen)
    print(f"loaded {len(seqs)} sequences of {a.seqlen} tokens from {ndoc} held-out documents")

    model.train()                      # so the sinkhorn branch runs and the EMA updates
    with torch.no_grad():
        for i in range(a.batches):
            b = seqs[i*a.batch_size:(i+1)*a.batch_size].to(dev)
            if b.shape[0] == 0: break
            model(input_ids=b)
            if (i+1) % 16 == 0:
                mu = moes[0].sinkhorn_mu
                # sinkhorn_mu_count is now per-iteration, shape (n_iter,), so .sum() not float()
                print(f"  batch {i+1:4d}/{a.batches}  |mu|max={mu.abs().max():.4f}  "
                      f"solves={float(moes[0].sinkhorn_mu_count.sum()):.0f}")
    for j, m in enumerate(moes):
        mu = m.sinkhorn_mu.float().cpu()
        print(f"  block {j}: counts={m.sinkhorn_mu_count.cpu().numpy().astype(int).tolist()}")
        for k in range(mu.shape[0]):
            print(f"    iter {k}: |mu|max={mu[k].abs().max():.4f}  mu[:6]={mu[k][:6].numpy().round(3)}")
        spread = (mu.max(0).values - mu.min(0).values)
        print(f"    per-expert spread ACROSS iterations: mean={spread.mean():.4f} max={spread.max():.4f}")
        print(f"    (a single averaged buffer would have had |mu|max={mu.mean(0).abs().max():.4f})")

    model.eval()
    model.save_pretrained(dst, safe_serialization=True)
    keys = [k for k in model.state_dict() if "sinkhorn_mu" in k]
    print(f"saved {dst}  (sinkhorn_mu keys in state_dict: {len(keys)})")
    print("now evaluate that directory instead of unsharded/ to see mu applied at eval")

if __name__ == "__main__":
    main()
