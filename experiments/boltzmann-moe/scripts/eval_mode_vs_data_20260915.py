#!/usr/bin/env python3
"""Is the pure+sinkhorn PPL blowup a MODEL effect or an EVAL-PATH bug?

Train loss 3.486 (token PPL ~33) on web vs WikiPPL 316 is not a credible domain shift, so
factorise the two things that differ between those numbers:
    MODE:    train() (sinkhorn active) vs eval() (sinkhorn inactive)
    DATASET: held-out web (what it trained on) vs wikitext (what PPL is reported on)
Computing loss on the SAME held-out web data in BOTH modes isolates the mode effect with the
dataset held fixed. Also reports any weights the loader did not find, since a partially
initialised model would produce exactly this signature.
"""
import argparse, torch, torch.nn.functional as F
import lm_engine.hf_models  # noqa: F401  registers model_type "energy"
from transformers import AutoModelForCausalLM
from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset

ap = argparse.ArgumentParser()
ap.add_argument("--ckpts", nargs="+", required=True)
ap.add_argument("--data_prefix", default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
ap.add_argument("--seqlen", type=int, default=4096)
ap.add_argument("--batches", type=int, default=8)
a = ap.parse_args()

ds = MMapIndexedDataset(a.data_prefix); n = len(ds); start = int(n*0.995)
buf, seqs, i = [], [], start
while len(seqs) < a.batches and i < n:
    buf.extend(ds[i].tolist()); i += 1
    while len(buf) >= a.seqlen + 1 and len(seqs) < a.batches:
        seqs.append(buf[:a.seqlen+1]); buf = buf[a.seqlen+1:]
data = torch.tensor(seqs, dtype=torch.long)
print(f"held-out WEB data: {len(data)} sequences of {a.seqlen}+1 tokens\n")

dev = "cuda" if torch.cuda.is_available() else "cpu"
for ck in a.ckpts:
    name = ck.rstrip("/").split("/")[-2]
    m, info = AutoModelForCausalLM.from_pretrained(
        ck, trust_remote_code=True, torch_dtype=torch.bfloat16, output_loading_info=True)
    m.to(dev)
    miss = [k for k in info.get("missing_keys", []) if "sinkhorn" not in k]
    unexp = info.get("unexpected_keys", [])
    print(f"=== {name}")
    print(f"    loader: missing(non-sinkhorn)={miss or 'none'}  unexpected={unexp or 'none'}")
    for mode in ("train", "eval"):
        getattr(m, mode)()
        if mode == "train":
            for x in m.modules():
                for at, v in (("repulsion_coef",0.0),("cos_probe_interval",0),("proxy_loss_coef",0.0)):
                    if hasattr(x, at): setattr(x, at, v)
        tot, ntok = 0.0, 0
        with torch.no_grad():
            for b in range(len(data)):
                s = data[b:b+1].to(dev)
                out = m(input_ids=s[:, :-1])
                lg = out.logits.float()
                l = F.cross_entropy(lg.view(-1, lg.size(-1)), s[:, 1:].reshape(-1), reduction="sum")
                tot += l.item(); ntok += s[:, 1:].numel()
        ce = tot/ntok
        print(f"    {mode:5s} mode: web CE = {ce:.4f}   token PPL = {torch.tensor(ce).exp():.2f}")
    del m
    if dev == "cuda": torch.cuda.empty_cache()
    print()
