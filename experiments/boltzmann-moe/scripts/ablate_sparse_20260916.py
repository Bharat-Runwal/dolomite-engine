#!/usr/bin/env python3
"""Which of sparse_forward's three approximations breaks the model? One rung per change.

sparse_forward on pure_hop_T12_sink measured 3.4346 bits/byte against the dense 1.0996 (+2.34
nats), while the proxy's top-2 agreement is 0.88. Agreement that high should not cost 2.34 nats,
so something OTHER than the selection is likely dominating. The candidates, from
test_sparse_forward_20260916.py:

  1. SELECTION      -- which k experts the proxy picks.
  2. DENOMINATOR    -- with renormalize_topk=False (as trained) the softmax denominator runs over
                       all K logits, K-k of which are never computed and are filled from the
                       proxy. A per-token error here RESCALES the whole FF branch, and this arm
                       applies that branch 12 times, so it compounds.
  3. ZSCORE MOMENTS -- per-token mean/std over all K energies, taken from the proxy.

Rung C is the decisive one: proxy SELECTS but everything else is computed densely and exactly, so
it isolates approximation 1 with 2 and 3 removed. Variants share the weights by symlink.
"""
import argparse
import json
import os
import shutil
from pathlib import Path

RUNGS = {
    # name:              (config overrides)
    "B_fused":           dict(fused_experts=True),
    "C_proxysel":        dict(fused_experts=True, proxy_route=True),
    # C' deconfounds D: D flipped renormalize_topk AND enabled the sparse machinery at once.
    # scale_ff was trained against sum(p) ~= 0.45, so forcing renormalisation doubles the FF
    # branch on its own. This rung isolates that model change with no sparse machinery.
    "Cp_proxysel_renorm": dict(fused_experts=True, proxy_route=True, renormalize_topk=True),
    "D_sparse_renorm":   dict(fused_experts=True, sparse_forward=True, renormalize_topk=True),
    "E_sparse_asis":     dict(fused_experts=True, sparse_forward=True),
    # F is the CORRECTNESS SELF-TEST that should have existed before any of the above: with every
    # expert a candidate, all K energies are exact, the denominator has no proxy term and the
    # zscore moments are exact -- so this must reproduce the dense 1.0996. If it does not, the bug
    # is in my dispatch/weighting code and not in any approximation.
    "F_sparse_pK":       dict(fused_experts=True, sparse_forward=True, sparse_candidates=16),
    "G_sparse_p3":       dict(fused_experts=True, sparse_forward=True, sparse_candidates=3),
    "H_sparse_p4":       dict(fused_experts=True, sparse_forward=True, sparse_candidates=4),
    "I_sparse_p6":       dict(fused_experts=True, sparse_forward=True, sparse_candidates=6),
    "J_sparse_p8":       dict(fused_experts=True, sparse_forward=True, sparse_candidates=8),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True, help="checkpoint WITH the fitted proxy tensors")
    ap.add_argument("--out_root", required=True)
    a = ap.parse_args()
    src = Path(a.src)
    cfg0 = json.loads((src / "config.json").read_text())
    root = Path(a.out_root)
    root.mkdir(parents=True, exist_ok=True)
    made = []
    for name, over in RUNGS.items():
        d = root / name
        if d.exists():
            shutil.rmtree(d)
        d.mkdir()
        for f in src.iterdir():
            if f.name == "config.json" or f.name.startswith("harness_results"):
                continue
            os.symlink(f.resolve(), d / f.name)      # share the weights, do not copy
        cfg = json.loads(json.dumps(cfg0))
        for blk in (cfg.get("mlp_blocks") or []):
            if isinstance(blk, dict) and "Boltzmann" in str(blk.get("mlp_type", "")):
                # start from the AS-TRAINED semantics, then apply only this rung's overrides
                blk["sparse_forward"] = False
                blk["proxy_route"] = False
                blk["renormalize_topk"] = False
                blk["sparse_candidates"] = 0
                blk.update(over)
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        made.append((name, d, over))
        print(f"{name:18s} {over}")
    print("\n".join(f"{n}\t{p}" for n, p, _ in made))


if __name__ == "__main__":
    main()
