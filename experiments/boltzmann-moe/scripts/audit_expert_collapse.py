"""Audit every Boltzmann/learned MoE arm for expert collapse, straight from DCP checkpoints.

WHY THIS EXISTS
---------------
Boltzmann-MoE has NO load-balancing loss and NO learned gate by design; its only
diversity pressure is the repulsion penalty on expert-weight cosine similarity. That
makes collapse a live risk, and until now we checked it exactly once, offline, on the
finished arms only -- because the in-training metrics (effective_n_experts,
max_expert_load) are computed behind ``if not torch.compiler.is_compiling()`` and every
arm runs with torch_compile: true, so they never reached wandb.

This reads ONLY the fused expert weight tensor out of the DCP shards (no unsharding, no
GPU, seconds per arm), so it can run on arms that are still training.

TWO INDEPENDENT FAILURE MODES, both reported:

1. WEIGHT collapse -- the experts have become copies of each other. Detected by pairwise
   cosine similarity between flattened expert blocks, and by each expert's norm (a dead
   expert decays toward zero). This is what repulsion is supposed to prevent.

2. ROUTING degeneracy -- the experts are distinct but assign nearly EQUAL energy to any
   input, so the router cannot discriminate and the mixture is uniform. This is the
   defect we already found in the shipped 33B checkpoint (effective_n_experts 7.999/8).
   It is invisible to weight-space checks. Probed here by pushing Gaussian inputs through
   the real energy functional and measuring the induced routing distribution.

Note (2) uses random inputs, so it measures the routing rule's INTRINSIC selectivity, not
its behaviour on the true hidden distribution. A model that looks selective here can still
route uniformly in situ; a model that looks degenerate here cannot be selective anywhere.
Treat a bad number as conclusive and a good number as necessary-but-not-sufficient.

Usage:
  python experiments/boltzmann-moe/scripts/audit_expert_collapse.py            # all arms
  python experiments/boltzmann-moe/scripts/audit_expert_collapse.py --only pure_hop
"""
from __future__ import annotations

import argparse, glob, json, math, os, sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict_from_keys

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
from lm_engine.arguments import TrainingArgs          # noqa: E402
from lm_engine.utils import load_yaml                  # noqa: E402

CFG_DIRS = ["iclr_flops", "iclr_moebase", "iclr_1blk", "iclr_big", "iclr_ctrl",
            "iclr_gptmoe", "iclr_slope"]


def arms() -> list[tuple[str, Path, dict]]:
    out = []
    for d in CFG_DIRS:
        for f in sorted(glob.glob(str(REPO / "configs" / d / "*.yml"))):
            a = TrainingArgs(**load_yaml(f))
            blk = a.model_args.pretrained_config["mlp_blocks"][-1]
            if "MoE" not in blk.get("mlp_type", ""):
                continue
            sp = Path(a.save_args.save_path)
            if not (sp / "latest_checkpointed_iteration.json").exists():
                continue
            step = json.load(open(sp / "latest_checkpointed_iteration.json"))["latest_checkpointed_iteration"]
            out.append((Path(f).stem, sp / f"global_step{step}" / "model", blk | {"_step": step}))
    return out


def read_expert_weight(model_dir: Path) -> dict[str, torch.Tensor]:
    md = FileSystemReader(str(model_dir)).read_metadata()
    want = [k for k in md.state_dict_metadata
            if ("expert_holder" in k or ".moe." in k or "W1.weight" in k or "W2.weight" in k
                or "c_fc" in k) and k.endswith("weight")]
    if not want:
        return {}
    sd = _load_state_dict_from_keys(keys=set(want), storage_reader=FileSystemReader(str(model_dir)))
    # unwrap the "state.model." prefix layer used by this trainer
    flat = {}
    def walk(d, pre=""):
        for k, v in d.items():
            if isinstance(v, dict): walk(v, f"{pre}{k}.")
            elif torch.is_tensor(v): flat[f"{pre}{k}"] = v
    walk(sd)
    return flat


def weight_diversity(W: torch.Tensor, K: int) -> dict:
    """W: (K*I_e, d) fused. Returns per-expert norms and pairwise cosine stats."""
    KI, d = W.shape
    Ie = KI // K
    E = W.float().reshape(K, Ie * d)
    norms = E.norm(dim=1)
    En = F.normalize(E, dim=1)
    C = En @ En.T
    off = C[~torch.eye(K, dtype=torch.bool)]
    return dict(K=K, Ie=Ie,
                norm_min=norms.min().item(), norm_max=norms.max().item(),
                norm_ratio=(norms.min() / norms.max().clamp_min(1e-9)).item(),
                cos_mean=off.mean().item(), cos_max=off.max().item(),
                dead=int((norms < 0.1 * norms.median()).sum().item()))


def routing_selectivity(W: torch.Tensor, K: int, temperature: float,
                        routing_norm: str = "none", n: int = 4096, seed: int = 0) -> dict:
    """Push Gaussian h through the Hopfield energy and measure the routing spread."""
    KI, d = W.shape
    Ie = KI // K
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(n, d, generator=g)
    h = h / h.norm(dim=-1, keepdim=True) * math.sqrt(d)      # typical RMS-normed scale
    Wk = W.float().reshape(K, Ie, d)
    # E_k = (1/I_e) * ||gelu(W_k h)||^2   (Hopfield form)
    E = torch.stack([(F.gelu(h @ Wk[k].T) ** 2).mean(-1) for k in range(K)], dim=-1)
    # MUST replicate BoltzmannMoEFFEnergy._logits exactly. Every current arm sets
    # routing_norm: zscore, which z-scores the logits ACROSS experts -- so the softmax
    # sharpness depends only on the SHAPE of the per-token energy spread, not on its
    # magnitude. Applying a raw -E/tau instead (as this function first did) reports a
    # wildly different and much more concentrated distribution than the model actually
    # uses; it made iclr_pure_hop_isoP look like max_load 0.99 single-expert collapse.
    s = -E                                                   # e_sign="neg"
    if routing_norm == "sqrt_width":
        s = s * (Ie ** 0.5)
    elif routing_norm == "zscore":
        s = (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp_min(1e-12)
    p = F.softmax(s / max(temperature, 1e-6), dim=-1)
    H = -(p * (p + 1e-9).log()).sum(-1)
    eff = H.mean().exp().item()
    dom = p.argmax(-1).bincount(minlength=K).float()
    return dict(eff_n=eff, eff_frac=eff / K,
                n_dominant=int((dom > 0).sum().item()),
                max_load=(dom / n).max().item(),
                routing_norm=routing_norm,
                E_spread=(E.std(-1).mean() / E.abs().mean().clamp_min(1e-9)).item())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default=None)
    ap.add_argument("--json_out", default=None)
    args = ap.parse_args()

    rows = []
    for name, mdir, blk in arms():
        if args.only and args.only not in name:
            continue
        if not mdir.exists():
            continue
        try:
            flat = read_expert_weight(mdir)
        except Exception as e:
            print(f"  {name:30s} READ FAILED {type(e).__name__}: {str(e)[:60]}")
            continue
        cand = [(k, v) for k, v in flat.items() if v.ndim == 2 and v.shape[0] > v.shape[1]]
        if not cand:
            print(f"  {name:30s} no fused expert tensor found (keys: {list(flat)[:2]})")
            continue
        k, W = max(cand, key=lambda kv: kv[1].numel())
        K = blk.get("n_experts") or blk.get("num_experts")
        r = dict(arm=name, step=blk["_step"], key=k.split(".")[-3],
                 kind=blk.get("expert_kind", blk["mlp_type"]))
        r |= weight_diversity(W, K)
        if blk.get("expert_kind") == "hopfield":
            r |= routing_selectivity(W, K, float(blk.get("temperature", 1.0)),
                                     str(blk.get("routing_norm", "none")))
        rows.append(r)

    print(f"\n{'arm':30s} {'step':>6s} {'K':>3s} {'cos_mean':>8s} {'cos_max':>7s} "
          f"{'nrm_ratio':>9s} {'dead':>4s} {'eff_n':>7s} {'eff/K':>6s} {'max_load':>8s}")
    print("-" * 108)
    for r in sorted(rows, key=lambda x: x["arm"]):
        print(f"{r['arm'][:30]:30s} {r['step']:6d} {r['K']:3d} {r['cos_mean']:8.4f} "
              f"{r['cos_max']:7.4f} {r['norm_ratio']:9.3f} {r['dead']:4d} "
              f"{r.get('eff_n', float('nan')):7.3f} {r.get('eff_frac', float('nan')):6.3f} "
              f"{r.get('max_load', float('nan')):8.3f}")
    print("""
READING THIS TABLE
  cos_mean / cos_max  pairwise cosine between flattened experts. -> 1.0 means the experts
                      have become copies (WEIGHT collapse). Random high-dim blocks sit
                      near 0, so anything above ~0.3 is worth investigating.
  nrm_ratio           min/max expert norm. A vanishing ratio means some expert is decaying
                      out of the mixture.
  dead                experts whose norm is under 10% of the median.
  eff_n / eff/K       effective number of experts from the routing distribution on
                      Gaussian probes (exp of mean entropy). eff/K -> 1.0 is UNIFORM
                      routing (the degenerate regime), -> 1/K is single-expert collapse.
                      Healthy is comfortably between.
  max_load            largest share of probes dominated by one expert. -> 1.0 is collapse.
Hopfield arms only for the routing columns; the w1w2 and learned arms report weight
diversity only, since their energy needs both matrices.""")
    if args.json_out:
        Path(args.json_out).write_text(json.dumps(rows, indent=1))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main()
