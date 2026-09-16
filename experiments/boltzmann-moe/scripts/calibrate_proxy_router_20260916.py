#!/usr/bin/env python3
"""Fit the rank-r proxy router on an ALREADY-TRAINED checkpoint, and measure what it costs.

WHY THIS EXISTS. `sparse_forward` skips the forward projection of the K-k experts a token was
not routed to, which is the only way past the 1/2*(1+k/K) floor. But then the router cannot see
what it skipped, so something cheap has to choose -- the rank-r proxy. Verified in
test_sparse_forward_20260916.py: with a PERFECT proxy the sparse path reproduces the dense
output to 4e-16 in every configuration. So the proxy's prediction error IS the entire
approximation, and this script measures it on real weights and real activations.

No retraining. `proxy_V` initialises for free from the SVD of the trained W_k -- which is why
this works at all: HANDOFF 7.3 measured the trained expert weights to be near rank-2, so a
rank-8 subspace keeps almost everything the router uses. The 2r+1 head coefficients per expert
are then a least-squares fit against the exact energies.

WHAT TO READ IN THE OUTPUT. `top-k agree` is the number that matters -- fraction of the exact
top-k set the proxy recovers, computed as a genuine per-row set overlap (HANDOFF 7.9: torch.isin
silently fakes this). The PER-ITERATION breakdown matters just as much on a recurrent stack: one
proxy serves all `layer_iterations` applications of a shared block, and the hidden state
distribution differs between them -- the same structure that made a single shared sinkhorn mu
buffer fail. If agreement falls off with iteration index, one proxy is not enough.

DATA. The training corpus, tail of the megatron index (the run's own validation split), not the
eval sets -- fitting the router on wikitext would be tuning it on the test distribution.
"""
import argparse
import json
import shutil
from pathlib import Path

import torch

import lm_engine.hf_models  # noqa: F401  -- registers model_type "energy" with AutoModel


def load_docs(prefix, n_seq, seqlen, tail_frac=0.005):
    from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset
    ds = MMapIndexedDataset(str(prefix))
    n = len(ds)
    start = int(n * (1.0 - tail_frac))
    buf, seqs = [], []
    i = start
    while len(seqs) < n_seq and i < n:
        buf.extend(ds[i].tolist())
        i += 1
        while len(buf) >= seqlen and len(seqs) < n_seq:
            seqs.append(buf[:seqlen])
            buf = buf[seqlen:]
    return torch.tensor(seqs, dtype=torch.long)


class Collector:
    """Capture (x, E_k) pairs per application of a shared MoE block, tagged by iteration."""

    def __init__(self, moe, per_call):
        self.moe, self.per_call = moe, per_call
        self.xs, self.Es, self.it = [], [], []
        self.call = 0
        self._orig_route = moe._route
        moe._route = self._route
        self._h = moe.register_forward_pre_hook(self._pre)
        self._pending_x = None

    def _pre(self, mod, inp):
        self._pending_x = inp[0].detach()

    def _route(self, E_k, proxy_E=None):
        x = self._pending_x
        with torch.no_grad():
            xf = x.reshape(-1, x.shape[-1])
            Ef = E_k.detach().reshape(-1, E_k.shape[-1])
            n = xf.shape[0]
            sel = torch.randperm(n, device=xf.device)[: self.per_call]
            self.xs.append(xf[sel].float())
            self.Es.append(Ef[sel].float())
            self.it.append(torch.full((sel.numel(),), self.call, dtype=torch.long,
                                      device=xf.device))
        self.call += 1
        return self._orig_route(E_k, proxy_E=proxy_E)

    def close(self):
        self.moe._route = self._orig_route
        self._h.remove()
        return (torch.cat(self.xs), torch.cat(self.Es), torch.cat(self.it))


def svd_basis(W, n_experts, I_e, r_max, device):
    """Top-r_max right singular vectors of each TRAINED expert weight.

    Computed ONCE and sliced per rank. The first version recomputed it inside the rank loop --
    80 float64 SVDs of a 4480x768 matrix on one CPU core, which blew the job's wall limit.
    This is also the whole reason the proxy works: HANDOFF 7.3 measured the trained expert
    weights near rank-2, so a rank-8 subspace keeps almost everything the router uses.
    """
    d = W.shape[-1]
    V = torch.empty(n_experts, d, r_max, dtype=torch.float64, device=device)
    for k in range(n_experts):
        Wk = W[k * I_e:(k + 1) * I_e].to(device=device, dtype=torch.float64)
        V[k] = torch.linalg.svd(Wk, full_matrices=False).Vh[:r_max].T
    return V


def fit_head(x, E, V, r, ridge=1e-6):
    """Least squares for the 2r+1 head coefficients per expert on the rank-r slice of V.

    Returns the parameters AND the fitted energies, so the caller does not recompute them.
    """
    Vr = V[..., :r].contiguous()                            # (K, d, r)
    K = Vr.shape[0]
    xd, Ed = x.double(), E.double()
    a = torch.einsum("nd,kdr->nkr", xd, Vr)                 # (N, K, r)
    ones = torch.ones(a.shape[0], 1, dtype=torch.float64, device=a.device)
    eye = torch.eye(2 * r + 1, dtype=torch.float64, device=a.device)
    quad = torch.empty(K, r, dtype=torch.float64, device=a.device)
    lin = torch.empty(K, r, dtype=torch.float64, device=a.device)
    bias = torch.empty(K, dtype=torch.float64, device=a.device)
    for k in range(K):
        ak = a[:, k]
        Fk = torch.cat([ak * ak, ak, ones], dim=1)
        c = torch.linalg.solve(Fk.T @ Fk + ridge * eye, Fk.T @ Ed[:, k])
        quad[k], lin[k], bias[k] = c[:r], c[r:2 * r], c[-1]
    E_hat = (quad * a * a).sum(-1) + (lin * a).sum(-1) + bias
    return Vr, quad, lin, bias, E_hat


def expert_I(moe):
    """Per-expert width. `_expert_I` is only set when fused_experts=True, and the trained
    checkpoints have it FALSE (the fused path is an exact optimisation added later), so read it
    off an expert instead."""
    return getattr(moe, "_expert_I", None) or moe.experts[0].intermediate_size


def agreement(E_hat, E, k, sign):
    """Genuine per-row top-k SET overlap, |A cap B| / k."""
    s_hat = E_hat if sign == "pos" else -E_hat
    s = E if sign == "pos" else -E
    a = s.topk(k, dim=-1).indices
    b = s_hat.topk(k, dim=-1).indices
    return (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1).mean().item() / k


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--data_prefix",
                    default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--ranks", type=int, nargs="*", default=None,
                    help="also report agreement at these ranks (fit is saved for --rank)")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--per_call", type=int, default=1024, help="tokens sampled per block call")
    ap.add_argument("--ckpt_name", default="unsharded",
                    help="subdir of run_dir holding the HF checkpoint; the mu-recalibrated arms "
                         "use unsharded_mucal2, NOT unsharded")
    ap.add_argument("--out_name", default=None)
    ap.add_argument("--enable_sparse", action="store_true",
                    help="also set sparse_forward: true in the saved config")
    ap.add_argument("--dry_run", action="store_true", help="measure only, write nothing")
    a = ap.parse_args()

    src = Path(a.run_dir) / a.ckpt_name
    assert src.is_dir(), f"missing {src}"

    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True,
                                             torch_dtype=torch.bfloat16).to(dev).eval()
    moes = [x for x in m.modules() if hasattr(x, "n_experts") and hasattr(x, "_fused_W")]
    assert moes, "no fused EnergyFF_BoltzmannMoE block in this checkpoint"
    print(f"{len(moes)} MoE block(s); K={moes[0].n_experts} I_e={expert_I(moes[0])} "
          f"dev={dev} "
          f"top_k={moes[0].top_k} e_sign={moes[0].e_sign} routing_norm={moes[0].routing_norm} "
          f"renormalize_topk={moes[0].renormalize_topk}")

    cols = [Collector(mo, a.per_call) for mo in moes]
    seqs = load_docs(a.data_prefix, a.batches, a.seqlen)
    with torch.no_grad():
        for i in range(a.batches):
            m(input_ids=seqs[i:i + 1].to(dev))
    data = [c.close() for c in cols]

    fits = []
    for bi, (mo, (x, E, it)) in enumerate(zip(moes, data)):
        W = mo._fused_W().detach().float()
        k = int(mo.top_k) if mo.top_k else 1
        print(f"\n--- block {bi}: {x.shape[0]} tokens over {int(it.max()) + 1} iteration(s)",
              flush=True)
        ranks = sorted({a.rank, *(a.ranks or [])})
        Vfull = svd_basis(W, mo.n_experts, expert_I(mo), max(ranks), x.device)
        print(f"    SVD basis ready (r_max={max(ranks)})", flush=True)
        best = None
        for r in ranks:
            V, q, l, b, E_hat = fit_head(x, E, Vfull, r)
            ov = agreement(E_hat, E.double(), k, mo.e_sign)
            r2 = 1 - ((E_hat - E.double()) ** 2).mean() / E.double().var()
            print(f"  r={r:2d}  top-{k} agree = {ov:.4f}   energy R^2 = {r2:.4f}", flush=True)
            if r == a.rank:
                best = (V, q, l, b)
                for i in range(int(it.max()) + 1):
                    msk = it == i
                    if msk.sum() > 0:
                        print(f"        iteration {i:2d}: agree = "
                              f"{agreement(E_hat[msk], E.double()[msk], k, mo.e_sign):.4f}"
                              f"  ({int(msk.sum())} tokens)", flush=True)
        fits.append(best)

    if a.dry_run:
        print("\n--dry_run: nothing written")
        return

    out = Path(a.run_dir) / (a.out_name or f"unsharded_proxy{a.rank}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("harness_results*"))
    cfgp = out / "config.json"
    cfg = json.loads(cfgp.read_text())
    blocks = cfg.get("mlp_blocks") or cfg.get("mlp_block_args") or []
    n = 0
    for blk in blocks:
        if isinstance(blk, dict) and "Boltzmann" in str(blk.get("mlp_type", "")):
            blk["proxy_rank"] = a.rank
            if a.enable_sparse:
                blk["sparse_forward"] = True
                # sparse_forward asserts fused_experts, and the trained arms have it FALSE.
                # Turning it on is an EXACT change (verified 1.227e-15, ACCEL_FINDINGS_20260915) --
                # the fused weight already backs the per-expert slices. The known 16-GPU/2-node
                # wedge does not apply: this checkpoint is for single-process eval.
                blk["fused_experts"] = True
                # removes the proxy-completed softmax denominator, one of the two smaller
                # approximations, at a measured cost of nothing (Avg11 44.54 vs 44.58). Comment
                # this out to measure the checkpoint under its AS-TRAINED masked softmax instead.
                blk["renormalize_topk"] = True
            n += 1
    cfgp.write_text(json.dumps(cfg, indent=2))
    print(f"\npatched {n} block(s) in {cfgp} (proxy_rank={a.rank}, "
          f"sparse_forward={a.enable_sparse})")

    # inject the fitted parameters into the saved weights
    from safetensors.torch import load_file, save_file
    shards = sorted(out.glob("*.safetensors"))
    assert len(shards) == 1, f"expected one safetensors shard, found {len(shards)}"
    sd = load_file(str(shards[0]))
    # map each MoE module to its state_dict prefix
    prefixes = [nm for nm, mod in m.named_modules() if mod in moes]
    assert len(prefixes) == len(moes), f"{len(prefixes)} prefixes for {len(moes)} blocks"
    for pre, (V, q, l, b) in zip(prefixes, fits):
        for name, t in (("proxy_V", V), ("proxy_quad", q), ("proxy_lin", l), ("proxy_bias", b)):
            sd[f"{pre}.{name}"] = t.detach().cpu().to(torch.bfloat16).contiguous()
    save_file(sd, str(shards[0]), metadata={"format": "pt"})
    print(f"wrote proxy parameters for {len(prefixes)} block(s) into {shards[0].name}")
    print(f"\nEVAL IT:  python experiments/eval_scripts/compute_avg11.py {out}")


if __name__ == "__main__":
    main()
