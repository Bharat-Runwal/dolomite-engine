#!/usr/bin/env python3
"""Fit the SUBSPACE proxy router on a trained checkpoint, and TRAIN it on the right objective.

Two things were wrong with the first attempt (calibrate_proxy_router_20260916.py, quad head):

1. WRONG FUNCTION CLASS. A diagonal quadratic in the r projection coefficients cannot represent
   mean(gelu(.)^2). Measured 0.348 top-2 at r=8 on pure_hop_T12_sink against a k/K = 0.125 chance
   floor. HANDOFF 7.6's 0.90 was for the EXACT energy restricted to a rank-r subspace, which is
   what `proxy_kind="subspace"` now implements.
2. WRONG OBJECTIVE. The fit was closed-form least squares on ENERGY MSE, but what we score -- and
   what routing actually needs -- is top-k SET agreement. LS on MSE spends its capacity on the
   energy magnitude of experts that will never be selected. So after the closed-form init this
   TRAINS the proxy by gradient descent on the routing KL, the same objective `_proxy_step` uses
   online.

Three fitting choices are measured separately so we learn which one carries the result:
  * whitened vs plain SVD basis. The best rank-r approximation of x -> W_k x in the DATA metric
    is the SVD of W_k * Sigma^(1/2), not of W_k. Free to compute, and it is the difference
    between "the directions W_k amplifies" and "the directions W_k amplifies that x actually has".
  * m = proxy_out_dim: E_k is a MEAN over I_e coordinates, so an m-row subsample of B_k is an
    unbiased estimator. Needed for cost -- the full-I_e form does the same elementwise work and
    holds the same (T,K,I_e) activations as the DENSE path.
  * shared vs per-iteration heads, because agreement varied 0.080-0.592 across the 12 iterations.

ALL AGREEMENT NUMBERS ARE ON A HELD-OUT SPLIT of the collected tokens. The refinement fits
thousands of parameters, so an in-sample number would be worth nothing.
"""
import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as F

import lm_engine.hf_models  # noqa: F401


def load_docs(prefix, n_seq, seqlen, tail_frac=0.005):
    from lm_engine.data.megatron.indexed_dataset import MMapIndexedDataset
    ds = MMapIndexedDataset(str(prefix))
    n = len(ds)
    buf, seqs, i = [], [], int(n * (1.0 - tail_frac))
    while len(seqs) < n_seq and i < n:
        buf.extend(ds[i].tolist())
        i += 1
        while len(buf) >= seqlen and len(seqs) < n_seq:
            seqs.append(buf[:seqlen])
            buf = buf[seqlen:]
    return torch.tensor(seqs, dtype=torch.long)


class Collector:
    """Capture (x, E_k) per application of the shared block, tagged by call index."""

    def __init__(self, moe, per_call):
        self.moe, self.per_call = moe, per_call
        self.xs, self.Es, self.it, self.call = [], [], [], 0
        self._orig = moe._route
        moe._route = self._route
        self._h = moe.register_forward_pre_hook(self._pre)
        self._x = None

    def _pre(self, mod, inp):
        self._x = inp[0].detach()

    def _route(self, E_k, proxy_E=None):
        with torch.no_grad():
            xf = self._x.reshape(-1, self._x.shape[-1])
            Ef = E_k.detach().reshape(-1, E_k.shape[-1])
            sel = torch.randperm(xf.shape[0], device=xf.device)[: self.per_call]
            self.xs.append(xf[sel].float())
            self.Es.append(Ef[sel].float())
            self.it.append(torch.full((sel.numel(),), self.call, dtype=torch.long,
                                      device=xf.device))
        self.call += 1
        return self._orig(E_k, proxy_E=proxy_E)

    def close(self):
        self.moe._route = self._orig
        self._h.remove()
        return torch.cat(self.xs), torch.cat(self.Es), torch.cat(self.it), self.call


def agreement(E_hat, E, k, sign):
    """Genuine per-row top-k SET overlap |A cap B| / k. HANDOFF 7.9: torch.isin fakes this."""
    s_hat, s = (E_hat, E) if sign == "pos" else (-E_hat, -E)
    a = s.topk(k, dim=-1).indices
    b = s_hat.topk(k, dim=-1).indices
    return ((a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1).float().sum(-1).mean() / k).item()


def recall_at(E_hat, E, k, p, sign):
    """Two numbers for the OVER-SELECT-THEN-RE-RANK design: the proxy nominates p candidates,
    the exact energies of those p (free -- a by-product of their forward projection) re-rank them.

    `frac` = mean fraction of the exact top-k found in the proxy's top-p.
    `all`  = fraction of TOKENS whose entire exact top-k is inside the proxy's top-p, i.e. the
             rate at which re-ranking recovers the dense routing EXACTLY. That second number is
             the one that maps to quality: on those tokens the sparse path is bit-comparable to
             dense, because the weights and the denominator over the selected set are exact.
    """
    s_hat, sx = (E_hat, E) if sign == "pos" else (-E_hat, -E)
    a = sx.topk(k, dim=-1).indices
    b = s_hat.topk(p, dim=-1).indices
    hit = (a.unsqueeze(-1) == b.unsqueeze(-2)).any(-1)          # (N, k) bool
    return (hit.float().sum(-1).mean() / k).item(), hit.all(-1).float().mean().item()


def sqrt_inv_sqrt(Sig, eps=1e-8):
    ev, U = torch.linalg.eigh(Sig)
    ev = ev.clamp_min(eps * ev.max())
    return (U * ev.sqrt()) @ U.T, (U * ev.rsqrt()) @ U.T


def group_basis(x, W, K, I_e, whiten, r_max):
    """The (whitened) SVD of every expert weight, computed ONCE per data group.

    It does NOT depend on r or m -- only on the group's covariance -- so it is cached and sliced.
    Recomputing it inside the (r, m) sweep is what made the first calibration attempt exceed its
    wall limit with nothing to show; 78 sweep points x 16 experts would be 1248 SVDs of a
    4480x768 float64 matrix instead of 16.
    """
    d, N = x.shape[-1], x.shape[0]
    xd = x.double()
    if whiten:
        L, Linv = sqrt_inv_sqrt((xd.T @ xd) / N)
    else:
        L = Linv = torch.eye(d, dtype=torch.float64, device=x.device)
    Vf = torch.empty(K, d, r_max, dtype=torch.float64, device=x.device)
    Bf = torch.empty(K, I_e, r_max, dtype=torch.float64, device=x.device)
    for k in range(K):
        Wk = W[k * I_e:(k + 1) * I_e].double()
        U, S, Rh = torch.linalg.svd(Wk @ L, full_matrices=False)
        Vf[k] = Linv @ Rh[:r_max].T
        Bf[k] = U[:, :r_max] * S[:r_max]
    return Vf, Bf


def init_subspace(x, E, basis, K, I_e, r, m, gen, ridge=1e-8):
    """Slice the cached basis to rank r and m rows, then least-squares the scale/bias."""
    Vf, Bf = basis
    rows = None if m <= 0 or m >= I_e else torch.randperm(I_e, generator=gen).to(x.device)[:m]
    V = Vf[..., :r].contiguous()
    B = (Bf[:, rows, :r] if rows is not None else Bf[:, :, :r]).contiguous()
    xd = x.double()
    raw = predict(xd, V, B, torch.ones(K, dtype=torch.float64, device=x.device),
                  torch.zeros(K, dtype=torch.float64, device=x.device))
    Ed = E.double()
    scale = torch.empty(K, dtype=torch.float64, device=x.device)
    bias = torch.empty(K, dtype=torch.float64, device=x.device)
    for k in range(K):
        A = torch.stack([raw[:, k], torch.ones_like(raw[:, k])], dim=1)
        c = torch.linalg.solve(A.T @ A + ridge * torch.eye(2, dtype=torch.float64,
                                                           device=x.device), A.T @ Ed[:, k])
        scale[k], bias[k] = c[0], c[1]
    return V, B, scale, bias


def logits_of(E, tau, sign, norm):
    s = E if sign == "pos" else -E
    if norm == "zscore":
        s = (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp_min(1e-12)
    return s / tau


def predict(x, V, B, scale, bias, chunk=None):
    """E_hat over all K experts, CHUNKED over tokens.

    The (n, K, m) intermediate is the whole cost: at m = I_e = 4480, K = 16 and 24k validation
    tokens it is 14 GiB in float64, so an unchunked version OOMs on the very configuration the
    sweep is meant to measure.
    """
    K, m = B.shape[0], B.shape[1]
    if chunk is None:
        chunk = max(256, min(x.shape[0], (1 << 23) // max(1, K * m)))
    out = []
    for i in range(0, x.shape[0], chunk):
        xi = x[i:i + chunk]
        a = torch.einsum("nd,kdr->nkr", xi, V)
        g = F.gelu(torch.einsum("nkr,kmr->nkm", a, B))
        out.append((g * g).mean(-1) * scale + bias)
    return torch.cat(out) if len(out) > 1 else out[0]


def refine(x, E, V, B, scale, bias, tau, sign, norm, steps, lr, refine_B, bs=8192,
           mse_coef=0.0):
    """TRAIN the proxy on the routing KL, optionally PLUS a calibration term on the energies.

    WHY BOTH. KL trains the proxy's RANKING, which is all that selection needs -- and selection
    turned out to cost only +0.0165 bits/byte. But the sparse path also uses the proxy's energies
    for two things that need their ABSOLUTE scale:
      * the all-K softmax denominator, when renormalize_topk is False (as trained);
      * the per-token zscore moments.
    Pure KL is invariant to a per-token shift and largely insensitive to scale, so it happily
    leaves the magnitudes wrong. Measured consequence: p-ladder bits/byte 3.43 / 3.28 / 3.26 /
    3.13 / 3.03 at p = 2/3/4/6/8, collapsing to the exact 1.0996 only at p=K, where the proxy
    leaves the denominator entirely. Over-selecting does NOT fix that, because the K-p terms it
    still supplies are miscalibrated at every p.

    `mse_coef` adds (E_hat - E)^2, normalised by var(E) so the coefficient is scale-free.
    """
    bs = max(256, min(bs, (1 << 22) // max(1, B.shape[0] * B.shape[1])))
    V, scale, bias = (t.clone().requires_grad_(True) for t in (V, scale, bias))
    B = B.clone().requires_grad_(refine_B)
    params = [V, scale, bias] + ([B] if refine_B else [])
    opt = torch.optim.Adam(params, lr=lr)
    tgt_all = F.softmax(logits_of(E, tau, sign, norm), dim=-1)
    Evar = E.var().clamp_min(1e-12)
    N = x.shape[0]
    for step in range(steps):
        idx = torch.randint(0, N, (min(bs, N),), device=x.device)
        Eh = predict(x[idx], V, B, scale, bias)
        lh = logits_of(Eh, tau, sign, norm)
        loss = F.kl_div(F.log_softmax(lh, dim=-1), tgt_all[idx], reduction="batchmean")
        if mse_coef > 0:
            loss = loss + mse_coef * ((Eh - E[idx]) ** 2).mean() / Evar
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step % max(1, steps // 4) == 0 or step == steps - 1:
            print(f"        refine step {step:5d}  KL = {loss.item():.5f}", flush=True)
    return (V.detach(), B.detach(), scale.detach(), bias.detach())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--ckpt_name", default="unsharded")
    ap.add_argument("--data_prefix",
                    default="/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0")
    ap.add_argument("--ranks", type=int, nargs="+", default=[8])
    ap.add_argument("--out_dims", type=int, nargs="+", default=[0],
                    help="m rows of B_k to keep; 0 = all I_e")
    ap.add_argument("--no_whiten", action="store_true")
    ap.add_argument("--per_iter", action="store_true", help="also fit one head per iteration")
    ap.add_argument("--refine_steps", type=int, default=0)
    ap.add_argument("--refine_lr", type=float, default=3e-3)
    ap.add_argument("--refine_B", action="store_true")
    ap.add_argument("--mse_coef", type=float, default=0.0,
                    help="weight on the energy-calibration term alongside the routing KL")
    ap.add_argument("--batches", type=int, default=8)
    ap.add_argument("--seqlen", type=int, default=4096)
    ap.add_argument("--per_call", type=int, default=1024)
    ap.add_argument("--val_frac", type=float, default=0.25)
    ap.add_argument("--write", action="store_true",
                    help="after the sweep, refit the FIRST (r, m) point on all collected tokens "
                         "and write a checkpoint with sparse_forward enabled")
    ap.add_argument("--write_per_iter", action="store_true", default=True)
    ap.add_argument("--out_name", default=None)
    ap.add_argument("--recall_at", type=int, nargs="*", default=None,
                    help="also report recall of the exact top-k inside the proxy's top-p")
    a = ap.parse_args()

    src = Path(a.run_dir) / a.ckpt_name
    assert src.is_dir(), f"missing {src}"
    from transformers import AutoModelForCausalLM
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    m_ = AutoModelForCausalLM.from_pretrained(src, trust_remote_code=True,
                                              torch_dtype=torch.bfloat16).to(dev).eval()
    moes = [z for z in m_.modules() if hasattr(z, "n_experts") and hasattr(z, "_fused_W")]
    assert moes, "no fused EnergyFF_BoltzmannMoE block found"
    moe = moes[0]
    I_e = getattr(moe, "_expert_I", None) or moe.experts[0].intermediate_size
    K, k = moe.n_experts, int(moe.top_k or 1)
    print(f"K={K} I_e={I_e} top_k={k} e_sign={moe.e_sign} routing_norm={moe.routing_norm} "
          f"tau={moe.temperature} blocks={len(moes)} dev={dev}", flush=True)
    print(f"chance floor for top-{k} of {K} = {k / K:.4f}", flush=True)

    col = Collector(moe, a.per_call)
    seqs = load_docs(a.data_prefix, a.batches, a.seqlen)
    with torch.no_grad():
        for i in range(a.batches):
            m_(input_ids=seqs[i:i + 1].to(dev))
    x, E, it, n_calls = col.close()
    n_iter = n_calls // a.batches
    cyc = it % n_iter
    print(f"collected {x.shape[0]} tokens; {n_calls} calls / {a.batches} batches "
          f"-> {n_iter} iterations per forward", flush=True)

    W = moe._fused_W().detach().float()
    gen = torch.Generator().manual_seed(1234)
    tau, sign, norm = moe.temperature, moe.e_sign, moe.routing_norm
    # held-out split, taken per cycle position so every group is represented in both halves
    perm = torch.randperm(x.shape[0], generator=gen).to(x.device)
    n_val = int(a.val_frac * x.shape[0])
    vi, ti = perm[:n_val], perm[n_val:]

    bases: dict = {}
    print(f"\n{'r':>3} {'m':>6} {'heads':>9} {'init(val)':>10} {'refined(val)':>13}", flush=True)
    print("-" * 48, flush=True)
    for r in a.ranks:
        for m in a.out_dims:
            groups = [("shared", None)] + ([("per-iter", n_iter)] if a.per_iter else [])
            for gname, ng in groups:
                ag_i, ag_r, wsum = 0.0, 0.0, 0
                rec: dict = {}
                for gi in range(ng or 1):
                    sel_t = ti if ng is None else ti[cyc[ti] == gi]
                    sel_v = vi if ng is None else vi[cyc[vi] == gi]
                    if sel_t.numel() < 64 or sel_v.numel() < 8:
                        continue
                    xt, Et = x[sel_t], E[sel_t]
                    xv, Ev = x[sel_v].double(), E[sel_v].double()
                    key = gi if ng is not None else -1
                    if key not in bases:
                        bases[key] = group_basis(xt, W, K, I_e, not a.no_whiten, max(a.ranks))
                    P = init_subspace(xt, Et, bases[key], K, I_e, r, m, gen)
                    ai = agreement(predict(xv, *P), Ev, k, sign)
                    ar = ai
                    if a.refine_steps > 0:
                        Pr = refine(xt.double(), Et.double(), *P, tau, sign, norm,
                                    a.refine_steps, a.refine_lr, a.refine_B,
                                    mse_coef=a.mse_coef)
                        Ehv = predict(xv, *Pr)
                        ar = agreement(Ehv, Ev, k, sign)
                        for pp in (a.recall_at or []):
                            f, al = recall_at(Ehv, Ev, k, pp, sign)
                            rec.setdefault(pp, [0.0, 0.0, 0])
                            rec[pp][0] += f * sel_v.numel()
                            rec[pp][1] += al * sel_v.numel()
                            rec[pp][2] += sel_v.numel()
                    w = sel_v.numel()
                    ag_i += ai * w
                    ag_r += ar * w
                    wsum += w
                if wsum:
                    print(f"{r:>3} {m if m else I_e:>6} {gname:>9} {ag_i / wsum:>10.4f} "
                          f"{ag_r / wsum:>13.4f}", flush=True)
                    for pp in sorted(rec):
                        f, al, w = rec[pp]
                        print(f"        proxy top-{pp}: recall of exact top-{k} = {f / w:.4f}"
                              f"   tokens with the FULL top-{k} inside = {al / w:.4f}", flush=True)
    if a.write:
        write_ckpt(a, src, m_, moes, x, E, cyc, n_iter, W, K, I_e, gen, tau, sign, norm)


def write_ckpt(a, src, m_, moes, x, E, cyc, n_iter, W, K, I_e, gen, tau, sign, norm):
    """Refit on ALL collected tokens and save a checkpoint that routes on the proxy.

    The sweep's held-out numbers validated the PROCEDURE; the artifact then uses every token,
    which is standard and strictly more data. Per-iteration heads are stacked on a leading axis
    in the order `_proxy_call` cycles them, so the buffer layout matches what eval will index.
    """
    import json
    import shutil

    from safetensors.torch import load_file, save_file

    r, m = a.ranks[0], a.out_dims[0]
    n_head = n_iter if a.write_per_iter else 1
    print(f"\nWRITE: r={r} m={m or I_e} heads={'per-iter x%d' % n_iter if n_head > 1 else 'shared'}",
          flush=True)
    Vs, Bs, Ss, Bi = [], [], [], []
    for gi in range(n_head):
        sel = torch.arange(x.shape[0], device=x.device) if n_head == 1 else (cyc == gi).nonzero(
            as_tuple=True)[0]
        xt, Et = x[sel], E[sel]
        basis = group_basis(xt, W, K, I_e, not a.no_whiten, r)
        P = init_subspace(xt, Et, basis, K, I_e, r, m, gen)
        if a.refine_steps > 0:
            P = refine(xt.double(), Et.double(), *P, tau, sign, norm,
                       a.refine_steps, a.refine_lr, a.refine_B, mse_coef=a.mse_coef)
        Eh = predict(xt.double(), *P)
        ag = agreement(Eh, Et.double(), int(moes[0].top_k or 1), sign)
        r2 = (1 - ((Eh - Et.double()) ** 2).mean() / Et.double().var()).item()
        print(f"  head {gi:2d}: agreement {ag:.4f}  energy R^2 {r2:.4f}  ({sel.numel()} tokens)",
              flush=True)
        for lst, t in zip((Vs, Bs, Ss, Bi), P):
            lst.append(t)
    stack = (lambda L: L[0] if n_head == 1 else torch.stack(L))
    V, B, sc, bi = map(stack, (Vs, Bs, Ss, Bi))

    out = Path(a.run_dir) / (a.out_name or f"unsharded_sparse_r{r}m{m or I_e}")
    if out.exists():
        shutil.rmtree(out)
    shutil.copytree(src, out, ignore=shutil.ignore_patterns("harness_results*"))
    cfgp = out / "config.json"
    cfg = json.loads(cfgp.read_text())
    n = 0
    for blk in (cfg.get("mlp_blocks") or []):
        if isinstance(blk, dict) and "Boltzmann" in str(blk.get("mlp_type", "")):
            blk.update(proxy_rank=r, proxy_kind="subspace", proxy_out_dim=(m or 0),
                       proxy_iters=n_head, sparse_forward=True,
                       # sparse_forward is built on the fused-weight view. Turning fused_experts
                       # on is EXACT (1.227e-15, ACCEL_FINDINGS_20260915); the known 2-node wedge
                       # does not apply to single-process eval.
                       fused_experts=True)
            # renormalize_topk deliberately LEFT AS TRAINED, so the only difference from the
            # dense checkpoint is which experts get selected.
            n += 1
    cfgp.write_text(json.dumps(cfg, indent=2))
    shards = sorted(out.glob("*.safetensors"))
    assert len(shards) == 1, f"expected 1 safetensors shard, found {len(shards)}"
    sd = load_file(str(shards[0]))
    pres = [nm for nm, mod in m_.named_modules() if mod in moes]
    assert len(pres) == len(moes) == 1, f"{len(pres)} prefixes / {len(moes)} blocks (expected 1)"
    for name, t in (("proxy_V", V), ("proxy_B", B), ("proxy_scale", sc), ("proxy_bias", bi)):
        sd[f"{pres[0]}.{name}"] = t.detach().cpu().to(torch.bfloat16).contiguous()
        print(f"    {pres[0]}.{name}  {tuple(t.shape)}", flush=True)
    save_file(sd, str(shards[0]), metadata={"format": "pt"})
    print(f"  patched {n} block(s); wrote {out}", flush=True)
    print(f"\nEVAL:  python experiments/eval_scripts/eval_harness.py --model hf \\\n"
          f"    --model_args pretrained={out},dtype=bfloat16,trust_remote_code=True \\\n"
          f"    --tasks wikitext --device cuda:0 --batch_size 4 --trust_remote_code \\\n"
          f"    --output_path {out}/harness_results.json", flush=True)


if __name__ == "__main__":
    main()
