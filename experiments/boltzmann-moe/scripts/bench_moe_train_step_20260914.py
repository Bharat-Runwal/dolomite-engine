"""Training-step (forward+BACKWARD) microbench of the Boltzmann-MoE energy FF, at the
EXACT scale32B_boltz_hop shape, plus the cost of repulsion and of running it 1-in-10.

WHY THIS EXISTS (what the 2026-09-12 benches did NOT cover)
----------------------------------------------------------
bench_moe_throughput_20260912.py measured FORWARD/prefill throughput only. But the run
we are trying to speed up (scale32B_boltz_hop, ~6.75 s/step, ~4.8 days for 32B tokens)
is bottlenecked by TRAINING step time = forward + backward. Backward roughly doubles the
arithmetic AND changes the balance between the paths:

  * dense/topk-masked backward computes a gradient for ALL K experts' weights (the mask
    p*mask zeros the OUTPUT but the graph still backprops through every expert's two
    matmuls), so it is ~2x the dense forward.
  * proxy/two-stage backward only touches the SELECTED experts' weights + a tiny router,
    so the sparse win should be AT LEAST as large in training as in inference.

We also never measured the repulsion loss. The user's hypothesis: repulsion may be
interfering, so run it 1-in-10 steps (optionally scaling its magnitude to compensate).
This script answers two separable questions:
  (1) COMPUTE cost of repulsion  -> marginal ms of the aux loss; amortized at 1/10.
  (2) COMPATIBILITY with sparsity -> output-repulsion needs ALL K expert outputs, which
      the sparse paths never compute. So the natural schedule is: the SAME 1-in-10 step
      that computes the exact Boltzmann energies (to supervise the proxy) ALSO computes
      the full dense output-repulsion; the other 9 steps run proxy-sparse with no
      repulsion. We also measure a WEIGHT-space repulsion that IS sparse-compatible.

PATHS (all timed fwd AND bwd, sharing one fused W [I_tot, d] that requires grad):
  shipped         topk-masked, all K experts, + output-repulsion (4 pairs, abs, coef .1)
                  == what scale32B_boltz_hop runs today.
  shipped_norep   topk-masked, all K, repulsion OFF                (isolates repulsion)
  two_stage       exact energy for all K (the fwd projection W h is irreducible), then a
                  GROUPED second matmul for the top-2 experts only. No repulsion.
  proxy           cheap linear router (d->K) picks top-2, then BOTH matmuls for the
                  selected experts only, grouped. + tiny router bwd. (the target design)

The grouped paths use an index_select / per-expert-loop / index_add_ scatter, all
autograd-safe. That loop is a CONSERVATIVE (slow) stand-in for a fused scattermoe/
grouped-GEMM kernel, so the measured proxy training speedup is a LOWER BOUND on what a
real kernel would give -- the forward bench already showed torch._grouped_mm within ~10%
of the loop, and a fused backward would only widen the gap in our favor.

Usage (needs 1 GPU; W at this shape is ~0.13 GB bf16, trivial):
  python experiments/boltzmann-moe/scripts/bench_moe_train_step_20260914.py
  python .../bench_moe_train_step_20260914.py --tokens 4096 8192 --iters 50
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "experiments/boltzmann-moe/results/router_analysis"

_SIG = (2.0 / math.pi) ** 0.5

# scale32B_boltz_hop block-7 config (see configs/iclr_scale/scale32B_boltz_hop.yml)
PROD = dict(d=1536, I_tot=40960, K=32, top_k=2, tau=0.35,
            rep_coef=0.1, rep_form="abs", n_rep_pairs=4)
# recurrence x grad-accum: block-7 is applied 6x per forward (layer_iterations[-1]=6)
# and there are ga=8 microbatches per optimizer step, so the MoE runs 6*8=48 times per
# step per GPU. Used only to project block-level ms -> per-step contribution.
RECURRENCE = 6
GRAD_ACCUM = 8
MEASURED_STEP_S = 6.75          # scale32B_boltz_hop, 16 GPU, 2026-09-14 (config header)
MEASURED_SWITCH_STEP_S = 1.34   # scale32B_gptswitch, same hardware (true top-2 dispatch)


def gelu_and_grad(x):
    """(phi, phi') for gelu_grad_method='sigmoid' (the production default)."""
    return F.gelu(x), torch.sigmoid(_SIG * x) * 0.5


def repulsion_penalty(cos, form):
    if form == "squared":
        return (cos ** 2).mean()
    if form == "abs":
        return cos.abs().mean()
    if form == "hinge":
        return F.relu(cos).mean()
    if form == "signed":
        return cos.mean()
    raise ValueError(form)


def zscore_logits(E, tau, e_sign="neg"):
    """logits = zscore_k(+-E)/tau  -- matches BoltzmannMoEFFEnergy._logits(routing_norm='zscore')."""
    s = -E if e_sign == "neg" else E
    s = (s - s.mean(-1, keepdim=True)) / s.std(-1, keepdim=True).clamp_min(1e-12)
    return s / tau


# --------------------------------------------------------------------------- #
# expert compute paths (each returns (out, aux_loss))                          #
# --------------------------------------------------------------------------- #


def _output_repulsion(per, K, d, n_pairs, coef, form, all_pairs):
    """Repulsion over random pairs of the K per-expert OUTPUT vectors.

    Mirrors BoltzmannMoEFFEnergy._add_repulsion_loss: cosine over expert 'grads'
    (the per-expert output vectors), NOT the weights. Needs all K outputs.
    """
    eg = per.reshape(-1, K, d)
    egn = F.normalize(eg, dim=-1)
    k = min(n_pairs, len(all_pairs))
    sampled = random.sample(all_pairs, k)
    i_idx = [p[0] for p in sampled]
    j_idx = [p[1] for p in sampled]
    cos = (egn[:, i_idx, :] * egn[:, j_idx, :]).sum(-1)
    return coef * repulsion_penalty(cos, form)


def _weight_repulsion(W, K, I_e, d, n_pairs, coef, form, all_pairs):
    """Repulsion over pairs of flattened expert WEIGHT matrices. Sparse-COMPATIBLE:
    depends only on W, so it can run on any step regardless of which experts fired."""
    Wf = W.view(K, I_e * d)
    Wn = F.normalize(Wf, dim=-1)
    k = min(n_pairs, len(all_pairs))
    sampled = random.sample(all_pairs, k)
    i_idx = [p[0] for p in sampled]
    j_idx = [p[1] for p in sampled]
    cos = (Wn[i_idx] * Wn[j_idx]).sum(-1)
    return coef * repulsion_penalty(cos, form)


def topk_masked(x, W, K, I_e, d, k, tau, pref, rep=None, all_pairs=None):
    """All K experts computed, then top-k MASK (what ships). Optional output-repulsion."""
    Wx = x @ W.t()
    g, gp = gelu_and_grad(Wx)
    E = (g.float() ** 2).view(-1, K, I_e).mean(-1)
    logits = zscore_logits(E, tau)
    p = F.softmax(logits, dim=-1)
    idx = logits.topk(k, dim=-1).indices
    mask = torch.zeros_like(p, dtype=torch.bool).scatter_(-1, idx, True)
    p = (p * mask).to(x.dtype)
    gated = (g * gp).view(-1, K, I_e)
    per = torch.einsum("nki,kih->nkh", gated, W.view(K, I_e, d))
    out = torch.einsum("nk,nkh->nh", p, per) * pref
    aux = torch.zeros((), device=x.device)
    if rep is not None:
        aux = _output_repulsion(per, K, d, rep["n"], rep["coef"], rep["form"], all_pairs)
    return out, aux


def _grouped_selected(x, We, idx, w, K, I_e, d, pref, gated_all=None):
    """Run only the selected experts, grouped by expert. autograd-safe.

    idx [N,k] expert ids, w [N,k] weights. If gated_all is given (exact-energy path
    already computed W h for all K) reuse it; else (proxy path) compute W h for the
    selected (token,expert) pairs only -- the k/K saving on BOTH matmuls.
    """
    N, k = idx.shape
    flat_e = idx.reshape(-1)
    flat_t = torch.arange(N, device=x.device).repeat_interleave(k)
    order = flat_e.argsort()
    flat_e, flat_t = flat_e[order], flat_t[order]
    flat_w = w.reshape(-1)[order].to(x.dtype)
    counts = torch.bincount(flat_e, minlength=K)
    bounds = torch.cat([torch.zeros(1, dtype=torch.long, device=x.device), counts.cumsum(0)])
    b = bounds.tolist()
    out = torch.zeros(N, d, dtype=x.dtype, device=x.device)
    for e in range(K):
        lo, hi = b[e], b[e + 1]
        if hi == lo:
            continue
        rows = flat_t[lo:hi]
        Wk = We[e]                                   # [I_e, d]
        if gated_all is None:
            z = x.index_select(0, rows) @ Wk.t()     # first matmul, selected only
            g, gp = gelu_and_grad(z)
            gated = g * gp
        else:
            gated = gated_all[:, e, :].index_select(0, rows)
        ye = gated @ Wk                              # second matmul
        out = out.index_add(0, rows, ye * flat_w[lo:hi, None])
    return out * pref


def two_stage(x, W, K, I_e, d, k, tau, pref):
    """Exact energy for ALL K (irreducible fwd projection) + grouped top-k 2nd matmul."""
    Wx = x @ W.t()
    g, gp = gelu_and_grad(Wx)
    E = (g.float() ** 2).view(-1, K, I_e).mean(-1)
    logits = zscore_logits(E, tau)
    p = F.softmax(logits, dim=-1)
    top = logits.topk(k, dim=-1)
    w = p.gather(-1, top.indices)
    out = _grouped_selected(x, W.view(K, I_e, d), top.indices, w, K, I_e, d, pref,
                            gated_all=(g * gp).view(-1, K, I_e))
    return out, torch.zeros((), device=x.device)


def proxy(x, W, router, K, I_e, d, k, tau, pref):
    """Cheap linear router (d->K) picks top-k; BOTH matmuls only for selected experts."""
    logits = router(x) / tau                          # [N, K]  (Kd MACs, tiny)
    p = F.softmax(logits, dim=-1)
    top = logits.topk(k, dim=-1)
    w = p.gather(-1, top.indices)
    out = _grouped_selected(x, W.view(K, I_e, d), top.indices, w, K, I_e, d, pref,
                            gated_all=None)
    return out, torch.zeros((), device=x.device)


# --------------------------------------------------------------------------- #
# timing                                                                       #
# --------------------------------------------------------------------------- #


def time_fwd_bwd(fn, params, warmup, iters):
    """Time forward and backward separately (CUDA events). loss = out^2.mean()+aux."""
    fwd, bwd = [], []
    for i in range(warmup + iters):
        for p in params:
            p.grad = None
        e0, e1, e2 = (torch.cuda.Event(enable_timing=True) for _ in range(3))
        torch.cuda.synchronize()
        e0.record()
        out, aux = fn()
        loss = out.float().pow(2).mean() + aux
        e1.record()
        loss.backward()
        e2.record()
        torch.cuda.synchronize()
        if i >= warmup:
            fwd.append(e0.elapsed_time(e1))
            bwd.append(e1.elapsed_time(e2))
    return statistics.median(fwd), statistics.median(bwd)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokens", type=int, nargs="+", default=[4096],
                    help="tokens/call (4096 = one micro_batch_size=1 x seq=4096 microbatch)")
    ap.add_argument("--iters", type=int, default=50)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()

    assert torch.cuda.is_available(), "needs a GPU"
    dev = "cuda"
    props = torch.cuda.get_device_properties(0)
    d, I_tot, K, k = PROD["d"], PROD["I_tot"], PROD["K"], PROD["top_k"]
    I_e = I_tot // K
    tau = PROD["tau"]
    pref = 4.0 * I_e ** -0.5                     # hopfield_grad_scale='sqrt_consistent'
    all_pairs = list(__import__("itertools").combinations(range(K), 2))
    rep = dict(n=PROD["n_rep_pairs"], coef=PROD["rep_coef"], form=PROD["rep_form"])

    print(f"device: {props.name}  {props.total_memory/1e9:.0f} GB")
    print(f"shape (scale32B_boltz_hop block-7): d={d} I_tot={I_tot} K={K} top_k={k} "
          f"I_e={I_e} tau={tau}  pref={pref:.5f}")
    print(f"repulsion: {rep['n']} pairs, form={rep['form']}, coef={rep['coef']}")
    print(f"anchors: boltz_hop {MEASURED_STEP_S}s/step  gptswitch {MEASURED_SWITCH_STEP_S}s/step "
          f"(16 GPU)\n")

    all_results = []
    for N in args.tokens:
        torch.manual_seed(0)
        random.seed(0)
        W = nn.Parameter((torch.randn(I_tot, d, device=dev, dtype=torch.bfloat16) * d ** -0.5))
        router = nn.Linear(d, K, bias=False).to(dev, torch.bfloat16)
        x = torch.randn(N, d, device=dev, dtype=torch.bfloat16)

        variants = {
            "shipped":       (lambda: topk_masked(x, W, K, I_e, d, k, tau, pref, rep=rep,
                                                  all_pairs=all_pairs),           [W]),
            "shipped_norep": (lambda: topk_masked(x, W, K, I_e, d, k, tau, pref, rep=None),
                                                                                  [W]),
            "two_stage":     (lambda: two_stage(x, W, K, I_e, d, k, tau, pref),   [W]),
            "proxy":         (lambda: proxy(x, W, router, K, I_e, d, k, tau, pref),
                                                                    [W] + list(router.parameters())),
        }

        print(f"===== N = {N} tokens =====")
        print(f"  {'path':16s} {'fwd ms':>9s} {'bwd ms':>9s} {'total ms':>9s} "
              f"{'speedup':>8s}   {'note'}")
        base = None
        res = {}
        for name, (fn, params) in variants.items():
            try:
                fms, bms = time_fwd_bwd(fn, params, args.warmup, args.iters)
            except torch.cuda.OutOfMemoryError:
                torch.cuda.empty_cache()
                print(f"  {name:16s}  OOM")
                continue
            tot = fms + bms
            if name == "shipped":
                base = tot
            note = ""
            if name == "shipped":
                note = "== what runs today"
            elif name == "shipped_norep":
                note = "repulsion OFF"
            elif name == "two_stage":
                note = "all-K energy + top-2 back"
            elif name == "proxy":
                note = "linear router + top-2 both matmuls"
            sp = base / tot if base else float("nan")
            print(f"  {name:16s} {fms:9.3f} {bms:9.3f} {tot:9.3f} {sp:7.2f}x   {note}")
            res[name] = dict(fwd_ms=fms, bwd_ms=bms, total_ms=tot, speedup=sp)

        # ---- repulsion cost (compute) + weight-space alternative ----
        rep_marginal = res["shipped"]["total_ms"] - res["shipped_norep"]["total_ms"]
        # amortized if the (already-computed) output-repulsion fires only 1-in-10 steps
        rep_1in10 = res["shipped_norep"]["total_ms"] + rep_marginal / 10.0
        # sparse-compatible weight repulsion: measure its own marginal cost on top of proxy
        def _proxy_wrep():
            out, _ = proxy(x, W, router, K, I_e, d, k, tau, pref)
            aux = _weight_repulsion(W, K, I_e, d, rep["n"], rep["coef"], rep["form"], all_pairs)
            return out, aux
        try:
            fms, bms = time_fwd_bwd(_proxy_wrep, [W] + list(router.parameters()),
                                    args.warmup, args.iters)
            proxy_wrep_tot = fms + bms
            wrep_marginal = proxy_wrep_tot - res["proxy"]["total_ms"]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            proxy_wrep_tot = wrep_marginal = float("nan")

        print(f"\n  repulsion (output, 4 pairs) marginal cost : {rep_marginal:+.3f} ms/call "
              f"({100*rep_marginal/res['shipped']['total_ms']:+.1f}% of shipped)")
        print(f"  shipped with repulsion 1-in-10 (amortized) : {rep_1in10:.3f} ms/call "
              f"(vs {res['shipped']['total_ms']:.3f} every step)")
        print(f"  weight-space repulsion marginal (on proxy) : {wrep_marginal:+.3f} ms/call "
              f"[sparse-COMPATIBLE alternative]")

        # ---- schedule blends & end-to-end projection ----
        proxy_tot = res["proxy"]["total_ms"]
        shipped_tot = res["shipped"]["total_ms"]
        # 1-in-10: 1 exact/dense step (teacher: exact energies for proxy targets + full
        # dense output-repulsion) + 9 proxy steps.
        blend_1in10 = (1 * shipped_tot + 9 * proxy_tot) / 10.0
        blend_1in20 = (1 * shipped_tot + 19 * proxy_tot) / 20.0
        block_speedup_blend = shipped_tot / blend_1in10

        print(f"\n  block-level blended step (1 dense teacher + 9 proxy)/10 : "
              f"{blend_1in10:.3f} ms/call  ->  {block_speedup_blend:.2f}x vs shipped")
        print(f"  block-level blended step (1 dense + 19 proxy)/20         : "
              f"{blend_1in20:.3f} ms/call  ->  {shipped_tot/blend_1in20:.2f}x vs shipped")

        # End-to-end projection. The block runs RECURRENCE*GRAD_ACCUM times per step per
        # GPU; call that fraction f of the boltz step. We don't know f exactly here, but
        # gptswitch (true top-2 dispatch, otherwise identical backbone) sets a hard floor:
        # if we make the MoE do true top-2 dispatch, the boltz step should approach the
        # gptswitch step plus energy-attention overhead. Report the block ratio and the
        # gptswitch floor; Phase B measures f end-to-end.
        print(f"\n  end-to-end context: boltz {MEASURED_STEP_S}s vs gptswitch "
              f"{MEASURED_SWITCH_STEP_S}s/step ({MEASURED_STEP_S/MEASURED_SWITCH_STEP_S:.1f}x). "
              f"gptswitch is the floor a true-sparse boltz approaches.")
        print()

        res["_meta"] = dict(N=N, d=d, I_tot=I_tot, K=K, top_k=k, I_e=I_e, tau=tau,
                            rep_marginal_ms=rep_marginal, rep_1in10_ms=rep_1in10,
                            weight_rep_marginal_ms=wrep_marginal,
                            blend_1in10_ms=blend_1in10, blend_1in20_ms=blend_1in20,
                            block_speedup_blend_1in10=block_speedup_blend)
        all_results.append(res)

        del W, router, x
        torch.cuda.empty_cache()

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "bench_moe_train_step_20260914.json").write_text(json.dumps(all_results, indent=1))
    print(f"wrote {OUT}/bench_moe_train_step_20260914.json")


if __name__ == "__main__":
    main()
