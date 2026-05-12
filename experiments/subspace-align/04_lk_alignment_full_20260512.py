"""Comprehensive LM-head alignment table (post-DL version).

New vs 03_lk_alignment.py:
  1. Computes BOTH Pi@J (full write op) AND J_alone (W_Q^T W_K scoring kernel)
     so we can ask: does the attention scoring subspace live in L or L^perp?
  2. Reports RAW alignment values alongside excess over sqrt(k/d)
  3. Uses friendly model display names and short config strings
  4. Adds random-input epsilon baseline (real model, random token IDs)

Outputs:
  plots/lk_alignment_full_20260512.json
  (print a LaTeX-ready table to stdout)

Usage (cluster job only, never login node):
    python 04_lk_alignment_full_20260512.py [--models all]
"""
from __future__ import annotations
import argparse, json, sys, math
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np

REPO = Path(__file__).resolve().parents[2]
BASE = REPO / "experiments/energy-inference/results/multi-block-ablation"
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

# ── Model registry ──────────────────────────────────────────────────────────
# (path, is_egpt, display_name, short_config)
MODELS = {
    "410m_egpt":   (BASE/"410m_hybrid_s8e4/unsharded",   True,
                    "EGPT (matched)",         "8GPT+4×[3,4,6,9], d=1280"),
    "410m_recgpt": (BASE/"410m_recgpt_s8e4/unsharded",   False,
                    "RecGPT (matched)",       "8GPT+4×[3,4,6,9], d=1280"),
    "v71_egpt_dual": (BASE/"v71_hybrid_8gpt_4egpt_rmsray_d1280/unsharded", True,
                    "EGPT-Dual+RMSRay",       "8GPT+4×[3,4,6,9], d=1280"),
    "v76_hybrid":  (BASE/"v76_4gpt_1egpt6x_rmsray_d1024_reg128/unsharded", True,
                    "Hybrid+RMSRay+Reg128",   "4GPT+1×6, d=1024, R=128"),
    "u4_rmsray":   (BASE/"u4_2gpt_4egpt3x_rmsray_d1024/unsharded", True,
                    "U4 EGPT+RMSRay",         "2GPT+4×3, d=1024"),
    "v9_gpt":      (BASE/"v9_gpt_baseline_d1024_lr1e3/unsharded", False,
                    "Deep GPT",               "24×1, d=1024"),
}

K_VALUES = [512, 128, 32]


def get_lm_head(model) -> torch.Tensor:
    for attr in ("lm_head", "embed_out", "output"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach().float().cpu()
    t = getattr(model, "transformer", None) or getattr(model, "model", None)
    if t is not None:
        wte = getattr(t, "wte", None)
        if wte is not None and hasattr(wte, "weight"):
            return wte.weight.detach().float().cpu()
    raise RuntimeError("Cannot find LM head")


def get_lm_svd_full(model, device="cuda"):
    """Full right-singular basis via W^T W eigendecomposition on GPU.

    svd_lowrank only gives top-k vectors; bottom-k requires the full basis.
    W^T W is d×d (e.g. 1280×1280) — eigh is fast even for large vocab.
    Returns (S, Vh) where Vh[0] = smallest SV direction, Vh[-1] = largest.
    """
    W = get_lm_head(model).to(device)  # [vocab, d]
    WtW = W.T @ W                       # [d, d]  O(vocab * d^2), fast on GPU
    eigenvals, V = torch.linalg.eigh(WtW)   # eigenvals ascending; V: [d, d]
    S = eigenvals.clamp(min=0).sqrt()        # singular values ascending
    Vh = V.T                                 # row i = right SV for i-th largest eigenval
    # Reverse so Vh[0] = largest SV (matches convention: Vh[-k:] = bottom-k)
    S = S.flip(0); Vh = Vh.flip(0)
    return S.cpu().numpy(), Vh.cpu().numpy()  # Vh[0]=top-1, Vh[-1]=bottom-1


def get_blocks(model):
    for attr in ("transformer", "model"):
        t = getattr(model, attr, None)
        if t is not None and hasattr(t, "h"):
            return t.h
        if t is not None and hasattr(t, "transformer"):
            return t.transformer.h
    raise AttributeError


def align_k(W_flat: np.ndarray, Vh: np.ndarray, k: int, top: bool) -> float:
    """||P_{L_k} W||_F / ||W||_F."""
    d = Vh.shape[1]
    Lk = torch.tensor(Vh[:k].T if top else Vh[-k:].T, dtype=torch.float32)
    W = torch.tensor(W_flat.reshape(d, -1), dtype=torch.float32)
    proj = Lk @ (Lk.T @ W)
    return (proj.norm() / W.norm()).item()


def get_write_ops(blk, d: int):
    """Returns dict of write operators for a block.

    Keys:
      'PiJ'     : Pi @ J  (EGPT write op, or None if standard attn)
      'J_alone' : J = W_Q^T W_K  (scoring kernel, same for all attn types)
      'WOV'     : W_O @ W_V  (RecGPT/GPT write op, or None if EGPT)
    """
    ops = {}

    # --- EGPT / energy attention block ---
    attn_e = getattr(blk, "attn", None)
    if attn_e is not None and hasattr(getattr(attn_e, "c_attn", None), "weight"):
        w = attn_e.c_attn.weight.detach().float().cpu()  # [2d, d] for energy
        if w.shape[0] >= 2 * d:
            J = w[:d].T @ w[d:2*d]  # [d, d]
            ops["J_alone"] = J.reshape(-1).numpy()
            # projection matrix
            Pi = None
            for attr in ("proj_attn", "proj"):
                p = getattr(blk, attr, None)
                if p is not None and hasattr(p, "weight"):
                    Pi = p.weight.detach().float().cpu()
                    break
            if Pi is not None:
                ops["PiJ"] = (Pi @ J).reshape(-1).numpy()
        return ops

    # --- GPT / RecGPT block ---
    seq = getattr(blk, "sequence_mixer", None)
    if seq is not None and hasattr(getattr(seq, "c_attn", None), "weight"):
        w = seq.c_attn.weight.detach().float().cpu()
        if w.shape[0] >= 2 * d:
            J = w[:d].T @ w[d:2*d]
            ops["J_alone"] = J.reshape(-1).numpy()
        if w.shape[0] >= 3 * d:
            c_proj = getattr(seq, "c_proj", None)
            if c_proj is not None and hasattr(c_proj, "weight"):
                WO = c_proj.weight.detach().float().cpu()
                WV = w[2*d:]
                ops["WOV"] = (WO @ WV).reshape(-1).numpy()
    return ops


def rand_eps_baseline(h_list: list, Lk: torch.Tensor) -> float:
    """ε of real h against random orthonormal Lk (5 repetitions)."""
    d = Lk.shape[0]; k = Lk.shape[1]; dev = Lk.device
    eps_vals = []
    for _ in range(5):
        G = torch.randn(d, k, device=dev)
        Lr, _ = torch.linalg.qr(G)
        for h in h_list:
            h = h.to(dev).float()
            proj = h @ Lr
            h_perp_norm = (h - proj @ Lr.T).norm(dim=-1)
            eps_vals.extend((h_perp_norm / h.norm(dim=-1).clamp(min=1e-8)).tolist())
    return float(np.mean(eps_vals))


def compute_eps(h: torch.Tensor, Lk: torch.Tensor) -> float:
    h = h.to(Lk.device).float()
    proj = h @ Lk
    perp = h - proj @ Lk.T
    return (perp.norm(dim=-1) / h.norm(dim=-1).clamp(min=1e-8)).mean().item()


def analyze(model_id: str, path: Path, display: str, config: str,
            is_egpt: bool, device: str = "cuda") -> dict | None:
    print(f"\n{'='*60}\n{display} ({config})\n{'='*60}")
    if not path.exists():
        print(f"  SKIP: {path}")
        return None

    model = AutoModelForCausalLM.from_pretrained(
        str(path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()

    d = model.config.hidden_size
    print(f"  d={d}")

    # LM head SVD — full basis via W^T W eigh (needed for correct bottom-k)
    S, Vh = get_lm_svd_full(model, device=device)
    total_S2 = float((S**2).sum())

    # Energy fraction
    for k in K_VALUES:
        frac = float((S[:k]**2).sum() / total_S2)
        print(f"  top-{k} energy frac: {frac:.3f}  rand=sqrt({k}/{d})={math.sqrt(k/d):.3f}")

    blocks = get_blocks(model)
    iters = getattr(model.config, "layer_iterations", [1]*len(blocks))

    # Determine which blocks to analyze (recurrent ones for EGPT, all for GPT)
    egpt_idxs = [i for i, it in enumerate(iters) if it > 1]
    if not egpt_idxs:
        egpt_idxs = list(range(len(blocks)))

    # Collect write operators
    all_ops: dict[str, list[np.ndarray]] = defaultdict(list)
    for i in egpt_idxs:
        ops = get_write_ops(blocks[i], d)
        for op_name, op_vec in ops.items():
            all_ops[op_name].append(op_vec)

    # Align each operator type
    results_by_op = {}
    for op_name, op_list in all_ops.items():
        if not op_list:
            continue
        op_arr = np.mean([op for op in op_list], axis=0)  # mean over blocks
        row = {"op_name": op_name, "n_blocks": len(op_list)}
        for k in K_VALUES:
            rand = math.sqrt(k / d)
            t = align_k(op_arr, Vh, k, top=True)
            b = align_k(op_arr, Vh, k, top=False)
            row[f"top{k}_raw"] = round(t, 4)
            row[f"bot{k}_raw"] = round(b, 4)
            row[f"top{k}_exc"] = round(t - rand, 4)
            row[f"bot{k}_exc"] = round(b - rand, 4)
            print(f"  {op_name:8s} k={k:4d}: top={t:.3f}({t-rand:+.3f}) bot={b:.3f}({b-rand:+.3f})")
        results_by_op[op_name] = row

    # ── Random-input epsilon baseline ──────────────────────────────────────
    print("  Computing random-input ε baseline...")
    tokenizer = AutoTokenizer.from_pretrained(str(path), trust_remote_code=True)
    vocab_size = model.config.vocab_size

    # Build Lk from top-512 right SVs (Vh[0]=top-1, so [:k_eps] = top-k)
    k_eps = 512
    Lk = torch.tensor(Vh[:k_eps].T, dtype=torch.float32).to(device)  # [d, k]
    gauss_baseline = math.sqrt((d - k_eps) / d)

    h_real_list, h_rand_list = [], []
    real_texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Scientists have discovered a new method for producing clean energy.",
    ]
    for text in real_texts:
        ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
        h_real_list.append(ids)

    # Random token IDs (same length as real texts)
    for ids in h_real_list:
        rand_ids = torch.randint(0, vocab_size, ids.shape, device=device)
        h_rand_list.append(rand_ids)

    # Hook to capture residual stream after embedding + first block
    captured = {"real": [], "rand": []}
    handles = []
    blk0 = blocks[egpt_idxs[0]]
    ln0 = getattr(blk0, "ln", None) or getattr(blk0, "ln_1", None)

    def make_hook(key):
        def hook(mod, inp):
            h = (inp[0] if isinstance(inp, tuple) else inp).detach().squeeze(0).float()
            captured[key].append(h)
        return hook

    handles.append(ln0.register_forward_pre_hook(make_hook("real")))
    with torch.no_grad():
        for ids in h_real_list:
            model(ids)
    for h in handles: h.remove(); handles.clear()

    handles.append(ln0.register_forward_pre_hook(make_hook("rand")))
    with torch.no_grad():
        for ids in h_rand_list:
            model(ids)
    for h in handles: h.remove()

    eps_real = float(np.mean([compute_eps(h, Lk) for h in captured["real"]]))
    eps_rand_input = float(np.mean([compute_eps(h, Lk) for h in captured["rand"]]))
    eps_rand_lk = rand_eps_baseline(captured["real"], Lk)
    print(f"  ε(real text, real Lk):  {eps_real:.4f}")
    print(f"  ε(rand tokens, real Lk):{eps_rand_input:.4f}  (random INPUT)")
    print(f"  ε(real text, rand Lk):  {eps_rand_lk:.4f}  (random SUBSPACE)")
    print(f"  ε(Gaussian, real Lk):   {gauss_baseline:.4f}  (theory)")

    del model
    torch.cuda.empty_cache()

    return {
        "model_id": model_id,
        "display": display,
        "config": config,
        "d": d,
        "is_egpt": is_egpt,
        "n_recurrent_blocks": len(egpt_idxs),
        "ops": results_by_op,
        "eps_real": eps_real,
        "eps_rand_input": eps_rand_input,
        "eps_rand_lk": eps_rand_lk,
        "eps_gauss_theory": gauss_baseline,
    }


def print_latex_table(results: dict):
    """Print LaTeX table rows for the paper."""
    print("\n\n" + "="*80)
    print("LATEX TABLE (raw values, excess in parens)")
    print("="*80)
    op_order = ["PiJ", "WOV", "J_alone"]
    op_labels = {"PiJ": r"$\bm\Pi J$", "WOV": r"$W_O W_V$", "J_alone": r"$J = W_Q^\top W_K$"}

    for mid, r in results.items():
        if r is None: continue
        print(f"\n% {r['display']} ({r['config']})")
        for op_name in op_order:
            if op_name not in r["ops"]: continue
            row = r["ops"][op_name]
            cells = []
            for k in K_VALUES:
                t = row[f'top{k}_raw']; te = row[f'top{k}_exc']
                b = row[f'bot{k}_raw']; be = row[f'bot{k}_exc']
                cells.append(f"{t:.3f}({te:+.3f})")
                cells.append(f"{b:.3f}({be:+.3f})")
            label = op_labels.get(op_name, op_name)
            print(f"  {r['display']:25s} & {label:30s} & " + " & ".join(cells) + r" \\")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="all")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    if args.models == "all":
        model_ids = list(MODELS.keys())
    else:
        model_ids = [m.strip() for m in args.models.split(",")]

    results = {}
    for mid in model_ids:
        if mid not in MODELS:
            print(f"Unknown model: {mid}")
            continue
        path, is_egpt, display, config = MODELS[mid]
        r = analyze(mid, path, display, config, is_egpt, device=args.device)
        results[mid] = r

    print_latex_table(results)

    # Save
    def to_json(obj):
        if isinstance(obj, (np.floating, float)): return float(obj)
        if isinstance(obj, (np.integer, int)): return int(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, dict): return {k: to_json(v) for k,v in obj.items()}
        if isinstance(obj, list): return [to_json(v) for v in obj]
        return obj

    out = PLOTS / "lk_alignment_full_20260512.json"
    out.write_text(json.dumps(to_json(results), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
