"""
Aligned layer-merge PPL analysis for V1 EGPT — three alignment strategies.

For EnergyAttention_QK (V=K, output uses W_Q):
  The full symmetry group is GL(d_head): W_Q_h → G W_Q_h, W_K_h → G^{-T} W_K_h
  leaves both QK^T scores and the attention output invariant for any invertible G.
  O(d_head) is the subgroup where G^{-T} = G (rotation-only).

Three strategies compared:
  1. Naive: average W_Q and W_K directly (no alignment).
  2. O(d_head) / Procrustes: head-perm + per-head orthogonal R via SVD.
  3. GL(d_head) / GD: head-perm + per-head G found by Adam gradient descent
     minimising ||Wq_a - G Wq_b||² + ||Wk_a - G^{-T} Wk_b||² + ε||G-I||².

Saved: results/multi-block-ablation/plots/layer_merge_aligned_ppl.{png,pdf}
"""

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment
from datasets import load_dataset

REPO = Path(__file__).parents[4]
RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer

CKPT = RESULTS_DIR / "v1_12x1_d768_lr2e3" / "unsharded"

SEQ_LEN = 512
PPL_TOKENS = 16384

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ── model helpers ──────────────────────────────────────────────────────────────

def get_transformer(model):
    if hasattr(model, "transformer"):
        return model.transformer
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer
    raise AttributeError(f"Cannot find transformer on {type(model)}")


def get_blocks(model):
    return get_transformer(model).h


# ── attention weight extraction ────────────────────────────────────────────────

def get_attn_weights(block):
    """
    Return (Wq, Wk) for EnergyAttention_QK.
    c_attn.weight shape: (2*d_model, d_model) — first half Q, second half K.
    V = K (implicit), output projection uses W_Q (no separate W_O).
    """
    attn = block.attn
    W = attn.c_attn.weight  # (2*d_model, d_model)
    d = W.shape[0] // 2
    return W[:d].clone(), W[d:].clone()  # Wq, Wk


def set_attn_weights(block, Wq, Wk):
    """Write back aligned Wq, Wk into c_attn.weight."""
    block.attn.c_attn.weight.data = torch.cat([Wq, Wk], dim=0)


# ── alignment ──────────────────────────────────────────────────────────────────

def align_block_b_to_a(block_a, block_b, num_heads: int, d_head: int):
    """
    Align block_b's attn weights to block_a for EnergyAttention_QK (V=K, no W_O).

    Symmetry: applying orthogonal R left-multiplying both W_Q_h and W_K_h simultaneously
    leaves QK^T AND the output (which uses W_Q as out-proj) invariant.

    Steps:
      1. Head permutation: Hungarian on per-head W_Q cosine similarity.
      2. Per-head Procrustes on W_Q: find R = UV^T from SVD(W_Q_a_h @ W_Q_b_h^T),
         apply R to both W_Q_b_h and W_K_b_h.

    Returns (Wq_b_aligned, Wk_b_aligned) as CPU float32 tensors.
    """
    Wq_a, Wk_a = [w.detach().float().cpu() for w in get_attn_weights(block_a)]
    Wq_b, Wk_b = [w.detach().float().cpu() for w in get_attn_weights(block_b)]

    d_model = Wq_a.shape[1]

    Qa = Wq_a.view(num_heads, d_head, d_model)
    Qb = Wq_b.view(num_heads, d_head, d_model)
    Kb = Wk_b.view(num_heads, d_head, d_model)

    # ── step 1: head permutation on W_Q ────────────────────────────────────
    Qa_n = F.normalize(Qa.reshape(num_heads, -1), dim=1)
    Qb_n = F.normalize(Qb.reshape(num_heads, -1), dim=1)
    sim = (Qa_n @ Qb_n.T).detach().numpy()
    _, col_perm = linear_sum_assignment(-sim)

    Qb = Qb[col_perm]
    Kb = Kb[col_perm]

    # ── step 2: per-head Procrustes on W_Q ─────────────────────────────────
    # R = argmin ||W_Q_a_h - R W_Q_b_h||_F  →  SVD(W_Q_a_h W_Q_b_h^T) = U Σ V^T, R = U V^T
    Qb_al = torch.zeros_like(Qb)
    Kb_al = torch.zeros_like(Kb)

    for h in range(num_heads):
        M = Qa[h] @ Qb[h].T          # (d_head, d_head)
        U, _, Vh = torch.linalg.svd(M)
        R = U @ Vh                    # orthogonal: R ∈ O(d_head)
        Qb_al[h] = R @ Qb[h]
        Kb_al[h] = R @ Kb[h]         # same R applied to K (V=K symmetry)

    return Qb_al.reshape(num_heads * d_head, d_model), Kb_al.reshape(num_heads * d_head, d_model)


def gl_align_block_b_to_a(block_a, block_b, num_heads: int, d_head: int,
                           n_steps: int = 300, lr: float = 1e-2, reg_eps: float = 1e-3):
    """
    GL(d_head) alignment via Adam GD.
    Symmetry: W_Q_h → G W_Q_h, W_K_h → G^{-T} W_K_h for G ∈ GL(d_head).
    Objective (per head): ||Wq_a - G Wq_b||² + ||Wk_a - G^{-T} Wk_b||² + ε||G-I||²

    Returns (Wq_b_aligned, Wk_b_aligned) as CPU float32 tensors.
    """
    Wq_a, Wk_a = [w.detach().float().cpu() for w in get_attn_weights(block_a)]
    Wq_b, Wk_b = [w.detach().float().cpu() for w in get_attn_weights(block_b)]
    d_model = Wq_a.shape[1]
    opt_dev = "cuda" if torch.cuda.is_available() else "cpu"

    Qa = Wq_a.view(num_heads, d_head, d_model)
    Ka = Wk_a.view(num_heads, d_head, d_model)
    Qb_raw = Wq_b.view(num_heads, d_head, d_model)
    Kb_raw = Wk_b.view(num_heads, d_head, d_model)

    # Step 1: head permutation (same as O-aligned)
    Qa_n = F.normalize(Qa.reshape(num_heads, -1), dim=1)
    Qb_n = F.normalize(Qb_raw.reshape(num_heads, -1), dim=1)
    sim = (Qa_n @ Qb_n.T).detach().numpy()
    _, col_perm = linear_sum_assignment(-sim)

    Qb = Qb_raw[col_perm].to(opt_dev)
    Kb = Kb_raw[col_perm].to(opt_dev)
    Qa_d = Qa.to(opt_dev)
    Ka_d = Ka.to(opt_dev)

    # Step 2: per-head GL optimization
    Qb_al = torch.zeros_like(Qb)
    Kb_al = torch.zeros_like(Kb)
    I = torch.eye(d_head, device=opt_dev)

    for h in range(num_heads):
        G = I.clone().requires_grad_(True)
        opt = torch.optim.Adam([G], lr=lr)
        for _ in range(n_steps):
            opt.zero_grad()
            loss_q = (Qa_d[h] - G @ Qb[h]).pow(2).sum()
            Ginv_T = torch.linalg.solve(G.T, I)  # G^{-T}
            loss_k = (Ka_d[h] - Ginv_T @ Kb[h]).pow(2).sum()
            loss_reg = reg_eps * (G - I).pow(2).sum()
            (loss_q + loss_k + loss_reg).backward()
            opt.step()
        with torch.no_grad():
            Qb_al[h] = G @ Qb[h]
            Ginv_T = torch.linalg.solve(G.T, I)
            Kb_al[h] = Ginv_T @ Kb[h]

    return Qb_al.cpu().reshape(num_heads * d_head, d_model), \
           Kb_al.cpu().reshape(num_heads * d_head, d_model)


def _get_proj_params(block):
    modules = [("attn", block.attn)]
    if hasattr(block, "proj") and block.proj is not None:
        modules.append(("proj", block.proj))
    if hasattr(block, "proj_attn") and block.proj_attn is not None:
        modules.append(("proj_attn", block.proj_attn))
    return [(f"{m}.{n}", p) for m, mod in modules for n, p in mod.named_parameters()]


class AlignedMergeContext:
    """
    Like MergeContext but aligns block B to block A before averaging attn weights.
    proj weights are averaged naively (no known symmetry to exploit there).
    """
    def __init__(self, model, merge_indices, num_heads, d_head):
        self.model = model
        self.merge_indices = merge_indices
        self.num_heads = num_heads
        self.d_head = d_head
        self.t = get_transformer(model)
        self._saved = {}
        self._orig_blocks = None
        self._orig_iter = None
        self._orig_smbt = None

    def __enter__(self):
        blocks = self.t.h
        nl = len(blocks)
        first_idx = self.merge_indices[0]
        first = blocks[first_idx]

        # Save originals for all blocks in the group
        for idx in self.merge_indices:
            self._saved[idx] = {n: p.data.clone() for n, p in _get_proj_params(blocks[idx])}

        # Align and average: for each subsequent block, align to the running average,
        # then fold it in.
        # For simplicity: align each block_i+1..n to block_0, then average all.
        Wq_a, Wk_a = get_attn_weights(first)
        Wq_a, Wk_a = Wq_a.detach().float().cpu(), Wk_a.detach().float().cpu()
        avg_Wq = Wq_a.clone()
        avg_Wk = Wk_a.clone()

        for idx in self.merge_indices[1:]:
            Wq_b_al, Wk_b_al = align_block_b_to_a(
                first, blocks[idx], self.num_heads, self.d_head
            )
            avg_Wq = avg_Wq + Wq_b_al
            avg_Wk = avg_Wk + Wk_b_al

        n = len(self.merge_indices)
        dev = first.attn.c_attn.weight.device
        set_attn_weights(first, (avg_Wq / n).to(dev), (avg_Wk / n).to(dev))

        # Average proj weights naively (no known closed-form symmetry)
        first_params = dict(_get_proj_params(first))
        for idx in self.merge_indices[1:]:
            for n_p, p in _get_proj_params(blocks[idx]):
                if n_p in first_params and "attn" not in n_p:
                    first_params[n_p].data.add_(p.data)
        for n_p, p in first_params.items():
            if "attn" not in n_p:
                p.data.div_(n)

        # Patch block list
        drop = set(self.merge_indices[1:])
        keep = [i for i in range(nl) if i not in drop]
        self._orig_blocks = self.t.h
        self._orig_iter  = getattr(self.t, "layer_iterations", None)
        self._orig_smbt  = getattr(self.t, "sequence_mixer_block_types", None)
        self.t.h = nn.ModuleList([blocks[i] for i in keep])
        if self._orig_iter is not None:
            self.t.layer_iterations = [self._orig_iter[i] for i in keep]
        if self._orig_smbt is not None:
            self.t.sequence_mixer_block_types = [self._orig_smbt[i] for i in keep]

        print(f"  Aligned+merged blocks {self.merge_indices}: {nl} → {len(keep)} blocks")
        return self

    def __exit__(self, *_):
        self.t.h = self._orig_blocks
        if self._orig_iter is not None:
            self.t.layer_iterations = self._orig_iter
        if self._orig_smbt is not None:
            self.t.sequence_mixer_block_types = self._orig_smbt
        blocks = self.t.h
        for idx, saved in self._saved.items():
            for n, p in _get_proj_params(blocks[idx]):
                if n in saved:
                    p.data = saved[n]


class GLAlignedMergeContext(AlignedMergeContext):
    """Same as AlignedMergeContext but uses GL(d_head) GD alignment instead of Procrustes."""

    def __enter__(self):
        blocks = self.t.h
        nl = len(blocks)
        first_idx = self.merge_indices[0]
        first = blocks[first_idx]

        for idx in self.merge_indices:
            self._saved[idx] = {n: p.data.clone() for n, p in _get_proj_params(blocks[idx])}

        Wq_a, Wk_a = get_attn_weights(first)
        avg_Wq = Wq_a.detach().float().cpu().clone()
        avg_Wk = Wk_a.detach().float().cpu().clone()

        for idx in self.merge_indices[1:]:
            Wq_b_al, Wk_b_al = gl_align_block_b_to_a(
                first, blocks[idx], self.num_heads, self.d_head
            )
            avg_Wq = avg_Wq + Wq_b_al
            avg_Wk = avg_Wk + Wk_b_al

        n = len(self.merge_indices)
        dev = first.attn.c_attn.weight.device
        set_attn_weights(first, (avg_Wq / n).to(dev), (avg_Wk / n).to(dev))

        first_params = dict(_get_proj_params(first))
        for idx in self.merge_indices[1:]:
            for n_p, p in _get_proj_params(blocks[idx]):
                if n_p in first_params and "attn" not in n_p:
                    first_params[n_p].data.add_(p.data)
        for n_p, p in first_params.items():
            if "attn" not in n_p:
                p.data.div_(n)

        drop = set(self.merge_indices[1:])
        keep = [i for i in range(nl) if i not in drop]
        self._orig_blocks = self.t.h
        self._orig_iter  = getattr(self.t, "layer_iterations", None)
        self._orig_smbt  = getattr(self.t, "sequence_mixer_block_types", None)
        self.t.h = nn.ModuleList([blocks[i] for i in keep])
        if self._orig_iter is not None:
            self.t.layer_iterations = [self._orig_iter[i] for i in keep]
        if self._orig_smbt is not None:
            self.t.sequence_mixer_block_types = [self._orig_smbt[i] for i in keep]

        print(f"  GL-aligned+merged blocks {self.merge_indices}: {nl} → {len(keep)} blocks")
        return self


# ── PPL ────────────────────────────────────────────────────────────────────────

def compute_ppl(model, input_ids: torch.Tensor) -> float:
    model.eval()
    total_nll, total_tokens = 0.0, 0
    ids = input_ids.to(DEVICE)
    T = ids.shape[1]
    with torch.no_grad():
        for start in range(0, min(T - 1, PPL_TOKENS), SEQ_LEN):
            end = min(start + SEQ_LEN + 1, T)
            chunk = ids[:, start:end]
            if chunk.shape[1] < 2:
                break
            logits = model(chunk[:, :-1]).logits
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                chunk[:, 1:].reshape(-1),
                reduction="sum"
            )
            total_nll += nll.item()
            total_tokens += chunk.shape[1] - 1
    return float(np.exp(total_nll / total_tokens))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

    print(f"Device: {DEVICE}")
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=True)
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"  WikiText-2 test: {input_ids.shape[1]} tokens (using first {PPL_TOKENS})")

    print(f"\nLoading V1 EGPT...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(CKPT), dtype=torch.float32, trust_remote_code=True, device_map=DEVICE
    )
    base_model.eval()
    blocks = get_blocks(base_model)
    n_base = len(blocks)

    # Detect num_heads and d_head from config
    cfg = base_model.config
    num_heads = getattr(cfg, "num_attention_heads", getattr(cfg, "n_head", 12))
    d_model   = getattr(cfg, "hidden_size",        getattr(cfg, "n_embd",  768))
    d_head    = d_model // num_heads
    print(f"  {n_base} blocks, {num_heads} heads, d_head={d_head}")

    candidates = [
        ("Baseline\n(12 blocks)", None),
        ("Aligned\n(8,9)→11b",   [8, 9]),
        ("Aligned\n(7,8)→11b",   [7, 8]),
        ("Aligned\n(9,10)→11b",  [9, 10]),
        ("Aligned\n(6,7)→11b",   [6, 7]),
        ("Aligned\n(7,8,9)→10b", [7, 8, 9]),
        ("Aligned\n(8,9,10)→10b",[8, 9, 10]),
        ("Aligned\n(6-9)→9b",    [6, 7, 8, 9]),
    ]

    # Naive (unaligned) PPL from analyze_layer_merge_ppl_20260420.py
    naive_ppls = {
        "Baseline\n(12 blocks)": 57.51,
        "(8,9)→11b":    99.99,
        "(7,8)→11b":    79.25,
        "(9,10)→11b":  135.85,
        "(6,7)→11b":    88.48,
        "(7,8,9)→10b": 189.41,
        "(8,9,10)→10b":237.43,
        "(6-9)→9b":    338.45,
    }

    # ── O(d_head) / Procrustes aligned pass ──────────────────────────────────
    print("\n── O(d_head) Procrustes-aligned merge ──")
    o_results = []
    for label, merge in candidates:
        if merge is None:
            ppl = compute_ppl(base_model, input_ids)
            n_blocks = n_base
        else:
            with AlignedMergeContext(base_model, merge, num_heads, d_head):
                n_blocks = len(get_blocks(base_model))
                ppl = compute_ppl(base_model, input_ids)
        print(f"  {label.replace(chr(10), ' '):<35}  PPL = {ppl:.2f}  ({n_blocks} blocks)", flush=True)
        o_results.append((label, ppl, n_blocks))

    # ── GL(d_head) / GD aligned pass ─────────────────────────────────────────
    print("\n── GL(d_head) GD-aligned merge ──")
    gl_results = []
    for label, merge in candidates:
        if merge is None:
            ppl = compute_ppl(base_model, input_ids)
            n_blocks = n_base
        else:
            with GLAlignedMergeContext(base_model, merge, num_heads, d_head):
                n_blocks = len(get_blocks(base_model))
                ppl = compute_ppl(base_model, input_ids)
        print(f"  {label.replace(chr(10), ' '):<35}  PPL = {ppl:.2f}  ({n_blocks} blocks)", flush=True)
        gl_results.append((label, ppl, n_blocks))

    # ── combined plot: naive / O-aligned / GL-aligned ────────────────────────
    from matplotlib.patches import Patch

    labels_plot = [r[0] for r in o_results]
    ppls_o  = [r[1] for r in o_results]
    ppls_gl = [r[1] for r in gl_results]
    nbl     = [r[2] for r in o_results]
    baseline = ppls_o[0]

    # Naive PPL lookup (keys match "Aligned\n..." → strip prefix)
    naive_vals = []
    for label in labels_plot:
        key = label.replace("Aligned\n", "").replace("Baseline\n", "Baseline\n")
        naive_vals.append(naive_ppls.get(key, None))

    x = np.arange(len(labels_plot))
    w = 0.25

    fig, ax = plt.subplots(figsize=(15, 5))

    # Naive bars (grey)
    for i, nv in enumerate(naive_vals):
        if nv is not None and i > 0:
            ax.bar(x[i] - w, nv, w, color="#aaaaaa", alpha=0.75, edgecolor="white")
            ax.text(x[i] - w, nv + 0.3, f"{nv:.0f}", ha="center",
                    va="bottom", fontsize=6.5, color="#555555")

    # O-aligned bars (orange)
    for i, (v, nb) in enumerate(zip(ppls_o, nbl)):
        if i == 0:
            ax.bar(x[i], v, w, color="#4878CF", alpha=0.85, edgecolor="white")
        else:
            ax.bar(x[i], v, w, color="#ff7f0e", alpha=0.85, edgecolor="white")
            ax.text(x[i], v + 0.3, f"{v:.0f}", ha="center",
                    va="bottom", fontsize=6.5, color="#993300")

    # GL-aligned bars (purple)
    for i, (v, nb) in enumerate(zip(ppls_gl, nbl)):
        if i == 0:
            continue  # skip baseline (same model)
        ax.bar(x[i] + w, v, w, color="#9467bd", alpha=0.85, edgecolor="white")
        ax.text(x[i] + w, v + 0.3, f"{v:.0f}", ha="center",
                va="bottom", fontsize=6.5, color="#4b0082")

    # Baseline value annotation
    ax.text(x[0], ppls_o[0] + 0.3, f"{ppls_o[0]:.1f}", ha="center",
            va="bottom", fontsize=7.5, fontweight="bold")

    ax.axhline(baseline, color="#4878CF", linewidth=1.2, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels_plot, fontsize=8)
    ax.set_ylabel("WikiText-2 PPL (↓ better)", fontsize=11)
    ax.set_ylim(baseline * 0.93, None)
    ax.set_title(
        "Layer-merge PPL: naive vs O(d_head) vs GL(d_head) alignment before averaging\n"
        "V1 EGPT lr=2e-3  |  attn only (FFN kept), blocks dropped",
        fontsize=11, fontweight="bold"
    )
    ax.legend(handles=[
        Patch(color="#aaaaaa", alpha=0.75, label="Naive (no alignment)"),
        Patch(color="#ff7f0e", alpha=0.85, label="O(d_head) Procrustes"),
        Patch(color="#9467bd", alpha=0.85, label="GL(d_head) GD"),
    ], fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()

    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"layer_merge_aligned_ppl.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Saved {out}")
    plt.close(fig)

    # ── table ─────────────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<30}  {'Naive':>7}  {'O-align':>7}  {'GL-align':>8}  {'Blocks':>6}")
    print("─" * 68)
    for (label, ppl_o, nb), (_, ppl_gl, _) in zip(o_results, gl_results):
        lk = label.replace("Aligned\n", "").replace("Baseline\n", "Baseline\n")
        nv = naive_ppls.get(lk, float("nan"))
        print(f"{label.replace(chr(10),' '):<30}  {nv:>7.2f}  {ppl_o:>7.2f}  {ppl_gl:>8.2f}  {nb:>6}")


if __name__ == "__main__":
    main()
