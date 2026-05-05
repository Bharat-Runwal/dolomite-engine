"""
Layer-merge PPL analysis for V1 EGPT.

Strategy: average ONLY the attn sub-block weights (and proj_attn) for high-similarity
adjacent layers, keeping the FFN of the first block and dropping the second block entirely.
FFN weights are NOT merged (consecutive cosine sim ~0 → they are doing different things).

Tests merge candidates ranked by peak consecutive attn cosine similarity:
  - Baseline (no merge, 12 blocks)
  - Merge single pairs: (7,8), (8,9), (9,10), (6,7)
  - Merge triples: (7,8,9), (8,9,10) → 10 blocks
  - Merge quad: (6,7,8,9) → 8 blocks

For each: compute word-level WikiText-2 PPL on the test split (~250k tokens).
Runtime: ~5-10s per candidate on CPU with seq_len=512.

Saved: results/multi-block-ablation/plots/layer_merge_ppl.{png,pdf}
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
PPL_TOKENS = 16384  # ~32 chunks of 512; fast enough and stable


# ── model helpers ──────────────────────────────────────────────────────────────

def get_transformer(model):
    if hasattr(model, "transformer"):
        return model.transformer
    if hasattr(model, "model") and hasattr(model.model, "transformer"):
        return model.model.transformer
    raise AttributeError(f"Cannot find transformer on {type(model)}")


def get_blocks(model):
    return get_transformer(model).h


def set_blocks(model, new_blocks: nn.ModuleList, keep_indices: list[int]):
    """Replace block list and patch layer_iterations / sequence_mixer_block_types."""
    t = get_transformer(model)
    t.h = new_blocks
    # Patch per-layer lists so the forward loop stays in sync
    if hasattr(t, "layer_iterations") and isinstance(t.layer_iterations, list):
        t.layer_iterations = [t.layer_iterations[i] for i in keep_indices]
    if hasattr(t, "sequence_mixer_block_types") and isinstance(t.sequence_mixer_block_types, list):
        t.sequence_mixer_block_types = [t.sequence_mixer_block_types[i] for i in keep_indices]


def _get_proj_params(block):
    """Return iterator of (name, param) for attn + projection sub-modules only."""
    modules = [("attn", block.attn)]
    if hasattr(block, "proj") and block.proj is not None:
        modules.append(("proj", block.proj))
    if hasattr(block, "proj_attn") and block.proj_attn is not None:
        modules.append(("proj_attn", block.proj_attn))
    return [(f"{m}.{n}", p) for m, mod in modules for n, p in mod.named_parameters()]


class MergeContext:
    """
    Context manager: merges attn+proj weights of merge_indices in-place,
    hides the dropped blocks from the transformer forward loop,
    then restores everything on exit — no deep copies needed.
    """
    def __init__(self, model, merge_indices):
        self.model = model
        self.merge_indices = merge_indices
        self.t = get_transformer(model)
        self._saved = {}          # {layer_idx: {param_name: original_tensor_clone}}
        self._orig_blocks = None
        self._orig_iter = None
        self._orig_smbt = None

    def __enter__(self):
        blocks = self.t.h
        nl = len(blocks)
        first_idx = self.merge_indices[0]
        first = blocks[first_idx]

        # Save original attn+proj weights for all blocks in the group
        for idx in self.merge_indices:
            self._saved[idx] = {n: p.data.clone() for n, p in _get_proj_params(blocks[idx])}

        # Average weights from blocks[1..] into blocks[0]
        first_params = dict(_get_proj_params(first))
        for idx in self.merge_indices[1:]:
            for n, p in _get_proj_params(blocks[idx]):
                if n in first_params:
                    first_params[n].data.add_(p.data).div_(
                        self.merge_indices.index(idx) + 1
                    )

        # Simpler: straightforward mean across the group
        for n, p in _get_proj_params(first):
            total = p.data.clone()
            for idx in self.merge_indices[1:]:
                other = dict(_get_proj_params(blocks[idx]))[n]
                total = total + other.data
            p.data = total / len(self.merge_indices)

        # Patch block list and per-layer metadata to hide dropped blocks
        drop = set(self.merge_indices[1:])
        keep = [i for i in range(nl) if i not in drop]
        self._orig_blocks = self.t.h
        self._orig_iter  = self.t.layer_iterations if hasattr(self.t, "layer_iterations") else None
        self._orig_smbt  = self.t.sequence_mixer_block_types if hasattr(self.t, "sequence_mixer_block_types") else None

        self.t.h = nn.ModuleList([blocks[i] for i in keep])
        if self._orig_iter is not None:
            self.t.layer_iterations = [self._orig_iter[i] for i in keep]
        if self._orig_smbt is not None:
            self.t.sequence_mixer_block_types = [self._orig_smbt[i] for i in keep]
        print(f"  Merged blocks {self.merge_indices}: {nl} → {len(keep)} blocks")
        return self

    def __exit__(self, *_):
        # Restore original block list
        self.t.h = self._orig_blocks
        if self._orig_iter is not None:
            self.t.layer_iterations = self._orig_iter
        if self._orig_smbt is not None:
            self.t.sequence_mixer_block_types = self._orig_smbt
        # Restore original attn+proj weights
        blocks = self.t.h
        for idx, saved in self._saved.items():
            for n, p in _get_proj_params(blocks[idx]):
                if n in saved:
                    p.data = saved[n]


# ── PPL computation ───────────────────────────────────────────────────────────

def compute_ppl(model, input_ids: torch.Tensor, seq_len: int = SEQ_LEN) -> float:
    """Sliding-window word PPL estimate; input_ids shape (1, T)."""
    model.eval()
    total_nll = 0.0
    total_tokens = 0
    T = input_ids.shape[1]

    with torch.no_grad():
        for start in range(0, min(T - 1, PPL_TOKENS), seq_len):
            end = min(start + seq_len + 1, T)
            chunk = input_ids[:, start:end]
            if chunk.shape[1] < 2:
                break
            logits = model(chunk[:, :-1]).logits  # (1, L, V)
            targets = chunk[:, 1:]                # (1, L)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)),
                targets.reshape(-1),
                reduction="sum"
            )
            total_nll += nll.item()
            total_tokens += targets.numel()

    return float(np.exp(total_nll / total_tokens))


# ── main ───────────────────────────────────────────────────────────────────────

def main():
    print(f"Loading tokenizer and WikiText-2 test split...")
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    tokenizer = AutoTokenizer.from_pretrained(str(CKPT), trust_remote_code=True)

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(ds["text"])
    input_ids = tokenizer(text, return_tensors="pt")["input_ids"]
    print(f"  WikiText-2 test: {input_ids.shape[1]} tokens (using first {PPL_TOKENS})")

    print(f"\nLoading V1 EGPT from {CKPT}...")
    base_model = AutoModelForCausalLM.from_pretrained(
        str(CKPT), dtype=torch.float32, trust_remote_code=True, device_map="cuda"
    )
    base_model.eval()
    n_base = len(get_blocks(base_model))
    print(f"  {n_base} blocks loaded")

    # Merge candidates: list of (label, merge_indices or None)
    # Ranked roughly by peak cosine sim: pairs 8→9 (0.53), 7→8 (0.49), 9→10 (0.49)
    candidates = [
        ("Baseline\n(12 blocks)", None),
        ("Merge (8,9)\n→11 blocks",  [8, 9]),
        ("Merge (7,8)\n→11 blocks",  [7, 8]),
        ("Merge (9,10)\n→11 blocks", [9, 10]),
        ("Merge (6,7)\n→11 blocks",  [6, 7]),
        ("Merge (7,8,9)\n→10 blocks",[7, 8, 9]),
        ("Merge (8,9,10)\n→10 blocks",[8, 9, 10]),
        ("Merge (6,7,8,9)\n→9 blocks",[6, 7, 8, 9]),
    ]

    results = []
    for label, merge in candidates:
        if merge is None:
            ppl = compute_ppl(base_model, input_ids)
            n_blocks = n_base
            print(f"  {label.replace(chr(10), ' '):<35}  PPL = {ppl:.2f}  ({n_blocks} blocks)")
        else:
            with MergeContext(base_model, merge) as ctx:
                n_blocks = len(get_blocks(base_model))
                ppl = compute_ppl(base_model, input_ids)
            print(f"  {label.replace(chr(10), ' '):<35}  PPL = {ppl:.2f}  ({n_blocks} blocks)")
        results.append((label, ppl, n_blocks))

    # ── plot ──────────────────────────────────────────────────────────────────
    labels, ppls, n_blocks_list = zip(*results)
    baseline_ppl = ppls[0]
    x = np.arange(len(labels))
    colors = ["#4878CF" if i == 0 else
              "#2ca02c" if ppls[i] <= baseline_ppl * 1.02 else
              "#D65F5F" if ppls[i] > baseline_ppl * 1.10 else
              "#ff7f0e"
              for i in range(len(ppls))]

    fig, ax = plt.subplots(figsize=(13, 5))
    bars = ax.bar(x, ppls, color=colors, alpha=0.85, edgecolor="white", width=0.6)
    for bar, v, nb in zip(bars, ppls, n_blocks_list):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.15,
                f"{v:.2f}\n({nb}b)", ha="center", va="bottom", fontsize=9,
                fontweight="bold")

    ax.axhline(baseline_ppl, color="#4878CF", linewidth=1.2, linestyle="--", alpha=0.7,
               label=f"Baseline PPL = {baseline_ppl:.2f}")
    ax.axhline(baseline_ppl * 1.02, color="green", linewidth=0.8, linestyle=":",
               alpha=0.6, label="±2% threshold")
    ax.axhline(baseline_ppl * 1.10, color="#D65F5F", linewidth=0.8, linestyle=":",
               alpha=0.6, label="+10% threshold")

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("WikiText-2 PPL (↓ better)", fontsize=11)
    ax.set_ylim(baseline_ppl * 0.97, max(ppls) * 1.12)
    ax.set_title(
        "Layer-merge PPL: average attn weights only (FFN kept, blocks dropped)\n"
        "V1 EGPT lr=2e-3  |  attn merged, FFN of first block kept",
        fontsize=11, fontweight="bold"
    )
    ax.legend(fontsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    for ext in ["png", "pdf"]:
        out = PLOTS_DIR / f"layer_merge_ppl.{ext}"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"\n  Saved {out}")
    plt.close(fig)

    # ── table ─────────────────────────────────────────────────────────────────
    print(f"\n{'Strategy':<38}  {'PPL':>7}  {'ΔPPL':>7}  {'Blocks':>6}")
    print("─" * 62)
    for label, ppl, nb in results:
        delta = ppl - baseline_ppl
        sign = "+" if delta >= 0 else ""
        flag = " ✓" if delta <= baseline_ppl * 0.02 else ""
        print(f"{label.replace(chr(10),' '):<38}  {ppl:>7.2f}  {sign}{delta:>6.2f}  {nb:>6}{flag}")


if __name__ == "__main__":
    main()
