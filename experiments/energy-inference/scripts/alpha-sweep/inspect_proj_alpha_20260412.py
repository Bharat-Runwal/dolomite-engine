"""
Inspect proj sym/anti norms and run an alpha sweep on the 410m hybrid checkpoint.

Alpha controls the sym/anti-symmetric mixing of proj.weight at inference time:
    W_eff = alpha * W_sym + (1 - alpha) * W_anti
    W_sym  = (W + W^T) / 2   (gradient / descent direction)
    W_anti = (W - W^T) / 2   (curl / rotation direction)

alpha = 1.0 -> pure symmetric (steepest descent on energy)
alpha = 0.0 -> pure anti-symmetric (rotation along energy contours)
alpha = 0.5 -> equal mix (current trained behaviour is close to this)

Usage (from repo root, on a GPU node):
    python experiments/energy-inference/scripts/alpha-sweep/inspect_proj_alpha_20260412.py \
        --checkpoint /proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k \
        --alphas 0.0 0.25 0.5 0.75 1.0 \
        --prompt "The key insight of the energy transformer is" \
        --max_new_tokens 50

Submit via bsub:
    bsub -Is -n 1 -G grp_ebm -gpu "num=1/task:mode=exclusive_process" /bin/bash
    tmux new -s work   # always start tmux first!
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
    uv pip install accelerate  # one-time, if not already installed
    cd /proj/dmfexp/nima/Code/dolomite-engine
    python experiments/energy-inference/scripts/alpha-sweep/inspect_proj_alpha_20260412.py
"""

import argparse
import json
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=(
        "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k"
    ))
    p.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 0.75, 1.0])
    p.add_argument("--prompt", type=str, default=(
        "Scientists recently discovered that the universe is expanding faster than"
    ))
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--eval_text", type=str, default=None,
                   help="If provided, measure perplexity on this text instead of generating.")
    p.add_argument("--out", type=str, default=None,
                   help="Path to write JSON results.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--norm_preserving", action="store_true",
                   help=(
                       "Norm-preserving interpolation: normalise W_sym and W_anti to unit "
                       "Frobenius norm, interpolate, then rescale to ||W||_F. "
                       "Since W_sym ⊥ W_anti in Frobenius, the rescaling factor is "
                       "1/sqrt(alpha^2 + (1-alpha)^2). This decouples direction from scale: "
                       "alpha=0 -> pure anti-sym at trained magnitude, "
                       "alpha=1 -> pure sym at trained magnitude, "
                       "alpha~0.5 -> trained direction (W_sym/||W_sym|| ≈ W_anti/||W_anti||)."
                   ))
    return p.parse_args()


# ── Proj norm analysis ────────────────────────────────────────────────────────

def analyze_proj_norms(model) -> dict:
    """Return sym/anti norm ratios for every energy block's proj.weight."""
    results = {}
    for name, module in model.named_modules():
        # Energy blocks have .proj (nn.Linear, square) and .scale_ff
        if (hasattr(module, 'proj') and module.proj is not None
                and hasattr(module.proj, 'weight')
                and module.proj.weight.ndim == 2
                and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
            W = module.proj.weight.float()
            W_sym  = (W + W.T) / 2
            W_anti = (W - W.T) / 2
            total = W.norm().item()
            sym   = W_sym.norm().item()
            anti  = W_anti.norm().item()
            sf = module.scale_ff.item() if hasattr(module, 'scale_ff') else float('nan')
            results[name] = dict(
                total=total, sym=sym, anti=anti,
                pct_sym=100 * sym / total, pct_anti=100 * anti / total,
                scale_ff=sf,
            )
    return results


def print_proj_table(norms: dict):
    print(f"\n{'Block':45s} {'total':8s} {'sym':8s} {'anti':8s} {'%sym':7s} {'%anti':7s}  scale_ff")
    print("-" * 100)
    for name, v in sorted(norms.items()):
        print(f"  {name:43s} {v['total']:8.3f} {v['sym']:8.3f} {v['anti']:8.3f} "
              f"{v['pct_sym']:6.1f}%  {v['pct_anti']:6.1f}%   {v['scale_ff']:.4f}")


# ── Alpha patching ────────────────────────────────────────────────────────────

class AlphaProjPatch:
    """Context manager that temporarily replaces each energy block's proj with
    an alpha-interpolated version.

    Two modes:

    Standard (norm_preserving=False):
        W_eff = alpha * W_sym + (1-alpha) * W_anti
        alpha=0.5 gives W/2 (half the trained magnitude).
        Tests both direction and scale jointly.

    Norm-preserving (norm_preserving=True):
        W_sym_hat = W_sym / ||W_sym||_F
        W_anti_hat = W_anti / ||W_anti||_F
        W_eff = ||W||_F * (alpha * W_sym_hat + (1-alpha) * W_anti_hat)
                         / sqrt(alpha^2 + (1-alpha)^2)
        Always ||W_eff||_F = ||W||_F. Decouples direction from scale.
        alpha=1 -> pure sym at trained magnitude
        alpha=0 -> pure anti-sym at trained magnitude
        The "trained" direction corresponds to alpha ~= ||W_sym||^2 / ||W||^2 ≈ 0.50
    """

    def __init__(self, model, alpha: float, norm_preserving: bool = False):
        self.model = model
        self.alpha = alpha
        self.norm_preserving = norm_preserving
        self._originals = {}

    def _make_W_eff(self, W: torch.Tensor) -> torch.Tensor:
        W_sym  = (W + W.T) / 2
        W_anti = (W - W.T) / 2
        a = self.alpha
        if not self.norm_preserving:
            return a * W_sym + (1 - a) * W_anti
        # Norm-preserving: normalise each component, interpolate, rescale
        W_norm = W.norm()
        sym_norm  = W_sym.norm()
        anti_norm = W_anti.norm()
        W_sym_hat  = W_sym  / (sym_norm  + 1e-12)
        W_anti_hat = W_anti / (anti_norm + 1e-12)
        # Since W_sym ⊥ W_anti, ||a*W_sym_hat + (1-a)*W_anti_hat||_F = sqrt(a²+(1-a)²)
        scale = W_norm / (a**2 + (1 - a)**2)**0.5
        return scale * (a * W_sym_hat + (1 - a) * W_anti_hat)

    def __enter__(self):
        for name, module in self.model.named_modules():
            if (hasattr(module, 'proj') and module.proj is not None
                    and hasattr(module.proj, 'weight')
                    and module.proj.weight.ndim == 2
                    and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
                W = module.proj.weight.data
                self._originals[name] = W.clone()
                module.proj.weight.data = self._make_W_eff(W)
        return self

    def __exit__(self, *_):
        for name, module in self.model.named_modules():
            if name in self._originals:
                module.proj.weight.data = self._originals[name]
        self._originals.clear()


# ── Perplexity helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str, max_len: int = 512) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    return math.exp(out.loss.item())


# ── Generation helper ─────────────────────────────────────────────────────────

@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    out = model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=1.0,
    )
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    # Import after potentially setting up the env
    from transformers import AutoTokenizer, AutoModelForCausalLM

    # Importing lm_engine.hf_models registers the energy model type with HF AutoClass.
    import lm_engine.hf_models  # noqa: F401

    print(f"\nLoading checkpoint: {args.checkpoint}")
    print(f"Device: {args.device}")

    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    # ── Step 1: proj norm analysis ────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("PROJ SYM / ANTI NORM ANALYSIS")
    print("=" * 60)
    norms = analyze_proj_norms(model)
    print_proj_table(norms)

    # ── Step 2: alpha sweep ───────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print(f"ALPHA SWEEP  (alphas = {args.alphas})")
    print("=" * 60)

    eval_text = args.eval_text or (
        "The capital of France is Paris. "
        "Albert Einstein developed the theory of relativity. "
        "Water molecules consist of two hydrogen atoms and one oxygen atom. "
        "The mitochondria is the powerhouse of the cell. "
        "In 1969, humans first landed on the Moon during the Apollo 11 mission."
    )

    sweep_results = []
    for alpha in args.alphas:
        with AlphaProjPatch(model, alpha, norm_preserving=args.norm_preserving):
            ppl = perplexity(model, tokenizer, eval_text, args.device)
            gen = generate(model, tokenizer, args.prompt, args.device, args.max_new_tokens)

        result = dict(alpha=alpha, perplexity=ppl, generation=gen)
        sweep_results.append(result)
        print(f"\nalpha={alpha:.2f}  PPL={ppl:.4f}")
        print(f"  Prompt:     {args.prompt!r}")
        print(f"  Completion: {gen!r}")

    # ── Step 3: summary ───────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("SWEEP SUMMARY")
    print("=" * 60)
    print(f"{'alpha':8s} {'PPL':10s}")
    for r in sweep_results:
        print(f"  {r['alpha']:.2f}    {r['perplexity']:.4f}")

    # ── Save results ──────────────────────────────────────────────────────────
    mode_tag = "_norm_preserving" if args.norm_preserving else ""
    out_path = args.out or (
        Path(__file__).parent.parent.parent / "results" / "alpha-sweep" /
        f"alpha_sweep{mode_tag}_{Path(args.checkpoint).name}_20260412.json"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = dict(
        checkpoint=args.checkpoint,
        alphas=args.alphas,
        prompt=args.prompt,
        norm_preserving=args.norm_preserving,
        proj_norms=norms,
        sweep=sweep_results,
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
