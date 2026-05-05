"""
Per-layer alpha sweep on the 410m hybrid checkpoint.

Tests whether different energy layers benefit from different sym/anti mixing ratios.
Energy layers: h.8 (3 iters), h.9 (4 iters), h.10 (6 iters), h.11 (9 iters).

All sweeps use norm-preserving interpolation (||W_eff||_F = ||W||_F always):
    W_eff_i = ||W_i||_F * (α_i·Ŵ_sym + (1-α_i)·Ŵ_anti) / sqrt(α_i²+(1-α_i)²)

Profiles tested:
  uniform-0.51     — NP optimum baseline from previous sweep
  explore-then-converge — early layers curl, late layers descent
  converge-then-explore — early layers descent, late layers curl
  all-curl-biased  — slightly anti-sym, best generation quality from fine sweep
  perturb-h8-anti  — only layer h.8 pushed to more anti-sym
  perturb-h11-anti — only layer h.11 (deepest recurrence) pushed to more anti-sym
  perturb-h11-sym  — only layer h.11 pushed to more sym

Usage:
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
    uv pip install accelerate -q
    cd /proj/dmfexp/nima/Code/dolomite-engine
    python experiments/energy-inference/scripts/perlayer-alpha/perlayer_alpha_20260412.py
"""

import argparse
import json
import math
from pathlib import Path

import torch


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=(
        "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k"
    ))
    p.add_argument("--prompt", type=str, default=(
        "Scientists recently discovered that the universe is expanding faster than"
    ))
    p.add_argument("--eval_text", type=str, default=None)
    p.add_argument("--max_new_tokens", type=int, default=80)
    p.add_argument("--out", type=str, default=None)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


# ── Sweep profiles ────────────────────────────────────────────────────────────

# Energy layers in the 410m hybrid checkpoint
ENERGY_LAYERS = ["transformer.h.8", "transformer.h.9", "transformer.h.10", "transformer.h.11"]

# Each profile maps layer_name -> alpha.
# alpha=0.5 → equal mix (trained behaviour at NP scale ≈ W)
# alpha<0.5  → more anti-sym (rotation/curl)
# alpha>0.5  → more sym (gradient descent)
PROFILES = {
    "uniform-0.51": {
        "transformer.h.8":  0.51,
        "transformer.h.9":  0.51,
        "transformer.h.10": 0.51,
        "transformer.h.11": 0.51,
    },
    "explore-then-converge": {
        "transformer.h.8":  0.40,   # early: more curl
        "transformer.h.9":  0.45,
        "transformer.h.10": 0.55,
        "transformer.h.11": 0.60,   # late:  more descent
    },
    "converge-then-explore": {
        "transformer.h.8":  0.60,   # early: more descent
        "transformer.h.9":  0.55,
        "transformer.h.10": 0.45,
        "transformer.h.11": 0.40,   # late:  more curl
    },
    "all-curl-biased": {
        "transformer.h.8":  0.47,
        "transformer.h.9":  0.47,
        "transformer.h.10": 0.47,
        "transformer.h.11": 0.47,
    },
    "perturb-h8-anti": {
        "transformer.h.8":  0.40,   # only h.8 perturbed
        "transformer.h.9":  0.51,
        "transformer.h.10": 0.51,
        "transformer.h.11": 0.51,
    },
    "perturb-h11-anti": {
        "transformer.h.8":  0.51,
        "transformer.h.9":  0.51,
        "transformer.h.10": 0.51,
        "transformer.h.11": 0.40,   # only h.11 perturbed toward curl
    },
    "perturb-h11-sym": {
        "transformer.h.8":  0.51,
        "transformer.h.9":  0.51,
        "transformer.h.10": 0.51,
        "transformer.h.11": 0.60,   # only h.11 perturbed toward descent
    },
}


# ── Per-layer alpha patching ──────────────────────────────────────────────────

class PerLayerAlphaPatch:
    """Context manager: applies a different norm-preserving alpha to each energy layer.

    profile: dict mapping module path (e.g. "transformer.h.8") -> alpha float.
    Layers not in the profile are left unchanged.
    """

    def __init__(self, model, profile: dict):
        self.model = model
        self.profile = profile
        self._originals = {}

    @staticmethod
    def _make_W_eff(W: torch.Tensor, alpha: float) -> torch.Tensor:
        W_sym  = (W + W.T) / 2
        W_anti = (W - W.T) / 2
        a = alpha
        W_norm    = W.norm()
        sym_norm  = W_sym.norm()
        anti_norm = W_anti.norm()
        W_sym_hat  = W_sym  / (sym_norm  + 1e-12)
        W_anti_hat = W_anti / (anti_norm + 1e-12)
        scale = W_norm / (a**2 + (1 - a)**2)**0.5
        return scale * (a * W_sym_hat + (1 - a) * W_anti_hat)

    def __enter__(self):
        for name, module in self.model.named_modules():
            if name not in self.profile:
                continue
            if not (hasattr(module, 'proj') and module.proj is not None
                    and hasattr(module.proj, 'weight')
                    and module.proj.weight.ndim == 2
                    and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
                continue
            alpha = self.profile[name]
            W = module.proj.weight.data
            self._originals[name] = W.clone()
            module.proj.weight.data = self._make_W_eff(W, alpha)
        return self

    def __exit__(self, *_):
        for name, module in self.model.named_modules():
            if name in self._originals:
                module.proj.weight.data = self._originals[name]
        self._originals.clear()


# ── Helpers ───────────────────────────────────────────────────────────────────

@torch.no_grad()
def perplexity(model, tokenizer, text: str, device: str, max_len: int = 512) -> float:
    enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len)
    input_ids = enc["input_ids"].to(device)
    if input_ids.shape[1] < 2:
        return float("nan")
    labels = input_ids.clone()
    out = model(input_ids, labels=labels)
    return math.exp(out.loss.item())


@torch.no_grad()
def generate(model, tokenizer, prompt: str, device: str, max_new_tokens: int) -> str:
    enc = tokenizer(prompt, return_tensors="pt")
    input_ids = enc["input_ids"].to(device)
    out = model.generate(input_ids, max_new_tokens=max_new_tokens, do_sample=False)
    new_tokens = out[0][input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()

    from transformers import AutoTokenizer, AutoModelForCausalLM
    import lm_engine.hf_models  # noqa: F401 — registers energy model type

    print(f"\nLoading checkpoint: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()
    print(f"Model loaded. Params: {sum(p.numel() for p in model.parameters()):,}")

    eval_text = args.eval_text or (
        "The capital of France is Paris. "
        "Albert Einstein developed the theory of relativity. "
        "Water molecules consist of two hydrogen atoms and one oxygen atom. "
        "The mitochondria is the powerhouse of the cell. "
        "In 1969, humans first landed on the Moon during the Apollo 11 mission."
    )

    print("\n" + "=" * 60)
    print("PER-LAYER ALPHA SWEEP (norm-preserving)")
    print("=" * 60)

    sweep_results = []
    for profile_name, profile in PROFILES.items():
        with PerLayerAlphaPatch(model, profile):
            ppl = perplexity(model, tokenizer, eval_text, args.device)
            gen = generate(model, tokenizer, args.prompt, args.device, args.max_new_tokens)

        result = dict(profile=profile_name, alphas=profile, perplexity=ppl, generation=gen)
        sweep_results.append(result)

        alpha_str = "  ".join(f"h.{n.split('.')[-1]}={a:.2f}"
                              for n, a in sorted(profile.items()))
        print(f"\n{profile_name}")
        print(f"  [{alpha_str}]  PPL={ppl:.4f}")
        print(f"  {gen[:120]!r}{'...' if len(gen) > 120 else ''}")

    # Summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Profile':28s}  {'h.8':5s}  {'h.9':5s}  {'h.10':5s}  {'h.11':5s}  {'PPL':>10s}")
    for r in sweep_results:
        p = r["alphas"]
        print(f"  {r['profile']:26s}  "
              f"{p['transformer.h.8']:.2f}   {p['transformer.h.9']:.2f}   "
              f"{p['transformer.h.10']:.2f}   {p['transformer.h.11']:.2f}   "
              f"{r['perplexity']:10.4f}")

    # Save
    ckpt_name = Path(args.checkpoint).name
    out_path = args.out or (
        Path(__file__).parent.parent.parent / "results" / "perlayer-alpha" /
        f"perlayer_alpha_{ckpt_name}_20260412.json"
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    output = dict(
        checkpoint=args.checkpoint,
        prompt=args.prompt,
        norm_preserving=True,
        profiles=PROFILES,
        sweep=sweep_results,
    )
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {out_path}")


if __name__ == "__main__":
    main()
