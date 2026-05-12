"""Dataset comparison for L⊥ trajectory: WikiText vs GSM8K math.

Runs the matched EGPT/RecGPT pair on two text datasets and prints mean ε(h)
for each (model × dataset) combination. Fast variant: n_batches=4, k_svd=128.

Question: is the bimodal LM-head alignment explained by data content?
If EGPT and RecGPT show the same ε on both datasets → architecture drives
the difference, not data. If RecGPT ε grows more on math → scratch-space
hypothesis for GSM8K gap is supported.

Usage:
    python analyze_lperp_dataset_compare_20260507.py [--n_batches 4]
"""
from __future__ import annotations

import argparse, json
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import sys

REPO = Path(__file__).resolve().parents[4]
BASE = Path(__file__).resolve().parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = BASE / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

sys.path.insert(0, str(REPO))
import lm_engine.hf_models  # noqa
from transformers import AutoModelForCausalLM, AutoTokenizer

MODELS = {
    "410m_egpt":   BASE / "410m_hybrid_s8e4"   / "unsharded",
    "410m_recgpt": BASE / "410m_recgpt_s8e4"   / "unsharded",
}

WIKI_PASSAGES = [
    "The tower is part of a complex of buildings that includes the Palace of Westminster. "
    "The tower stands 315 feet tall at the north end of the Palace of Westminster.",
    "Scientists discovered that certain proteins fold into specific shapes that determine "
    "their function in the cell. The process of protein folding is guided by molecular chaperones.",
    "The stock market experienced significant volatility as investors weighed the impact of "
    "rising interest rates on technology company valuations and future earnings projections.",
    "Deep learning models have achieved remarkable performance on image recognition tasks "
    "by learning hierarchical features from large amounts of labeled training data.",
    "The Amazon rainforest plays a crucial role in regulating the global climate by absorbing "
    "carbon dioxide and releasing oxygen through the process of photosynthesis.",
    "Quantum computing leverages quantum mechanical phenomena such as superposition and "
    "entanglement to process information in fundamentally different ways than classical computers.",
    "The human brain contains approximately 86 billion neurons connected by trillions of "
    "synapses that enable complex cognitive functions including memory, language, and reasoning.",
    "Medieval European cities were typically built around a central marketplace where merchants "
    "and craftsmen gathered to trade goods and services with local residents.",
]

MATH_PASSAGES = [
    "Janet has 3 dozen eggs. She uses 7 eggs each morning for breakfast. "
    "How many eggs does she have left after 5 days? Answer: 3*12=36 total. 7*5=35 used. 36-35=1.",
    "A store sells apples for $2 each and oranges for $3 each. "
    "If Tom buys 4 apples and 3 oranges, how much does he spend? Answer: 4*2=8 for apples. 3*3=9 for oranges. 8+9=17.",
    "A train travels at 60 miles per hour. How long does it take to travel 240 miles? "
    "Answer: 240 divided by 60 equals 4 hours.",
    "Maria has 5 times as many marbles as John. John has 12 marbles. "
    "How many marbles do they have together? Answer: Maria has 5*12=60 marbles. Together: 60+12=72.",
    "A rectangle has length 8 cm and width 5 cm. What is the perimeter? "
    "Answer: 2*(8+5) = 2*13 = 26 cm.",
    "Sarah earns $15 per hour. She worked 6 hours on Monday and 4 hours on Tuesday. "
    "How much did she earn in total? Answer: (6+4)*15 = 10*15 = $150.",
    "A class has 30 students. 12 are girls. What fraction are boys? "
    "Answer: 30-12=18 boys. 18/30 = 3/5.",
    "If a car uses 8 gallons of fuel per 100 miles, how many gallons are needed for 350 miles? "
    "Answer: 350/100 * 8 = 3.5 * 8 = 28 gallons.",
]


def get_lm_head(model) -> torch.Tensor:
    for attr in ("lm_head", "embed_out", "output"):
        m = getattr(model, attr, None)
        if m is not None and hasattr(m, "weight"):
            return m.weight.detach().float()
    # fallback: tied embedding
    t = getattr(model, "transformer", None) or getattr(model, "model", None)
    if t is not None:
        wte = getattr(t, "wte", None)
        if wte is not None and hasattr(wte, "weight"):
            return wte.weight.detach().float()
    raise RuntimeError("Cannot find LM head")


def get_blocks(model):
    for attr in ("transformer", "model"):
        t = getattr(model, attr, None)
        if t is not None and hasattr(t, "h"):
            return t.h
        if t is not None and hasattr(t, "transformer"):
            return t.transformer.h
    raise AttributeError


def compute_l_perp(h: torch.Tensor, Lk: torch.Tensor) -> float:
    """Mean ε = ||h_⊥|| / ||h|| over token positions."""
    h_f = h.float()
    proj = (h_f @ Lk) @ Lk.T
    perp = h_f - proj
    eps = perp.norm(dim=-1) / h_f.norm(dim=-1).clamp(min=1e-8)
    return eps.mean().item()


def gaussian_eps_baseline(d: int, k: int) -> float:
    """Analytical ε for random Gaussian h: E[ε] = sqrt((d-k)/d)."""
    return float(((d - k) / d) ** 0.5)


def random_lk_eps(h_all: list[torch.Tensor], d: int, k: int,
                  device: str, n_reps: int = 5) -> float:
    """ε of real h against a random orthonormal Lk (equivalent to shuffled-W_U
    but O(d*k) instead of O(vocab*d^2) SVD).  Average over n_reps random bases."""
    eps_list = []
    for _ in range(n_reps):
        G = torch.randn(d, k, device=device)
        Lk_rand, _ = torch.linalg.qr(G)  # [d, k], orthonormal
        for h in h_all:
            eps_list.append(compute_l_perp(h.to(device), Lk_rand))
    return float(np.mean(eps_list)) if eps_list else float("nan")


def run_one_model(model_name: str, model_path: Path, texts: list[str],
                  Lk: torch.Tensor, device: str,
                  collect_h: list | None = None) -> dict:
    """Returns per-(block, iter) mean ε averaged over all texts.
    If collect_h is a list, appends recurrent-block h tensors to it (for shuffle baseline)."""
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
    ).to(device).eval()
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), trust_remote_code=True)

    blocks = get_blocks(model)
    layer_iters = model.config.layer_iterations
    block_types = [
        "energy" if (hasattr(b, "ln") and hasattr(b, "attn") and hasattr(b, "ffwd")) else "gpt"
        for b in blocks
    ]

    # Accumulate ε[block][iter] = list of floats
    eps_acc: dict = defaultdict(lambda: defaultdict(list))

    for text in texts:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=256).to(device)
        handles = []
        block_calls: dict = defaultdict(int)  # block_idx → call count

        def make_hook(blk_idx, is_recurrent: bool):
            def hook(mod, inp):  # pre-hook: (module, input) only
                h = (inp[0] if isinstance(inp, tuple) else inp).detach().squeeze(0)  # [T, d]
                call_n = block_calls[blk_idx]
                block_calls[blk_idx] += 1
                eps_acc[blk_idx][call_n].append(compute_l_perp(h, Lk))
                if collect_h is not None and is_recurrent and call_n == 0:
                    collect_h.append(h.cpu())
            return hook

        is_recurrent = [layer_iters[i] > 1 for i in range(len(blocks))]
        for i, blk in enumerate(blocks):
            # Hook at LN input to capture h before normalization
            # For EGPT blocks: blk.ln is called once per iteration
            # For GPT blocks: blk.ln_1 is called once per block
            ln = getattr(blk, "ln", None) or getattr(blk, "ln_1", None)
            if ln is not None:
                handles.append(ln.register_forward_pre_hook(make_hook(i, is_recurrent[i])))

        with torch.no_grad():
            model(enc["input_ids"])

        for h in handles:
            h.remove()

    del model
    torch.cuda.empty_cache()

    # Build trajectory: only recurrent (multi-iter) blocks
    trajectory = []
    for i, blk_type in enumerate(block_types):
        n_iters = layer_iters[i]
        if n_iters > 1:
            for j in range(n_iters):
                vals = eps_acc[i][j]
                if vals:
                    trajectory.append({
                        "block": i, "iter": j, "n_iters": n_iters,
                        "mean_eps": float(np.mean(vals)),
                    })

    # Summary: mean ε across all recurrent block-iterations
    all_eps = [t["mean_eps"] for t in trajectory]
    mean_recurrent_eps = float(np.mean(all_eps)) if all_eps else float("nan")

    # First-to-last change within each multi-iter block
    block_delta = {}
    for i in set(t["block"] for t in trajectory):
        blk_traj = sorted([t for t in trajectory if t["block"] == i], key=lambda x: x["iter"])
        if len(blk_traj) >= 2:
            block_delta[i] = blk_traj[-1]["mean_eps"] - blk_traj[0]["mean_eps"]

    return {
        "trajectory": trajectory,
        "mean_recurrent_eps": mean_recurrent_eps,
        "block_delta_eps": block_delta,  # positive = ε increases within block
        "block_types": block_types,
        "layer_iters": layer_iters,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n_batches", type=int, default=4,
                        help="Passages per dataset (max 8)")
    parser.add_argument("--k_svd", type=int, default=512)
    parser.add_argument("--device", type=str,
                        default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    n = min(args.n_batches, 8)
    wiki_texts = WIKI_PASSAGES[:n]
    math_texts = MATH_PASSAGES[:n]
    datasets = {"wiki": wiki_texts, "math": math_texts}

    results = {}  # model → dataset → stats

    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"SKIP {model_name}: {model_path} not found")
            continue
        print(f"\n{'='*50}\n{model_name}\n{'='*50}")

        # Load LM head once, compute Lk via randomized SVD on GPU (fast)
        print("  Loading model for Lk computation...")
        m_tmp = AutoModelForCausalLM.from_pretrained(
            str(model_path), torch_dtype=torch.bfloat16, trust_remote_code=True
        ).cpu().eval()
        W_U = get_lm_head(m_tmp).to(args.device)  # float32 on GPU
        del m_tmp
        # Randomized SVD: O(vocab * k) not O(vocab * d^2). W_U: [vocab, d]
        # svd_lowrank returns (U, S, V) where V: [d, q], q = k + oversampling
        q = min(args.k_svd + 16, W_U.shape[1])
        _, S_top, V_top = torch.svd_lowrank(W_U, q=q, niter=4)
        # V_top: [d, q], columns are right singular vectors sorted by S descending
        Lk = V_top[:, :args.k_svd].contiguous()  # [d, k]
        d = Lk.shape[0]
        energy_frac = float((S_top[:args.k_svd]**2).sum() / (S_top**2).sum())
        gauss_baseline = gaussian_eps_baseline(d, args.k_svd)
        del W_U, S_top, V_top
        print(f"  Lk: k={args.k_svd}, d={d}, energy_frac={energy_frac:.3f}")
        print(f"  Gaussian random h baseline: ε={gauss_baseline:.4f}")

        results[model_name] = {}
        all_recurrent_h: list[torch.Tensor] = []  # collect h for shuffle baseline

        for ds_name, texts in datasets.items():
            print(f"\n  --- Dataset: {ds_name} ({len(texts)} passages) ---")
            stats = run_one_model(model_name, model_path, texts, Lk, args.device,
                                  collect_h=all_recurrent_h if ds_name == "wiki" else None)
            results[model_name][ds_name] = stats
            print(f"  mean ε (recurrent blocks): {stats['mean_recurrent_eps']:.4f}")
            for entry in stats["trajectory"]:
                print(f"    B{entry['block']}(×{entry['n_iters']}) iter{entry['iter']:2d}: "
                      f"ε={entry['mean_eps']:.4f}")

        # Random-Lk baseline: ε of real h against random orthonormal basis
        # (equivalent to shuffled W_U but O(d*k) not O(vocab*d^2))
        if all_recurrent_h:
            print(f"\n  Computing random-Lk baseline (n_reps=5)...")
            rand_lk_eps = random_lk_eps(all_recurrent_h, d, args.k_svd, args.device)
            print(f"  Random-Lk baseline: ε={rand_lk_eps:.4f}")
            results[model_name]["random_lk_baseline"] = rand_lk_eps
        results[model_name]["gaussian_baseline"] = gauss_baseline
        results[model_name]["k_svd"] = args.k_svd
        results[model_name]["d"] = d
        results[model_name]["energy_frac"] = energy_frac

    # ── Summary table ──────────────────────────────────────────────────────
    print("\n" + "="*75)
    print("DATASET COMPARISON: mean ε(h) in recurrent blocks")
    print("(ε=0 → fully in LM-head subspace; ε=1 → fully orthogonal)")
    print("="*75)
    header = f"{'Model':20s}  {'Wiki':>8s}  {'Math':>8s}  {'Δ(math-wiki)':>14s}  {'RandLk':>8s}  {'Gauss':>7s}"
    print(header)
    print("-" * len(header))
    for model_name in results:
        r = results[model_name]
        wiki_eps = r.get("wiki", {}).get("mean_recurrent_eps", float("nan"))
        math_eps = r.get("math", {}).get("mean_recurrent_eps", float("nan"))
        delta = math_eps - wiki_eps
        rand_lk = r.get("random_lk_baseline", float("nan"))
        gauss = r.get("gaussian_baseline", float("nan"))
        print(f"{model_name:20s}  {wiki_eps:8.4f}  {math_eps:8.4f}  {delta:+14.4f}  {rand_lk:8.4f}  {gauss:7.4f}")

    print("\nPer-block delta ε (last_iter - first_iter), math dataset:")
    for model_name in results:
        stats = results[model_name].get("math", {})
        deltas = stats.get("block_delta_eps", {})
        if deltas:
            parts = ", ".join(f"B{k}:{v:+.3f}" for k, v in sorted(deltas.items()))
            print(f"  {model_name}: {parts}")

    print("\nPer-block delta ε, wiki dataset:")
    for model_name in results:
        stats = results[model_name].get("wiki", {})
        deltas = stats.get("block_delta_eps", {})
        if deltas:
            parts = ", ".join(f"B{k}:{v:+.3f}" for k, v in sorted(deltas.items()))
            print(f"  {model_name}: {parts}")

    # Save
    def to_json(obj):
        if isinstance(obj, (np.floating, float)):  return float(obj)
        if isinstance(obj, (np.integer, int)):      return int(obj)
        if isinstance(obj, dict):  return {str(k): to_json(v) for k, v in obj.items()}
        if isinstance(obj, list):  return [to_json(v) for v in obj]
        return obj

    out = PLOTS_DIR / "lperp_dataset_compare_20260507.json"
    out.write_text(json.dumps(to_json(results), indent=2))
    print(f"\nSaved: {out}")


if __name__ == "__main__":
    main()
