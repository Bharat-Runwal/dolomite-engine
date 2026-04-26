"""analyze_v52v53_cosim_20260425e.py

Computes same-input delta consecutive cosim metrics for V52 and V53 cosreg models,
prints a summary table of all cosreg models (V1/V39/V52/V53), and regenerates
cosreg_performance_comparison.pdf with V52 and V53 included.

Metrics computed (same-input successive-pair averages):
  cos(Δ)     = full-block delta cosim
  cos(raw_a) = raw attn output cosim
  cos(p(a))  = proj_attn(a_i) cosim
  cos(raw_f) = raw ffn output cosim
  cos(p(f))  = proj_mlp(f_i) cosim

Output:
  prints JSON summary to stdout
  results/multi-block-ablation/plots/cosreg_performance_comparison.pdf  (updated)
"""

import sys, json, warnings
warnings.filterwarnings("ignore")
import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

REPO = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO)
import lm_engine.hf_models  # noqa: F401

from transformers import AutoModelForCausalLM, AutoTokenizer

BASE  = Path(f"{REPO}/experiments/energy-inference/results/multi-block-ablation")
PLOTS = BASE / "plots"
PLOTS.mkdir(exist_ok=True)

TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"
EVAL_TEXT = """
The transformer architecture has become the dominant paradigm in natural language processing.
Self-attention mechanisms allow models to relate different positions in a sequence to compute
representations. The key insight is that language understanding requires tracking long-range
dependencies between words and phrases. Energy-based models offer an alternative perspective,
where the attention computation corresponds to gradient descent on an energy function. This
connection between energy minimization and attention is not merely an analogy but a precise
mathematical relationship: the output of energy attention equals the gradient of the energy
with respect to the hidden state. Recurrent computation in transformer blocks may thus
implement an iterative optimization process, converging toward a low-energy state that captures
the semantic content of the input. Layer similarity then measures whether different blocks are
performing redundant optimization steps or distinct transformations.
""".strip()
N_TOKENS = 256


def flat_cosim(a, b):
    return F.cosine_similarity(a.reshape(1, -1), b.reshape(1, -1)).item()


def succ_cosim(lst):
    result = []
    for i in range(len(lst) - 1):
        a, b = lst[i], lst[i + 1]
        if a is not None and b is not None:
            result.append(flat_cosim(a, b))
        else:
            result.append(float("nan"))
    return result


def nanmean(arr):
    vals = [v for v in arr if not np.isnan(v)]
    return float(np.mean(vals)) if vals else float("nan")


def load_model(model_path):
    print(f"  Loading {model_path.name} ...", flush=True)
    model = AutoModelForCausalLM.from_pretrained(str(model_path), trust_remote_code=True)
    model.eval()
    return model


def get_input(model_path):
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    ids = tok.encode(EVAL_TEXT)[:N_TOKENS]
    input_ids = torch.tensor([ids])
    with torch.no_grad():
        emb = model_placeholder(input_ids)
    return emb, input_ids


def compute_rope(model):
    """Return the RoPE module (or None)."""
    blocks = model.transformer.h if hasattr(model, "transformer") else model.model.transformer.h
    blk = blocks[0]
    if hasattr(blk, "attn") and hasattr(blk.attn, "rotary_emb"):
        return blk.attn.rotary_emb
    return None


def get_embedding(model, input_ids):
    """Get token embeddings (before transformer blocks)."""
    with torch.no_grad():
        if hasattr(model, "transformer"):
            emb = model.transformer.wte(input_ids)
        else:
            emb = model.model.transformer.wte(input_ids)
    return emb


def same_delta_consec_metrics(model, h_base):
    """
    Run each block on the same fixed input h_base independently (no residual accumulation).
    Compute successive cosim for:
      full delta, raw attn, proj_attn output, raw ffn, proj_mlp output
    """
    if hasattr(model, "transformer"):
        blocks = model.transformer.h
    else:
        blocks = model.model.transformer.h

    T = h_base.shape[1]

    full_deltas, attn_raws, proj_attn_outs, ffn_raws, proj_ffn_outs = [], [], [], [], []

    with torch.no_grad():
        for blk in blocks:
            # Collect submodule outputs via hooks
            attn_buf, proj_attn_buf, ffn_buf, proj_ffn_buf = [], [], [], []

            def _hook(buf):
                def fn(m, inp, out):
                    buf.append(out[0] if isinstance(out, tuple) else out)
                return fn

            hooks = []
            # raw attn (before proj_attn) — capture output of the attention kernel
            if hasattr(blk, "attn"):
                attn_mod = blk.attn
                # try inner sub-modules
                if hasattr(attn_mod, "attn"):
                    hooks.append(attn_mod.attn.register_forward_hook(_hook(attn_buf)))
                else:
                    hooks.append(attn_mod.register_forward_hook(_hook(attn_buf)))
            # proj_attn
            if hasattr(blk, "proj_attn"):
                hooks.append(blk.proj_attn.register_forward_hook(_hook(proj_attn_buf)))
            elif hasattr(blk, "attn") and hasattr(blk.attn, "proj"):
                hooks.append(blk.attn.proj.register_forward_hook(_hook(proj_attn_buf)))
            # raw ffn
            if hasattr(blk, "ffwd"):
                hooks.append(blk.ffwd.register_forward_hook(_hook(ffn_buf)))
            elif hasattr(blk, "mlp"):
                hooks.append(blk.mlp.register_forward_hook(_hook(ffn_buf)))
            # proj_mlp
            if hasattr(blk, "proj_mlp"):
                hooks.append(blk.proj_mlp.register_forward_hook(_hook(proj_ffn_buf)))

            try:
                out = blk(h_base)
                h_out = out[0] if isinstance(out, tuple) else out
            finally:
                for h in hooks:
                    h.remove()

            delta = (h_out - h_base).detach().reshape(-1)
            full_deltas.append(delta)

            a_raw = attn_buf[0].detach().reshape(-1) if attn_buf else None
            attn_raws.append(a_raw)

            pa = proj_attn_buf[0].detach().reshape(-1) if proj_attn_buf else None
            proj_attn_outs.append(pa)

            f_raw = ffn_buf[0].detach().reshape(-1) if ffn_buf else None
            ffn_raws.append(f_raw)

            pf = proj_ffn_buf[0].detach().reshape(-1) if proj_ffn_buf else None
            proj_ffn_outs.append(pf)

    return {
        "cos_delta":   nanmean(succ_cosim(full_deltas)),
        "cos_raw_attn": nanmean(succ_cosim(attn_raws)),
        "cos_proj_attn": nanmean(succ_cosim(proj_attn_outs)),
        "cos_raw_ffn": nanmean(succ_cosim(ffn_raws)),
        "cos_proj_ffn": nanmean(succ_cosim(proj_ffn_outs)),
    }


MODELS_TO_ANALYZE = {
    "V52": BASE / "v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3" / "unsharded",
    "V53": BASE / "v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3" / "unsharded",
}

# Known metrics for V1 and V39 (from paper Table, consecutive avg)
KNOWN = {
    "V1":  {"cos_delta": 0.536, "cos_raw_attn": 0.198, "cos_proj_attn": 0.034, "cos_raw_ffn": 0.001, "cos_proj_ffn": 0.496},
    "V39": {"cos_delta": 0.529, "cos_raw_attn": 0.212, "cos_proj_attn": 0.021, "cos_raw_ffn": 0.000, "cos_proj_ffn": 0.490},
}

results = {}
for name, path in MODELS_TO_ANALYZE.items():
    print(f"\nAnalyzing {name} ({path}) ...", flush=True)
    model = load_model(path)
    model.eval()

    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH, trust_remote_code=True)
    ids = tok.encode(EVAL_TEXT)[:N_TOKENS]
    input_ids = torch.tensor([ids])

    with torch.no_grad():
        h_base = get_embedding(model, input_ids)

    metrics = same_delta_consec_metrics(model, h_base)
    results[name] = metrics
    print(f"  {name}: {json.dumps({k: round(v, 4) for k, v in metrics.items()})}", flush=True)
    del model
    torch.cuda.empty_cache()

# Save metrics
out_json = PLOTS / "v52v53_cosim_metrics.json"
with open(out_json, "w") as f:
    json.dump(results, f, indent=2)
print(f"\nSaved metrics to {out_json}", flush=True)

# Print summary table
print("\n=== COSREG ALIGNMENT SUMMARY ===")
print(f"{'Model':<12} {'cos(Δ)':>8} {'cos(raw_a)':>10} {'cos(p(a))':>10} {'cos(raw_f)':>10} {'cos(p(f))':>10}")
print("-" * 60)
for name, m in [*KNOWN.items(), *results.items()]:
    print(f"{name:<12} {m['cos_delta']:>8.3f} {m['cos_raw_attn']:>10.3f} {m['cos_proj_attn']:>10.3f} {m['cos_raw_ffn']:>10.3f} {m['cos_proj_ffn']:>10.3f}")


# ── Regenerate cosreg_performance_comparison.pdf with V52/V53 ─────────────────

def load_harness(path_glob_pattern):
    """Try to load harness results from a glob pattern."""
    p = Path(path_glob_pattern).parent
    stem = Path(path_glob_pattern).name
    # stem may contain wildcard
    matches = sorted(p.glob(stem))
    if not matches:
        return {}
    with open(matches[-1]) as f:
        d = json.load(f)
    res = d.get("results", {})
    out = {}
    for k in ["arc_challenge", "arc_easy", "boolq", "copa", "hellaswag", "mmlu", "winogrande", "piqa"]:
        if k in res:
            v = res[k]
            out[k] = v.get("acc,none") or v.get("acc_norm,none") or v.get("exact_match,none") or 0.0
    return out


BENCH_TASKS = ["arc_challenge", "arc_easy", "boolq", "hellaswag", "mmlu", "piqa", "winogrande"]

def avg_bench(m):
    vals = [m[k] for k in BENCH_TASKS if k in m]
    return sum(vals) / len(vals) if vals else None

def savefig(fig, stem):
    for ext in ["png", "pdf"]:
        fig.savefig(PLOTS / f"{stem}.{ext}", dpi=150, bbox_inches="tight")
    print(f"  Saved {stem}")
    plt.close(fig)


models_176m = [
    {
        "label": "V0\nGPT",
        "loss": 3.121,
        "color": "#4C72B0", "hatch": "",
        "metrics": load_harness(str(BASE / "v0_gpt_baseline_d768/unsharded/harness_results_*.json")),
    },
    {
        "label": "V1\nEGPT",
        "loss": 3.271,
        "color": "#DD8452", "hatch": "",
        "metrics": load_harness(str(BASE / "v1_12x1_d768_lr2e3/unsharded/harness_results_*.json")),
    },
    {
        "label": "V39\nact-cosreg\n(λ=0.01)",
        "loss": 3.273,
        "color": "#AA3377", "hatch": "///",
        "metrics": load_harness(str(BASE / "v39_egpt_cosreg_12x1_d768_lr2e3/unsharded/harness_results_*.json")),
    },
    {
        "label": "V52\nact-cosreg\n(λ=0.1)",
        "loss": 3.2733,
        "color": "#CC5500", "hatch": "///",
        "metrics": load_harness(str(BASE / "v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3/unsharded/harness_results_*.json")),
        "incomplete": False,
    },
    {
        "label": "V53\nact-cosreg\n(ramp→1.0)",
        "loss": 3.2789,
        "color": "#8800CC", "hatch": "///",
        "metrics": load_harness(str(BASE / "v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3/unsharded/harness_results_*.json")),
        "incomplete": False,
    },
    {
        "label": "V48\nwt-cosreg\n(λ=0.01)",
        "loss": 3.287,
        "color": "#DD8452", "hatch": "xxx",
        "metrics": {},
        "incomplete": True,
    },
]

models_400m = [
    {
        "label": "V9\nGPT",
        "loss": 2.931,
        "color": "#4C72B0", "hatch": "",
        "metrics": load_harness(str(BASE / "v9_gpt_baseline_d1024_lr1e3/unsharded/harness_results_*.json")),
    },
    {
        "label": "V1\nEGPT",
        "loss": 3.117,
        "color": "#DD8452", "hatch": "",
        "metrics": load_harness(str(BASE / "v1_400m_d1024_lr7e4/unsharded/harness_results_*.json")),
    },
    {
        "label": "V40\nact-cosreg\n(λ=0.01)*",
        "loss": 3.225,
        "color": "#AA3377", "hatch": "///",
        "metrics": {},
        "incomplete": True,
    },
    {
        "label": "V50\nwt-cosreg\n(λ=0.01)*",
        "loss": 3.259,
        "color": "#DD8452", "hatch": "xxx",
        "metrics": {},
        "incomplete": True,
    },
]

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("Cosine Similarity Regulariser: Performance Impact", fontsize=12, fontweight="bold")

# Panel 1: loss 176M
ax = axes[0]
ax.set_title("Training Loss — 176M", fontsize=10)
x = np.arange(len(models_176m))
bars = ax.bar(x, [m["loss"] for m in models_176m],
              color=[m["color"] for m in models_176m],
              hatch=[m["hatch"] for m in models_176m],
              edgecolor="white", width=0.6)
for bar, m in zip(bars, models_176m):
    if m.get("incomplete"):
        bar.set_alpha(0.55)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{m['loss']:.3f}", ha="center", va="bottom", fontsize=7,
            alpha=0.6 if m.get("incomplete") else 1.0)
ax.set_xticks(x)
ax.set_xticklabels([m["label"] for m in models_176m], fontsize=7)
ax.set_ylabel("Train loss (lower = better)")
ax.set_ylim(3.05, 3.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel 2: loss 400M
ax = axes[1]
ax.set_title("Training Loss — 400M  (* = incomplete)", fontsize=10)
x = np.arange(len(models_400m))
bars = ax.bar(x, [m["loss"] for m in models_400m],
              color=[m["color"] for m in models_400m],
              hatch=[m["hatch"] for m in models_400m],
              edgecolor="white", width=0.55)
for bar, m in zip(bars, models_400m):
    if m.get("incomplete"):
        bar.set_alpha(0.55)
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.003,
            f"{m['loss']:.3f}", ha="center", va="bottom", fontsize=8,
            alpha=0.6 if m.get("incomplete") else 1.0)
ax.set_xticks(x)
ax.set_xticklabels([m["label"] for m in models_400m], fontsize=8)
ax.set_ylabel("Train loss (lower = better)")
ax.set_ylim(2.85, 3.35)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel 3: benchmark accuracy
ax = axes[2]
ax.set_title("Avg Zero-Shot Accuracy (7 tasks)", fontsize=10)
bench_models = [m for m in (models_176m + models_400m) if avg_bench(m.get("metrics", {})) is not None]
accs = [avg_bench(m["metrics"]) for m in bench_models]
scale_labels = []
for m in bench_models:
    in_400m = any(m is bm for bm in models_400m)
    scale_labels.append(m["label"].replace("\n", " ") + (" 400M" if in_400m else " 176M"))
x = np.arange(len(bench_models))
bars = ax.bar(x, accs,
              color=[m["color"] for m in bench_models],
              hatch=[m["hatch"] for m in bench_models],
              edgecolor="white", width=0.6)
for bar, acc in zip(bars, accs):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
            f"{acc:.3f}", ha="center", va="bottom", fontsize=8)
ax.set_xticks(x)
ax.set_xticklabels(scale_labels, fontsize=7, rotation=25, ha="right")
ax.set_ylabel("Avg accuracy (higher = better)")
ax.set_ylim(0.42, 0.56)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Legend
legend_items = [
    mpatches.Patch(color="#4C72B0", label="GPT baseline"),
    mpatches.Patch(color="#DD8452", label="EGPT"),
    mpatches.Patch(color="#AA3377", hatch="///", label="act-cosreg λ=0.01"),
    mpatches.Patch(color="#CC5500", hatch="///", label="act-cosreg λ=0.1 (V52)"),
    mpatches.Patch(color="#8800CC", hatch="///", label="act-cosreg ramp→1.0 (V53)"),
    mpatches.Patch(color="#DD8452", hatch="xxx", label="wt-cosreg λ=0.01"),
]
fig.legend(handles=legend_items, loc="lower center", ncol=3, fontsize=8,
           frameon=False, bbox_to_anchor=(0.5, -0.04))

plt.tight_layout(rect=[0, 0.08, 1, 1])
savefig(fig, "cosreg_performance_comparison")
print("Done.")
