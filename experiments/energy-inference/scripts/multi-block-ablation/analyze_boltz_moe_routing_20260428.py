"""
BoltzmannMoE routing collapse analysis.

Parses bsub stdout log files for B1/B2/B3/B4 and extracts routing metrics
(effective_n_experts, n_dominant_experts, max_expert_load, mean_token_entropy_norm)
per layer per step.  Generates:
  - routing_evolution_{B1,B2,B3,B4}.pdf  — per-layer effective_n_experts vs step
  - routing_collapse_comparison.pdf       — all models, mean across layers
  - training_loss_boltz.pdf               — LM loss comparison

Run on interactive node:
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  python analyze_boltz_moe_routing_20260428.py [--update]

With --update, overwrites existing plots (default: only regenerate if log is newer).
"""

import re
import sys
import argparse
from pathlib import Path
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

REPO = Path(__file__).parents[4]
LOG_DIR = Path.home() / "bsub_logs"
CACHE_DIR = Path("/tmp/boltz_moe_logs")
CACHE_DIR.mkdir(exist_ok=True)
RESULTS_DIR = Path(__file__).parents[2] / "results" / "multi-block-ablation"
PLOTS_DIR = RESULTS_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

N_LAYERS = 12
N_EXPERTS = 16

# Map run name → (bsub job name, stdout glob pattern)
# LSF buffers stdout until job completion; we use bpeek for live data and
# fall back to the stdout file once the job is done.
RUNS = {
    "B1 (baseline)":            ("egpt_b1_boltz_moe_16x1024_d768_lr2e3",         "egpt_b1_boltz_moe_16x1024_d768_lr2e3_*.stderr"),
    "B2 (repulsion 0.01)":      ("egpt_b2_boltz_moe_repulsion_16x1024_d768_lr2e3","egpt_b2_boltz_moe_repulsion_16x1024_d768_lr2e3_*.stderr"),
    "B3 (dropout+WD=0.3)":      ("egpt_b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3","egpt_b3_boltz_moe_dropout_wd_16x1024_d768_lr2e3_*.stderr"),
    "B4 (repulsion 0.1)":       ("egpt_b4_boltz_moe_repulsion_strong",            "egpt_b4_boltz_moe_repulsion_strong_*.stderr"),
    "B5 (rep+drop+WD)":         ("egpt_b5_boltz_moe_rep_strong_dropout_wd",       "egpt_b5_boltz_moe_rep_strong_dropout_wd_*.stderr"),
}

# Colours consistent with other paper plots
COLORS = {
    "B1 (baseline)":        "#e74c3c",
    "B2 (repulsion 0.01)":  "#f39c12",
    "B3 (dropout+WD=0.3)": "#2ecc71",
    "B4 (repulsion 0.1)":   "#3498db",
}


import subprocess

def get_job_id(job_name: str) -> str | None:
    """Return job ID of a running/pending job by name, or None."""
    try:
        out = subprocess.check_output(
            ["bjobs", "-noheader", "-o", "jobid job_name stat"],
            text=True, stderr=subprocess.DEVNULL
        )
        for line in out.strip().splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[1] == job_name and parts[2] in ("RUN", "PEND"):
                return parts[0]
    except Exception:
        pass
    return None


def fetch_log(job_name: str, stdout_pattern: str) -> Path | None:
    """Return path to log data for this run.

    Priority:
     1. Pre-built combined log in CACHE_DIR (covers multi-segment runs)
     2. bpeek (live output from running job)
     3. Latest finished stderr file
    """
    # Check for pre-built combined log
    tag = job_name.replace("egpt_", "").split("_16x")[0].split("_d768")[0]
    # map job_name prefix to combined file tag
    for btag in ["b1", "b2", "b3", "b4", "b5"]:
        combined = CACHE_DIR / f"{btag}_combined.stderr"
        if combined.exists() and btag in job_name:
            return combined

    # Try bpeek (live)
    job_id = get_job_id(job_name)
    if job_id:
        cache = CACHE_DIR / f"{job_name}_{job_id}.txt"
        try:
            out = subprocess.check_output(["bpeek", job_id], text=True,
                                          stderr=subprocess.DEVNULL)
            cache.write_text(out)
            return cache
        except Exception:
            pass

    # Fall back to latest stderr file
    files = sorted(LOG_DIR.glob(stdout_pattern), key=lambda p: p.stat().st_mtime)
    return files[-1] if files else None


def parse_log(path: Path) -> dict:
    """Parse a training stdout log into structured data.

    Returns dict with keys:
      steps       : list[int]
      lm_loss     : list[float]
      aux_loss    : list[float]
      eff_n       : dict[layer_idx] -> list[float]  (effective_n_experts)
      n_dom       : dict[layer_idx] -> list[float]  (n_dominant_experts)
      max_load    : dict[layer_idx] -> list[float]  (max_expert_load)
      entropy_norm: dict[layer_idx] -> list[float]  (mean_token_entropy_norm)
    """
    data = {
        "steps": [], "lm_loss": [], "aux_loss": [],
        "eff_n": defaultdict(list),
        "n_dom": defaultdict(list),
        "max_load": defaultdict(list),
        "entropy_norm": defaultdict(list),
    }

    re_step     = re.compile(r"step = (\d+)")
    re_lmloss   = re.compile(r"train-lm_loss = ([0-9.]+)")
    re_auxloss  = re.compile(r"train-aux_loss = ([0-9.]+)")
    re_eff      = re.compile(r"h\.(\d+)\.ffwd/effective_n_experts = ([0-9.]+)")
    re_ndom     = re.compile(r"h\.(\d+)\.ffwd/n_dominant_experts = ([0-9.]+)")
    re_maxload  = re.compile(r"h\.(\d+)\.ffwd/max_expert_load = ([0-9.]+)")
    re_ent      = re.compile(r"h\.(\d+)\.ffwd/mean_token_entropy_norm = ([0-9.]+)")

    with open(path) as f:
        for line in f:
            if "▶ step =" not in line:
                continue
            m = re_step.search(line)
            if not m:
                continue
            step = int(m.group(1))
            data["steps"].append(step)

            m = re_lmloss.search(line)
            data["lm_loss"].append(float(m.group(1)) if m else float("nan"))

            m = re_auxloss.search(line)
            data["aux_loss"].append(float(m.group(1)) if m else 0.0)

            for layer, val in re_eff.findall(line):
                data["eff_n"][int(layer)].append(float(val))
            for layer, val in re_ndom.findall(line):
                data["n_dom"][int(layer)].append(float(val))
            for layer, val in re_maxload.findall(line):
                data["max_load"][int(layer)].append(float(val))
            for layer, val in re_ent.findall(line):
                data["entropy_norm"][int(layer)].append(float(val))

    return data


def mean_across_layers(per_layer: dict, n_steps: int) -> np.ndarray:
    """Mean over available layers, NaN where a layer has no data."""
    out = np.full((n_steps,), np.nan)
    arrays = []
    for layer in range(N_LAYERS):
        vals = per_layer.get(layer, [])
        if vals:
            arr = np.array(vals[:n_steps])
            arrays.append(arr)
    if arrays:
        min_len = min(len(a) for a in arrays)
        out = np.nanmean(np.stack([a[:min_len] for a in arrays], axis=0), axis=0)
    return out


# ── Plot 1: routing evolution per layer for each model ─────────────────────

def plot_per_layer_evolution(name: str, data: dict) -> None:
    steps = np.array(data["steps"])
    if not len(steps):
        return

    fig, axes = plt.subplots(3, 4, figsize=(14, 9), sharey=True, sharex=True)
    axes = axes.flatten()
    cmap = cm.get_cmap("plasma", N_LAYERS)

    for layer in range(N_LAYERS):
        ax = axes[layer]
        vals = data["eff_n"].get(layer)
        if vals:
            s = steps[:len(vals)]
            ax.plot(s, vals, color=cmap(layer), lw=1.5)
            ax.axhline(1.0, color="gray", lw=0.8, ls="--", alpha=0.5)
            ax.axhline(N_EXPERTS, color="gray", lw=0.8, ls=":", alpha=0.5)
        ax.set_title(f"Layer {layer}", fontsize=9)
        ax.set_ylim(0.8, N_EXPERTS + 0.5)
        ax.tick_params(labelsize=7)

    for ax in axes[N_LAYERS:]:
        ax.set_visible(False)
    fig.supxlabel("Training step", fontsize=11)
    fig.supylabel("Effective # experts", fontsize=11)
    fig.suptitle(f"{name} — effective experts per layer", fontsize=12, fontweight="bold")
    plt.tight_layout()
    tag = name.split()[0].lower()
    out = PLOTS_DIR / f"boltz_routing_evolution_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Plot 2: cross-model comparison (mean ± std across layers) ───────────────

def plot_comparison(all_data: dict) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for name, data in all_data.items():
        steps = np.array(data["steps"])
        if not len(steps):
            continue
        color = COLORS.get(name, "black")

        # Panel A: mean effective_n_experts across layers
        means = mean_across_layers(data["eff_n"], len(steps))
        axes[0].plot(steps[:len(means)], means, label=name, color=color, lw=2)

        # Panel B: mean max_expert_load across layers
        loads = mean_across_layers(data["max_load"], len(steps))
        axes[1].plot(steps[:len(loads)], loads, label=name, color=color, lw=2)

        # Panel C: LM loss
        axes[2].plot(steps, data["lm_loss"], label=name, color=color, lw=2)

    axes[0].axhline(1.0, color="gray", ls="--", lw=0.8, alpha=0.7, label="Collapsed (1)")
    axes[0].axhline(N_EXPERTS, color="gray", ls=":", lw=0.8, alpha=0.7, label="Uniform (16)")
    axes[0].set_ylabel("Effective # experts (mean over layers)")
    axes[0].set_title("Routing diversity over training")
    axes[0].legend(fontsize=8)
    axes[0].set_ylim(0.8, N_EXPERTS + 0.5)

    axes[1].axhline(1.0 / N_EXPERTS, color="gray", ls=":", lw=0.8, alpha=0.7)
    axes[1].set_ylabel("Max expert load (mean over layers)")
    axes[1].set_title("Expert load imbalance")

    axes[2].set_ylabel("LM training loss")
    axes[2].set_title("Training loss")

    for ax in axes:
        ax.set_xlabel("Training step")
        ax.grid(alpha=0.25)

    fig.suptitle("BoltzmannMoE ablation — routing collapse analysis", fontsize=13, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "boltz_routing_collapse_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Plot 3: effective_n_experts heatmap (layers × time) ─────────────────────

def plot_heatmap(name: str, data: dict) -> None:
    steps = np.array(data["steps"])
    if len(steps) < 10:
        return

    # Build (N_LAYERS, T) matrix
    T = min(len(steps), min((len(data["eff_n"].get(l, [])) for l in range(N_LAYERS)), default=0))
    if T == 0:
        return
    mat = np.full((N_LAYERS, T), np.nan)
    for layer in range(N_LAYERS):
        vals = data["eff_n"].get(layer, [])
        mat[layer, :len(vals[:T])] = vals[:T]

    fig, ax = plt.subplots(figsize=(10, 4))
    # Subsample steps for display (at most 300 columns)
    stride = max(1, T // 300)
    im = ax.imshow(mat[:, ::stride], aspect="auto", origin="lower", vmin=1, vmax=N_EXPERTS,
                   cmap="RdYlGn", extent=[steps[0], steps[min(T-1, len(steps)-1)], -0.5, N_LAYERS - 0.5])
    plt.colorbar(im, ax=ax, label="Effective # experts")
    ax.set_xlabel("Training step")
    ax.set_ylabel("Layer")
    ax.set_yticks(range(N_LAYERS))
    ax.set_title(f"{name} — effective experts per layer over time", fontweight="bold")
    plt.tight_layout()
    tag = name.split()[0].lower()
    out = PLOTS_DIR / f"boltz_heatmap_{tag}.pdf"
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(str(out).replace(".pdf", ".png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  saved {out.name}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--update", action="store_true", help="Force regeneration of all plots")
    args = parser.parse_args()

    all_data = {}
    for name, (job_name, pattern) in RUNS.items():
        log = fetch_log(job_name, pattern)
        if log is None:
            print(f"  [SKIP] {name}: no log found")
            continue
        print(f"  Parsing {name}: {log.name}")
        data = parse_log(log)
        n_steps = len(data["steps"])
        n_layers = len(data["eff_n"])
        latest_step = data["steps"][-1] if n_steps else 0
        mean_eff = np.nanmean([data["eff_n"][l][-1] for l in data["eff_n"]]) if data["eff_n"] else float("nan")
        print(f"    {n_steps} log entries, latest step {latest_step}, mean_eff_n={mean_eff:.3f}, {n_layers} layers")
        all_data[name] = data

    print("\nGenerating plots...")
    for name, data in all_data.items():
        plot_per_layer_evolution(name, data)
        plot_heatmap(name, data)
    plot_comparison(all_data)

    # Print summary table
    print("\n── Routing summary at latest logged step ──")
    print(f"{'Model':<28} {'Step':>6} {'MeanEff':>9} {'MinEff':>9} {'MeanLoad':>10} {'LMLoss':>8}")
    print("-" * 75)
    for name, data in all_data.items():
        if not data["steps"]:
            continue
        step = data["steps"][-1]
        loss = data["lm_loss"][-1] if data["lm_loss"] else float("nan")
        effs = [data["eff_n"][l][-1] for l in range(N_LAYERS) if data["eff_n"].get(l)]
        loads = [data["max_load"][l][-1] for l in range(N_LAYERS) if data["max_load"].get(l)]
        mean_eff = np.mean(effs) if effs else float("nan")
        min_eff  = np.min(effs) if effs else float("nan")
        mean_load = np.mean(loads) if loads else float("nan")
        print(f"{name:<28} {step:>6} {mean_eff:>9.3f} {min_eff:>9.3f} {mean_load:>10.3f} {loss:>8.4f}")

    print(f"\nPlots written to {PLOTS_DIR}")


if __name__ == "__main__":
    main()
