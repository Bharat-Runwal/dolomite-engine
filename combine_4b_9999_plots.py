"""Combine energy plots for 4 EGPT_4b_9_9_9_9 models (30k-70k, skipping 60k) and
generate cross-step analysis figures.

Produces:
  1. Combined per-block energy profiles (7 rows, one per step)
  2. Combined energy heatmaps (7 rows)
  3. Combined convergence analysis (7 rows)
  4. Cross-step analysis: energy delta, convergence rate, initial/final energy vs steps
  5. Per-block energy trajectory across training steps (iter-level detail)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

MODEL_PREFIX = "EGPT_4b_9_9_9_9"
NUM_BLOCKS = 4


# =========================================================================
# PART 1: Combine existing per-model plots into single stacked figures
# =========================================================================

def combine_vertical(image_paths, labels, output_path, title=None):
    """Stack images vertically with labels on the left."""
    imgs = [Image.open(p) for p in image_paths]

    target_w = max(img.size[0] for img in imgs)
    label_w = 180

    resized = []
    for img in imgs:
        w, h = img.size
        if w != target_w:
            new_h = int(h * target_w / w)
            img = img.resize((target_w, new_h), Image.LANCZOS)
        resized.append(img)

    title_h = 60 if title else 0
    total_h = sum(img.size[1] for img in resized) + title_h
    canvas = Image.new("RGB", (label_w + target_w, total_h), "white")

    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 18)
        title_font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 28)
    except (OSError, IOError):
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 18)
            title_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 28)
        except (OSError, IOError):
            font = ImageFont.load_default()
            title_font = font

    draw = ImageDraw.Draw(canvas)

    if title:
        bbox = draw.textbbox((0, 0), title, font=title_font)
        tw = bbox[2] - bbox[0]
        draw.text(((label_w + target_w - tw) // 2, 15), title, fill="black", font=title_font)

    y_offset = title_h
    for img, label in zip(resized, labels):
        h = img.size[1]
        text_y = y_offset + h // 2
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        th = bbox[3] - bbox[1]
        draw.text(((label_w - tw) // 2, text_y - th // 2), label, fill="black", font=font)
        canvas.paste(img, (label_w, y_offset))
        draw.line([(0, y_offset), (label_w + target_w, y_offset)], fill="gray", width=1)
        y_offset += h

    canvas.save(output_path, dpi=(150, 150))
    print(f"  Saved: {output_path}  ({canvas.size[0]}x{canvas.size[1]})")


def combine_existing_plots():
    base = Path(f"tools/bs/energy_landscape/{MODEL_PREFIX}")
    steps = ["30k", "40k", "50k", "70k"]

    plot_types = {
        "per_block_energy_wikitext.png": "Per-Block Energy Profiles",
        "energy_heatmaps_wikitext.png": "Energy Heatmaps",
        "convergence_analysis_wikitext.png": "Convergence Analysis",
    }

    for plot_file, title in plot_types.items():
        paths = []
        labels = []
        for step in steps:
            p = base / f"{MODEL_PREFIX}_{step}" / plot_file
            if p.exists():
                paths.append(p)
                labels.append(f"Step {step}")

        if paths:
            out_name = f"combined_{plot_file}"
            print(f"\nCombining {title} ({len(paths)} models)...")
            combine_vertical(paths, labels, base / out_name,
                           title=f"{MODEL_PREFIX}: {title} (30k-70k)")


# =========================================================================
# PART 2: Cross-step analysis figures using model inference
# =========================================================================

def load_energy_data():
    """Load all 7 models and compute energy data across steps."""
    from lm_engine.hf_models.register_hf import register_model_classes
    register_model_classes()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from datasets import load_dataset
    import torch

    BASE = "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_test"
    STEPS = [30000, 40000, 50000, 70000]
    STEP_LABELS = ["30k", "40k", "50k", "70k"]

    tokenizer = AutoTokenizer.from_pretrained('/proj/dmfexp/energy-gpt/data/granite-4.0-tiktoken')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:30]
    print(f"Using {len(texts)} samples")

    all_data = {}

    for step, label in zip(STEPS, STEP_LABELS):
        model_name = f"{MODEL_PREFIX}_{label}"
        model_path = os.path.join(BASE, model_name)
        if not os.path.exists(model_path):
            print(f"  SKIP: {model_path}")
            continue

        print(f"\nLoading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        model.config.use_cache = False
        model.eval()
        model.to(device)

        base_model = model.transformer
        num_iters_list = list(base_model.layer_iterations)

        step_data = defaultdict(lambda: defaultdict(list))

        for idx, text in enumerate(texts):
            input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
            if input_ids.shape[1] < 10:
                continue

            with torch.no_grad():
                hidden_states = base_model.wte(input_ids)
                if base_model.m_emb is not None:
                    hidden_states = hidden_states * base_model.m_emb

                rope_cos_sin = None
                if base_model.position_embedding_type == "rope":
                    position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
                    rope_cos_sin = base_model._get_rope_cos_sin(
                        key_length=input_ids.shape[1], position_ids=position_ids, dtype=hidden_states.dtype)

                for i, num_iter in enumerate(base_model.layer_iterations):
                    block = base_model.h[i]
                    has_energy = hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention"
                    for j in range(num_iter):
                        hidden_states = block(
                            hidden_states, past_key_values=None, attention_mask=None,
                            rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)
                        if has_energy:
                            e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
                            step_data[i][j].append(e.float().mean().item())

            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(texts)}")

        all_data[label] = {
            'energies': {(b, it): vals for b, iters in step_data.items() for it, vals in iters.items()},
            'num_iters': num_iters_list,
        }

        del model
        torch.cuda.empty_cache()

    return all_data, STEP_LABELS


def plot_cross_step_figures(all_data, step_labels):
    out_dir = Path(f"tools/bs/energy_landscape/{MODEL_PREFIX}")
    out_dir.mkdir(parents=True, exist_ok=True)

    available = [s for s in step_labels if s in all_data]
    step_nums = [int(s.replace('k', '')) for s in available]
    num_blocks = NUM_BLOCKS
    cmap = plt.cm.tab10(np.linspace(0, 1, num_blocks))

    # Precompute per-block stats for each step
    stats = {}
    for label in available:
        data = all_data[label]
        stats[label] = {}
        for b in range(num_blocks):
            max_iter = max((it for (bi, it) in data['energies'] if bi == b), default=0)
            e_first = np.mean(data['energies'].get((b, 0), [np.nan]))
            e_last = np.mean(data['energies'].get((b, max_iter), [np.nan]))
            e_mid = np.mean(data['energies'].get((b, max_iter // 2), [np.nan]))

            iter_energies = []
            for it in range(max_iter + 1):
                vals = data['energies'].get((b, it), [])
                if vals:
                    iter_energies.append((it, np.mean(vals), np.std(vals)))

            deltas = []
            for it in range(1, max_iter + 1):
                prev = data['energies'].get((b, it - 1), [])
                curr = data['energies'].get((b, it), [])
                if prev and curr:
                    pm, cm = np.mean(prev), np.mean(curr)
                    deltas.append(abs(cm - pm) / (abs(pm) + 1e-8))

            stats[label][b] = {
                'e_first': e_first,
                'e_last': e_last,
                'e_mid': e_mid,
                'delta': e_last - e_first,
                'conv_rate': np.mean(deltas) if deltas else 0,
                'iter_energies': iter_energies,
                'max_iter': max_iter,
            }

    # =====================================================================
    # FIGURE A: Comprehensive cross-step dashboard (2x3 grid)
    # =====================================================================
    fig, axes = plt.subplots(2, 3, figsize=(24, 14))

    ax = axes[0, 0]
    for b in range(num_blocks):
        y = [stats[s][b]['delta'] for s in available]
        ax.plot(step_nums, y, 'o-', label=f'Block {b}', color=cmap[b], linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy Change (last - first iter)', fontsize=12)
    ax.set_title('Energy Change per Block vs Training Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.4)

    ax = axes[0, 1]
    for b in range(num_blocks):
        y = [stats[s][b]['conv_rate'] for s in available]
        ax.plot(step_nums, y, 's-', label=f'Block {b}', color=cmap[b], linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Mean Relative |Energy Change|', fontsize=12)
    ax.set_title('Convergence Rate per Block vs Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    for b in range(num_blocks):
        y = [stats[s][b]['e_first'] for s in available]
        ax.plot(step_nums, y, 'o-', label=f'Block {b}', color=cmap[b], linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy at Iteration 0', fontsize=12)
    ax.set_title('Initial Energy per Block vs Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    for b in range(num_blocks):
        y = [stats[s][b]['e_last'] for s in available]
        ax.plot(step_nums, y, 'o-', label=f'Block {b}', color=cmap[b], linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy at Last Iteration', fontsize=12)
    ax.set_title('Final Energy per Block vs Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    total_first = [sum(stats[s][b]['e_first'] for b in range(num_blocks)) for s in available]
    total_last = [sum(stats[s][b]['e_last'] for b in range(num_blocks)) for s in available]
    total_delta = [sum(stats[s][b]['delta'] for b in range(num_blocks)) for s in available]
    ax.plot(step_nums, total_first, 'o-', label='Sum Initial E', color='steelblue', linewidth=2, markersize=7)
    ax.plot(step_nums, total_last, 's-', label='Sum Final E', color='coral', linewidth=2, markersize=7)
    ax.plot(step_nums, total_delta, '^--', label='Sum Delta E', color='green', linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Total Energy (summed over blocks)', fontsize=12)
    ax.set_title('Aggregate Energy vs Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.4)

    ax = axes[1, 2]
    spread_first = [max(stats[s][b]['e_first'] for b in range(num_blocks)) - min(stats[s][b]['e_first'] for b in range(num_blocks)) for s in available]
    spread_last = [max(stats[s][b]['e_last'] for b in range(num_blocks)) - min(stats[s][b]['e_last'] for b in range(num_blocks)) for s in available]
    ax.plot(step_nums, spread_first, 'o-', label='Spread at Iter 0', color='steelblue', linewidth=2, markersize=7)
    ax.plot(step_nums, spread_last, 's-', label='Spread at Last Iter', color='coral', linewidth=2, markersize=7)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy Spread (max - min block)', fontsize=12)
    ax.set_title('Inter-Block Energy Spread vs Steps', fontsize=13, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{MODEL_PREFIX}: Cross-Step Energy Analysis Dashboard\n(wikitext, 30 samples)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'cross_step_dashboard.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =====================================================================
    # FIGURE B: Per-block energy trajectory heatmap (blocks x steps x iters)
    # =====================================================================
    fig, axes = plt.subplots(1, num_blocks, figsize=(5 * num_blocks, 6))
    max_iter = 9

    for b in range(num_blocks):
        ax = axes[b]
        matrix = np.full((len(available), max_iter), np.nan)
        for si, label in enumerate(available):
            for it, mean_e, _ in stats[label][b]['iter_energies']:
                if it < max_iter:
                    matrix[si, it] = mean_e

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Training Step', fontsize=10)
        ax.set_yticks(range(len(available)))
        ax.set_yticklabels(available, fontsize=9)
        ax.set_xticks(range(max_iter))
        ax.set_title(f'Block {b}', fontsize=12, fontweight='bold')
        plt.colorbar(im, ax=ax, shrink=0.8, label='Energy')

    fig.suptitle(f'{MODEL_PREFIX}: Energy per (Step, Iteration) for Each Block\n(wikitext, 30 samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'energy_heatmap_steps_x_iters.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =====================================================================
    # FIGURE C: Per-block energy curves overlaid across steps
    # =====================================================================
    fig, axes = plt.subplots(1, num_blocks, figsize=(5 * num_blocks, 5))
    step_cmap = plt.cm.viridis(np.linspace(0, 1, len(available)))

    for b in range(num_blocks):
        ax = axes[b]
        for si, label in enumerate(available):
            iters = [it for it, _, _ in stats[label][b]['iter_energies']]
            means = [m for _, m, _ in stats[label][b]['iter_energies']]
            stds = [s for _, _, s in stats[label][b]['iter_energies']]
            if iters:
                ax.plot(iters, means, 'o-', label=label, color=step_cmap[si], linewidth=1.5, markersize=4)
                ax.fill_between(iters, np.array(means) - np.array(stds),
                              np.array(means) + np.array(stds), alpha=0.1, color=step_cmap[si])
        ax.set_xlabel('Iteration', fontsize=10)
        ax.set_ylabel('Mean Energy', fontsize=10)
        ax.set_title(f'Block {b}', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f'{MODEL_PREFIX}: Energy Curves Across Training Steps\n(wikitext, 30 samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'energy_curves_across_steps.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =====================================================================
    # FIGURE D: Energy dynamics - early vs late training
    # =====================================================================
    from matplotlib.lines import Line2D

    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    ax = axes[0]
    for b in range(num_blocks):
        for si, (label, marker) in enumerate([(available[0], 'o'), (available[-1], 's')]):
            ie = stats[label][b]['iter_energies']
            if len(ie) > 1:
                deltas = [ie[i+1][1] - ie[i][1] for i in range(len(ie)-1)]
                iters = [ie[i][0] + 0.5 for i in range(len(ie)-1)]
                ls = '-' if si == 0 else '--'
                ax.plot(iters, deltas, marker=marker, linestyle=ls,
                       color=cmap[b], alpha=0.8 if si == 0 else 0.4,
                       linewidth=2 if si == 1 else 1, markersize=5)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Energy Change (iter N+1 - iter N)', fontsize=12)
    ax.set_title(f'Per-Iteration Energy Change: {available[0]} (solid) vs {available[-1]} (dashed)',
                fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.3)
    handles = [Line2D([0], [0], color=cmap[b], linewidth=2, label=f'Block {b}') for b in range(num_blocks)]
    handles.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='-', label=available[0]))
    handles.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label=available[-1]))
    ax.legend(handles=handles, fontsize=8, ncol=2)

    ax = axes[1]
    for b in range(num_blocks):
        for si, label in enumerate([available[0], available[-1]]):
            ie = stats[label][b]['iter_energies']
            if ie:
                base_e = ie[0][1]
                cum_delta = [m - base_e for _, m, _ in ie]
                iters = [it for it, _, _ in ie]
                marker = 'o' if si == 0 else 's'
                ls = '-' if si == 0 else '--'
                ax.plot(iters, cum_delta, marker=marker, linestyle=ls,
                       color=cmap[b], linewidth=2 if si == 1 else 1,
                       alpha=0.8 if si == 0 else 0.5, markersize=4)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_ylabel('Cumulative Energy Change from Iter 0', fontsize=12)
    ax.set_title(f'Cumulative Energy Trajectory: {available[0]} vs {available[-1]}',
                fontsize=11, fontweight='bold')
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.4)
    ax.grid(True, alpha=0.3)
    handles2 = [Line2D([0], [0], color=cmap[b], linewidth=2, label=f'Block {b}') for b in range(num_blocks)]
    handles2.append(Line2D([0], [0], color='gray', linewidth=1, linestyle='-', label=available[0]))
    handles2.append(Line2D([0], [0], color='gray', linewidth=2, linestyle='--', label=available[-1]))
    ax.legend(handles=handles2, fontsize=8, ncol=2)

    fig.suptitle(f'{MODEL_PREFIX}: Energy Dynamics Comparison (Early vs Late Training)\n(wikitext, 30 samples)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = out_dir / 'energy_dynamics_early_vs_late.png'
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def main():
    print("=" * 60)
    print("PART 1: Combining existing per-model plots")
    print("=" * 60)
    combine_existing_plots()

    print("\n" + "=" * 60)
    print("PART 2: Cross-step analysis figures")
    print("=" * 60)
    all_data, step_labels = load_energy_data()
    plot_cross_step_figures(all_data, step_labels)

    print("\nAll done!")


if __name__ == '__main__':
    main()
