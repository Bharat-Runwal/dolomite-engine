"""Visualize energy profiles across ablation models (3-block and 4-block with varying iterations).

Generates per-block energy trajectories for each model and comparison plots showing
how energy landscape changes with iteration count.

Usage:
    python visualize_ablation_energy.py [--output_dir DIR] [--num_samples N]
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

from lm_engine.hf_models.register_hf import register_model_classes
register_model_classes()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_energy_per_iteration(model, input_ids):
    base_model = model.transformer
    with torch.no_grad():
        hidden_states = base_model.wte(input_ids)
        if base_model.m_emb is not None:
            hidden_states = hidden_states * base_model.m_emb

        rope_cos_sin = None
        if base_model.position_embedding_type == "rope":
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            rope_cos_sin = base_model._get_rope_cos_sin(
                key_length=input_ids.shape[1], position_ids=position_ids, dtype=hidden_states.dtype)

        energies = []
        for i, num_iter in enumerate(base_model.layer_iterations):
            block = base_model.h[i]
            has_energy = hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention"
            for j in range(num_iter):
                hidden_states = block(hidden_states, past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)
                # Compute energy AFTER each iteration (block's output energy)
                if has_energy:
                    e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
                    energies.append((i, j, e.float().mean().item(), e.float().std().item()))

    return energies


def profile_model(model_path, tokenizer, texts):
    print(f"  Loading {model_path.split('/')[-1]}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.eval()

    all_energies = defaultdict(list)
    all_deltas = defaultdict(list)

    for idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        if input_ids.shape[1] < 10:
            continue
        energies = compute_energy_per_iteration(model, input_ids)
        prev_by_block = {}
        for block_idx, iter_idx, mean_e, std_e in energies:
            all_energies[(block_idx, iter_idx)].append(mean_e)
            if block_idx in prev_by_block:
                delta = mean_e - prev_by_block[block_idx]
                all_deltas[(block_idx, iter_idx)].append(delta)
            prev_by_block[block_idx] = mean_e

    del model
    torch.cuda.empty_cache()
    return dict(all_energies), dict(all_deltas)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_dir", default="/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_test")
    parser.add_argument("--output_dir", default="workshop_results/ablation_energy")
    parser.add_argument("--num_samples", type=int, default=30)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained('/proj/checkpoints/dmf-lh-checkpoints/granite-4.0-tiktoken')

    print("Loading WikiText data...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:args.num_samples]
    print(f"Using {len(texts)} samples")

    # Discover models - only uniform "all" configs for 3-block and 4-block
    models_3b = sorted([d for d in os.listdir(args.model_dir) if d.startswith("EGPT_3_blocks_all_")],
                       key=lambda x: int(x.split("_all_")[1].split("_")[0]))
    models_4b = sorted([d for d in os.listdir(args.model_dir) if d.startswith("EGPT_4_blocks_all_")],
                       key=lambda x: int(x.split("_all_")[1].split("_")[0]))

    # Profile all models
    results = {}
    for model_name in models_3b + models_4b:
        model_path = os.path.join(args.model_dir, model_name)
        energies, deltas = profile_model(model_path, tokenizer, texts)
        num_iters = int(model_name.split("_all_")[1].split("_")[0])
        num_blocks = 3 if "3_blocks" in model_name else 4
        results[model_name] = {
            'energies': energies, 'deltas': deltas,
            'num_iters': num_iters, 'num_blocks': num_blocks
        }

    # =========================================================================
    # FIGURE 1: 3-block models - per-block energy vs iteration for each model
    # =========================================================================
    print("Generating 3-block comparison...")
    iter_counts_3 = sorted(set(r['num_iters'] for r in results.values() if r['num_blocks'] == 3))
    fig, axes = plt.subplots(len(iter_counts_3), 3, figsize=(18, 4 * len(iter_counts_3)), squeeze=False)

    for row, n_iter in enumerate(iter_counts_3):
        model_name = f"EGPT_3_blocks_all_{n_iter}_30k"
        r = results[model_name]
        for block_idx in range(3):
            ax = axes[row][block_idx]
            iters_x = []
            means_y = []
            stds_y = []
            for it in range(n_iter + 1):
                key = (block_idx, it)
                if key in r['energies']:
                    iters_x.append(it)
                    means_y.append(np.mean(r['energies'][key]))
                    stds_y.append(np.std(r['energies'][key]))
            if iters_x:
                means_y = np.array(means_y)
                stds_y = np.array(stds_y)
                ax.plot(iters_x, means_y, 'o-', color='steelblue', linewidth=2, markersize=4)
                ax.fill_between(iters_x, means_y - stds_y, means_y + stds_y, alpha=0.2, color='steelblue')
            ax.set_title(f'Block {block_idx} (T={n_iter})', fontsize=10)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Energy')
            ax.grid(True, alpha=0.3)

    fig.suptitle('3-Block Models: Energy per Block vs Iteration\n(WikiText, varying T)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'energy_3blocks_per_model.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 2: 4-block models - same layout
    # =========================================================================
    print("Generating 4-block comparison...")
    iter_counts_4 = sorted(set(r['num_iters'] for r in results.values() if r['num_blocks'] == 4))
    fig, axes = plt.subplots(len(iter_counts_4), 4, figsize=(22, 4 * len(iter_counts_4)), squeeze=False)

    for row, n_iter in enumerate(iter_counts_4):
        model_name = f"EGPT_4_blocks_all_{n_iter}_30k"
        r = results[model_name]
        for block_idx in range(4):
            ax = axes[row][block_idx]
            iters_x = []
            means_y = []
            stds_y = []
            for it in range(n_iter + 1):
                key = (block_idx, it)
                if key in r['energies']:
                    iters_x.append(it)
                    means_y.append(np.mean(r['energies'][key]))
                    stds_y.append(np.std(r['energies'][key]))
            if iters_x:
                means_y = np.array(means_y)
                stds_y = np.array(stds_y)
                ax.plot(iters_x, means_y, 'o-', color='coral', linewidth=2, markersize=4)
                ax.fill_between(iters_x, means_y - stds_y, means_y + stds_y, alpha=0.2, color='coral')
            ax.set_title(f'Block {block_idx} (T={n_iter})', fontsize=10)
            ax.set_xlabel('Iteration')
            ax.set_ylabel('Mean Energy')
            ax.grid(True, alpha=0.3)

    fig.suptitle('4-Block Models: Energy per Block vs Iteration\n(WikiText, varying T)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'energy_4blocks_per_model.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 3: Convergence comparison - final energy and convergence rate vs T
    # =========================================================================
    print("Generating convergence comparison...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    for nb, color, label in [(3, 'steelblue', '3-block'), (4, 'coral', '4-block')]:
        iter_counts = sorted(set(r['num_iters'] for r in results.values() if r['num_blocks'] == nb))
        n_blocks = nb

        # Panel A: Final energy (at last iteration) per block vs T
        ax = axes[0, 0]
        for block_idx in range(n_blocks):
            final_energies = []
            for n_iter in iter_counts:
                model_name = f"EGPT_{nb}_blocks_all_{n_iter}_30k"
                r = results[model_name]
                last_key = (block_idx, n_iter)
                if last_key in r['energies']:
                    final_energies.append(np.mean(r['energies'][last_key]))
                else:
                    # Try n_iter-1
                    last_key2 = (block_idx, n_iter - 1)
                    if last_key2 in r['energies']:
                        final_energies.append(np.mean(r['energies'][last_key2]))
                    else:
                        final_energies.append(np.nan)
            style = '-' if nb == 3 else '--'
            ax.plot(iter_counts, final_energies, f'o{style}', label=f'{nb}B-B{block_idx}',
                    linewidth=1.5, markersize=5)
        ax.set_xlabel('Iterations per Block (T)')
        ax.set_ylabel('Final Energy (after T iterations)')
        ax.set_title('Final Energy vs Iteration Count')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

        # Panel B: Mean convergence rate per block vs T
        ax = axes[0, 1]
        for block_idx in range(n_blocks):
            conv_rates = []
            for n_iter in iter_counts:
                model_name = f"EGPT_{nb}_blocks_all_{n_iter}_30k"
                r = results[model_name]
                abs_deltas = []
                for it in range(1, n_iter + 1):
                    key = (block_idx, it)
                    if key in r['deltas']:
                        abs_deltas.extend([abs(d) for d in r['deltas'][key]])
                conv_rates.append(np.mean(abs_deltas) if abs_deltas else np.nan)
            style = '-' if nb == 3 else '--'
            ax.plot(iter_counts, conv_rates, f'o{style}', label=f'{nb}B-B{block_idx}',
                    linewidth=1.5, markersize=5)
        ax.set_xlabel('Iterations per Block (T)')
        ax.set_ylabel('Mean |Energy Change| per Iteration')
        ax.set_title('Convergence Rate vs Iteration Count')
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    # Panel C: Total energy change (first to last) vs T
    ax = axes[1, 0]
    for nb, color, marker in [(3, 'steelblue', 'o'), (4, 'coral', 's')]:
        iter_counts = sorted(set(r['num_iters'] for r in results.values() if r['num_blocks'] == nb))
        total_changes = []
        for n_iter in iter_counts:
            model_name = f"EGPT_{nb}_blocks_all_{n_iter}_30k"
            r = results[model_name]
            total = 0
            for block_idx in range(nb):
                first_key = (block_idx, 0)
                last_key = (block_idx, n_iter)
                if last_key not in r['energies']:
                    last_key = (block_idx, n_iter - 1)
                if first_key in r['energies'] and last_key in r['energies']:
                    total += np.mean(r['energies'][last_key]) - np.mean(r['energies'][first_key])
            total_changes.append(total)
        ax.plot(iter_counts, total_changes, f'{marker}-', color=color, label=f'{nb}-block',
                linewidth=2, markersize=7)
    ax.set_xlabel('Iterations per Block (T)')
    ax.set_ylabel('Total Energy Change (sum across blocks)')
    ax.set_title('Total Energy Change vs Iteration Count')
    ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel D: Energy at iteration 0 vs T (does initial energy change with training?)
    ax = axes[1, 1]
    for nb, color, marker in [(3, 'steelblue', 'o'), (4, 'coral', 's')]:
        iter_counts = sorted(set(r['num_iters'] for r in results.values() if r['num_blocks'] == nb))
        for block_idx in range(nb):
            init_energies = []
            for n_iter in iter_counts:
                model_name = f"EGPT_{nb}_blocks_all_{n_iter}_30k"
                r = results[model_name]
                key = (block_idx, 0)
                if key in r['energies']:
                    init_energies.append(np.mean(r['energies'][key]))
                else:
                    init_energies.append(np.nan)
            style = '-' if nb == 3 else '--'
            ax.plot(iter_counts, init_energies, f'{marker}{style}', label=f'{nb}B-B{block_idx}',
                    linewidth=1.5, markersize=5)
    ax.set_xlabel('Iterations per Block (T)')
    ax.set_ylabel('Energy at Iteration 0')
    ax.set_title('Initial Energy vs Iteration Count (effect of training)')
    ax.legend(fontsize=7, ncol=2)
    ax.grid(True, alpha=0.3)

    fig.suptitle('Energy Convergence Analysis: 3-Block vs 4-Block\n(WikiText, 30k steps training)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'convergence_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 4: Energy heatmaps for selected models
    # =========================================================================
    print("Generating energy heatmaps...")
    selected = ['EGPT_3_blocks_all_6_30k', 'EGPT_3_blocks_all_12_30k', 'EGPT_3_blocks_all_24_30k',
                'EGPT_4_blocks_all_6_30k', 'EGPT_4_blocks_all_12_30k', 'EGPT_4_blocks_all_24_30k']
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))

    for idx, model_name in enumerate(selected):
        if model_name not in results:
            continue
        r = results[model_name]
        nb = r['num_blocks']
        ni = r['num_iters']
        ax = axes[idx // 3][idx % 3]

        matrix = np.full((nb, ni + 1), np.nan)
        for (b, i), vals in r['energies'].items():
            if b < nb and i <= ni:
                matrix[b, i] = np.mean(vals)

        im = ax.imshow(matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Block')
        ax.set_title(f'{nb}B x T={ni}', fontsize=11, fontweight='bold')
        ax.set_yticks(range(nb))
        plt.colorbar(im, ax=ax, shrink=0.8)

    fig.suptitle('Energy Heatmaps: Block x Iteration\n(WikiText samples)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, 'energy_heatmaps_selected.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # Print summary table
    print(f"\n{'Model':>35} | {'Blocks':>6} | {'T':>4} | {'E(0) avg':>10} | {'E(final) avg':>12} | {'Delta':>10} | {'Conv Rate':>10}")
    print("-" * 100)
    for model_name in sorted(results.keys(), key=lambda x: (results[x]['num_blocks'], results[x]['num_iters'])):
        r = results[model_name]
        nb = r['num_blocks']
        ni = r['num_iters']
        init_es = []
        final_es = []
        conv_rates = []
        for block_idx in range(nb):
            k0 = (block_idx, 0)
            kf = (block_idx, ni)
            if kf not in r['energies']:
                kf = (block_idx, ni - 1)
            if k0 in r['energies']:
                init_es.append(np.mean(r['energies'][k0]))
            if kf in r['energies']:
                final_es.append(np.mean(r['energies'][kf]))
            abs_d = []
            for it in range(1, ni + 1):
                key = (block_idx, it)
                if key in r['deltas']:
                    abs_d.extend([abs(d) for d in r['deltas'][key]])
            if abs_d:
                conv_rates.append(np.mean(abs_d))

        avg_init = np.mean(init_es) if init_es else float('nan')
        avg_final = np.mean(final_es) if final_es else float('nan')
        avg_delta = avg_final - avg_init if init_es and final_es else float('nan')
        avg_conv = np.mean(conv_rates) if conv_rates else float('nan')
        print(f"{model_name:>35} | {nb:>6} | {ni:>4} | {avg_init:>10.2f} | {avg_final:>12.2f} | {avg_delta:>+10.2f} | {avg_conv:>10.4f}")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
