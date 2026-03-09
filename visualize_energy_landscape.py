"""Comprehensive energy landscape visualization for Energy Transformers.

Generates per-block energy profiles, energy change heatmaps, convergence
analysis, and per-token energy distributions across WikiText samples.

Usage:
    python visualize_energy_landscape.py [model_path] [--output_dir DIR] [--num_samples N] [--dataset wikitext]
"""

import sys
import os
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

from lm_engine.hf_models.register_hf import register_model_classes
register_model_classes()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_energy_per_iteration(model, input_ids):
    """Run forward pass, collecting per-token energy at each (block, iteration)."""
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

        energies = []       # (block_idx, iter_idx, mean_energy, std_energy)
        token_energies = [] # (block_idx, iter_idx, energy_per_token_tensor)
        hidden_norms = []   # (block_idx, iter_idx, hidden_state_norm)

        for i, num_iter in enumerate(base_model.layer_iterations):
            block = base_model.h[i]
            has_energy = hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention"

            for j in range(num_iter):
                h_norm = hidden_states.float().norm(dim=-1).mean().item()

                if has_energy:
                    e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
                    energies.append((i, j, e.float().mean().item(), e.float().std().item()))
                    token_energies.append((i, j, e.float().cpu()))

                hidden_norms.append((i, j, h_norm))

                hidden_states = block(
                    hidden_states, past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)

        # Final energy after last iteration
        last_idx = len(base_model.h) - 1
        block = base_model.h[last_idx]
        if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention":
            e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
            last_iter = base_model.layer_iterations[last_idx]
            energies.append((last_idx, last_iter, e.float().mean().item(), e.float().std().item()))
            token_energies.append((last_idx, last_iter, e.float().cpu()))
            hidden_norms.append((last_idx, last_iter, hidden_states.float().norm(dim=-1).mean().item()))

    return energies, token_energies, hidden_norms


def load_texts(dataset_name, num_samples):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    elif dataset_name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        return [f"{row['ctx']} {row['endings'][int(row['label'])]}" for row in dataset][:num_samples]
    elif dataset_name == "lambada":
        dataset = load_dataset("EleutherAI/lambada_openai", "default", split="test")
        return [row["text"] for row in dataset if len(row["text"]) > 50][:num_samples]
    elif dataset_name == "arc":
        dataset = load_dataset("allenai/ai2_arc", "ARC-Easy", split="validation")
        return [row["question"] + " " + row["choices"]["text"][0] for row in dataset][:num_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path", nargs="?",
        default="/proj/checkpoints/bharat/personal/energy_gpt/model_unshard_FINAL_30k/compositional_12EGPT_6iter_gelu_30k_AGAIN")
    parser.add_argument("--output_dir", default="workshop_results/energy_landscape")
    parser.add_argument("--num_samples", type=int, default=50)
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "hellaswag", "lambada", "arc"])
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    model_name = args.model_path.rstrip("/").split("/")[-1]

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained('/proj/checkpoints/dmf-lh-checkpoints/granite-4.0-tiktoken')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.eval()

    print(f"Loading {args.dataset} data...")
    texts = load_texts(args.dataset, args.num_samples)
    print(f"Using {len(texts)} samples")

    # Collect data across all samples
    all_energies = defaultdict(list)       # (block, iter) -> [mean_energy, ...]
    all_stds = defaultdict(list)
    all_deltas = defaultdict(list)         # (block, iter) -> [delta, ...]
    all_rel_deltas = defaultdict(list)     # (block, iter) -> [relative delta, ...]
    all_token_energies = defaultdict(list) # (block, iter) -> [token_energy_tensor, ...]
    all_hidden_norms = defaultdict(list)
    per_sample_trajectories = []           # list of per-sample energy trajectories

    for idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        if input_ids.shape[1] < 10:
            continue

        energies, token_energies, hidden_norms = compute_energy_per_iteration(model, input_ids)

        sample_traj = []
        prev_by_block = {}
        for block_idx, iter_idx, mean_e, std_e in energies:
            key = (block_idx, iter_idx)
            all_energies[key].append(mean_e)
            all_stds[key].append(std_e)
            sample_traj.append((block_idx, iter_idx, mean_e))

            if block_idx in prev_by_block:
                delta = mean_e - prev_by_block[block_idx]
                all_deltas[key].append(delta)
                rel_delta = abs(delta) / (abs(prev_by_block[block_idx]) + 1e-8)
                all_rel_deltas[key].append(rel_delta)
            prev_by_block[block_idx] = mean_e

        for block_idx, iter_idx, e_tensor in token_energies:
            all_token_energies[(block_idx, iter_idx)].append(e_tensor)

        for block_idx, iter_idx, h_norm in hidden_norms:
            all_hidden_norms[(block_idx, iter_idx)].append(h_norm)

        per_sample_trajectories.append(sample_traj)

        if (idx + 1) % 10 == 0:
            print(f"  Processed {idx+1}/{len(texts)} samples")

    num_blocks = max(k[0] for k in all_energies.keys()) + 1
    max_iter = max(k[1] for k in all_energies.keys())
    num_iters = model.transformer.layer_iterations

    # =========================================================================
    # FIGURE 1: Per-block energy profiles (individual subplots)
    # =========================================================================
    print("Generating per-block energy profiles...")
    cols = 4
    rows = (num_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for block_idx in range(num_blocks):
        ax = axes[block_idx // cols][block_idx % cols]
        iters_x = []
        means_y = []
        stds_y = []
        for iter_idx in range(max_iter + 1):
            key = (block_idx, iter_idx)
            if key in all_energies:
                iters_x.append(iter_idx)
                means_y.append(np.mean(all_energies[key]))
                stds_y.append(np.std(all_energies[key]))

        if iters_x:
            means_y = np.array(means_y)
            stds_y = np.array(stds_y)
            ax.plot(iters_x, means_y, 'o-', color='steelblue', linewidth=2, markersize=6)
            ax.fill_between(iters_x, means_y - stds_y, means_y + stds_y, alpha=0.2, color='steelblue')

            # Mark energy change on secondary y-axis
            if len(iters_x) > 1:
                ax2 = ax.twinx()
                deltas = np.diff(means_y)
                ax2.bar([x + 0.5 for x in iters_x[:-1]], deltas, width=0.3, alpha=0.4, color='coral', label='Delta')
                ax2.set_ylabel('Energy Change', fontsize=8, color='coral')
                ax2.tick_params(axis='y', labelcolor='coral', labelsize=7)
                ax2.axhline(y=0, color='coral', linestyle=':', alpha=0.3)

        ax.set_title(f'Block {block_idx} ({num_iters[block_idx]} iters)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration', fontsize=9)
        ax.set_ylabel('Mean Energy', fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.tick_params(labelsize=8)

    # Hide unused axes
    for i in range(num_blocks, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(f'Per-Block Energy Profiles ({args.dataset}, {len(texts)} samples)\n{model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'per_block_energy_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 2: Energy change heatmap (blocks x iterations)
    # =========================================================================
    print("Generating energy change heatmap...")
    fig, axes = plt.subplots(1, 3, figsize=(22, 6))

    # Panel A: Absolute energy heatmap
    energy_matrix = np.full((num_blocks, max_iter + 1), np.nan)
    for (b, i), vals in all_energies.items():
        energy_matrix[b, i] = np.mean(vals)

    ax = axes[0]
    im = ax.imshow(energy_matrix, aspect='auto', cmap='RdYlBu_r', interpolation='nearest')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Block')
    ax.set_title('Mean Energy per (Block, Iteration)')
    ax.set_yticks(range(num_blocks))
    ax.set_xticks(range(max_iter + 1))
    plt.colorbar(im, ax=ax, shrink=0.8, label='Energy')

    # Panel B: Energy change (delta) heatmap
    delta_matrix = np.full((num_blocks, max_iter + 1), np.nan)
    for (b, i), vals in all_deltas.items():
        delta_matrix[b, i] = np.mean(vals)

    ax = axes[1]
    vmax = np.nanmax(np.abs(delta_matrix))
    im = ax.imshow(delta_matrix, aspect='auto', cmap='RdBu_r', interpolation='nearest',
                   vmin=-vmax, vmax=vmax)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Block')
    ax.set_title('Mean Energy Change (Delta)')
    ax.set_yticks(range(num_blocks))
    ax.set_xticks(range(max_iter + 1))
    plt.colorbar(im, ax=ax, shrink=0.8, label='Energy Delta')

    # Panel C: Relative energy change heatmap
    rel_delta_matrix = np.full((num_blocks, max_iter + 1), np.nan)
    for (b, i), vals in all_rel_deltas.items():
        rel_delta_matrix[b, i] = np.mean(vals)

    ax = axes[2]
    im = ax.imshow(rel_delta_matrix, aspect='auto', cmap='YlOrRd', interpolation='nearest')
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Block')
    ax.set_title('Mean Relative |Energy Change|')
    ax.set_yticks(range(num_blocks))
    ax.set_xticks(range(max_iter + 1))
    plt.colorbar(im, ax=ax, shrink=0.8, label='|Delta E| / |E|')

    fig.suptitle(f'Energy Landscape Heatmaps ({args.dataset})\n{model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'energy_heatmaps_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 3: Convergence analysis
    # =========================================================================
    print("Generating convergence analysis...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Panel A: Per-block convergence rate (bar chart)
    ax = axes[0, 0]
    block_conv_rates = []
    block_labels = []
    for block_idx in range(num_blocks):
        abs_deltas = []
        for iter_idx in range(1, max_iter + 1):
            key = (block_idx, iter_idx)
            if key in all_rel_deltas:
                abs_deltas.extend(all_rel_deltas[key])
        if abs_deltas:
            block_conv_rates.append(np.mean(abs_deltas))
        else:
            block_conv_rates.append(0)
        block_labels.append(f'B{block_idx}')

    colors_conv = plt.cm.RdYlGn_r(np.array(block_conv_rates) / (max(block_conv_rates) + 1e-8))
    bars = ax.bar(range(num_blocks), block_conv_rates, color=colors_conv, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels(block_labels)
    ax.set_ylabel('Mean Relative |Energy Change|')
    ax.set_title('Block Convergence Rate (lower = faster convergence)')
    ax.axhline(y=np.median(block_conv_rates), color='red', linestyle='--', alpha=0.7, label=f'Median={np.median(block_conv_rates):.4f}')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Energy trajectory across all blocks (global)
    ax = axes[0, 1]
    global_means = []
    global_stds = []
    global_labels = []
    for block_idx in range(num_blocks):
        for iter_idx in range(max_iter + 1):
            key = (block_idx, iter_idx)
            if key in all_energies:
                global_means.append(np.mean(all_energies[key]))
                global_stds.append(np.std(all_energies[key]))
                global_labels.append(f'B{block_idx}_I{iter_idx}')

    x = range(len(global_means))
    ax.plot(x, global_means, '-', linewidth=1.5, color='steelblue')
    ax.fill_between(x, np.array(global_means) - np.array(global_stds),
                     np.array(global_means) + np.array(global_stds), alpha=0.15, color='steelblue')
    # Mark block boundaries
    pos = 0
    for block_idx in range(num_blocks):
        n = sum(1 for k in all_energies if k[0] == block_idx)
        if pos > 0:
            ax.axvline(x=pos - 0.5, color='gray', linestyle=':', alpha=0.4)
        pos += n
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Mean Energy')
    ax.set_title('Global Energy Trajectory')
    ax.set_xticks(x[::2])
    ax.set_xticklabels(global_labels[::2], rotation=45, fontsize=6)
    ax.grid(True, alpha=0.3)

    # Panel C: Per-block energy overlaid
    ax = axes[1, 0]
    cmap = plt.cm.tab20(np.linspace(0, 1, num_blocks))
    for block_idx in range(num_blocks):
        iters_x = []
        means_y = []
        for iter_idx in range(max_iter + 1):
            key = (block_idx, iter_idx)
            if key in all_energies:
                iters_x.append(iter_idx)
                means_y.append(np.mean(all_energies[key]))
        if iters_x:
            ax.plot(iters_x, means_y, 'o-', label=f'B{block_idx}', color=cmap[block_idx],
                    linewidth=1.5, markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Energy')
    ax.set_title('All Blocks: Energy vs Iteration')
    ax.legend(fontsize=7, ncol=3, loc='best')
    ax.grid(True, alpha=0.3)

    # Panel D: Hidden state norm trajectory
    ax = axes[1, 1]
    norm_matrix = np.full((num_blocks, max_iter + 1), np.nan)
    for (b, i), vals in all_hidden_norms.items():
        norm_matrix[b, i] = np.mean(vals)

    for block_idx in range(num_blocks):
        iters_x = []
        norms_y = []
        for iter_idx in range(max_iter + 1):
            if not np.isnan(norm_matrix[block_idx, iter_idx]):
                iters_x.append(iter_idx)
                norms_y.append(norm_matrix[block_idx, iter_idx])
        if iters_x:
            ax.plot(iters_x, norms_y, 'o-', label=f'B{block_idx}', color=cmap[block_idx],
                    linewidth=1.5, markersize=4)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Mean Hidden State Norm')
    ax.set_title('Hidden State Norm vs Iteration')
    ax.legend(fontsize=7, ncol=3, loc='best')
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'Convergence Analysis ({args.dataset}, {len(texts)} samples)\n{model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'convergence_analysis_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 4: Per-token energy distribution (sampled)
    # =========================================================================
    print("Generating per-token energy distributions...")
    fig, axes = plt.subplots(3, 4, figsize=(20, 12), squeeze=False)

    for block_idx in range(min(num_blocks, 12)):
        ax = axes[block_idx // 4][block_idx % 4]

        # Get first and last iteration token energies
        first_key = (block_idx, 0)
        last_iter_idx = max(i for (b, i) in all_token_energies if b == block_idx) if any(b == block_idx for b, _ in all_token_energies) else 0
        last_key = (block_idx, last_iter_idx)

        if first_key in all_token_energies and len(all_token_energies[first_key]) > 0:
            # Aggregate token energies across samples
            first_vals = torch.cat([e.flatten() for e in all_token_energies[first_key][:20]]).numpy()
            ax.hist(first_vals, bins=50, alpha=0.5, density=True, color='blue', label=f'Iter 0')

        if last_key in all_token_energies and last_iter_idx > 0 and len(all_token_energies[last_key]) > 0:
            last_vals = torch.cat([e.flatten() for e in all_token_energies[last_key][:20]]).numpy()
            ax.hist(last_vals, bins=50, alpha=0.5, density=True, color='red', label=f'Iter {last_iter_idx}')

        # Also show middle iteration if available
        mid_iter = last_iter_idx // 2
        mid_key = (block_idx, mid_iter)
        if mid_iter > 0 and mid_key in all_token_energies and mid_key != last_key and len(all_token_energies[mid_key]) > 0:
            mid_vals = torch.cat([e.flatten() for e in all_token_energies[mid_key][:20]]).numpy()
            ax.hist(mid_vals, bins=50, alpha=0.4, density=True, color='green', label=f'Iter {mid_iter}')

        ax.set_title(f'Block {block_idx}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Energy per Token', fontsize=8)
        ax.set_ylabel('Density', fontsize=8)
        ax.legend(fontsize=7)
        ax.tick_params(labelsize=7)

    for i in range(num_blocks, 12):
        axes[i // 4][i % 4].set_visible(False)

    fig.suptitle(f'Per-Token Energy Distribution: First vs Last Iteration ({args.dataset})\n{model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'token_energy_dist_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 5: Per-sample trajectory variability
    # =========================================================================
    print("Generating per-sample trajectory variability...")
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Panel A: Overlay of individual sample trajectories (subsample for clarity)
    ax = axes[0]
    max_show = min(20, len(per_sample_trajectories))
    for s_idx in range(max_show):
        traj = per_sample_trajectories[s_idx]
        x_vals = list(range(len(traj)))
        y_vals = [e for _, _, e in traj]
        ax.plot(x_vals, y_vals, alpha=0.3, linewidth=0.8)

    # Overlay mean
    if global_means:
        ax.plot(range(len(global_means)), global_means, 'k-', linewidth=2.5, label='Mean', zorder=10)
    ax.set_xlabel('Global Step (Block_Iteration)')
    ax.set_ylabel('Energy')
    ax.set_title(f'Per-Sample Energy Trajectories (showing {max_show})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Panel B: Variance of energy across samples at each step
    ax = axes[1]
    step_vars = []
    step_labels = []
    for block_idx in range(num_blocks):
        for iter_idx in range(max_iter + 1):
            key = (block_idx, iter_idx)
            if key in all_energies and len(all_energies[key]) > 1:
                step_vars.append(np.var(all_energies[key]))
                step_labels.append(f'B{block_idx}_I{iter_idx}')

    ax.bar(range(len(step_vars)), step_vars, color='mediumpurple', alpha=0.7)
    ax.set_xlabel('Global Step')
    ax.set_ylabel('Variance of Energy Across Samples')
    ax.set_title('Cross-Sample Energy Variance per Step')
    ax.set_xticks(range(0, len(step_labels), 2))
    ax.set_xticklabels(step_labels[::2], rotation=45, fontsize=6)
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Sample Variability ({args.dataset}, {len(texts)} samples)\n{model_name}',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'sample_variability_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"Energy Landscape Summary ({args.dataset})")
    print(f"{'='*60}")
    print(f"\n{'Block':>6} | {'Iters':>5} | {'E(iter=0)':>10} | {'E(final)':>10} | {'Total Delta':>11} | {'Conv Rate':>10} | {'Category':>10}")
    print("-" * 75)

    median_rate = np.median(block_conv_rates)
    for block_idx in range(num_blocks):
        first_e = np.mean(all_energies.get((block_idx, 0), [0]))
        last_iter = max((i for (b, i) in all_energies if b == block_idx), default=0)
        last_e = np.mean(all_energies.get((block_idx, last_iter), [0]))
        total_delta = last_e - first_e
        rate = block_conv_rates[block_idx]

        if rate < median_rate * 0.7:
            cat = "FAST"
        elif rate > median_rate * 1.3:
            cat = "SLOW"
        else:
            cat = "MEDIUM"

        print(f"{block_idx:>6} | {num_iters[block_idx]:>5} | {first_e:>10.2f} | {last_e:>10.2f} | {total_delta:>+11.4f} | {rate:>10.4f} | {cat:>10}")

    print(f"\nMedian convergence rate: {median_rate:.4f}")
    print(f"Fast blocks (can halt early): {[i for i in range(num_blocks) if block_conv_rates[i] < median_rate * 0.7]}")
    print(f"Slow blocks (need full iters): {[i for i in range(num_blocks) if block_conv_rates[i] > median_rate * 1.3]}")

    print(f"\nAll figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
