"""Cross-step energy change visualization for EGPT_5b_9_9_9_9_9 models.

Plots energy change per block (y-axis) vs training steps (x-axis).
One line per block across the 7 checkpoints (30k-90k).

Usage:
    python visualize_energy_vs_steps.py [--num_samples N] [--output_dir DIR]
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

        energies = []
        for i, num_iter in enumerate(base_model.layer_iterations):
            block = base_model.h[i]
            has_energy = hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention"

            for j in range(num_iter):
                hidden_states = block(
                    hidden_states, past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)

                if has_energy:
                    e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
                    energies.append((i, j, e.float().mean().item()))

    return energies


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_samples", type=int, default=30)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "hellaswag", "lambada", "arc"])
    parser.add_argument("--model_prefix", default="EGPT_5b_9_9_9_9_9",
                        help="Model name prefix (e.g. EGPT_5b_9_9_9_9_9 or EGPT_6b_9_9_9_9_9_9)")
    args = parser.parse_args()

    MODEL_PREFIX = args.model_prefix
    if args.output_dir is None:
        args.output_dir = f"tools/bs/energy_landscape/{MODEL_PREFIX}"
    os.makedirs(args.output_dir, exist_ok=True)

    BASE_UNSHARDED = "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_test"
    STEPS = [30000, 40000, 50000, 60000, 70000, 80000, 90000]
    STEP_LABELS = ["30k", "40k", "50k", "60k", "70k", "80k", "90k"]

    tokenizer = AutoTokenizer.from_pretrained('/proj/dmfexp/energy-gpt/data/granite-4.0-tiktoken')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load texts once
    print(f"Loading {args.dataset} data...")
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:args.num_samples]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")
    print(f"Using {len(texts)} samples")

    # For each step, compute per-block energy change (delta across iterations within each block)
    # energy_change_per_block[step_label] = {block_idx: mean_total_delta}
    # Also track absolute energy at first/last iter per block
    energy_first_per_block = {}   # step_label -> {block_idx: mean_energy_at_iter0}
    energy_last_per_block = {}    # step_label -> {block_idx: mean_energy_at_last_iter}
    energy_delta_per_block = {}   # step_label -> {block_idx: mean(last - first)}

    for step, label in zip(STEPS, STEP_LABELS):
        model_name = f"{MODEL_PREFIX}_{label}"
        model_path = os.path.join(BASE_UNSHARDED, model_name)

        if not os.path.exists(model_path):
            print(f"  SKIP (not found): {model_path}")
            continue

        print(f"\nLoading {model_name}...")
        model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
        model.config.use_cache = False
        model.eval()
        model.to(device)

        # Collect energies across samples
        all_energies = defaultdict(list)  # (block, iter) -> [mean_energy, ...]

        for idx, text in enumerate(texts):
            input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
            if input_ids.shape[1] < 10:
                continue
            energies = compute_energy_per_iteration(model, input_ids)
            for block_idx, iter_idx, mean_e in energies:
                all_energies[(block_idx, iter_idx)].append(mean_e)
            if (idx + 1) % 10 == 0:
                print(f"  Processed {idx+1}/{len(texts)} samples")

        num_blocks = max(k[0] for k in all_energies.keys()) + 1

        first_e = {}
        last_e = {}
        delta_e = {}
        for block_idx in range(num_blocks):
            # First iteration energy
            if (block_idx, 0) in all_energies:
                first_e[block_idx] = np.mean(all_energies[(block_idx, 0)])
            # Last iteration energy
            last_iter = max((it for (b, it) in all_energies if b == block_idx), default=0)
            if (block_idx, last_iter) in all_energies:
                last_e[block_idx] = np.mean(all_energies[(block_idx, last_iter)])
            # Delta
            if block_idx in first_e and block_idx in last_e:
                delta_e[block_idx] = last_e[block_idx] - first_e[block_idx]

        energy_first_per_block[label] = first_e
        energy_last_per_block[label] = last_e
        energy_delta_per_block[label] = delta_e

        # Free GPU memory
        del model
        torch.cuda.empty_cache()

        print(f"  {model_name}: blocks={num_blocks}, deltas={delta_e}")

    # =========================================================================
    # FIGURE: Energy change per block vs training steps
    # =========================================================================
    print("\nGenerating cross-step energy change figure...")

    # Determine all block indices
    all_blocks = set()
    for deltas in energy_delta_per_block.values():
        all_blocks.update(deltas.keys())
    all_blocks = sorted(all_blocks)

    available_steps = [l for l in STEP_LABELS if l in energy_delta_per_block]
    step_nums = [int(l.replace('k', '')) for l in available_steps]

    fig, axes = plt.subplots(1, 3, figsize=(24, 7))
    cmap = plt.cm.tab10(np.linspace(0, 1, max(len(all_blocks), 1)))

    # Panel A: Energy change (delta) per block vs steps
    ax = axes[0]
    for bi, block_idx in enumerate(all_blocks):
        y_vals = []
        x_vals = []
        for label, step_n in zip(available_steps, step_nums):
            if block_idx in energy_delta_per_block.get(label, {}):
                x_vals.append(step_n)
                y_vals.append(energy_delta_per_block[label][block_idx])
        if x_vals:
            ax.plot(x_vals, y_vals, 'o-', label=f'Block {block_idx}', color=cmap[bi],
                    linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy Change (last iter - first iter)', fontsize=12)
    ax.set_title('Energy Change per Block vs Training Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.axhline(y=0, color='black', linestyle=':', alpha=0.3)

    # Panel B: Absolute energy at first iteration per block vs steps
    ax = axes[1]
    for bi, block_idx in enumerate(all_blocks):
        y_vals = []
        x_vals = []
        for label, step_n in zip(available_steps, step_nums):
            if block_idx in energy_first_per_block.get(label, {}):
                x_vals.append(step_n)
                y_vals.append(energy_first_per_block[label][block_idx])
        if x_vals:
            ax.plot(x_vals, y_vals, 'o-', label=f'Block {block_idx}', color=cmap[bi],
                    linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy at Iteration 0', fontsize=12)
    ax.set_title('Initial Energy per Block vs Training Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Panel C: Absolute energy at last iteration per block vs steps
    ax = axes[2]
    for bi, block_idx in enumerate(all_blocks):
        y_vals = []
        x_vals = []
        for label, step_n in zip(available_steps, step_nums):
            if block_idx in energy_last_per_block.get(label, {}):
                x_vals.append(step_n)
                y_vals.append(energy_last_per_block[label][block_idx])
        if x_vals:
            ax.plot(x_vals, y_vals, 'o-', label=f'Block {block_idx}', color=cmap[bi],
                    linewidth=2, markersize=6)
    ax.set_xlabel('Training Steps (k)', fontsize=12)
    ax.set_ylabel('Energy at Last Iteration', fontsize=12)
    ax.set_title('Final Energy per Block vs Training Steps', fontsize=14, fontweight='bold')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(f'{MODEL_PREFIX}: Energy Landscape Across Training Steps\n'
                 f'({args.dataset}, {len(texts)} samples)',
                 fontsize=16, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'energy_change_vs_steps_{MODEL_PREFIX}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {path}")


if __name__ == '__main__':
    main()
