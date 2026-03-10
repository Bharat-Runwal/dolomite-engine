"""Systematic per-block energy analysis for ablation models.

Extracts numerical energy trajectories per (block, iteration) across WikiText
samples, prints detailed analysis, and saves results as CSV + summary.

Usage:
    python analyze_ablation_energy.py [--model_dir DIR] [--output_dir DIR] [--num_samples N]
"""

import sys
import os
import torch
import json
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
                    energies.append((i, j, e.float().mean().item()))
    return energies


def analyze_model(model_path, tokenizer, texts):
    model_name = model_path.rstrip("/").split("/")[-1]
    print(f"  Loading {model_name}...")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.eval()

    nb = len(model.transformer.layer_iterations)
    ni = model.transformer.layer_iterations[0]

    all_e = defaultdict(list)
    for idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True)
        if input_ids.shape[1] < 10:
            continue
        energies = compute_energy_per_iteration(model, input_ids)
        for b, it, e in energies:
            all_e[(b, it)].append(e)

    del model
    torch.cuda.empty_cache()
    return model_name, nb, ni, dict(all_e)


def print_block_analysis(model_name, nb, ni, all_e):
    print(f"\n{'='*100}")
    print(f"{model_name}  ({nb} blocks x T={ni})")
    print(f"{'='*100}")

    block_summaries = []

    for block_idx in range(nb):
        iters = sorted([it for (b, it) in all_e if b == block_idx])
        if not iters:
            continue
        means = [np.mean(all_e[(block_idx, it)]) for it in iters]
        stds = [np.std(all_e[(block_idx, it)]) for it in iters]

        e_first = means[0]
        e_last = means[-1]
        total_delta = e_last - e_first

        # Early delta (iter 0->1)
        early_delta = means[1] - means[0] if len(means) > 1 else 0

        # Late delta (second half average change)
        mid = len(means) // 2
        if mid > 0 and mid < len(means) - 1:
            late_changes = [means[i+1] - means[i] for i in range(mid, len(means)-1)]
            late_avg_delta = np.mean(late_changes) if late_changes else 0
        else:
            late_avg_delta = 0

        # Monotonicity
        ups = sum(1 for i in range(1, len(means)) if means[i] > means[i-1])
        downs = sum(1 for i in range(1, len(means)) if means[i] < means[i-1])
        total_steps = ups + downs

        # Find min and max energy and when they occur
        min_idx = np.argmin(means)
        max_idx = np.argmax(means)
        e_min = means[min_idx]
        e_max = means[max_idx]
        iter_at_min = iters[min_idx]
        iter_at_max = iters[max_idx]

        # Classify the trajectory shape
        if total_delta > 2.0 and ups > downs:
            shape = "RISING"
        elif total_delta < -2.0 and downs > ups:
            shape = "FALLING"
        elif early_delta < -2.0 and total_delta > -2.0:
            shape = "DIP-RECOVER"
        elif early_delta > 2.0 and total_delta < 2.0:
            shape = "JUMP-SETTLE"
        elif abs(total_delta) < 2.0:
            shape = "FLAT"
        else:
            shape = "OSCILLATING"

        # Direction of each phase
        n_phases = min(4, len(means) - 1)
        phase_size = max(1, (len(means) - 1) // n_phases)
        phases = []
        for p in range(n_phases):
            start = p * phase_size
            end = min(start + phase_size, len(means) - 1)
            if start < end:
                phase_delta = means[end] - means[start]
                phases.append(("+" if phase_delta > 0.5 else ("-" if phase_delta < -0.5 else "~")))

        phase_str = "".join(phases)

        # Print trajectory at key iterations
        if ni <= 9:
            sample_iters = list(range(ni + 1))
        else:
            sample_iters = [0, 1, 2, ni//4, ni//2, 3*ni//4, ni]
        sample_iters = sorted(set(it for it in sample_iters if (block_idx, it) in all_e))
        traj_str = " -> ".join([f"{np.mean(all_e[(block_idx, it)]):.1f}" for it in sample_iters])
        iter_labels = ",".join([str(it) for it in sample_iters])

        print(f"\n  Block {block_idx}: [{shape:>12}] total_delta={total_delta:+.2f}  ups={ups} downs={downs}  phases=[{phase_str}]")
        print(f"    E(0)={e_first:.2f}  E({ni})={e_last:.2f}  E_min={e_min:.2f}@iter{iter_at_min}  E_max={e_max:.2f}@iter{iter_at_max}")
        print(f"    early_delta(0->1)={early_delta:+.2f}  late_avg_delta={late_avg_delta:+.2f}")
        print(f"    trajectory (iters {iter_labels}): {traj_str}")

        block_summaries.append({
            'model': model_name, 'num_blocks': nb, 'T': ni, 'block': block_idx,
            'shape': shape, 'E_first': e_first, 'E_last': e_last,
            'total_delta': total_delta, 'early_delta': early_delta,
            'late_avg_delta': late_avg_delta, 'ups': ups, 'downs': downs,
            'E_min': e_min, 'iter_at_min': iter_at_min,
            'E_max': e_max, 'iter_at_max': iter_at_max,
            'phases': phase_str,
        })

    return block_summaries


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

    # All models
    all_models = sorted(os.listdir(args.model_dir))
    all_summaries = []

    for model_name in all_models:
        model_path = os.path.join(args.model_dir, model_name)
        if not os.path.isdir(model_path) or not os.path.exists(os.path.join(model_path, 'config.json')):
            continue
        name, nb, ni, all_e = analyze_model(model_path, tokenizer, texts)
        summaries = print_block_analysis(name, nb, ni, all_e)
        all_summaries.extend(summaries)

        # Save raw energy trajectories per model
        raw_out = {}
        for (b, it), vals in all_e.items():
            raw_out[f"block{b}_iter{it}"] = {"mean": float(np.mean(vals)), "std": float(np.std(vals)), "n": len(vals)}
        with open(os.path.join(args.output_dir, f"{name}_energy_raw.json"), 'w') as f:
            json.dump(raw_out, f, indent=2)

    # Save summary CSV
    if all_summaries:
        csv_path = os.path.join(args.output_dir, "block_analysis_summary.csv")
        header = "model,num_blocks,T,block,shape,E_first,E_last,total_delta,early_delta,late_avg_delta,ups,downs,E_min,iter_at_min,E_max,iter_at_max,phases"
        rows = [header]
        for s in all_summaries:
            row = f"{s['model']},{s['num_blocks']},{s['T']},{s['block']},{s['shape']},{s['E_first']:.2f},{s['E_last']:.2f},{s['total_delta']:.2f},{s['early_delta']:.2f},{s['late_avg_delta']:.2f},{s['ups']},{s['downs']},{s['E_min']:.2f},{s['iter_at_min']},{s['E_max']:.2f},{s['iter_at_max']},{s['phases']}"
            rows.append(row)
        with open(csv_path, 'w') as f:
            f.write("\n".join(rows) + "\n")
        print(f"\nSummary CSV saved to: {csv_path}")

    # Print cross-model comparison table
    print(f"\n\n{'='*120}")
    print("CROSS-MODEL SHAPE SUMMARY")
    print(f"{'='*120}")
    print(f"{'Model':>35} | {'Block 0':>14} | {'Block 1':>14} | {'Block 2':>14} | {'Block 3':>14}")
    print("-" * 100)
    for model_name in sorted(set(s['model'] for s in all_summaries),
                              key=lambda x: (int(x.split('_')[1]), int(x.split('_all_')[1].split('_')[0]))):
        blocks = {s['block']: s for s in all_summaries if s['model'] == model_name}
        cols = []
        for b in range(4):
            if b in blocks:
                s = blocks[b]
                cols.append(f"{s['shape']:>12} {s['total_delta']:+.1f}")
            else:
                cols.append(f"{'':>14}")
        print(f"{model_name:>35} | {cols[0]} | {cols[1]} | {cols[2]} | {cols[3]}")

    print(f"\nAll results saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
