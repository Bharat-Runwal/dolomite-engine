"""Helmholtz decomposition analysis of energy block dynamics.

Measures the alignment between the actual update (Δh) and the energy gradient (∇E)
at each (block, iteration) to decompose the dynamics into:
  - Gradient component (cos θ ≈ -1): energy descent
  - Curl component (cos θ ≈ 0): feature rotation/transformation
  - Anti-gradient (cos θ ≈ +1): energy ascent

IMPORTANT CAVEAT (per Ben Hoover):
  cos θ = g^T W_P g / (|W_P g| |g|)
  Decompose W_P = S + A (S symmetric, A antisymmetric).
  Then g^T W_P g = g^T S g + g^T A g = g^T S g  (since x^T A x = 0 for all x).
  So cos θ is BLIND to the antisymmetric (curl) component of W_P.
  cos θ ≈ 0 can occur simply because ||A||_F >> ||S||_F.

  The CORRECT measure of curl vs descent is the data-independent weight ratio:
      ||A||_F / ||S||_F   where S = (W+W^T)/2, A = (W-W^T)/2
  This is now printed as an additional table after the main Helmholtz summary.

Usage:
    python analyze_helmholtz.py <model_path> [--dataset wikitext|gsm8k] [--num_samples 20]
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


def load_texts(dataset_name, num_samples):
    if dataset_name == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        return [t for t in dataset["text"] if len(t.strip()) > 100][:num_samples]
    elif dataset_name == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main", split="test")
        exemplars = list(dataset)[:5]
        test_examples = list(dataset)[5:5 + num_samples]
        prompt_prefix = ""
        for ex in exemplars:
            prompt_prefix += f"Question: {ex['question']}\nAnswer: {ex['answer']}\n\n"
        return [prompt_prefix + f"Question: {ex['question']}\nAnswer: {ex['answer']}" for ex in test_examples]
    elif dataset_name == "hellaswag":
        dataset = load_dataset("Rowan/hellaswag", split="validation")
        return [f"{row['ctx']} {row['endings'][int(row['label'])]}" for row in dataset][:num_samples]
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def compute_helmholtz(model, input_ids, apply_ln_jacobian=False):
    """Compute gradient alignment for each (block, iteration).
    Returns list of (block_idx, iter_idx, cos_theta, delta_norm, grad_norm, energy_delta)
    """
    base_model = model.transformer

    hidden_states = base_model.wte(input_ids).detach().requires_grad_(False)
    if base_model.m_emb is not None:
        hidden_states = hidden_states * base_model.m_emb

    rope_cos_sin = None
    if base_model.position_embedding_type == "rope":
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        rope_cos_sin = base_model._get_rope_cos_sin(
            key_length=input_ids.shape[1], position_ids=position_ids, dtype=hidden_states.dtype)

    results = []

    for i, num_iter in enumerate(base_model.layer_iterations):
        block = base_model.h[i]
        has_proj = hasattr(block, 'proj_type') and block.proj_type != 'none'
        has_energy_fn = hasattr(block, 'energy_per_token') and hasattr(block, 'ffwd') and hasattr(block.ffwd, 'energy_per_token')

        if not has_proj:
            # Skip blocks without a projection (standard GPT, additive residual)
            for j in range(num_iter):
                hidden_states = block(
                    hidden_states.detach(), past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)
            continue

        is_dual = getattr(block, 'proj_type', '') == 'dual_unconstrained'

        for j in range(num_iter):
            h_in = hidden_states.detach().clone()

            # Compute the causal-correct gradient: what the forward pass treats as ∇E
            # (attn_out + scale_ff * ffwd_out, before projection)
            # This avoids the autograd key-role bias from future tokens attending to k_t.
            grad_E = block.forward_gradient(h_in, rope_cos_sin=rope_cos_sin, apply_ln_jacobian=apply_ln_jacobian)

            # Run the block to get the actual update (applies projection on top)
            with torch.no_grad():
                h_out = block(
                    h_in, past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)

            # Compute delta h (the actual update)
            delta_h = (h_out - h_in.detach()).float()
            grad_E_float = grad_E.float()

            # Flatten to vectors for dot product (per token, then average)
            dot = (delta_h * grad_E_float).sum(dim=-1)
            delta_norm = delta_h.norm(dim=-1)
            grad_norm = grad_E_float.norm(dim=-1)

            denom = (delta_norm * grad_norm).clamp(min=1e-8)
            cos_theta = dot / denom

            # Gradient and curl magnitudes
            grad_component = dot / grad_norm.clamp(min=1e-8)
            curl_component = torch.sqrt((delta_norm ** 2 - grad_component ** 2).clamp(min=0))

            # Compute energy delta (mean per-token, not sum) -- only for energy MLP models
            energy_delta = 0.0
            if has_energy_fn:
                with torch.no_grad():
                    energy_before = block.energy_per_token(h_in, rope_cos_sin=rope_cos_sin).mean()
                    energy_after = block.energy_per_token(h_out, rope_cos_sin=rope_cos_sin).mean()
                    energy_delta = (energy_after - energy_before).item()

            # Excess descent: dimension-independent z-score vs random projection
            # For random W with entries ~N(0, sigma^2), fixed v:
            #   E[v^T W v] = 0,  std[v^T W v] = sigma * ||v||^2
            # We use sigma = ||W||_F / d (entry-level std from Frobenius norm).
            # excess = (g^T W g) / (sigma_W * ||g||^2)
            # For random W, excess ~ N(0, 1). Positive = descent on g.
            # Note: identity gives excess ≈ sqrt(d) because ||I||_F/d = 1/sqrt(d).
            d = grad_E_float.shape[-1]
            g_norm_sq = (grad_E_float ** 2).sum(dim=-1)  # ||g||^2 per token
            gWg = -dot  # g^T W g per token; positive = descent on g

            # Get W Frobenius norm for normalization
            pt = getattr(block, 'proj_type', 'unconstrained')
            if pt == "dual_unconstrained":
                w_norm = (block.proj_attn.weight.data.float().norm() + block.proj_mlp.weight.data.float().norm()) / 2
            elif pt == "mlp_only":
                w_norm = block.proj_mlp.weight.data.float().norm()
            elif hasattr(block, 'proj') and hasattr(block.proj, 'weight'):
                w_norm = block.proj.weight.data.float().norm()
            elif pt == "identity":
                w_norm = torch.tensor(d ** 0.5)  # ||I||_F = sqrt(d)
            elif pt == "pos_scalar":
                # pos_scalar: Pi(g) = weight^2 * g, so effective W = weight^2 * I
                # ||W||_F = weight^2 * ||I||_F = weight^2 * sqrt(d)
                weight_sq = block.proj.weight.data.float() ** 2
                w_norm = weight_sq * (d ** 0.5)
            else:
                w_norm = torch.tensor(d ** 0.5)  # default to identity-scale

            sigma_W = w_norm / d  # entry-level std
            random_std = sigma_W * g_norm_sq  # std of v^T W_random v (corrected: no sqrt(d))
            excess_descent = gWg / random_std.clamp(min=1e-12)  # z-score vs random

            result_entry = {
                'block': i,
                'iter': j,
                'cos_theta_mean': cos_theta.mean().item(),
                'cos_theta_std': cos_theta.std().item(),
                'cos_theta_median': cos_theta.median().item(),
                'delta_norm': delta_norm.mean().item(),
                'grad_norm': grad_norm.mean().item(),
                'grad_component': grad_component.mean().item(),
                'curl_component': curl_component.mean().item(),
                'curl_to_grad_ratio': (curl_component / grad_component.abs().clamp(min=1e-8)).mean().item(),
                'energy_delta': energy_delta,
                'excess_descent_mean': excess_descent.mean().item(),
                'excess_descent_std': excess_descent.std().item(),
            }

            # For dual projection: decompose attn and MLP components separately
            if is_dual and hasattr(block, 'forward_gradient_separate'):
                attn_grad, mlp_grad = block.forward_gradient_separate(h_in, rope_cos_sin=rope_cos_sin)
                with torch.no_grad():
                    proj_attn_out = block.proj_attn(attn_grad)
                    proj_mlp_out = block.proj_mlp(mlp_grad)

                attn_grad_f = attn_grad.float()
                mlp_grad_f = mlp_grad.float()
                proj_attn_f = proj_attn_out.float()
                proj_mlp_f = proj_mlp_out.float()

                # cos theta between proj_attn(a) and a
                dot_attn = (proj_attn_f * attn_grad_f).sum(dim=-1)
                norm_pa = proj_attn_f.norm(dim=-1)
                norm_a = attn_grad_f.norm(dim=-1)
                cos_attn = dot_attn / (norm_pa * norm_a).clamp(min=1e-8)

                # cos theta between proj_mlp(m) and m
                dot_mlp = (proj_mlp_f * mlp_grad_f).sum(dim=-1)
                norm_pm = proj_mlp_f.norm(dim=-1)
                norm_m = mlp_grad_f.norm(dim=-1)
                cos_mlp = dot_mlp / (norm_pm * norm_m).clamp(min=1e-8)

                # curl/grad for each
                grad_comp_attn = dot_attn / norm_a.clamp(min=1e-8)
                curl_comp_attn = torch.sqrt((norm_pa ** 2 - grad_comp_attn ** 2).clamp(min=0))
                grad_comp_mlp = dot_mlp / norm_m.clamp(min=1e-8)
                curl_comp_mlp = torch.sqrt((norm_pm ** 2 - grad_comp_mlp ** 2).clamp(min=0))

                result_entry.update({
                    'attn_cos_theta': cos_attn.mean().item(),
                    'mlp_cos_theta': cos_mlp.mean().item(),
                    'attn_grad_norm': norm_a.mean().item(),
                    'mlp_grad_norm': norm_m.mean().item(),
                    'attn_proj_norm': norm_pa.mean().item(),
                    'mlp_proj_norm': norm_pm.mean().item(),
                    'attn_curl_grad': (curl_comp_attn / grad_comp_attn.abs().clamp(min=1e-8)).mean().item(),
                    'mlp_curl_grad': (curl_comp_mlp / grad_comp_mlp.abs().clamp(min=1e-8)).mean().item(),
                })

            results.append(result_entry)
            hidden_states = h_out.detach()

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "gsm8k", "hellaswag"])
    parser.add_argument("--apply_ln_jacobian", action="store_true",
                        help="Multiply gradient by J_RMSNorm^T to get dE/dh in pre-norm space")
    args = parser.parse_args()

    if args.output_dir is None:
        model_name = args.model_path.rstrip("/").split("/")[-1]
        args.output_dir = f"workshop_results/helmholtz_FIXCAUSAL_{model_name}_{args.dataset}"
    os.makedirs(args.output_dir, exist_ok=True)

    model_name = args.model_path.rstrip("/").split("/")[-1]

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained('/proj/checkpoints/dmf-lh-checkpoints/tokenizers/granite-4.0-tiktoken')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.eval()

    print(f"Loading {args.dataset} data...")
    texts = load_texts(args.dataset, args.num_samples)
    print(f"Using {len(texts)} samples")

    # Collect results across samples
    all_results = defaultdict(list)  # (block, iter) -> [result_dict, ...]

    for idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).cuda()
        if input_ids.shape[1] < 10:
            continue

        results = compute_helmholtz(model, input_ids, apply_ln_jacobian=args.apply_ln_jacobian)
        for r in results:
            all_results[(r['block'], r['iter'])].append(r)

        if (idx + 1) % 5 == 0:
            print(f"  Processed {idx + 1}/{len(texts)} samples")

    # Aggregate
    blocks = sorted(set(b for b, _ in all_results.keys()))
    num_blocks = len(blocks)

    # =========================================================================
    # FIGURE 1: Cos theta per (block, iteration)
    # =========================================================================
    print("Generating cos theta plots...")
    if num_blocks == 0:
        print("No blocks with projections found. Skipping plots.")
        return
    cols = min(4, num_blocks)
    rows = (num_blocks + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows), squeeze=False)

    for idx, block_idx in enumerate(blocks):
        ax = axes[idx // cols][idx % cols]
        iters = sorted([it for (b, it) in all_results if b == block_idx])
        means = [np.mean([r['cos_theta_mean'] for r in all_results[(block_idx, it)]]) for it in iters]
        stds = [np.std([r['cos_theta_mean'] for r in all_results[(block_idx, it)]]) for it in iters]
        means = np.array(means)
        stds = np.array(stds)

        ax.plot(iters, means, 'o-', color='steelblue', linewidth=2, markersize=6)
        ax.fill_between(iters, means - stds, means + stds, alpha=0.2, color='steelblue')
        ax.axhline(y=0, color='gray', linestyle=':', alpha=0.5)
        ax.axhline(y=-1, color='green', linestyle='--', alpha=0.3, label='Pure descent')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.3, label='Pure ascent')

        num_iters_cfg = model.transformer.layer_iterations[block_idx]
        ax.set_title(f'Block {block_idx} ({num_iters_cfg} iters)', fontsize=11, fontweight='bold')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('cos θ (Δh · ∇E)')
        ax.set_ylim(-1.5, 1.5)
        ax.grid(True, alpha=0.3)
        if idx == 0:
            ax.legend(fontsize=7)

    for i in range(num_blocks, rows * cols):
        axes[i // cols][i % cols].set_visible(False)

    fig.suptitle(f'Gradient Alignment: cos θ(Δh, ∇E) per Block ({args.dataset})\n'
                 f'{model_name}\n'
                 f'cos θ ≈ -1: descent | cos θ ≈ 0: rotation | cos θ ≈ +1: ascent',
                 fontsize=12, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'cos_theta_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 2: Curl-to-gradient ratio per block (bar chart)
    # =========================================================================
    print("Generating curl/gradient decomposition...")
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    # Panel A: Mean cos_theta per block
    ax = axes[0]
    block_cos = []
    for block_idx in blocks:
        all_cos = []
        for it in sorted([i for b, i in all_results if b == block_idx]):
            all_cos.extend([r['cos_theta_mean'] for r in all_results[(block_idx, it)]])
        block_cos.append(np.mean(all_cos))
    colors = ['green' if c < -0.3 else ('red' if c > 0.3 else 'gold') for c in block_cos]
    ax.bar(range(num_blocks), block_cos, color=colors, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels([f'B{b}' for b in blocks])
    ax.set_ylabel('Mean cos θ')
    ax.set_title('Gradient Alignment per Block')
    ax.axhline(y=0, color='gray', linestyle=':')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel B: Curl-to-gradient ratio per block
    ax = axes[1]
    block_cgr = []
    for block_idx in blocks:
        all_cgr = []
        for it in sorted([i for b, i in all_results if b == block_idx]):
            all_cgr.extend([r['curl_to_grad_ratio'] for r in all_results[(block_idx, it)]])
        block_cgr.append(np.mean(all_cgr))
    ax.bar(range(num_blocks), block_cgr, color='mediumpurple', edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(num_blocks))
    ax.set_xticklabels([f'B{b}' for b in blocks])
    ax.set_ylabel('Curl / |Gradient| ratio')
    ax.set_title('Curl-to-Gradient Ratio per Block\n(higher = more rotation, less descent)')
    ax.grid(True, alpha=0.3, axis='y')

    # Panel C: Gradient vs curl component magnitudes
    ax = axes[2]
    block_grad_mag = []
    block_curl_mag = []
    for block_idx in blocks:
        all_grad = []
        all_curl = []
        for it in sorted([i for b, i in all_results if b == block_idx]):
            all_grad.extend([abs(r['grad_component']) for r in all_results[(block_idx, it)]])
            all_curl.extend([r['curl_component'] for r in all_results[(block_idx, it)]])
        block_grad_mag.append(np.mean(all_grad))
        block_curl_mag.append(np.mean(all_curl))
    x = np.arange(num_blocks)
    w = 0.35
    ax.bar(x - w / 2, block_grad_mag, w, label='|Gradient component|', color='steelblue', edgecolor='black', linewidth=0.5)
    ax.bar(x + w / 2, block_curl_mag, w, label='Curl component', color='coral', edgecolor='black', linewidth=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f'B{b}' for b in blocks])
    ax.set_ylabel('Magnitude')
    ax.set_title('Gradient vs Curl Decomposition')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')

    fig.suptitle(f'Helmholtz Decomposition ({args.dataset})\n{model_name}',
                 fontsize=13, fontweight='bold')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'helmholtz_decomp_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # FIGURE 3: Cos theta heatmap (block x iteration)
    # =========================================================================
    print("Generating cos theta heatmap...")
    max_iter = max(it for _, it in all_results.keys())
    cos_matrix = np.full((num_blocks, max_iter + 1), np.nan)
    for (b, it), rs in all_results.items():
        b_idx = blocks.index(b)
        cos_matrix[b_idx, it] = np.mean([r['cos_theta_mean'] for r in rs])

    fig, ax = plt.subplots(1, 1, figsize=(12, 5))
    im = ax.imshow(cos_matrix, aspect='auto', cmap='RdYlGn_r', interpolation='nearest',
                   vmin=-1, vmax=1)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Block')
    ax.set_yticks(range(num_blocks))
    ax.set_yticklabels([f'B{b}' for b in blocks])
    ax.set_title(f'Gradient Alignment Heatmap: cos θ(Δh, ∇E)\n'
                 f'{model_name} | {args.dataset}\n'
                 f'Green = descent (-1) | Yellow = rotation (0) | Red = ascent (+1)')
    plt.colorbar(im, ax=ax, label='cos θ')
    plt.tight_layout()
    path = os.path.join(args.output_dir, f'cos_theta_heatmap_{args.dataset}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")

    # =========================================================================
    # Print summary
    # =========================================================================
    print(f"\n{'='*100}")
    print(f"Helmholtz Decomposition Summary ({args.dataset})")
    print(f"  Excess = (g^T W g) / (sigma_W * ||g||^2)  [z-score vs random W; O(1) for any d]")
    print(f"  Excess >> 0: descent on partial grad g | ~0: random-like | << 0: anti-descent on g")
    print(f"  NOTE: Identity gives max excess (perfect descent on g) yet energy ASCENDS --")
    print(f"        because the partial gradient g is misaligned with the true energy gradient.")
    print(f"{'='*100}")
    print(f"\n{'Block':>6} | {'Iters':>5} | {'cos θ':>8} | {'|Grad|':>8} | {'|Curl|':>8} | {'Curl/Grad':>10} | {'E delta':>10} | {'Excess':>8} | {'Role':>12}")
    print("-" * 100)

    for block_idx in blocks:
        iters_for_block = sorted([it for b, it in all_results if b == block_idx])
        all_cos = [r['cos_theta_mean'] for it in iters_for_block for r in all_results[(block_idx, it)]]
        all_grad = [abs(r['grad_component']) for it in iters_for_block for r in all_results[(block_idx, it)]]
        all_curl = [r['curl_component'] for it in iters_for_block for r in all_results[(block_idx, it)]]
        all_cgr = [r['curl_to_grad_ratio'] for it in iters_for_block for r in all_results[(block_idx, it)]]
        all_ed = [r['energy_delta'] for it in iters_for_block for r in all_results[(block_idx, it)]]
        all_excess = [r['excess_descent_mean'] for it in iters_for_block for r in all_results[(block_idx, it)]]

        mean_cos = np.mean(all_cos)
        mean_grad = np.mean(all_grad)
        mean_curl = np.mean(all_curl)
        mean_cgr = np.mean(all_cgr)
        mean_ed = np.mean(all_ed)
        mean_excess = np.mean(all_excess)

        # Role classification based on excess descent (dimension-independent)
        # excess >> 0: descent on g (but g is misaligned, so energy may go UP)
        # excess ~ 0: random-like; excess << 0: anti-descent on g
        if abs(mean_excess) < 2.0:
            role = "RANDOM-LIKE"
        elif mean_excess > 0:
            role = f"g-DESCENT"
        else:
            role = f"g-ANTI-DESC"

        num_iters_cfg = model.transformer.layer_iterations[block_idx]
        print(f"{block_idx:>6} | {num_iters_cfg:>5} | {mean_cos:>+8.4f} | {mean_grad:>8.4f} | {mean_curl:>8.4f} | {mean_cgr:>10.4f} | {mean_ed:>+10.2f} | {mean_excess:>+8.2f} | {role:>12}")

    # Print dual projection decomposition if available
    has_dual = any('attn_cos_theta' in r for rs in all_results.values() for r in rs)
    if has_dual:
        print(f"\n{'='*100}")
        print(f"Dual Projection Decomposition: Attn vs MLP ({args.dataset})")
        print(f"{'='*100}")
        print(f"\n{'Block':>6} | {'Iters':>5} | {'Attn cos θ':>10} | {'MLP cos θ':>10} | {'Attn C/G':>9} | {'MLP C/G':>9} | {'|Attn grad|':>11} | {'|MLP grad|':>10} | {'|Attn proj|':>11} | {'|MLP proj|':>10}")
        print("-" * 115)

        for block_idx in blocks:
            iters_for_block = sorted([it for b, it in all_results if b == block_idx])
            all_r = [r for it in iters_for_block for r in all_results[(block_idx, it)]]

            if not all_r or 'attn_cos_theta' not in all_r[0]:
                continue

            attn_cos = np.mean([r['attn_cos_theta'] for r in all_r])
            mlp_cos = np.mean([r['mlp_cos_theta'] for r in all_r])
            attn_cg = np.mean([r['attn_curl_grad'] for r in all_r])
            mlp_cg = np.mean([r['mlp_curl_grad'] for r in all_r])
            attn_gn = np.mean([r['attn_grad_norm'] for r in all_r])
            mlp_gn = np.mean([r['mlp_grad_norm'] for r in all_r])
            attn_pn = np.mean([r['attn_proj_norm'] for r in all_r])
            mlp_pn = np.mean([r['mlp_proj_norm'] for r in all_r])

            num_iters_cfg = model.transformer.layer_iterations[block_idx]
            print(f"{block_idx:>6} | {num_iters_cfg:>5} | {attn_cos:>+10.4f} | {mlp_cos:>+10.4f} | {attn_cg:>9.2f} | {mlp_cg:>9.2f} | {attn_gn:>11.4f} | {mlp_gn:>10.4f} | {attn_pn:>11.4f} | {mlp_pn:>10.4f}")

    # =========================================================================
    # Weight-space decomposition: ||A||_F / ||S||_F for projection matrices
    # This is the CORRECT measure of rotationality per Ben Hoover's observation:
    #   cos θ = g^T W_P g / (|W_P g| |g|) = g^T S g / (|W_P g| |g|)  (g^T A g = 0 always)
    # So cos θ is BLIND to the antisymmetric component.
    # The proper diagnostic is the data-independent Frobenius ratio on the weights.
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"Weight-space Sym/Antisym Decomposition  W_P = S + A")
    print(f"  S = (W + W^T)/2  (symmetric, gradient descent component)")
    print(f"  A = (W - W^T)/2  (antisymmetric, curl/rotation component)")
    print(f"  Ratio ||A||_F / ||S||_F: >1 means rotation dominates")
    print(f"NOTE: cos θ is blind to A (x^T A x = 0 for all x), so this is the")
    print(f"      correct measure of curl vs descent in the projection.")
    print(f"{'='*80}")

    def _sym_antisym_ratio(W):
        """W: [out, in] weight matrix. Returns ||A||_F/||S||_F and norms."""
        W = W.float()
        S = (W + W.T) / 2
        A = (W - W.T) / 2
        norm_S = S.norm().item()
        norm_A = A.norm().item()
        norm_W = W.norm().item()
        ratio = norm_A / (norm_S + 1e-8)
        return ratio, norm_S, norm_A, norm_W

    header_printed = False
    for block_idx in blocks:
        block = model.transformer.h[block_idx]
        if not hasattr(block, 'proj_type'):
            continue

        pt = block.proj_type
        rows_to_print = []

        if pt == "dual_unconstrained":
            W_attn = block.proj_attn.weight.data   # [H, H]
            W_mlp  = block.proj_mlp.weight.data
            r_a, s_a, a_a, n_a = _sym_antisym_ratio(W_attn)
            r_m, s_m, a_m, n_m = _sym_antisym_ratio(W_mlp)
            rows_to_print.append((block_idx, "proj_attn", pt, r_a, s_a, a_a, n_a))
            rows_to_print.append((block_idx, "proj_mlp",  pt, r_m, s_m, a_m, n_m))
        elif pt == "mlp_only":
            W_mlp = block.proj_mlp.weight.data
            r_m, s_m, a_m, n_m = _sym_antisym_ratio(W_mlp)
            rows_to_print.append((block_idx, "proj_mlp", pt, r_m, s_m, a_m, n_m))
        elif pt in ("unconstrained", "identity", "pos_scalar"):
            if hasattr(block, 'proj') and hasattr(block.proj, 'weight'):
                W = block.proj.weight.data
                r, s, a, n = _sym_antisym_ratio(W)
                rows_to_print.append((block_idx, "proj", pt, r, s, a, n))
        elif pt in ("antisymmetric", "low_rank_antisymmetric"):
            # Antisymmetric by construction: A = J - J^T, S = 0
            rows_to_print.append((block_idx, "proj", pt, float('inf'), 0.0, float('nan'), float('nan')))
        else:
            rows_to_print.append((block_idx, "proj", pt, float('nan'), float('nan'), float('nan'), float('nan')))

        if rows_to_print:
            if not header_printed:
                print(f"\n{'Block':>6} | {'Matrix':>10} | {'proj_type':>22} | {'||A||/||S||':>12} | {'||S||_F':>9} | {'||A||_F':>9} | {'||W||_F':>9}")
                print("-" * 90)
                header_printed = True
            for (bi, mname, ptype, ratio, ns, na, nw) in rows_to_print:
                r_str = f"{ratio:>12.4f}" if not (isinstance(ratio, float) and (ratio != ratio or ratio == float('inf'))) else f"{'inf':>12}"
                ns_str = f"{ns:>9.4f}" if ns == ns else f"{'n/a':>9}"
                na_str = f"{na:>9.4f}" if na == na else f"{'n/a':>9}"
                nw_str = f"{nw:>9.4f}" if nw == nw else f"{'n/a':>9}"
                print(f"{bi:>6} | {mname:>10} | {ptype:>22} | {r_str} | {ns_str} | {na_str} | {nw_str}")

    if not header_printed:
        print("  (no unconstrained projection weights found in this model)")

    # Save CSV
    csv_header = "block,iters,iter_idx,cos_theta_mean,cos_theta_std,delta_norm,grad_norm,grad_component,curl_component,curl_to_grad_ratio,energy_delta"
    if has_dual:
        csv_header += ",attn_cos_theta,mlp_cos_theta,attn_grad_norm,mlp_grad_norm,attn_proj_norm,mlp_proj_norm,attn_curl_grad,mlp_curl_grad"
    csv_rows = [csv_header]
    for (b, it) in sorted(all_results.keys()):
        rs = all_results[(b, it)]
        row = (f"{b},{model.transformer.layer_iterations[b]},{it},"
               f"{np.mean([r['cos_theta_mean'] for r in rs]):.6f},"
               f"{np.mean([r['cos_theta_std'] for r in rs]):.6f},"
               f"{np.mean([r['delta_norm'] for r in rs]):.6f},"
               f"{np.mean([r['grad_norm'] for r in rs]):.6f},"
               f"{np.mean([r['grad_component'] for r in rs]):.6f},"
               f"{np.mean([r['curl_component'] for r in rs]):.6f},"
               f"{np.mean([r['curl_to_grad_ratio'] for r in rs]):.6f},"
               f"{np.mean([r['energy_delta'] for r in rs]):.6f}")
        if has_dual and 'attn_cos_theta' in rs[0]:
            row += (f",{np.mean([r['attn_cos_theta'] for r in rs]):.6f}"
                    f",{np.mean([r['mlp_cos_theta'] for r in rs]):.6f}"
                    f",{np.mean([r['attn_grad_norm'] for r in rs]):.6f}"
                    f",{np.mean([r['mlp_grad_norm'] for r in rs]):.6f}"
                    f",{np.mean([r['attn_proj_norm'] for r in rs]):.6f}"
                    f",{np.mean([r['mlp_proj_norm'] for r in rs]):.6f}"
                    f",{np.mean([r['attn_curl_grad'] for r in rs]):.6f}"
                    f",{np.mean([r['mlp_curl_grad'] for r in rs]):.6f}")
        csv_rows.append(row)
    csv_path = os.path.join(args.output_dir, f'helmholtz_{args.dataset}.csv')
    with open(csv_path, 'w') as f:
        f.write("\n".join(csv_rows) + "\n")
    print(f"\nSaved CSV: {csv_path}")
    print(f"All figures saved to: {args.output_dir}/")


if __name__ == '__main__':
    main()
