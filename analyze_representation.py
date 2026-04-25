"""Representation-level analysis of EGPT vs RecGPT iterative blocks.

Analyses:
  1. LogitLens: Project hidden states at each iteration through ln_f + lm_head
     to see how token predictions evolve across iterations.
  2. Prediction entropy: How confident is the model at each iteration?
  3. Representation convergence: ||h^(t+1) - h^(t)|| / ||h^(t)|| per iteration.
  4. CKA similarity: How similar are representations across iterations?
  5. Token rank tracking: Does the correct next-token rank improve with iterations?

Usage:
    python analyze_representation.py <model_path> [--num_samples 20] [--max_seq_len 256]
"""

import sys
import os
import torch
import torch.nn.functional as F
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


def logitlens(hidden_states, model):
    """Project hidden states through ln_f + lm_head to get logits."""
    h = model.transformer.ln_f(hidden_states)
    if model._tied_word_embeddings:
        logits = F.linear(h, model.transformer.wte.weight)
    else:
        logits = model.lm_head(h)
    return logits


def top_k_tokens(logits, tokenizer, k=5):
    """Get top-k token predictions from logits [T, V]."""
    probs = F.softmax(logits, dim=-1)
    topk = torch.topk(probs, k, dim=-1)
    results = []
    for t in range(logits.shape[0]):
        tokens = [tokenizer.decode([idx]) for idx in topk.indices[t].tolist()]
        probs_t = topk.values[t].tolist()
        results.append(list(zip(tokens, probs_t)))
    return results


def compute_entropy(logits):
    """Compute entropy of prediction distribution per token. logits: [T, V]."""
    probs = F.softmax(logits.float(), dim=-1)
    log_probs = F.log_softmax(logits.float(), dim=-1)
    entropy = -(probs * log_probs).sum(dim=-1)  # [T]
    return entropy


def compute_correct_rank(logits, target_ids):
    """Compute rank of the correct next token. logits: [T, V], target_ids: [T]."""
    # For each position t, rank of target_ids[t] in logits[t]
    sorted_indices = logits.argsort(dim=-1, descending=True)  # [T, V]
    ranks = []
    for t in range(min(logits.shape[0], target_ids.shape[0])):
        rank = (sorted_indices[t] == target_ids[t]).nonzero(as_tuple=True)[0]
        if len(rank) > 0:
            ranks.append(rank[0].item() + 1)  # 1-indexed
        else:
            ranks.append(logits.shape[-1])  # worst case
    return ranks


def linear_cka(X, Y):
    """Linear CKA between two representations X, Y of shape [n, d]."""
    X = X - X.mean(dim=0, keepdim=True)
    Y = Y - Y.mean(dim=0, keepdim=True)
    hsic_xy = (X @ Y.T).pow(2).sum()
    hsic_xx = (X @ X.T).pow(2).sum()
    hsic_yy = (Y @ Y.T).pow(2).sum()
    return (hsic_xy / (hsic_xx.sqrt() * hsic_yy.sqrt() + 1e-12)).item()


def run_with_iteration_tracking(model, input_ids):
    """Run model forward, capturing hidden states at every iteration of every block.

    Returns:
        per_iter_states: dict of (block_idx, iter_idx) -> hidden_states [1, T, D]
        final_hidden: final hidden states after all blocks
    """
    base_model = model.transformer

    hidden_states = base_model.wte(input_ids).detach()
    if base_model.m_emb is not None:
        hidden_states = hidden_states * base_model.m_emb

    rope_cos_sin = None
    if base_model.position_embedding_type == "rope":
        position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
        rope_cos_sin = base_model._get_rope_cos_sin(
            key_length=input_ids.shape[1], position_ids=position_ids, dtype=hidden_states.dtype)

    per_iter_states = {}

    for i, num_iter in enumerate(base_model.layer_iterations):
        block = base_model.h[i]
        is_iterative = num_iter > 1

        # Store state before this block's iterations
        per_iter_states[(i, 0)] = hidden_states.detach().clone()

        for j in range(num_iter):
            with torch.no_grad():
                hidden_states = block(
                    hidden_states.detach(), past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)

            # Store after each iteration
            per_iter_states[(i, j + 1)] = hidden_states.detach().clone()

    # Apply final layer norm
    final_hidden = base_model.ln_f(hidden_states)

    return per_iter_states, final_hidden


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--num_samples", type=int, default=20)
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "c4"])
    args = parser.parse_args()

    model_name = args.model_path.rstrip("/").split("/")[-1]
    if args.output_dir is None:
        args.output_dir = f"egpt_paper/representation_analysis/{model_name}"
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained('/proj/checkpoints/dmf-lh-checkpoints/tokenizers/granite-4.0-tiktoken')
    model = AutoModelForCausalLM.from_pretrained(args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.eval()

    print(f"Model: {model_name}")
    print(f"Layer iterations: {model.transformer.layer_iterations}")
    print(f"Tied embeddings: {model._tied_word_embeddings}")

    # Load data
    if args.dataset == "wikitext":
        dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        texts = [t for t in dataset["text"] if len(t.strip()) > 100][:args.num_samples]
    elif args.dataset == "c4":
        dataset = load_dataset("allenai/c4", "en", split="validation", streaming=True)
        texts = []
        for sample in dataset:
            if len(sample["text"].strip()) > 200:
                texts.append(sample["text"])
            if len(texts) >= args.num_samples:
                break
    print(f"Using {len(texts)} {args.dataset} samples, max_seq_len={args.max_seq_len}")

    # Find iterative blocks (num_iter > 1)
    iter_blocks = [(i, n) for i, n in enumerate(model.transformer.layer_iterations) if n > 1]
    print(f"Iterative blocks: {iter_blocks}")

    if not iter_blocks:
        print("No iterative blocks found. This is a standard GPT model.")
        return

    # =====================================================================
    # Collect per-iteration statistics across samples
    # =====================================================================
    # Per (block, iter): lists of per-sample values
    all_entropy = defaultdict(list)       # prediction entropy
    all_correct_rank = defaultdict(list)  # rank of correct next token
    all_convergence = defaultdict(list)   # ||h^(t+1) - h^(t)|| / ||h^(t)||
    all_cka = defaultdict(list)           # CKA between consecutive iterations
    all_top1_match = defaultdict(list)    # does top-1 match the final iteration's top-1?
    all_logit_kl = defaultdict(list)      # KL(iter_t || final_iter) for prediction distribution

    for sample_idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt',
                                     max_length=args.max_seq_len, truncation=True).cuda()
        if input_ids.shape[1] < 10:
            continue
        target_ids = input_ids[0, 1:]  # shifted by 1 for next-token prediction

        # Run forward with iteration tracking
        per_iter_states, final_hidden = run_with_iteration_tracking(model, input_ids)

        for block_idx, num_iter in iter_blocks:
            # Get final iteration's logits for this block (for comparison)
            h_final_iter = per_iter_states[(block_idx, num_iter)]
            logits_final = logitlens(h_final_iter, model).squeeze(0).float()  # [T, V]
            top1_final = logits_final.argmax(dim=-1)  # [T]

            for j in range(num_iter + 1):  # 0 = before iterations, 1..num_iter = after each
                h = per_iter_states[(block_idx, j)]

                # LogitLens: project through ln_f + lm_head
                logits = logitlens(h, model).squeeze(0).float()  # [T, V]

                # Entropy of prediction
                entropy = compute_entropy(logits)
                all_entropy[(block_idx, j)].append(entropy.mean().item())

                # Rank of correct next token
                ranks = compute_correct_rank(logits[:-1], target_ids)
                all_correct_rank[(block_idx, j)].append(np.median(ranks))

                # Top-1 match with final iteration
                top1_match = (logits.argmax(dim=-1) == top1_final).float().mean().item()
                all_top1_match[(block_idx, j)].append(top1_match)

                # KL(p_t || p_final): how much info is lost using iter t instead of final
                # F.kl_div(input=log_q, target=p) computes KL(p || q)
                # So we need input=log_p_final, target=p_t to get KL(p_t || p_final)
                if j < num_iter:
                    kl = F.kl_div(
                        F.log_softmax(logits_final, dim=-1),
                        F.softmax(logits, dim=-1),
                        reduction='batchmean'
                    ).item()
                    all_logit_kl[(block_idx, j)].append(kl)

                # Convergence rate (for j >= 1)
                if j >= 1:
                    h_prev = per_iter_states[(block_idx, j - 1)]
                    delta = (h.float() - h_prev.float()).norm(dim=-1)
                    h_norm = h_prev.float().norm(dim=-1).clamp(min=1e-8)
                    conv_rate = (delta / h_norm).mean().item()
                    all_convergence[(block_idx, j)].append(conv_rate)

                # CKA with previous iteration
                if j >= 1:
                    h_prev = per_iter_states[(block_idx, j - 1)]
                    # Use a subset of tokens for CKA (computational cost)
                    n_tokens = min(h.shape[1], 128)
                    cka_val = linear_cka(
                        h.squeeze(0)[:n_tokens].float(),
                        h_prev.squeeze(0)[:n_tokens].float()
                    )
                    all_cka[(block_idx, j)].append(cka_val)

        if (sample_idx + 1) % 5 == 0:
            print(f"  Processed {sample_idx + 1}/{len(texts)} samples")

    # =====================================================================
    # Print summary tables
    # =====================================================================
    print(f"\n{'='*100}")
    print(f"Representation Analysis: {model_name}")
    print(f"{'='*100}")

    for block_idx, num_iter in iter_blocks:
        block = model.transformer.h[block_idx]
        block_type = "energy" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "softmax"

        print(f"\n--- Block {block_idx} ({block_type}, {num_iter} iterations) ---")
        print(f"{'Iter':>5} | {'Entropy':>8} | {'Med Rank':>9} | {'Conv Rate':>10} | {'CKA':>6} | {'Top1 Match':>11} | {'KL(t||final)':>12}")
        print("-" * 75)

        for j in range(num_iter + 1):
            ent = np.mean(all_entropy[(block_idx, j)]) if all_entropy[(block_idx, j)] else float('nan')
            rank = np.mean(all_correct_rank[(block_idx, j)]) if all_correct_rank[(block_idx, j)] else float('nan')
            conv = np.mean(all_convergence[(block_idx, j)]) if all_convergence[(block_idx, j)] else float('nan')
            cka = np.mean(all_cka[(block_idx, j)]) if all_cka[(block_idx, j)] else float('nan')
            top1 = np.mean(all_top1_match[(block_idx, j)]) if all_top1_match[(block_idx, j)] else float('nan')
            kl = np.mean(all_logit_kl[(block_idx, j)]) if all_logit_kl[(block_idx, j)] else float('nan')

            label = f"t={j}" if j > 0 else "input"
            print(f"{label:>5} | {ent:>8.2f} | {rank:>9.1f} | {conv:>10.4f} | {cka:>6.4f} | {top1:>11.4f} | {kl:>12.4f}")

    # =====================================================================
    # Generate plots
    # =====================================================================
    # Plot 1: Entropy across iterations per block
    fig, axes = plt.subplots(1, len(iter_blocks), figsize=(5 * len(iter_blocks), 4), squeeze=False)
    for idx, (block_idx, num_iter) in enumerate(iter_blocks):
        ax = axes[0][idx]
        block = model.transformer.h[block_idx]
        block_type = "energy" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "softmax"

        iters = range(num_iter + 1)
        entropies = [np.mean(all_entropy[(block_idx, j)]) for j in iters]
        ax.plot(list(iters), entropies, 'o-', linewidth=2, markersize=6)
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Prediction Entropy')
        ax.set_title(f'B{block_idx} ({block_type}, T={num_iter})')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'LogitLens Entropy per Iteration\n{model_name}', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'entropy_per_iteration.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"\nSaved: {fig_path}")

    # Plot 2: Correct token rank across iterations
    fig, axes = plt.subplots(1, len(iter_blocks), figsize=(5 * len(iter_blocks), 4), squeeze=False)
    for idx, (block_idx, num_iter) in enumerate(iter_blocks):
        ax = axes[0][idx]
        block = model.transformer.h[block_idx]
        block_type = "energy" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "softmax"

        iters = range(num_iter + 1)
        ranks = [np.mean(all_correct_rank[(block_idx, j)]) for j in iters]
        ax.plot(list(iters), ranks, 'o-', linewidth=2, markersize=6, color='coral')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('Median Rank of Correct Token')
        ax.set_title(f'B{block_idx} ({block_type}, T={num_iter})')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'LogitLens: Correct Token Rank per Iteration\n{model_name}', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'correct_rank_per_iteration.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Plot 3: Convergence rate + CKA
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    ax = axes[0]
    for block_idx, num_iter in iter_blocks:
        block = model.transformer.h[block_idx]
        block_type = "E" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "S"
        iters = range(1, num_iter + 1)
        conv = [np.mean(all_convergence[(block_idx, j)]) for j in iters]
        ax.plot(list(iters), conv, 'o-', label=f'B{block_idx} ({block_type})', linewidth=2, markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('||h(t+1)-h(t)|| / ||h(t)||')
    ax.set_title('Convergence Rate')
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for block_idx, num_iter in iter_blocks:
        block = model.transformer.h[block_idx]
        block_type = "E" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "S"
        iters = range(1, num_iter + 1)
        cka = [np.mean(all_cka[(block_idx, j)]) for j in iters]
        ax.plot(list(iters), cka, 'o-', label=f'B{block_idx} ({block_type})', linewidth=2, markersize=5)
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Linear CKA with previous iteration')
    ax.set_title('Inter-iteration Similarity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0.9, 1.001)

    plt.suptitle(f'{model_name}', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'convergence_cka.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # Plot 4: KL divergence from final iteration
    fig, axes = plt.subplots(1, len(iter_blocks), figsize=(5 * len(iter_blocks), 4), squeeze=False)
    for idx, (block_idx, num_iter) in enumerate(iter_blocks):
        ax = axes[0][idx]
        block = model.transformer.h[block_idx]
        block_type = "energy" if hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention" else "softmax"

        iters = range(num_iter)  # 0 to num_iter-1
        kls = [np.mean(all_logit_kl[(block_idx, j)]) for j in iters]
        ax.plot(list(iters), kls, 'o-', linewidth=2, markersize=6, color='green')
        ax.set_xlabel('Iteration')
        ax.set_ylabel('KL(iter_t || final)')
        ax.set_title(f'B{block_idx} ({block_type}, T={num_iter})')
        ax.grid(True, alpha=0.3)

    plt.suptitle(f'KL Divergence from Final Iteration\n{model_name}', fontsize=12)
    plt.tight_layout()
    fig_path = os.path.join(args.output_dir, 'kl_from_final.png')
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {fig_path}")

    # =====================================================================
    # LogitLens qualitative example: show top-5 predictions per iteration
    # for a few tokens of the first sample
    # =====================================================================
    print(f"\n{'='*100}")
    print(f"LogitLens Qualitative Example (first sample, last iterative block)")
    print(f"{'='*100}")

    # Use first sample
    text = texts[0]
    input_ids = tokenizer.encode(text, return_tensors='pt',
                                 max_length=args.max_seq_len, truncation=True).cuda()
    per_iter_states, _ = run_with_iteration_tracking(model, input_ids)

    last_block_idx, last_num_iter = iter_blocks[-1]
    tokens = [tokenizer.decode([tid]) for tid in input_ids[0].tolist()]

    # Show positions 10-20
    show_positions = list(range(10, min(20, input_ids.shape[1] - 1)))

    for pos in show_positions:
        target_tok = tokens[pos + 1] if pos + 1 < len(tokens) else "?"
        print(f"\nPosition {pos}: '{tokens[pos]}' -> target: '{target_tok}'")

        for j in [0, 1, last_num_iter // 2, last_num_iter]:
            if j > last_num_iter:
                continue
            h = per_iter_states[(last_block_idx, j)]
            logits = logitlens(h, model).squeeze(0).float()
            probs = F.softmax(logits[pos], dim=-1)
            topk = torch.topk(probs, 5)
            top_toks = [f"'{tokenizer.decode([idx])}' ({p:.3f})" for idx, p in zip(topk.indices.tolist(), topk.values.tolist())]
            label = f"iter {j}" if j > 0 else "input"
            print(f"  {label:>8}: {', '.join(top_toks)}")

    print(f"\n{'='*100}")
    print(f"All results saved to: {args.output_dir}/")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
