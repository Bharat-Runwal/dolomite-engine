#!/usr/bin/env python3
"""Evaluate adaptive halting criteria for iterative transformer blocks.

Tests several halting criteria that stop iterations early per-block:
1. Convergence rate: ||h_t - h_{t-1}|| / ||h_{t-1}|| < threshold
2. Token stability: fraction of positions where argmax doesn't change
3. Fixed early exit: stop all blocks at a fixed iteration

Measures WikiText PPL and average iterations used per block.

Usage:
    python eval_halting.py <model_path> [--num_samples 100] [--max_seq_len 512]
"""

import sys
import os
import json
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

from lm_engine.hf_models.register_hf import register_model_classes
register_model_classes()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def logitlens(hidden_states, model):
    h = model.transformer.ln_f(hidden_states)
    if model._tied_word_embeddings:
        logits = F.linear(h, model.transformer.wte.weight)
    else:
        logits = model.lm_head(h)
    return logits


def eval_ppl_with_halting(model, tokenizer, texts, max_seq_len, halt_fn, halt_name):
    """Evaluate perplexity with a per-block halting criterion.

    halt_fn(block_idx, t, h_prev, h_curr, model) -> bool
        Returns True if block should stop at iteration t.
    """
    model.eval()
    device = next(model.parameters()).device

    total_loss = 0.0
    total_tokens = 0
    iters_used = defaultdict(list)

    layer_iterations = model.transformer.layer_iterations
    iter_blocks = [(i, layer_iterations[i]) for i in range(len(layer_iterations)) if layer_iterations[i] > 1]

    for sample_idx, text in enumerate(texts):
        tokens = tokenizer(text, return_tensors='pt', max_length=max_seq_len,
                          truncation=True).to(device)
        input_ids = tokens['input_ids']
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            hidden_states = model.transformer.wte(input_ids)

            if hasattr(model.transformer, 'embedding_dropout'):
                hidden_states = model.transformer.embedding_dropout(hidden_states)

            rope_cos_sin = None
            if hasattr(model.transformer, 'rope'):
                rope_cos_sin = model.transformer.rope(input_ids.shape[1], dtype=hidden_states.dtype)

            layer_id = 0
            for block_idx in range(len(model.transformer.h)):
                block = model.transformer.h[block_idx]
                num_iter = layer_iterations[block_idx]

                if num_iter == 1:
                    hidden_states = block(
                        hidden_states,
                        rope_cos_sin=rope_cos_sin,
                        layer_id=layer_id,
                    )
                    layer_id += 1
                else:
                    actual_iters = 0
                    for t in range(num_iter):
                        h_prev = hidden_states.clone()
                        hidden_states = block(
                            hidden_states,
                            rope_cos_sin=rope_cos_sin,
                            layer_id=layer_id,
                        )
                        layer_id += 1
                        actual_iters += 1

                        if t > 0 and halt_fn(block_idx, t, h_prev, hidden_states, model):
                            layer_id += (num_iter - t - 1)
                            break

                    iters_used[block_idx].append(actual_iters)

        if (sample_idx + 1) % 20 == 0:
            print(f"  [{halt_name}] {sample_idx+1}/{len(texts)}", flush=True)

    # Final LN + LM head for loss
    # Actually we need to recompute -- the above loop doesn't compute loss.
    # Let me fix: compute loss inside the loop.
    # Rewrite: accumulate loss properly.

    # -- Rerun with loss computation --
    total_loss = 0.0
    total_tokens = 0
    iters_used = defaultdict(list)

    for sample_idx, text in enumerate(texts):
        tokens = tokenizer(text, return_tensors='pt', max_length=max_seq_len,
                          truncation=True).to(device)
        input_ids = tokens['input_ids']
        if input_ids.shape[1] < 2:
            continue

        with torch.no_grad():
            hidden_states = model.transformer.wte(input_ids)

            if hasattr(model.transformer, 'embedding_dropout'):
                hidden_states = model.transformer.embedding_dropout(hidden_states)

            rope_cos_sin = None
            if hasattr(model.transformer, 'rope'):
                rope_cos_sin = model.transformer.rope(input_ids.shape[1], dtype=hidden_states.dtype)

            layer_id = 0
            for block_idx in range(len(model.transformer.h)):
                block = model.transformer.h[block_idx]
                num_iter = layer_iterations[block_idx]

                if num_iter == 1:
                    hidden_states = block(
                        hidden_states,
                        rope_cos_sin=rope_cos_sin,
                        layer_id=layer_id,
                    )
                    layer_id += 1
                else:
                    actual_iters = 0
                    for t in range(num_iter):
                        h_prev = hidden_states.clone()
                        hidden_states = block(
                            hidden_states,
                            rope_cos_sin=rope_cos_sin,
                            layer_id=layer_id,
                        )
                        layer_id += 1
                        actual_iters += 1

                        if t > 0 and halt_fn(block_idx, t, h_prev, hidden_states, model):
                            layer_id += (num_iter - t - 1)
                            break

                    iters_used[block_idx].append(actual_iters)

            # Final LN + LM head
            hidden_states = model.transformer.ln_f(hidden_states)
            if model._tied_word_embeddings:
                logits = F.linear(hidden_states, model.transformer.wte.weight)
            else:
                logits = model.lm_head(hidden_states)

            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                   shift_labels.view(-1), reduction='sum')
            total_loss += loss.item()
            total_tokens += shift_labels.numel()

    ppl = torch.exp(torch.tensor(total_loss / total_tokens)).item()

    avg_iters = {}
    for block_idx, iters_list in iters_used.items():
        avg_iters[block_idx] = np.mean(iters_list)

    total_iters = sum(avg_iters.get(i, layer_iterations[i]) for i in range(len(layer_iterations)))

    return ppl, avg_iters, total_iters


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model_path")
    parser.add_argument("--num_samples", type=int, default=100)
    parser.add_argument("--max_seq_len", type=int, default=512)
    parser.add_argument("--output_dir", default=None)
    parser.add_argument("--dataset", default="wikitext", choices=["wikitext", "c4"],
                        help="Dataset for evaluation: wikitext (validation) or c4")
    args = parser.parse_args()

    model_name = os.path.basename(args.model_path.rstrip("/"))
    if args.output_dir is None:
        args.output_dir = os.path.join(args.model_path, f"halting_results_{args.dataset}")
    os.makedirs(args.output_dir, exist_ok=True)

    print(f"Loading model from {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(
        '/proj/checkpoints/dmf-lh-checkpoints/tokenizers/granite-4.0-tiktoken')
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.bfloat16, device_map="cuda")
    model.config.use_cache = False
    model.eval()

    layer_iters = model.transformer.layer_iterations
    iter_block_indices = [i for i in range(len(layer_iters)) if layer_iters[i] > 1]
    print(f"Layer iterations: {layer_iters}")
    print(f"Iterative blocks: {iter_block_indices}")

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

    results = []

    # ---- Baseline: full iterations ----
    def no_halt(block_idx, t, h_prev, h_curr, model):
        return False

    print("\n=== Baseline ===")
    ppl, avg_iters, total = eval_ppl_with_halting(
        model, tokenizer, texts, args.max_seq_len, no_halt, "baseline")
    print(f"Baseline: PPL={ppl:.2f}, total_iters={total:.1f}")
    results.append({"criterion": "baseline", "ppl": ppl, "avg_iters": {str(k): v for k, v in avg_iters.items()}, "total_iters": total})

    # ---- Criterion 1: Convergence rate ----
    print("\n=== Convergence Rate ===")
    for threshold in [0.3, 0.2, 0.15, 0.1, 0.05, 0.03]:
        def conv_halt(block_idx, t, h_prev, h_curr, model, thr=threshold):
            diff = (h_curr - h_prev).float()
            prev_norm = h_prev.float().norm()
            if prev_norm < 1e-8:
                return False
            return (diff.norm() / prev_norm).item() < thr

        ppl, avg_iters, total = eval_ppl_with_halting(
            model, tokenizer, texts, args.max_seq_len, conv_halt, f"conv<{threshold}")
        per_block = [f"B{i}:{avg_iters.get(i, layer_iters[i]):.1f}/{layer_iters[i]}" for i in iter_block_indices]
        print(f"conv_rate < {threshold}: PPL={ppl:.2f}, iters={total:.1f}, {' '.join(per_block)}")
        results.append({"criterion": f"conv_rate<{threshold}", "ppl": ppl,
                        "avg_iters": {str(k): v for k, v in avg_iters.items()}, "total_iters": total})

    # ---- Criterion 2: Token prediction stability ----
    print("\n=== Token Stability ===")
    for stability in [0.98, 0.95, 0.9, 0.85]:
        def token_halt(block_idx, t, h_prev, h_curr, model, thr=stability):
            logits_prev = logitlens(h_prev, model)
            logits_curr = logitlens(h_curr, model)
            pred_prev = logits_prev.argmax(dim=-1)
            pred_curr = logits_curr.argmax(dim=-1)
            return (pred_prev == pred_curr).float().mean().item() >= thr

        ppl, avg_iters, total = eval_ppl_with_halting(
            model, tokenizer, texts, args.max_seq_len, token_halt, f"stable>{stability}")
        per_block = [f"B{i}:{avg_iters.get(i, layer_iters[i]):.1f}/{layer_iters[i]}" for i in iter_block_indices]
        print(f"token_stable > {stability}: PPL={ppl:.2f}, iters={total:.1f}, {' '.join(per_block)}")
        results.append({"criterion": f"token_stable>{stability}", "ppl": ppl,
                        "avg_iters": {str(k): v for k, v in avg_iters.items()}, "total_iters": total})

    # ---- Criterion 3: Fixed early exit ----
    print("\n=== Fixed Early Exit ===")
    for max_t in [2, 3, 4, 6]:
        def fixed_halt(block_idx, t, h_prev, h_curr, model, mt=max_t):
            return t >= mt

        ppl, avg_iters, total = eval_ppl_with_halting(
            model, tokenizer, texts, args.max_seq_len, fixed_halt, f"fixed_T={max_t}")
        print(f"Fixed T={max_t}: PPL={ppl:.2f}, total_iters={total:.1f}")
        results.append({"criterion": f"fixed_T={max_t}", "ppl": ppl,
                        "avg_iters": {str(k): v for k, v in avg_iters.items()}, "total_iters": total})

    # ---- Summary ----
    baseline_ppl = results[0]["ppl"]
    baseline_total = results[0]["total_iters"]

    print(f"\n{'='*80}")
    print(f"HALTING RESULTS: {model_name}")
    print(f"{'='*80}")
    print(f"{'Criterion':<30} {'PPL':>8} {'Degrad%':>8} {'Iters':>7} {'Saved%':>7}")
    print("-" * 65)
    for r in results:
        degrad = (r["ppl"] - baseline_ppl) / baseline_ppl * 100
        saved = (1 - r["total_iters"] / baseline_total) * 100
        print(f"{r['criterion']:<30} {r['ppl']:8.2f} {degrad:+7.1f}% {r['total_iters']:7.1f} {saved:6.1f}%")

    output = {
        'model': model_name,
        'layer_iterations': layer_iters,
        'baseline_ppl': baseline_ppl,
        'baseline_iters': baseline_total,
        'results': results
    }
    with open(os.path.join(args.output_dir, 'halting_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=float)
    print(f"\nSaved to {args.output_dir}/halting_results.json")


if __name__ == '__main__':
    main()
