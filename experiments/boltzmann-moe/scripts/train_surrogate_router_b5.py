#!/usr/bin/env python3
"""Post-hoc surrogate router training for a pretrained BoltzmannMoE checkpoint.

Loads a trained BoltzmannMoE model (e.g. B5), freezes all parameters except
newly-added linear surrogate routers (d → n_experts per FFN layer), then trains
on the same data using the KL divergence loss:

    L = KL(p_boltz(x) || p_surr(x)) = -Σ_i p_boltz_i * log(p_surr_i)

where p_boltz = softmax(E_i(x) / T) from the frozen experts and
      p_surr  = softmax(W_router x / T) from the trainable linear layer.

After training, the surrogate router can replace the Boltzmann routing at inference,
reducing O(d * expert_I * n_experts) routing cost to O(d * n_experts).

Usage:
    python train_surrogate_router_b5.py \
        --checkpoint /path/to/b5/unsharded \
        --data_path /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0 \
        --steps 5000 \
        --lr 1e-3 \
        --output /path/to/b5_surrogate

The script attaches linear routers to each BoltzmannMoE_Energy_MLP layer,
trains for `steps` steps, and saves the full model (frozen weights + trained router).
"""

import argparse
import math
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.optim import AdamW
import lm_engine.hf_models  # registers energy model type with HF AutoModel  # noqa: F401
from transformers import AutoModelForCausalLM, AutoTokenizer


def attach_surrogate_routers(model: torch.nn.Module, temperature: float = 1.0) -> list:
    """Add a linear surrogate_router to every BoltzmannMoE_Energy_MLP layer.

    Freezes all existing parameters; only the new routers are trainable.
    Returns the list of newly-added parameter groups.
    """
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP

    # Freeze everything
    for p in model.parameters():
        p.requires_grad_(False)

    router_params = []
    n_attached = 0
    for name, module in model.named_modules():
        if isinstance(module, BoltzmannMoE_Energy_MLP):
            hidden = module.hidden_size
            n_exp  = module.n_experts
            router = torch.nn.Linear(hidden, n_exp, bias=False)
            torch.nn.init.normal_(router.weight, std=0.01)
            router = router.to(next(model.parameters()).device).to(torch.bfloat16)
            module.surrogate_router = router
            module.temperature = temperature
            router_params.extend(router.parameters())
            n_attached += 1

    print(f"Attached {n_attached} surrogate routers "
          f"({sum(p.numel() for p in router_params):,} trainable params).")
    return router_params


def compute_surrogate_loss(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    """Forward pass collecting KL losses from all attached surrogate routers."""
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP

    # Hooks to collect (p_boltz, p_surr) pairs
    kl_losses = []

    def make_hook(module):
        def hook(mod, inp, out):
            x = inp[0]  # (..., hidden)
            # Recompute Boltzmann weights (frozen experts, no grad needed for p_boltz)
            with torch.no_grad():
                W1_e = mod.W1.weight.view(mod.n_experts, mod.expert_I, mod.hidden_size)
                W1x  = mod.W1(x).view(*x.shape[:-1], mod.n_experts, mod.expert_I)
                phi  = F.gelu(W1x)
                term1 = torch.einsum("...ei,eih->...eh", phi, W1_e)  # actually need W2_e
                # Simplified: use term1 energy (first gradient term)
                W2_e = mod.W2.weight.view(mod.n_experts, mod.expert_I, mod.hidden_size)
                W2x  = mod.W2(x).view(*x.shape[:-1], mod.n_experts, mod.expert_I)
                term1_full = torch.einsum("...ei,eih->...eh", phi, W2_e)
                E = torch.einsum("...h,...eh->...e", x, term1_full)
                p_boltz = F.softmax(E / mod.temperature, dim=-1).detach()

            # Surrogate routing weights (trainable)
            p_surr = F.softmax(mod.surrogate_router(x) / mod.temperature, dim=-1)

            # KL(p_boltz || p_surr) = -sum p_boltz * log(p_surr)
            kl = -(p_boltz * (p_surr + 1e-8).log()).sum(-1).mean()
            kl_losses.append(kl)
        return hook

    handles = []
    for module in model.modules():
        if isinstance(module, BoltzmannMoE_Energy_MLP) and hasattr(module, 'surrogate_router'):
            handles.append(module.register_forward_hook(make_hook(module)))

    # Do NOT use no_grad here — p_surr inside hooks needs grad_fn for backprop.
    # Frozen expert weights have requires_grad=False so they don't accumulate grad anyway.
    model(input_ids)

    for h in handles:
        h.remove()

    if not kl_losses:
        raise RuntimeError("No BoltzmannMoE layers found with surrogate_router.")
    return torch.stack(kl_losses).mean()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="Path to unsharded B5 checkpoint")
    parser.add_argument("--steps", type=int, default=5000, help="Training steps")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate for surrogate router")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--seq_len", type=int, default=1024, help="Sequence length for surrogate training")
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--output", required=True, help="Where to save the updated model")
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--data_path", default=None, help="Optional: token bin file for real data")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading checkpoint from {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    ).to(device)
    model.eval()

    # Attach trainable surrogate routers, freeze everything else
    router_params = attach_surrogate_routers(model, temperature=args.temperature)
    optimizer = AdamW(router_params, lr=args.lr, weight_decay=0.0)

    # Simple cosine LR decay
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.steps, eta_min=args.lr * 0.1)

    vocab_size = model.config.vocab_size if hasattr(model.config, 'vocab_size') else 100352

    print(f"Training surrogate routers for {args.steps} steps...")
    running_loss = 0.0
    for step in range(1, args.steps + 1):
        # Generate random token batch (or load from data_path if provided)
        if args.data_path and os.path.exists(args.data_path):
            # Simple random offset into the binary file
            import numpy as np
            n_tokens = args.batch_size * args.seq_len
            f = open(args.data_path, "rb")
            file_size = os.path.getsize(args.data_path) // 2  # uint16
            offset = torch.randint(0, max(1, file_size - n_tokens - 1), (1,)).item()
            f.seek(offset * 2)
            tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16).astype(np.int64)
            f.close()
            input_ids = torch.tensor(tokens).view(args.batch_size, args.seq_len).to(device)
        else:
            input_ids = torch.randint(0, vocab_size, (args.batch_size, args.seq_len), device=device)

        optimizer.zero_grad()
        loss = compute_surrogate_loss(model, input_ids)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(router_params, 1.0)
        optimizer.step()
        scheduler.step()

        running_loss += loss.item()
        if step % args.log_interval == 0:
            avg = running_loss / args.log_interval
            print(f"step {step:5d}/{args.steps}  kl_loss={avg:.4f}  lr={scheduler.get_last_lr()[0]:.2e}")
            running_loss = 0.0

    # Convert BoltzmannMoE layers → SurrogateBoltzmannMoE with use_surrogate=True,
    # then update the config so eval harness loads the surrogate forward by default.
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import (
        BoltzmannMoE_Energy_MLP, SurrogateBoltzmannMoE_Energy_MLP
    )
    print("Converting BoltzmannMoE layers to SurrogateBoltzmannMoE (use_surrogate=True)...")
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, BoltzmannMoE_Energy_MLP):
                continue
            if not hasattr(child_module, 'surrogate_router'):
                continue
            m = child_module
            surrogate = SurrogateBoltzmannMoE_Energy_MLP(
                hidden_size=m.hidden_size,
                intermediate_size=m.intermediate_size,
                n_experts=m.n_experts,
                temperature=m.temperature,
                repulsion_coef=m.repulsion_coef,
                n_repulsion_pairs=m.n_repulsion_pairs,
                surrogate_coef=0.0,       # no KL loss at inference
                use_surrogate=True,       # use linear router at eval
                activation_function="gelu_pytorch_tanh",
                add_bias=False,
                dropout=0.0,
                init_method="normal",
                initializer_range=0.02,
                m_width=m.hidden_size,
                num_layers=1,
            ).to(next(m.parameters()).device).to(next(m.parameters()).dtype)
            # Copy weights
            surrogate.W1.weight.data.copy_(m.W1.weight.data)
            surrogate.W2.weight.data.copy_(m.W2.weight.data)
            surrogate.surrogate_router.weight.data.copy_(m.surrogate_router.weight.data)
            setattr(parent_module, child_name, surrogate)
            print(f"  Converted {parent_name}.{child_name}")

    # Update config so the checkpoint is self-describing
    if hasattr(model.config, 'mlp_blocks'):
        for blk in model.config.mlp_blocks:
            if hasattr(blk, 'mlp_type') and blk.mlp_type == 'BoltzmannMoE_Energy_MLP':
                blk.mlp_type = 'SurrogateBoltzmannMoE_Energy_MLP'
                blk.surrogate_coef = 0.0
                blk.use_surrogate = True

    # Save model (frozen weights + trained routers) + tokenizer from source checkpoint
    os.makedirs(args.output, exist_ok=True)
    model.save_pretrained(args.output)
    # Copy tokenizer files so the output directory is self-contained for eval harness
    import shutil
    for fname in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        src = Path(args.checkpoint) / fname
        if src.exists():
            shutil.copy2(src, Path(args.output) / fname)
    print(f"Saved to {args.output}")

    # Print summary of final KL per layer
    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import BoltzmannMoE_Energy_MLP
    print("\nFinal surrogate router weight norms per layer:")
    for name, module in model.named_modules():
        if isinstance(module, BoltzmannMoE_Energy_MLP) and hasattr(module, 'surrogate_router'):
            w_norm = module.surrogate_router.weight.norm().item()
            print(f"  {name}: router weight norm = {w_norm:.3f}")


if __name__ == "__main__":
    main()
