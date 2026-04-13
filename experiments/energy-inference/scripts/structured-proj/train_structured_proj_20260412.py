"""
Retrain the energy block projections of the 410m hybrid checkpoint using
the dual_low_rank_port_hamiltonian parametrisation.

What this does
--------------
1. Loads the trained checkpoint (HF safetensors format).
2. For each energy block (h.8–h.11), replaces the unconstrained proj.weight
   with a DualLowRankPortHamiltonianProjection, warm-starting its parameters
   from the SVD/eigendecomposition of the trained weight.
3. Freezes everything except the new proj parameters.
4. Trains with alpha noise: each forward pass randomly scales the J (curl) and
   R (descent) components by a factor drawn from Uniform(alpha_min, alpha_max),
   so the model must learn a landscape that is robust to different traversal strategies.
5. Logs to wandb and saves checkpoints periodically.

The alpha noise schedule
------------------------
alpha controls the descent/curl balance:
    alpha=0.5 → equal mix (≈ trained behaviour after warm-start)
    alpha<0.5 → more curl
    alpha>0.5 → more descent

At each step we draw alpha ~ U(alpha_min, alpha_max) and scale:
    J_scale = (1 - alpha) / 0.5     # relative curl strength
    R_scale = alpha / 0.5            # relative descent strength

So training with alpha_min=0.35, alpha_max=0.65 exposes the model to the full
range that currently causes catastrophic degeneration, forcing the landscape to
adapt so that it works across the whole window.

Usage
-----
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install accelerate datasets wandb tqdm -q
  cd /proj/dmfexp/nima/Code/dolomite-engine
  python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py

bsub submission
---------------
  bsub -q standard -G grp_ebm -J struct_proj_retrain \\
      -gpu "num=1/task:mode=exclusive_process" -n 1 -M 32G -W 04:00 \\
      -o $HOME/%J.stdout -e $HOME/%J.stderr << 'EOF'
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install accelerate datasets wandb tqdm -q
  cd /proj/dmfexp/nima/Code/dolomite-engine
  python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py \\
      --steps 10000 --alpha_min 0.35 --alpha_max 0.65
  EOF
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=(
        "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k"
    ))
    p.add_argument("--rank",             type=int,   default=32)
    p.add_argument("--dissipation_rank", type=int,   default=16)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--steps",            type=int,   default=10_000)
    p.add_argument("--batch_size",       type=int,   default=4)
    p.add_argument("--seq_len",          type=int,   default=512)
    p.add_argument("--grad_accum",       type=int,   default=4)
    p.add_argument("--alpha_min",        type=float, default=0.35,
                   help="Minimum alpha for trajectory diversity training.")
    p.add_argument("--alpha_max",        type=float, default=0.65,
                   help="Maximum alpha for trajectory diversity training.")
    p.add_argument("--no_alpha_noise",   action="store_true",
                   help="Disable alpha noise (train at fixed alpha=0.5).")
    p.add_argument("--residual",          action="store_true",
                   help="Residual mode: keep frozen trained W as base, train structured "
                        "J-R delta on top. Guarantees initial loss = trained loss (~2.4). "
                        "The structured part starts near zero and learns corrections.")
    p.add_argument("--no_warmstart",     action="store_true",
                   help="Random init for structured proj (no decomposition from trained W). "
                        "Ignored when --residual is set.")
    p.add_argument("--psd_only",         action="store_true",
                   help="Freeze J (U/V=0) and train only the PSD descent component R=LL^T.")
    p.add_argument("--learnable_alpha",  action="store_true",
                   help="Add a per-stream learnable alpha (sigmoid-parametrised) to each proj. "
                        "Logs alpha convergence to wandb. Disables alpha noise.")
    p.add_argument("--backbone_lr",      type=float, default=0.0,
                   help="If >0, unfreeze backbone (non-proj) params and train them at this LR. "
                        "Use e.g. 1e-5 for two-LR setup.")
    p.add_argument("--norm_log_interval", type=int, default=200,
                   help="Steps between per-layer J/R norm / alpha logging to wandb.")
    p.add_argument("--save_dir",         type=str,   default=None)
    p.add_argument("--save_interval",    type=int,   default=2000)
    p.add_argument("--log_interval",     type=int,   default=50)
    p.add_argument("--wandb_project",    type=str,   default="energy-gpt")
    p.add_argument("--wandb_name",       type=str,   default="410m_structured_proj_retrain")
    p.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset",          type=str,   default="wikitext",
                   help="HF dataset name. 'wikitext' uses wikitext-103-raw-v1.")
    return p.parse_args()


# ── Weight initialisation from trained W ──────────────────────────────────────

def init_from_trained_weight(proj, W: torch.Tensor, scale_ff: float = 0.1):
    """Warm-start DualLowRankPortHamiltonianProjection from a trained unconstrained W.

    Goal: make the new model's forward pass approximate the trained model's update.

    Trained:  h = h - W @ (attn_out + scale_ff * ffwd_out)
    New:      h = h + (J_attn - R_attn) @ attn_out
                     + (J_ff   - R_ff  ) @ ffwd_out

    For these to match we need:
      K_attn = J_attn - R_attn = -W
      K_ff   = J_ff   - R_ff   = -scale_ff * W

    J = UV^T - VU^T ≈ -W_anti  →  negate U relative to the SVD so J ≈ -W_anti
    R = LL^T        ≈  W_sym   →  L from eigendecomposition of W_sym (positive part)
    Combined: K = J - R ≈ -W_anti - W_sym = -W  ✓

    The ff stream gets the same decomposition scaled by scale_ff.
    """
    W = W.float()
    r_j = proj.rank
    r_d = proj.dissipation_rank

    W_sym  = (W + W.T) / 2
    W_anti = (W - W.T) / 2

    # ── Init R (PSD / descent) from W_sym ────────────────────────────────────
    # Eigendecompose, keep top-r_d eigenvectors with positive eigenvalues.
    # R = L L^T ≈ W_sym_+  (truncated to rank r_d)
    eigvals, eigvecs = torch.linalg.eigh(W_sym)   # ascending order
    # Take the r_d largest-magnitude positive eigenvalues
    pos_mask = eigvals > 0
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]
    if pos_vals.numel() >= r_d:
        idx = torch.argsort(pos_vals, descending=True)[:r_d]
        L_init = pos_vecs[:, idx] * pos_vals[idx].sqrt().unsqueeze(0)
    else:
        # Fewer positive eigenvalues than r_d — pad with small random columns
        L_init = torch.cat([
            pos_vecs * pos_vals.sqrt().unsqueeze(0),
            torch.randn(W.shape[0], r_d - pos_vals.numel()) * 0.01
        ], dim=1)

    # ── Init J (antisym / curl) from W_anti ──────────────────────────────────
    # We need J = UV^T - VU^T = -W_anti.
    #
    # Using SVD: W_anti = U_svd @ diag(S) @ Vh_svd  (full_matrices=False)
    # Set: U = -U_svd[:,:k] @ diag(sqrt(S[:k]) / sqrt(2))
    #      V =  Vh_svd[:k].T @ diag(sqrt(S[:k]) / sqrt(2))
    # Then:
    #   UV^T = -1/2 * U_svd @ diag(S) @ Vh_svd = -W_anti/2
    #   VU^T = -(W_anti.T)/2 = +W_anti/2  (since W_anti.T = -W_anti)
    # → J = UV^T - VU^T = -W_anti/2 - W_anti/2 = -W_anti  ✓
    U_svd, S_svd, Vh_svd = torch.linalg.svd(W_anti, full_matrices=False)
    k = min(r_j, S_svd.numel())
    sqrt_S = (S_svd[:k] / 2.0).sqrt()          # sqrt(σ_i / 2)
    U_init = -U_svd[:, :k] * sqrt_S.unsqueeze(0)   # (d, k)
    V_init =  Vh_svd[:k].T * sqrt_S.unsqueeze(0)   # (d, k)
    if k < r_j:
        pad = torch.randn(W.shape[0], r_j - k) * 0.01
        U_init = torch.cat([U_init, pad], dim=1)
        V_init = torch.cat([V_init, pad.clone()], dim=1)

    dtype = proj.U_attn.dtype
    with torch.no_grad():
        # Attn stream: K_attn ≈ -W
        proj.L_attn.copy_(L_init.to(dtype))
        proj.U_attn.copy_(U_init.to(dtype))
        proj.V_attn.copy_(V_init.to(dtype))
        # FF stream: K_ff ≈ -scale_ff * W  (scale_ff ≈ 0.09–0.15 for these layers)
        proj.L_ff.copy_((L_init * scale_ff).to(dtype))
        proj.U_ff.copy_((U_init * scale_ff).to(dtype))
        proj.V_ff.copy_((V_init * scale_ff).to(dtype))


# ── Alpha noise hooks ─────────────────────────────────────────────────────────

class AlphaNoiseHook:
    """Forward pre-hook: scales J and R components by alpha-derived factors.

    Registered on DualLowRankPortHamiltonianProjection modules. At each
    forward call during training, samples alpha and monkey-patches _K to
    apply the scaling.  Removed on exit or when model is set to eval mode.
    """

    def __init__(self, alpha_min: float, alpha_max: float):
        self.alpha_min = alpha_min
        self.alpha_max = alpha_max
        self._current_alpha = 0.5

    def __call__(self, module, args):
        if not module.training:
            return  # no-op at eval
        alpha = random.uniform(self.alpha_min, self.alpha_max)
        self._current_alpha = alpha
        j_scale = (1.0 - alpha) / 0.5
        r_scale = alpha / 0.5

        # Patch _K for this forward call by temporarily overriding it
        original_K = module._K

        def scaled_K(U, V, L):
            J = torch.matmul(U, V.T) - torch.matmul(V, U.T)
            R = torch.matmul(L, L.T)
            return j_scale * J - r_scale * R

        module._K = scaled_K
        # Restore after forward via a post-hook stored on the module
        module._restore_K = original_K

    @staticmethod
    def make_post_hook():
        def post_hook(module, args, output):
            if hasattr(module, '_restore_K'):
                module._K = module._restore_K
                del module._restore_K
        return post_hook


# ── Swap proj in energy blocks ────────────────────────────────────────────────

def swap_to_structured_proj(model, rank: int, dissipation_rank: int,
                             init_from_weights: bool = True,
                             learnable_alpha: bool = False,
                             residual: bool = False) -> list:
    """Replace unconstrained proj in each energy block with DualLowRankPH.

    Returns list of (block_name, structured_proj) for the patched blocks.
    """
    from lm_engine.hf_models.models.energy.layer import DualLowRankPortHamiltonianProjection

    patched = []
    for name, module in model.named_modules():
        if not (hasattr(module, 'proj') and module.proj is not None
                and hasattr(module.proj, 'weight')
                and module.proj.weight.ndim == 2
                and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
            continue

        d = module.proj.weight.shape[0]
        W_trained = module.proj.weight.data.float()
        scale_ff_val = float(module.scale_ff.item()) if hasattr(module, 'scale_ff') else 1.0
        dtype = module.proj.weight.dtype
        device = module.proj.weight.device

        if residual:
            # Residual mode: frozen base W + trainable structured delta starting near 0.
            # base_weight stored as buffer so it participates in forward but not in grad.
            structured = DualLowRankPortHamiltonianProjection(
                d, rank=rank, dissipation_rank=dissipation_rank,
                learnable_alpha=learnable_alpha,
                base_weight=W_trained.to(dtype),
                base_scale_ff=scale_ff_val,
            ).to(dtype=dtype, device=device)
            # All J/R params already have tiny std (1e-3) from constructor
        else:
            structured = DualLowRankPortHamiltonianProjection(
                d, rank=rank, dissipation_rank=dissipation_rank,
                learnable_alpha=learnable_alpha,
            ).to(dtype=dtype, device=device)
            if init_from_weights:
                init_from_trained_weight(structured, W_trained, scale_ff=scale_ff_val)

        # Replace the proj attribute and update proj_type so forward uses the right branch
        module.proj = structured
        module.proj_type = "dual_low_rank_port_hamiltonian"
        if hasattr(module, 'scale_ff'):
            module.scale_ff.requires_grad_(False)

        patched.append((name, structured))
        mode = "residual" if residual else ("warmstart" if init_from_weights else "random")
        print(f"  Swapped proj in {name}: {d}x{d} → "
              f"rank={rank}, dissipation_rank={dissipation_rank}, "
              f"mode={mode}  "
              f"[{sum(p.numel() for p in structured.parameters() if p.requires_grad):,} trainable params]")

    return patched


# ── Data loading ──────────────────────────────────────────────────────────────

def make_dataloader(tokenizer, dataset_name: str, seq_len: int, batch_size: int,
                    device: str):
    """Stream batches of token sequences from a HuggingFace text dataset."""
    from datasets import load_dataset

    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        text_col = "text"
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        text_col = "text"

    token_buffer = []

    def batch_gen():
        nonlocal token_buffer
        for sample in ds:
            text = sample[text_col].strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            token_buffer.extend(ids)
            token_buffer.append(tokenizer.eos_token_id)
            while len(token_buffer) >= (seq_len + 1) * batch_size:
                chunk = token_buffer[: (seq_len + 1) * batch_size]
                token_buffer = token_buffer[(seq_len + 1) * batch_size :]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                # Yield (input_ids, input_ids) — NOT pre-shifted labels.
                # HF AutoModelForCausalLM shifts internally; pre-shifting here
                # would cause a double-shift and predict t[i+2] instead of t[i+1].
                inp = t[:, :seq_len].to(device)
                yield inp, inp

    return batch_gen()


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    import wandb
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import lm_engine.hf_models  # noqa: F401

    # ── Load model ────────────────────────────────────────────────────────────
    print(f"\nLoading checkpoint: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Total params: {total_params:,}")

    # ── Swap proj ─────────────────────────────────────────────────────────────
    print("\nSwapping energy block projections → dual_low_rank_port_hamiltonian")
    patched = swap_to_structured_proj(
        model, args.rank, args.dissipation_rank,
        init_from_weights=not args.no_warmstart and not args.residual,
        learnable_alpha=args.learnable_alpha,
        residual=args.residual,
    )
    if not patched:
        raise RuntimeError("No energy blocks found — check model architecture.")

    # ── Freeze everything, then re-enable only structured proj params ─────────
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    proj_params = []
    for _, structured in patched:
        for p in structured.parameters():
            p.requires_grad_(True)
            proj_params.append(p)

    # ── PSD-only: zero and freeze J (U/V) params after enabling all above ─────
    if args.psd_only:
        print("PSD-only mode: zeroing J factors (U, V); training only R = LL^T")
        for _, structured in patched:
            for pname in ("U_attn", "V_attn", "U_ff", "V_ff"):
                param = getattr(structured, pname)
                with torch.no_grad():
                    param.zero_()
                param.requires_grad_(False)
        proj_params = [p for p in proj_params if p.requires_grad]

    # ── Optionally unfreeze backbone with a smaller LR ────────────────────────
    backbone_params = []
    if args.backbone_lr > 0:
        proj_param_ids = {id(p) for p in proj_params}
        for name, param in model.named_parameters():
            if id(param) not in proj_param_ids:
                param.requires_grad_(True)
                backbone_params.append(param)
        print(f"Backbone LR: {args.backbone_lr:.1e}  "
              f"({sum(p.numel() for p in backbone_params):,} backbone params unfrozen)")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.2f}%)")

    # ── Alpha noise hooks ─────────────────────────────────────────────────────
    # Disable noise if learnable_alpha is active (alpha is trained directly).
    use_alpha_noise = not args.no_alpha_noise and not args.learnable_alpha
    pre_hooks, post_hooks = [], []
    if use_alpha_noise:
        print(f"Alpha noise enabled: U({args.alpha_min}, {args.alpha_max})")
        hook_obj = AlphaNoiseHook(args.alpha_min, args.alpha_max)
        for _, structured in patched:
            ph = structured.register_forward_pre_hook(hook_obj)
            poh = structured.register_forward_hook(AlphaNoiseHook.make_post_hook())
            pre_hooks.append(ph)
            post_hooks.append(poh)
    elif args.learnable_alpha:
        print("Learnable alpha enabled — alpha noise disabled.")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    param_groups = [{"params": proj_params, "lr": args.lr}]
    if backbone_params:
        param_groups.append({"params": backbone_params, "lr": args.backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=0.1, betas=(0.9, 0.95))
    warmup_steps = max(1, args.steps // 20)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nStreaming data from: {args.dataset}")
    data_gen = make_dataloader(tokenizer, args.dataset, args.seq_len,
                               args.batch_size, args.device)

    # ── Save dir ──────────────────────────────────────────────────────────────
    ckpt_name = Path(args.checkpoint).name
    save_dir = Path(args.save_dir or (
        Path(__file__).parent.parent.parent / "results" / "structured-proj" /
        f"{ckpt_name}_rank{args.rank}_dr{args.dissipation_rank}"
    ))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        config=vars(args),
    )

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0

    print(f"\nTraining for {args.steps} steps "
          f"(grad_accum={args.grad_accum}, effective batch={args.batch_size * args.grad_accum})")

    for input_ids, labels in data_gen:
        out = model(input_ids, labels=labels)
        loss = out.loss / args.grad_accum
        loss.backward()
        accum_loss += loss.item()

        if (step + 1) % args.grad_accum == 0:
            nn.utils.clip_grad_norm_(proj_params, 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            global_step = (step + 1) // args.grad_accum

            if global_step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                # accum_loss = sum of (out.loss / grad_accum) over log_interval*grad_accum
                # micro-steps.  Dividing by log_interval recovers the per-step average.
                avg_loss = accum_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 20))
                print(f"  step {global_step:6d}/{args.steps}  "
                      f"loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr_now:.2e}")
                wandb.log({"train/loss": avg_loss,
                           "train/ppl": ppl,
                           "train/lr": lr_now,
                           "step": global_step})
                accum_loss = 0.0

            # Per-layer J/R norm and alpha logging
            if global_step % args.norm_log_interval == 0:
                log_dict = {"step": global_step}
                with torch.no_grad():
                    for blk_name, structured in patched:
                        tag = blk_name.replace(".", "_")
                        j_norm_a = structured.get_J_attn().norm().item()
                        r_norm_a = structured.get_R_attn().norm().item()
                        j_norm_f = structured.get_J_ff().norm().item()
                        r_norm_f = structured.get_R_ff().norm().item()
                        log_dict[f"norms/{tag}/J_attn"] = j_norm_a
                        log_dict[f"norms/{tag}/R_attn"] = r_norm_a
                        log_dict[f"norms/{tag}/J_ff"]   = j_norm_f
                        log_dict[f"norms/{tag}/R_ff"]   = r_norm_f
                        log_dict[f"norms/{tag}/J_R_ratio_attn"] = (
                            j_norm_a / (r_norm_a + 1e-8))
                        log_dict[f"norms/{tag}/J_R_ratio_ff"] = (
                            j_norm_f / (r_norm_f + 1e-8))
                        if args.learnable_alpha:
                            log_dict[f"alpha/{tag}/attn"] = structured.get_alpha_attn()
                            log_dict[f"alpha/{tag}/ff"]   = structured.get_alpha_ff()
                wandb.log(log_dict)

            if global_step % args.save_interval == 0:
                ckpt_path = save_dir / f"step_{global_step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved to {ckpt_path}")

            if global_step >= args.steps:
                break

        step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    for h in pre_hooks + post_hooks:
        h.remove()

    final_path = save_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)

    # Save training config
    with open(save_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\nTraining complete. Model saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
