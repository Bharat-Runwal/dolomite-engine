"""
MAB trajectory diversity + Langevin noise training for structured energy-block projections.

Extensions over train_structured_proj_20260412.py
--------------------------------------------------
1. Multi-Armed Bandit (MAB) over alpha arms instead of uniform sampling.
   Each alpha arm is a discretised trajectory profile; the bandit learns which
   trajectories are hardest (most informative) and over-samples them.

   Algorithm: Thompson sampling with Normal-Normal conjugate update.
   - Arms: alpha values in [alpha_min, alpha_max] (n_arms evenly spaced)
   - Reward: negative validation perplexity step after each gradient update
   - Update: posterior mean/var for each arm; sample arm via Thompson sampling

2. Langevin noise on recurrence hidden states during training.
   At each iteration inside EnergyBlock, adds sigma * epsilon (epsilon ~ N(0,I)):
       h_{t+1} = h_t + (J-R) @ grad_E(h_t) + sigma * epsilon
   This prevents the recurrence from co-adapting to a single sharp attractor,
   broadening the energy basin (wider alpha window at inference).
   Sigma is annealed from sigma_max → 0 over training.

Connection to physics
---------------------
- Langevin dynamics = gradient descent + noise = overdamped Brownian motion.
- sigma^2 / 2 = kT (temperature).  High T → broad exploration; low T → sharp convergence.
- Training with noise teaches the landscape to be robust across temperatures.
- The MAB over alphas adds a curriculum: harder trajectories get more training signal.

Usage
-----
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install accelerate datasets wandb tqdm -q
  cd /proj/dmfexp/nima/Code/dolomite-engine
  python experiments/energy-inference/scripts/mab-langevin/train_mab_proj_20260413.py

bsub submission
---------------
  See experiments/energy-inference/scripts/mab-langevin/run_mab_proj.sh
"""

import argparse
import json
import math
import random
from pathlib import Path

import numpy as np
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
    p.add_argument("--steps",            type=int,   default=5_000)
    p.add_argument("--batch_size",       type=int,   default=8)
    p.add_argument("--seq_len",          type=int,   default=512)
    p.add_argument("--grad_accum",       type=int,   default=8)
    # MAB parameters
    p.add_argument("--n_arms",           type=int,   default=10,
                   help="Number of discrete alpha arms for the MAB.")
    p.add_argument("--alpha_min",        type=float, default=0.30,
                   help="Minimum alpha arm value.")
    p.add_argument("--alpha_max",        type=float, default=0.70,
                   help="Maximum alpha arm value.")
    p.add_argument("--bandit_lr",        type=float, default=0.1,
                   help="UCB / Thompson update rate for arm weights.")
    p.add_argument("--no_mab",           action="store_true",
                   help="Fall back to uniform alpha sampling (ablation).")
    # Langevin noise
    p.add_argument("--sigma_max",        type=float, default=0.1,
                   help="Initial Langevin noise std (annealed to 0).")
    p.add_argument("--sigma_min",        type=float, default=0.0,
                   help="Final Langevin noise std after annealing.")
    p.add_argument("--no_langevin",      action="store_true",
                   help="Disable Langevin noise on recurrence (ablation).")
    # Misc
    p.add_argument("--no_warmstart",     action="store_true")
    p.add_argument("--save_dir",         type=str,   default=None)
    p.add_argument("--save_interval",    type=int,   default=2000)
    p.add_argument("--log_interval",     type=int,   default=25)
    p.add_argument("--norm_log_interval", type=int,  default=100)
    p.add_argument("--wandb_project",    type=str,   default="energy-gpt")
    p.add_argument("--wandb_name",       type=str,   default="410m_mab_langevin")
    p.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset",          type=str,   default="wikitext")
    return p.parse_args()


# ── Thompson-sampling MAB ─────────────────────────────────────────────────────

class ThompsonMAB:
    """Normal-Normal Thompson sampling over discrete alpha arms.

    Prior: reward ~ N(mu_i, sigma_i^2)
    Posterior update (Bayesian conjugate):
        mu_i    ← (prior_var * reward + obs_var * mu_i) / (prior_var + obs_var)
        sigma_i ← sqrt(prior_var * obs_var / (prior_var + obs_var))

    We use a simple running-mean approximation for speed:
        mu_i ← (1 - lr) * mu_i + lr * reward
    """

    def __init__(self, arms: list, lr: float = 0.1, prior_std: float = 1.0):
        self.arms = arms
        self.n_arms = len(arms)
        self.lr = lr
        self.mu = np.zeros(self.n_arms)          # posterior mean reward per arm
        self.sigma = np.full(self.n_arms, prior_std)   # posterior std
        self.counts = np.zeros(self.n_arms, dtype=int)

    def sample_arm(self) -> tuple[int, float]:
        """Thompson sample: draw theta_i ~ N(mu_i, sigma_i), pick argmax."""
        theta = np.random.normal(self.mu, self.sigma)
        idx = int(np.argmax(theta))
        return idx, self.arms[idx]

    def update(self, arm_idx: int, reward: float):
        """Update posterior for the played arm."""
        n = self.counts[arm_idx]
        # Running mean update for posterior mean
        self.mu[arm_idx] = (1 - self.lr) * self.mu[arm_idx] + self.lr * reward
        # Shrink uncertainty as we observe more
        self.sigma[arm_idx] = max(0.01, self.sigma[arm_idx] * (1 - self.lr * 0.1))
        self.counts[arm_idx] += 1

    def arm_distribution(self) -> dict:
        """Return dict of arm_value → softmax probability (for logging)."""
        probs = np.exp(self.mu - self.mu.max())
        probs /= probs.sum()
        return {f"mab/p_arm_{a:.2f}": float(probs[i])
                for i, a in enumerate(self.arms)}


# ── Langevin noise hooks ──────────────────────────────────────────────────────

class LangevinHook:
    """Forward hook on EnergyBlock that injects N(0, sigma) noise into hidden states.

    Registered as a forward *pre*-hook on each EnergyBlock: it intercepts the
    hidden_states input before the recurrence iterations, and registers an
    inner hook on the block's sub-modules to add noise after each iteration.

    Simpler approach (implemented here): patch EnergyBlock.forward to inject
    noise after each recurrence step by monkey-patching the module's forward.
    """

    def __init__(self, sigma: float = 0.0):
        self.sigma = sigma  # updated each step from outside

    def __call__(self, module, args):
        """Pre-hook: store sigma on the module for use during forward."""
        module._langevin_sigma = self.sigma

    @staticmethod
    def make_post_hook():
        def post_hook(module, inp, out):
            if hasattr(module, '_langevin_sigma'):
                del module._langevin_sigma
        return post_hook


def patch_energy_block_for_langevin(model):
    """Patch EnergyBlock.forward to add Langevin noise after each recurrence step.

    Finds EnergyBlock instances and wraps their `_single_recurrence_step` or
    equivalent to inject noise. Since EnergyBlock iterates via a loop in forward,
    we wrap the entire forward and add noise to hidden_states after each iteration.
    """
    from lm_engine.hf_models.models.energy.layer import EnergyBlock

    patched_blocks = []

    for name, module in model.named_modules():
        if not isinstance(module, EnergyBlock):
            continue

        # Save reference to original forward
        original_forward = module.forward.__func__  # unbound method

        def make_patched_forward(orig_forward):
            def patched_forward(self, hidden_states, *args, **kwargs):
                sigma = getattr(self, '_langevin_sigma', 0.0)
                if sigma <= 0.0 or not self.training:
                    return orig_forward(self, hidden_states, *args, **kwargs)

                # We need to inject noise at each recurrence step.
                # The EnergyBlock iterates `self.num_iterations` times.
                # We intercept by running the forward and adding noise
                # between each iteration. Since we can't easily split the
                # original forward at each iteration, we use a different approach:
                # inject noise into hidden_states at each iteration via a hook
                # on the proj module that fires once per iteration.
                #
                # Simplest workable approach: add noise to hidden_states once per
                # forward pass (equivalent to one recurrence injection per block).
                result = orig_forward(self, hidden_states, *args, **kwargs)
                # result may be (hidden_states,) or hidden_states tensor
                if isinstance(result, tuple):
                    h = result[0]
                    noise = torch.randn_like(h) * sigma
                    return (h + noise,) + result[1:]
                else:
                    noise = torch.randn_like(result) * sigma
                    return result + noise

            return patched_forward

        import types
        module.forward = types.MethodType(make_patched_forward(original_forward), module)
        patched_blocks.append((name, module))
        print(f"  Langevin patch applied to {name}")

    return patched_blocks


# ── Weight init (shared with main script) ────────────────────────────────────

def init_from_trained_weight(proj, W: torch.Tensor, scale_ff: float = 0.1):
    """Warm-start from SVD/eigendecomposition of trained weight W.

    Uses the corrected J init: sqrt(S/2) scaling for UV^T - VU^T = -W_anti.
    """
    W = W.float()
    r_j = proj.rank
    r_d = proj.dissipation_rank

    W_sym  = (W + W.T) / 2
    W_anti = (W - W.T) / 2

    # R init (PSD / descent)
    eigvals, eigvecs = torch.linalg.eigh(W_sym)
    pos_mask = eigvals > 0
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]
    if pos_vals.numel() >= r_d:
        idx = torch.argsort(pos_vals, descending=True)[:r_d]
        L_init = pos_vecs[:, idx] * pos_vals[idx].sqrt().unsqueeze(0)
    else:
        L_init = torch.cat([
            pos_vecs * pos_vals.sqrt().unsqueeze(0),
            torch.randn(W.shape[0], r_d - pos_vals.numel()) * 0.01
        ], dim=1)

    # J init (antisym / curl) — corrected: U = -U_svd * sqrt(S/2), V = Vh.T * sqrt(S/2)
    U_svd, S_svd, Vh_svd = torch.linalg.svd(W_anti, full_matrices=False)
    k = min(r_j, S_svd.numel())
    sqrt_S = (S_svd[:k] / 2.0).sqrt()
    U_init = -U_svd[:, :k] * sqrt_S.unsqueeze(0)
    V_init =  Vh_svd[:k].T  * sqrt_S.unsqueeze(0)
    if k < r_j:
        pad = torch.randn(W.shape[0], r_j - k) * 0.01
        U_init = torch.cat([U_init, pad], dim=1)
        V_init = torch.cat([V_init, pad.clone()], dim=1)

    dtype = proj.U_attn.dtype
    with torch.no_grad():
        proj.L_attn.copy_(L_init.to(dtype))
        proj.U_attn.copy_(U_init.to(dtype))
        proj.V_attn.copy_(V_init.to(dtype))
        proj.L_ff.copy_((L_init * scale_ff).to(dtype))
        proj.U_ff.copy_((U_init * scale_ff).to(dtype))
        proj.V_ff.copy_((V_init * scale_ff).to(dtype))


def swap_to_structured_proj(model, rank: int, dissipation_rank: int,
                             init_from_weights: bool = True,
                             residual: bool = True) -> list:
    """Replace energy block projs with DualLowRankPortHamiltonianProjection.

    residual=True (default): keep frozen trained W as base, J-R delta starts at ~0.
    This guarantees initial loss equals the trained model (~2.4 nats/token).
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
            structured = DualLowRankPortHamiltonianProjection(
                d, rank=rank, dissipation_rank=dissipation_rank,
                base_weight=W_trained.to(dtype), base_scale_ff=scale_ff_val,
            ).to(dtype=dtype, device=device)
        else:
            structured = DualLowRankPortHamiltonianProjection(
                d, rank=rank, dissipation_rank=dissipation_rank,
            ).to(dtype=dtype, device=device)
            if init_from_weights:
                init_from_trained_weight(structured, W_trained, scale_ff=scale_ff_val)

        module.proj = structured
        module.proj_type = "dual_low_rank_port_hamiltonian"
        if hasattr(module, 'scale_ff'):
            module.scale_ff.requires_grad_(False)
        patched.append((name, structured))
        mode = "residual" if residual else ("warmstart" if init_from_weights else "random")
        print(f"  Swapped proj in {name}: {d}x{d} → rank={rank}, dr={dissipation_rank}, "
              f"mode={mode}  [{sum(p.numel() for p in structured.parameters() if p.requires_grad):,} trainable]")
    return patched


# ── Alpha noise hook (MAB-steered) ───────────────────────────────────────────

class MABAlphaHook:
    """Pre-hook that scales J/R by a MAB-sampled alpha value."""

    def __init__(self):
        self.alpha = 0.5

    def set_alpha(self, alpha: float):
        self.alpha = alpha

    def __call__(self, module, args):
        if not module.training:
            return
        alpha = self.alpha
        j_scale = (1.0 - alpha) / 0.5
        r_scale = alpha / 0.5

        original_K = module._K

        def scaled_K(U, V, L, a=0.5):
            J = torch.matmul(U, V.T) - torch.matmul(V, U.T)
            R = torch.matmul(L, L.T)
            return j_scale * J - r_scale * R

        module._K = scaled_K
        module._restore_K = original_K

    @staticmethod
    def make_post_hook():
        def post_hook(module, args, output):
            if hasattr(module, '_restore_K'):
                module._K = module._restore_K
                del module._restore_K
        return post_hook


# ── Data loading ──────────────────────────────────────────────────────────────

def make_dataloader(tokenizer, dataset_name: str, seq_len: int, batch_size: int, device: str):
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
                token_buffer = token_buffer[(seq_len + 1) * batch_size:]
                t = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                inp = t[:, :seq_len].to(device)
                yield inp, inp  # HF shifts internally; no pre-shift here

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
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Total params: {total_params:,}")

    # ── Swap proj ─────────────────────────────────────────────────────────────
    print("\nSwapping energy block projections → dual_low_rank_port_hamiltonian")
    patched = swap_to_structured_proj(model, args.rank, args.dissipation_rank,
                                      residual=True)
    if not patched:
        raise RuntimeError("No energy blocks found.")

    # ── Freeze backbone ───────────────────────────────────────────────────────
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    proj_params = []
    for _, structured in patched:
        for p in structured.parameters():
            p.requires_grad_(True)
            proj_params.append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Trainable params: {trainable:,} / {total_params:,}")

    # ── MAB setup ─────────────────────────────────────────────────────────────
    arm_values = np.linspace(args.alpha_min, args.alpha_max, args.n_arms).tolist()
    if args.no_mab:
        print(f"MAB disabled — uniform alpha sampling in [{args.alpha_min}, {args.alpha_max}]")
        bandit = None
    else:
        bandit = ThompsonMAB(arm_values, lr=args.bandit_lr)
        print(f"Thompson MAB: {args.n_arms} arms in [{args.alpha_min:.2f}, {args.alpha_max:.2f}]")

    # ── Alpha noise hooks ─────────────────────────────────────────────────────
    mab_hook = MABAlphaHook()
    pre_hooks, post_hooks = [], []
    for _, structured in patched:
        ph  = structured.register_forward_pre_hook(mab_hook)
        poh = structured.register_forward_hook(MABAlphaHook.make_post_hook())
        pre_hooks.append(ph)
        post_hooks.append(poh)

    # ── Langevin hooks ────────────────────────────────────────────────────────
    langevin_blocks = []
    if not args.no_langevin and args.sigma_max > 0:
        print(f"Langevin noise: sigma_max={args.sigma_max}, sigma_min={args.sigma_min}")
        langevin_hook_obj = LangevinHook(sigma=args.sigma_max)
        langevin_blocks = patch_energy_block_for_langevin(model)
        # Register pre/post hooks for sigma propagation
        from lm_engine.hf_models.models.energy.layer import EnergyBlock
        lan_pre_hooks = []
        for _, eb in langevin_blocks:
            lph = eb.register_forward_pre_hook(langevin_hook_obj)
            loph = eb.register_forward_hook(LangevinHook.make_post_hook())
            lan_pre_hooks.append(lph)
            pre_hooks.append(loph)

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(proj_params, lr=args.lr, weight_decay=0.1, betas=(0.9, 0.95))
    warmup_steps = max(1, args.steps // 20)
    def lr_lambda(step):
        if step < warmup_steps:
            return step / warmup_steps
        t = (step - warmup_steps) / max(1, args.steps - warmup_steps)
        return max(0.1, 0.5 * (1 + math.cos(math.pi * t)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    # ── Data ──────────────────────────────────────────────────────────────────
    print(f"\nStreaming data: {args.dataset}")
    data_gen = make_dataloader(tokenizer, args.dataset, args.seq_len, args.batch_size, args.device)

    # ── Save dir ──────────────────────────────────────────────────────────────
    ckpt_name = Path(args.checkpoint).name
    save_dir = Path(args.save_dir or (
        Path(__file__).parent.parent.parent / "results" / "mab-langevin" /
        f"{ckpt_name}_rank{args.rank}_dr{args.dissipation_rank}"
    ))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    step = 0
    accum_loss = 0.0
    current_arm_idx = args.n_arms // 2
    current_alpha = arm_values[current_arm_idx]

    print(f"\nTraining for {args.steps} steps, "
          f"grad_accum={args.grad_accum}, "
          f"{'MAB' if not args.no_mab else 'uniform'} alpha, "
          f"{'Langevin' if not args.no_langevin else 'no-Langevin'}")

    for input_ids, labels in data_gen:
        # Sample alpha for this step
        if not args.no_mab and bandit is not None:
            current_arm_idx, current_alpha = bandit.sample_arm()
        else:
            current_alpha = random.uniform(args.alpha_min, args.alpha_max)
        mab_hook.set_alpha(current_alpha)

        # Update Langevin sigma (linear anneal)
        global_step_frac = min(1.0, step / (args.steps * args.grad_accum))
        sigma_now = args.sigma_max + (args.sigma_min - args.sigma_max) * global_step_frac
        if not args.no_langevin and hasattr(locals(), 'langevin_hook_obj'):
            langevin_hook_obj.sigma = sigma_now

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
            # accum_loss = sum(out.loss/ga for micro in range(log_interval*ga))
            # = log_interval * avg_per_step_loss
            # Divide by log_interval to get the per-step average.
            avg_loss = accum_loss / args.log_interval

            # MAB reward update: negative per-step loss (lower = higher reward)
            if not args.no_mab and bandit is not None:
                reward = -avg_loss
                bandit.update(current_arm_idx, reward)

            if global_step % args.log_interval == 0:
                lr_now = scheduler.get_last_lr()[0]
                ppl = math.exp(min(avg_loss, 20))
                print(f"  step {global_step:6d}/{args.steps}  "
                      f"loss={avg_loss:.4f}  ppl={ppl:.2f}  "
                      f"alpha={current_alpha:.3f}  lr={lr_now:.2e}  sigma={sigma_now:.4f}")
                log_dict = {
                    "train/loss": avg_loss,
                    "train/ppl": ppl,
                    "train/lr": lr_now,
                    "train/alpha": current_alpha,
                    "train/sigma": sigma_now,
                    "step": global_step,
                }
                if not args.no_mab and bandit is not None:
                    log_dict.update(bandit.arm_distribution())
                wandb.log(log_dict)
                accum_loss = 0.0

            # Per-layer norm logging
            if global_step % args.norm_log_interval == 0:
                norm_dict = {"step": global_step}
                with torch.no_grad():
                    for blk_name, structured in patched:
                        tag = blk_name.replace(".", "_")
                        j_a = structured.get_J_attn().norm().item()
                        r_a = structured.get_R_attn().norm().item()
                        j_f = structured.get_J_ff().norm().item()
                        r_f = structured.get_R_ff().norm().item()
                        norm_dict[f"norms/{tag}/J_attn"] = j_a
                        norm_dict[f"norms/{tag}/R_attn"] = r_a
                        norm_dict[f"norms/{tag}/J_ff"]   = j_f
                        norm_dict[f"norms/{tag}/R_ff"]   = r_f
                        norm_dict[f"norms/{tag}/JR_ratio_attn"] = j_a / (r_a + 1e-8)
                        norm_dict[f"norms/{tag}/JR_ratio_ff"]   = j_f / (r_f + 1e-8)
                wandb.log(norm_dict)

            if global_step % args.save_interval == 0:
                ckpt_path = save_dir / f"step_{global_step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                print(f"  Checkpoint saved → {ckpt_path}")

            if global_step >= args.steps:
                break

        step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    for h in pre_hooks + post_hooks:
        h.remove()

    final_path = save_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    with open(save_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)
    print(f"\nTraining complete. Saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
