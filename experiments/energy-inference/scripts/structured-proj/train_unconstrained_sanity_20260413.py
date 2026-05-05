"""
Sanity check: fine-tune only the 4 energy block proj.weight matrices
(unconstrained nn.Linear, exactly as trained) to verify:
  1. Initial loss ≈ 2.4 nats/token (≈ PPL 10.83 on wikitext)
  2. Loss decreases with gradient steps
  3. The training script infra (data, tokenizer, loss scaling) is correct

If initial loss is >> 2.4, the problem is in data/tokenizer mismatch,
NOT in the structured proj architecture.

Usage:
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install accelerate datasets wandb tqdm -q
  python experiments/energy-inference/scripts/structured-proj/train_unconstrained_sanity_20260413.py
"""

import argparse
import math
import torch
import torch.nn as nn
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=(
        "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k"
    ))
    p.add_argument("--lr",          type=float, default=3e-4)
    p.add_argument("--steps",       type=int,   default=500)
    p.add_argument("--batch_size",  type=int,   default=8)
    p.add_argument("--seq_len",     type=int,   default=512)
    p.add_argument("--grad_accum",  type=int,   default=8)
    p.add_argument("--log_interval",type=int,   default=5)
    p.add_argument("--wandb_project",type=str,  default="energy-gpt")
    p.add_argument("--wandb_name",  type=str,   default="410m_unconstrained_sanity")
    p.add_argument("--dataset",     type=str,   default="wikitext")
    p.add_argument("--frozen",      action="store_true",
                   help="If set, freeze EVERYTHING (zero-grad eval mode) to measure "
                        "pure eval loss without any training. This should give ~2.4.")
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def make_dataloader(tokenizer, dataset_name, seq_len, batch_size, device):
    from datasets import load_dataset
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
        text_col = "text"
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
        text_col = "text"
    buf = []

    def gen():
        nonlocal buf
        for s in ds:
            t = s[text_col].strip()
            if not t:
                continue
            buf.extend(tokenizer.encode(t, add_special_tokens=False))
            buf.append(tokenizer.eos_token_id)
            while len(buf) >= (seq_len + 1) * batch_size:
                chunk = buf[:(seq_len+1)*batch_size]
                buf = buf[(seq_len+1)*batch_size:]
                tk = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len+1)
                yield tk[:, :seq_len].to(device), tk[:, 1:].to(device)

    return gen()


def train(args):
    import wandb
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import lm_engine.hf_models  # noqa

    print(f"\nLoading: {args.checkpoint}")
    tokenizer = AutoTokenizer.from_pretrained(args.checkpoint)
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16, device_map=args.device,
    )
    total = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total:,}")

    # Find energy block proj weights (unconstrained nn.Linear)
    proj_params = []
    proj_names = []
    for name, module in model.named_modules():
        if (hasattr(module, 'proj') and module.proj is not None
                and hasattr(module.proj, 'weight')
                and module.proj.weight.ndim == 2
                and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
            proj_names.append(name)

    print(f"\nFound {len(proj_names)} energy blocks with unconstrained proj:")
    for nm in proj_names:
        print(f"  {nm}")

    if args.frozen:
        # Eval-only: measure true eval loss without any training
        print("\n[FROZEN mode] Measuring eval loss with NO gradient updates")
        for p in model.parameters():
            p.requires_grad_(False)
        model.eval()
    else:
        # Freeze everything, then unfreeze only proj.weight
        for p in model.parameters():
            p.requires_grad_(False)
        for name, module in model.named_modules():
            if name in proj_names and hasattr(module.proj, 'weight'):
                module.proj.weight.requires_grad_(True)
                proj_params.append(module.proj.weight)
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"\nTrainable (proj.weight only): {trainable:,} / {total:,} "
              f"({100*trainable/total:.3f}%)")
        model.train()

    wandb.init(project=args.wandb_project, name=args.wandb_name, config=vars(args))

    if not args.frozen:
        optimizer = torch.optim.AdamW(proj_params, lr=args.lr, weight_decay=0.1)
        warmup = max(1, args.steps // 20)
        scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer,
            lambda s: min(1.0, s/warmup) if s < warmup
                      else max(0.1, 0.5*(1+math.cos(math.pi*(s-warmup)/max(1,args.steps-warmup))))
        )
        optimizer.zero_grad()

    data = make_dataloader(tokenizer, args.dataset, args.seq_len, args.batch_size, args.device)

    step = 0
    accum = 0.0
    print(f"\nRunning {'eval' if args.frozen else 'training'} for {args.steps} steps...")

    for inp, lbl in data:
        with (torch.no_grad() if args.frozen else torch.enable_grad()):
            # Pass inp as labels so HF does the single causal shift internally.
            # (lbl from the dataloader is already shifted; passing it causes a
            #  double-shift: model predicts t[i+2] instead of t[i+1].)
            out = model(inp, labels=inp)
        loss_val = out.loss.item()

        if not args.frozen:
            (out.loss / args.grad_accum).backward()
        accum += loss_val

        micro = step + 1
        if micro % args.grad_accum == 0:
            if not args.frozen:
                nn.utils.clip_grad_norm_(proj_params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()

            gs = micro // args.grad_accum
            if gs % args.log_interval == 0:
                avg_loss = accum / (args.log_interval * args.grad_accum)
                ppl = math.exp(min(avg_loss, 20))
                lr_now = scheduler.get_last_lr()[0] if not args.frozen else 0.0
                print(f"  step {gs:5d}/{args.steps}  loss={avg_loss:.4f}  ppl={ppl:.2f}"
                      + (f"  lr={lr_now:.2e}" if not args.frozen else ""))
                wandb.log({"train/loss": avg_loss, "train/ppl": ppl, "step": gs,
                           "train/lr": lr_now})
                accum = 0.0
            if gs >= args.steps:
                break

        step += 1

    wandb.finish()
    print("\nDone.")


if __name__ == "__main__":
    train(parse_args())
