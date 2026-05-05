"""
Per-iteration alpha experiment: structured proj on h.10 only (penultimate block, 6 iters).

Motivation
----------
Previous learnable-alpha runs used a single scalar alpha per energy block.  That alpha
converged to ~0.5 for all blocks because the model captured its curl preference via the
antisymmetric structure of V^TU (99% antisymmetric) rather than through the scalar alpha.
Alpha was a flat direction in the loss landscape.

This experiment asks a different question: within a SINGLE recurrence block (h.10, which
runs 6 iterations), does a per-iteration α_t (t=0..5) converge to an interesting pattern?

Hypotheses:
  - Early iters: more curl  (explore the manifold)
  - Late  iters: more descent (converge to attractor)
  → α_t should INCREASE from 0 to 1 as t increases

Only h.10 is modified:
  - DualLowRankPortHamiltonianProjection with num_iters=6, learnable_alpha=True
  - Trained on the 6 per-iter alpha values + J/R matrix params

All other energy blocks (h.8, h.9, h.11) remain at their original unconstrained proj weights
(frozen — no gradient flows through them).

The J/R decomposition for h.10 is warm-started from the trained unconstrained W (same as
_20260412.py) so the initial loss matches the trained model.

Usage
-----
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install accelerate datasets wandb tqdm -q
  cd /proj/dmfexp/nima/Code/dolomite-engine
  python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260413.py

bsub submission
---------------
  see run_per_iter_alpha.sh in the same directory.
"""

import argparse
import json
import math
import random
from pathlib import Path

import torch
import torch.nn as nn


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", default=(
        "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_400m/410m_hybrid_s8e4_lr1e3_60k"
    ))
    p.add_argument("--target_block",     type=int,   default=10,
                   help="Block index to apply per-iter structured proj. Default h.10 (6 iters).")
    p.add_argument("--num_iters",        type=int,   default=6,
                   help="Number of recurrence iterations for target block. Must match config.")
    p.add_argument("--rank",             type=int,   default=256)
    p.add_argument("--dissipation_rank", type=int,   default=128)
    p.add_argument("--lr",               type=float, default=3e-4)
    p.add_argument("--steps",            type=int,   default=10_000)
    p.add_argument("--batch_size",       type=int,   default=2)
    p.add_argument("--seq_len",          type=int,   default=4096)
    p.add_argument("--grad_accum",       type=int,   default=8)
    p.add_argument("--no_warmstart",     action="store_true",
                   help="Random init instead of SVD decomposition from trained W.")
    p.add_argument("--save_dir",         type=str,   default=None)
    p.add_argument("--save_interval",    type=int,   default=2000)
    p.add_argument("--log_interval",     type=int,   default=50)
    p.add_argument("--norm_log_interval",type=int,   default=200)
    p.add_argument("--eval_interval",    type=int,   default=250)
    p.add_argument("--eval_batches",     type=int,   default=50)
    p.add_argument("--wandb_project",    type=str,   default="energy-gpt")
    p.add_argument("--wandb_name",       type=str,   default="410m_per_iter_alpha_h10")
    p.add_argument("--device",           type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--dataset",          type=str,   default="nematron")
    return p.parse_args()


# ── Reuse weight init from _20260412 ─────────────────────────────────────────

def init_from_trained_weight(proj, W: torch.Tensor, scale_ff: float = 0.1):
    """Warm-start DualLowRankPortHamiltonianProjection from a trained unconstrained W.

    Identical to the version in train_structured_proj_20260412.py.
    """
    import math as _math
    W = W.float()
    r_j = proj.rank
    r_d = proj.dissipation_rank

    W_sym  = (W + W.T) / 2
    W_anti = (W - W.T) / 2

    eigvals, eigvecs = torch.linalg.eigh(W_sym)
    pos_mask = eigvals > 0
    pos_vals = eigvals[pos_mask]
    pos_vecs = eigvecs[:, pos_mask]
    _sqrt2 = _math.sqrt(2)
    if pos_vals.numel() >= r_d:
        idx = torch.argsort(pos_vals, descending=True)[:r_d]
        L_init = pos_vecs[:, idx] * (pos_vals[idx].sqrt() * _sqrt2).unsqueeze(0)
    else:
        L_init = torch.cat([
            pos_vecs * (pos_vals.sqrt() * _sqrt2).unsqueeze(0),
            torch.randn(W.shape[0], r_d - pos_vals.numel()) * 0.01
        ], dim=1)

    U_svd, S_svd, Vh_svd = torch.linalg.svd(W_anti, full_matrices=False)
    k = min(r_j, S_svd.numel())
    sqrt_S = S_svd[:k].sqrt()
    U_init = -U_svd[:, :k] * sqrt_S.unsqueeze(0)
    V_init =  Vh_svd[:k].T * sqrt_S.unsqueeze(0)
    if k < r_j:
        pad = torch.randn(W.shape[0], r_j - k) * 0.01
        U_init = torch.cat([U_init, pad], dim=1)
        V_init = torch.cat([V_init, pad.clone()], dim=1)

    dtype = proj.U_attn.dtype
    with torch.no_grad():
        proj.L_attn.copy_(L_init.to(dtype))
        proj.U_attn.copy_(U_init.to(dtype))
        proj.V_attn.copy_(V_init.to(dtype))
        sf = _math.sqrt(abs(scale_ff))
        proj.L_ff.copy_((L_init * sf).to(dtype))
        proj.U_ff.copy_((U_init * sf).to(dtype))
        proj.V_ff.copy_((V_init * sf).to(dtype))
    # log_alpha_attn / log_alpha_ff initialised to 0 (sigmoid → 0.5) by default.
    # That mirrors the trained model's α=0.5 balanced mix.


# ── Swap only the target block ────────────────────────────────────────────────

def swap_target_block(model, target_block: int, num_iters: int,
                      rank: int, dissipation_rank: int,
                      init_from_weights: bool = True):
    """Replace the proj of block h.{target_block} with DualLowRankPH (num_iters per-iter alpha).

    Returns (block_name, structured_proj) for the single patched block.
    """
    from lm_engine.hf_models.models.energy.layer import DualLowRankPortHamiltonianProjection

    for name, module in model.named_modules():
        # Detect by module name: "transformer.h.{target_block}"
        suffix = f".{target_block}"
        if not (name.endswith(suffix) and hasattr(module, 'proj')
                and module.proj is not None
                and hasattr(module.proj, 'weight')
                and module.proj.weight.ndim == 2
                and module.proj.weight.shape[0] == module.proj.weight.shape[1]):
            continue

        d = module.proj.weight.shape[0]
        W_trained = module.proj.weight.data.float()
        scale_ff_val = float(module.scale_ff.item()) if hasattr(module, 'scale_ff') else 1.0
        dtype = module.proj.weight.dtype
        device = module.proj.weight.device

        structured = DualLowRankPortHamiltonianProjection(
            d, rank=rank, dissipation_rank=dissipation_rank,
            learnable_alpha=True,
            num_iters=num_iters,
        ).to(dtype=dtype, device=device)

        if init_from_weights:
            init_from_trained_weight(structured, W_trained, scale_ff=scale_ff_val)

        module.proj = structured
        module.proj_type = "dual_low_rank_port_hamiltonian"
        if hasattr(module, 'scale_ff'):
            module.scale_ff.requires_grad_(False)

        n_params = sum(p.numel() for p in structured.parameters() if p.requires_grad)
        print(f"  Swapped proj in {name}: {d}x{d} → "
              f"rank={rank}, dr={dissipation_rank}, num_iters={num_iters}  "
              f"[{n_params:,} trainable params (incl. {2*num_iters} alpha scalars)]")
        return name, structured

    raise RuntimeError(
        f"Block h.{target_block} not found or already swapped — check model structure."
    )


# ── Held-out evaluation (same as _20260412) ──────────────────────────────────

def eval_loss(model, tokenizer, seq_len, batch_size, device, n_batches=50):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation", streaming=True)
    buf = []
    total_loss = 0.0
    count = 0
    total_chars = total_words = total_bytes = total_tokens = 0
    model.eval()
    with torch.no_grad():
        for sample in ds:
            text = sample["text"].strip()
            if not text:
                continue
            ids = tokenizer.encode(text, add_special_tokens=False)
            if not ids:
                continue
            buf.extend(ids)
            buf.append(tokenizer.eos_token_id)
            total_chars  += len(text)
            total_words  += len(text.split())
            total_bytes  += len(text.encode("utf-8"))
            total_tokens += len(ids)
            while len(buf) >= (seq_len + 1) * batch_size and count < n_batches:
                chunk = buf[:(seq_len + 1) * batch_size]
                buf   = buf[(seq_len + 1) * batch_size:]
                t   = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                inp = t[:, :seq_len].to(device)
                out = model(inp, labels=inp)
                total_loss += out.loss.item()
                count += 1
            if count >= n_batches:
                break
    model.train()
    avg_loss = total_loss / max(count, 1)
    cpt = total_chars  / max(total_tokens, 1)
    wpt = total_words  / max(total_tokens, 1)
    bpt = total_bytes  / max(total_tokens, 1)
    return {
        "loss":     avg_loss,
        "ppl":      math.exp(min(avg_loss, 20)),
        "bpd":      avg_loss / (math.log(2) * max(cpt, 1e-9)),
        "byte_ppl": 2 ** (avg_loss / (math.log(2) * max(bpt, 1e-9))),
        "word_ppl": math.exp(min(avg_loss / max(wpt, 1e-9), 20)),
    }


# ── Data loading (same as _20260412) ─────────────────────────────────────────

_NEMATRON_PATHS = [
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0",
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1",
]
_MEGATRON_DTYPES = {
    1: "uint8", 2: "int8",  3: "int16", 4: "int32", 5: "int64",
    6: "float64", 7: "float32", 8: "uint16",
}
_MEGATRON_HEADER = b"MMIDIDX\x00\x00"


def _megatron_batch_gen(paths, seq_len, batch_size, device):
    import struct, numpy as np
    token_buffer = []
    for path in paths:
        idx_path = path + ".idx"
        bin_path = path + ".bin"
        with open(idx_path, "rb") as f:
            header = f.read(9)
            assert header == _MEGATRON_HEADER, f"Bad Megatron header: {idx_path}"
            _version = struct.unpack("<Q", f.read(8))[0]
            dtype_code = struct.unpack("<B", f.read(1))[0]
            dtype = np.dtype(_MEGATRON_DTYPES[dtype_code])
            seq_count = struct.unpack("<Q", f.read(8))[0]
            _doc_count = struct.unpack("<Q", f.read(8))[0]
            offset = f.tell()
        idx_mm  = np.memmap(idx_path, mode="r", order="C")
        buf_view = memoryview(idx_mm)
        seq_lengths  = np.frombuffer(buf_view, dtype=np.int32, count=seq_count, offset=offset)
        seq_pointers = np.frombuffer(buf_view, dtype=np.int64, count=seq_count,
                                     offset=offset + seq_lengths.nbytes)
        bin_buf = memoryview(np.memmap(bin_path, mode="r", dtype=np.uint8))
        indices = list(range(seq_count))
        random.shuffle(indices)
        for doc_idx in indices:
            ptr    = int(seq_pointers[doc_idx])
            length = int(seq_lengths[doc_idx])
            tokens = np.frombuffer(bin_buf, dtype=dtype, count=length, offset=ptr)
            token_buffer.extend(tokens.tolist())
            while len(token_buffer) >= (seq_len + 1) * batch_size:
                chunk = token_buffer[:(seq_len + 1) * batch_size]
                token_buffer = token_buffer[(seq_len + 1) * batch_size:]
                t   = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
                inp = t[:, :seq_len].to(device)
                yield inp, inp


def _hf_batch_gen(tokenizer, dataset_name, seq_len, batch_size, device):
    from datasets import load_dataset
    if dataset_name == "wikitext":
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="train", streaming=True)
    else:
        ds = load_dataset(dataset_name, split="train", streaming=True)
    token_buffer = []
    for sample in ds:
        text = sample["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text, add_special_tokens=False)
        token_buffer.extend(ids)
        token_buffer.append(tokenizer.eos_token_id)
        while len(token_buffer) >= (seq_len + 1) * batch_size:
            chunk = token_buffer[:(seq_len + 1) * batch_size]
            token_buffer = token_buffer[(seq_len + 1) * batch_size:]
            t   = torch.tensor(chunk, dtype=torch.long).view(batch_size, seq_len + 1)
            inp = t[:, :seq_len].to(device)
            yield inp, inp


def make_dataloader(tokenizer, dataset_name, seq_len, batch_size, device):
    if dataset_name == "nematron":
        return _megatron_batch_gen(_NEMATRON_PATHS, seq_len, batch_size, device)
    return _hf_batch_gen(tokenizer, dataset_name, seq_len, batch_size, device)


# ── Training loop ─────────────────────────────────────────────────────────────

def train(args):
    import wandb
    from datetime import datetime
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import lm_engine.hf_models  # noqa: F401

    # ── Save dir ──────────────────────────────────────────────────────────────
    base_dir = Path(args.save_dir) if args.save_dir else (
        Path(__file__).parent.parent.parent / "results" / "structured-proj"
    )
    resume_step = 0
    resume_ckpt_path = None
    resume_wandb_id = None
    save_dir = None

    if not args.save_dir:
        existing_runs = sorted(
            [d for d in base_dir.glob(f"{args.wandb_name}_20*") if d.is_dir()],
            key=lambda d: d.name,
        )
        for run_dir in reversed(existing_runs):
            ckpts = sorted(
                [p for p in run_dir.glob("step_*") if (p / "training_state.pt").exists()],
                key=lambda p: int(p.name.split("_")[1]),
            )
            if ckpts:
                save_dir = run_dir
                resume_ckpt_path = ckpts[-1]
                resume_step = int(resume_ckpt_path.name.split("_")[1])
                saved_state = torch.load(resume_ckpt_path / "training_state.pt",
                                         map_location="cpu")
                resume_wandb_id = saved_state.get("wandb_run_id")
                print(f"\nResuming from {save_dir.name} (step {resume_step})")
                break

    if save_dir is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        save_dir = base_dir / f"{args.wandb_name}_{ts}"
        print(f"\nFresh start: {save_dir.name}")
    save_dir.mkdir(parents=True, exist_ok=True)

    # ── Load model ────────────────────────────────────────────────────────────
    load_path = str(resume_ckpt_path) if resume_ckpt_path else args.checkpoint
    print(f"Loading: {load_path}")
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForCausalLM.from_pretrained(
        load_path,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded. Total params: {total_params:,}")

    # ── Swap only h.{target_block} ────────────────────────────────────────────
    print(f"\nSwapping h.{args.target_block} → DualLowRankPH (num_iters={args.num_iters})")
    blk_name, structured = swap_target_block(
        model,
        target_block=args.target_block,
        num_iters=args.num_iters,
        rank=args.rank,
        dissipation_rank=args.dissipation_rank,
        init_from_weights=not args.no_warmstart,
    )

    # ── Restore structured proj from checkpoint on resume ─────────────────────
    if resume_ckpt_path:
        proj_state_path = resume_ckpt_path / "proj_state.pt"
        if proj_state_path.exists():
            saved_proj = torch.load(proj_state_path, map_location=args.device)
            missing, unexpected = model.load_state_dict(saved_proj, strict=False)
            print(f"  Loaded proj_state.pt  missing={len(missing)}  unexpected={len(unexpected)}")
        else:
            print(f"  WARNING: {proj_state_path} not found — reinitialised from W.")

    # ── Freeze everything, then re-enable only structured proj params ─────────
    for name, param in model.named_parameters():
        param.requires_grad_(False)
    proj_params = []
    for p in structured.parameters():
        p.requires_grad_(True)
        proj_params.append(p)

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable params: {trainable:,} / {total_params:,} "
          f"({100 * trainable / total_params:.3f}%)")
    print("  All alpha scalars initialised to sigmoid(0)=0.5  (balanced mix).")

    # ── Optimizer & scheduler ─────────────────────────────────────────────────
    optimizer = torch.optim.AdamW(proj_params, lr=args.lr, weight_decay=0.1,
                                  betas=(0.9, 0.95))
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
    if resume_step > 0:
        skip = resume_step * args.grad_accum
        print(f"Fast-forwarding data stream by {skip} batches...")
        for _ in range(skip):
            next(data_gen)
        print("Data stream ready.")

    # ── wandb ─────────────────────────────────────────────────────────────────
    wandb.init(
        project=args.wandb_project,
        name=args.wandb_name,
        id=resume_wandb_id,
        resume="allow",
        config=vars(args),
    )

    if resume_step > 0:
        optimizer.load_state_dict(saved_state["optimizer"])
        scheduler.load_state_dict(saved_state["scheduler"])
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        print(f"Restored optimizer + scheduler from step {resume_step}.")

    # ── Training ──────────────────────────────────────────────────────────────
    model.train()
    optimizer.zero_grad()
    step = resume_step * args.grad_accum
    accum_loss = 0.0

    print(f"\nTraining for {args.steps} steps  "
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
                avg_loss = accum_loss / args.log_interval
                ppl = math.exp(min(avg_loss, 20))
                print(f"  step {global_step:6d}/{args.steps}  "
                      f"loss={avg_loss:.4f}  ppl={ppl:.2f}  lr={lr_now:.2e}")
                wandb.log({"train/loss": avg_loss,
                           "train/ppl": ppl,
                           "train/lr": lr_now,
                           "step": global_step})
                accum_loss = 0.0

            if global_step % args.eval_interval == 0:
                vm = eval_loss(model, tokenizer, args.seq_len, args.batch_size,
                               args.device, n_batches=args.eval_batches)
                print(f"  [val] step {global_step:6d}  "
                      f"loss={vm['loss']:.4f}  ppl={vm['ppl']:.2f}  "
                      f"word_ppl={vm['word_ppl']:.2f}  bpd={vm['bpd']:.4f}")
                wandb.log({"val/loss":     vm["loss"],
                           "val/ppl":      vm["ppl"],
                           "val/word_ppl": vm["word_ppl"],
                           "val/bpd":      vm["bpd"],
                           "val/byte_ppl": vm["byte_ppl"],
                           "step": global_step})

            if global_step % args.norm_log_interval == 0:
                log_dict = {"step": global_step}
                with torch.no_grad():
                    tag = blk_name.replace(".", "_")
                    # Per-iter alpha: log each α_t for attn and ff streams
                    alpha_attn_list = structured.get_alpha_attn()   # list of T floats
                    alpha_ff_list   = structured.get_alpha_ff()
                    if isinstance(alpha_attn_list, float):
                        alpha_attn_list = [alpha_attn_list]
                        alpha_ff_list   = [alpha_ff_list] if isinstance(alpha_ff_list, float) else alpha_ff_list
                    for t, (a_a, a_f) in enumerate(zip(alpha_attn_list, alpha_ff_list)):
                        log_dict[f"alpha/{tag}/iter{t}/attn"] = a_a
                        log_dict[f"alpha/{tag}/iter{t}/ff"]   = a_f
                    # Effective curl/descent ratio (norm-adjusted): (1-α)||J|| / (α||R||)
                    j_norm_a = structured.get_J_attn().norm().item()
                    r_norm_a = structured.get_R_attn().norm().item()
                    j_norm_f = structured.get_J_ff().norm().item()
                    r_norm_f = structured.get_R_ff().norm().item()
                    log_dict[f"norms/{tag}/J_attn"] = j_norm_a
                    log_dict[f"norms/{tag}/R_attn"] = r_norm_a
                    log_dict[f"norms/{tag}/J_ff"]   = j_norm_f
                    log_dict[f"norms/{tag}/R_ff"]   = r_norm_f
                    log_dict[f"norms/{tag}/J_R_ratio_attn"] = j_norm_a / (r_norm_a + 1e-8)
                    log_dict[f"norms/{tag}/J_R_ratio_ff"]   = j_norm_f / (r_norm_f + 1e-8)
                    # Norm-adjusted alpha: α_balanced = ||J|| / (||J|| + ||R||)
                    log_dict[f"norms/{tag}/alpha_balanced_attn"] = j_norm_a / (j_norm_a + r_norm_a + 1e-8)
                    log_dict[f"norms/{tag}/alpha_balanced_ff"]   = j_norm_f / (j_norm_f + r_norm_f + 1e-8)
                wandb.log(log_dict)

            if global_step % args.save_interval == 0:
                ckpt_path = save_dir / f"step_{global_step}"
                model.save_pretrained(ckpt_path)
                tokenizer.save_pretrained(ckpt_path)
                _PROJ_KEYS = ("L_attn", "U_attn", "V_attn", "L_ff", "U_ff", "V_ff",
                              "base_weight", "log_alpha")
                proj_state = {k: v.clone()
                              for k, v in model.state_dict().items()
                              if any(x in k for x in _PROJ_KEYS)}
                torch.save(proj_state, ckpt_path / "proj_state.pt")
                torch.save({
                    "global_step": global_step,
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(),
                    "wandb_run_id": wandb.run.id,
                }, ckpt_path / "training_state.pt")
                print(f"  Checkpoint saved → {ckpt_path}")

            if global_step >= args.steps:
                break
        step += 1

    # ── Final save ────────────────────────────────────────────────────────────
    final_path = save_dir / "final"
    model.save_pretrained(final_path)
    tokenizer.save_pretrained(final_path)
    _PROJ_KEYS = ("L_attn", "U_attn", "V_attn", "L_ff", "U_ff", "V_ff",
                  "base_weight", "log_alpha", "log_alpha_attn", "log_alpha_ff")
    proj_state = {k: v.clone()
                  for k, v in model.state_dict().items()
                  if any(x in k for x in _PROJ_KEYS)}
    torch.save(proj_state, final_path / "proj_state.pt")
    print(f"FINAL_CKPT={final_path}")

    with open(save_dir / "train_config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    # ── Print final per-iter alpha distribution ───────────────────────────────
    print("\nFinal per-iteration alpha distribution (h.10):")
    print(f"  {'iter':>4}  {'alpha_attn':>10}  {'alpha_ff':>10}")
    with torch.no_grad():
        alphas_a = structured.get_alpha_attn()
        alphas_f = structured.get_alpha_ff()
        if isinstance(alphas_a, float):
            alphas_a = [alphas_a]
            alphas_f = [alphas_f]
        for t, (a_a, a_f) in enumerate(zip(alphas_a, alphas_f)):
            print(f"  {t:>4}  {a_a:>10.4f}  {a_f:>10.4f}")

    print(f"\nTraining complete. Model saved to {final_path}")
    wandb.finish()


if __name__ == "__main__":
    args = parse_args()
    train(args)
