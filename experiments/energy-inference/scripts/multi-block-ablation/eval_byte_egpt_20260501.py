"""
lm-evaluation-harness adapter for the byte-EGPT model.

The byte-EGPT operates on raw UTF-8 bytes (vocab=256) with a w4s2 linear-pool
encoder that compresses the sequence by 2×. This module implements the lm-eval
LM interface so we can run standard benchmarks (arc, hellaswag, piqa, etc.)
on the byte model without any BPE tokenizer.

Key mapping (window=4, stride=2):
  - compressed token k sees input bytes [k*stride - window + stride : k*stride + stride]
    (causal, left-padded)
  - compressed token k predicts bytes at positions [k*stride + window : k*stride + window + stride]
    i.e. the next `stride` bytes AFTER the current window

For loglikelihood(context, continuation):
  1. Encode context + continuation as UTF-8 bytes
  2. Run a single forward pass
  3. Sum log-probs for the continuation byte positions

Usage (called from run_byte_egpt_20260501.sh after training):
  python eval_byte_egpt_20260501.py \\
      --checkpoint <save_dir>/best.pt \\
      --output_path <save_dir>/harness_results.json \\
      --tasks arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,winogrande,wikitext
"""

import argparse
import json
import math
import sys
import os
from pathlib import Path

import torch
import torch.nn.functional as F
from tqdm import tqdm

# lm-eval imports
from lm_eval.api.model import LM
from lm_eval.api.registry import register_model
from lm_eval import evaluator


# ── Load model from checkpoint ──────────────────────────────────────────────

def load_model(checkpoint_path: str, device: str = "cuda"):
    """Load ByteEGPT from a training checkpoint (.pt file)."""
    # The training script lives next to this eval script
    script_dir = Path(__file__).parent
    sys.path.insert(0, str(script_dir))
    from train_byte_egpt_20260501 import ByteEGPT

    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    args = ckpt["args"]

    variant = args.get("variant", "layernorm")
    is_rayleigh  = variant in ("rmsnorm_rayleigh", "rmsnorm_reileigh", "deep_egpt_rayleigh")
    use_rmsnorm    = is_rayleigh
    apply_reileigh = is_rayleigh
    center_block   = {
        "layernorm":          "egpt",
        "rmsnorm_rayleigh":   "egpt",
        "rmsnorm_reileigh":   "egpt",
        "rec_gpt":            "gpt_rec",
        "deep_gpt":           "none",
        "deep_egpt_rayleigh": "egpt_deep",
    }.get(variant, "egpt")

    model = ByteEGPT(
        d_model=args["d_model"],
        d_local=args["d_local"],
        n_head=args["n_head"],
        vocab_size=256,
        window_size=args["window_size"],
        stride=args["stride"],
        block_size=args["block_size"],
        n_pre=args["n_pre"],
        n_post=args["n_post"],
        n_egpt_iter=args["n_egpt_iter"],
        use_rmsnorm=use_rmsnorm,
        apply_reileigh=apply_reileigh,
        center_block=center_block,
    )

    # Strip torch.compile prefix if present
    state = ckpt["model"]
    state = {k.replace("_orig_mod.", ""): v for k, v in state.items()}
    model.load_state_dict(state)
    model.eval()
    model.to(device)

    total = sum(p.numel() for p in model.parameters())
    print(f"Loaded {checkpoint_path}  ({total/1e6:.2f}M params, variant={variant})", flush=True)
    return model


# ── Byte log-prob computation ────────────────────────────────────────────────

def byte_logprobs(model, byte_seq: list[int], device: str,
                  block_size: int = 1024) -> torch.Tensor:
    """Return per-byte log-probs for every byte in byte_seq that the model predicts.

    Returns a tensor of shape (len(byte_seq),) where positions the model
    cannot predict (first window_size bytes) are filled with 0.0.
    """
    W = model.window_size   # 4
    S = model.stride        # 2
    T = len(byte_seq)

    if T > block_size:
        byte_seq = byte_seq[-block_size:]
        T = block_size

    x = torch.tensor(byte_seq, dtype=torch.long).unsqueeze(0).to(device)  # [1, T]
    with torch.no_grad():
        with torch.autocast(device_type="cuda" if "cuda" in device else "cpu",
                            dtype=torch.bfloat16):
            logits, _ = model(x)  # [1, N, stride, 256]

    # logits[0, k, p, :] = distribution for byte at position k*S + W + p
    lps = torch.zeros(T, dtype=torch.float32)
    N = logits.shape[1]
    log_softmax = F.log_softmax(logits[0].float(), dim=-1)  # [N, stride, 256]

    for k in range(N):
        for p in range(S):
            pos = k * S + W + p
            if pos >= T:
                break
            lps[pos] = log_softmax[k, p, byte_seq[pos] if pos < T else 0]

    return lps  # shape (T,), zero where not predicted


# ── lm-eval LM adapter ───────────────────────────────────────────────────────

@register_model("byte_egpt")
class ByteEGPTLM(LM):
    """lm-eval LM wrapper for byte-level EGPT models."""

    def __init__(self, checkpoint: str, device: str = "cuda",
                 block_size: int = 1024, batch_size: int = 1):
        super().__init__()
        self.model = load_model(checkpoint, device)
        self.device = device
        self.block_size = block_size
        self._batch_size = batch_size

    @property
    def eot_token_id(self):
        return 0  # null byte as EOT

    @property
    def max_length(self):
        return self.block_size

    @property
    def max_gen_toks(self):
        return 256

    @property
    def batch_size(self):
        return self._batch_size

    def tok_encode(self, string: str) -> list[int]:
        return list(string.encode("utf-8"))

    def tok_decode(self, tokens: list[int]) -> str:
        return bytes(tokens).decode("utf-8", errors="replace")

    def _get_loglikelihood(self, context: str, continuation: str) -> tuple[float, bool]:
        ctx_bytes  = list(context.encode("utf-8"))
        cont_bytes = list(continuation.encode("utf-8"))

        if not cont_bytes:
            return (0.0, True)

        full = ctx_bytes + cont_bytes
        if len(full) > self.block_size:
            # Keep full continuation, truncate context from the left
            keep_ctx = self.block_size - len(cont_bytes)
            if keep_ctx < 0:
                # Continuation itself exceeds block_size — score only what fits
                cont_bytes = cont_bytes[-self.block_size:]
                ctx_bytes  = []
                full = cont_bytes
            else:
                ctx_bytes = ctx_bytes[-keep_ctx:]
                full = ctx_bytes + cont_bytes

        lps = byte_logprobs(self.model, full, self.device, self.block_size)

        ctx_len = len(ctx_bytes)
        log_prob  = 0.0
        is_greedy = True

        x_tensor = torch.tensor(full, dtype=torch.long).to(self.device)
        W, S = self.model.window_size, self.model.stride

        with torch.no_grad():
            with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu",
                                dtype=torch.bfloat16):
                logits, _ = self.model(x_tensor.unsqueeze(0))  # [1, N, S, 256]

        N = logits.shape[1]
        for i, bv in enumerate(cont_bytes):
            pos = ctx_len + i
            if pos < W:
                continue
            k = (pos - W) // S
            p = (pos - W) % S
            if k >= N:
                continue
            lp_vec = F.log_softmax(logits[0, k, p].float(), dim=-1)
            log_prob += lp_vec[bv].item()
            if lp_vec.argmax().item() != bv:
                is_greedy = False

        return (log_prob, is_greedy)

    def loglikelihood(self, requests) -> list[tuple[float, bool]]:
        results = []
        for req in tqdm(requests, desc="loglikelihood", disable=len(requests) < 20):
            context, continuation = req.args
            results.append(self._get_loglikelihood(context, continuation))
        return results

    def loglikelihood_rolling(self, requests) -> list[float]:
        results = []
        for req in tqdm(requests, desc="loglikelihood_rolling", disable=len(requests) < 20):
            string = req.args[0]
            byte_seq = list(string.encode("utf-8"))
            # Process in block_size windows with 50% overlap for smoothness
            W, S = self.model.window_size, self.model.stride
            total_lp = 0.0
            step = self.block_size - W * 4  # small overlap
            start = 0
            while start < len(byte_seq):
                chunk = byte_seq[start:start + self.block_size]
                lps = byte_logprobs(self.model, chunk, self.device, self.block_size)
                # Sum log-probs for positions that are newly covered (not in previous window)
                new_start = W if start == 0 else 0
                total_lp += lps[new_start:].sum().item()
                start += step
            results.append(total_lp)
        return results

    def generate_until(self, requests) -> list[str]:
        results = []
        W, S = self.model.window_size, self.model.stride
        for req in tqdm(requests, desc="generate_until", disable=len(requests) < 5):
            context    = req.args[0]
            until      = req.args[1].get("until", ["\n"])
            max_new    = req.args[1].get("max_gen_toks", self.max_gen_toks)

            ctx_bytes = list(context.encode("utf-8"))
            generated = []

            for _ in range(max_new):
                full = ctx_bytes + generated
                full = full[-self.block_size:]  # keep within block_size
                x = torch.tensor(full, dtype=torch.long).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    with torch.autocast(device_type="cuda" if "cuda" in self.device else "cpu",
                                        dtype=torch.bfloat16):
                        logits, _ = self.model(x)
                # Get last predicted byte
                T = len(full)
                pos = T  # predict next byte
                k = max(0, (pos - W) // S - 1)
                p = (pos - W) % S if (pos - W) >= 0 else 0
                k = min(k, logits.shape[1] - 1)
                next_byte = logits[0, k, p].argmax().item()
                generated.append(next_byte)

                # Check stop sequences
                gen_str = bytes(generated).decode("utf-8", errors="replace")
                if any(gen_str.endswith(u) for u in until):
                    break

            results.append(bytes(generated).decode("utf-8", errors="replace"))
        return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True, help="Path to best.pt checkpoint")
    p.add_argument("--output_path", default=None, help="Where to write results JSON")
    p.add_argument("--tasks", default=(
        "arc_challenge,arc_easy,boolq,copa,hellaswag,"
        "openbookqa,piqa,sciq,winogrande,mmlu,wikitext"
    ))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_fewshot", type=int, default=0)
    p.add_argument("--iter_sweep", action="store_true",
                   help="Also evaluate at multiple iteration counts to measure test-time "
                        "compute scaling. Saves to iter_sweep_results.json next to output_path.")
    args = p.parse_args()

    if args.output_path is None:
        args.output_path = str(Path(args.checkpoint).parent / "harness_results.json")

    print(f"Evaluating {args.checkpoint}", flush=True)
    print(f"Tasks: {args.tasks}", flush=True)
    print(f"Output: {args.output_path}", flush=True)

    lm = ByteEGPTLM(
        checkpoint=args.checkpoint,
        device=args.device,
        batch_size=args.batch_size,
    )

    task_list = [t.strip() for t in args.tasks.split(",")]

    results = evaluator.simple_evaluate(
        model=lm,
        tasks=task_list,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        device=args.device,
    )

    # Print summary table
    print("\n=== Eval Results ===", flush=True)
    for task, res in sorted(results["results"].items()):
        metrics = {k: v for k, v in res.items() if not k.endswith("_stderr") and k != "alias"}
        print(f"  {task}: {metrics}", flush=True)

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to {args.output_path}", flush=True)

    # ── Test-time compute scaling sweep ──────────────────────────────────────
    # Only meaningful for recurrent models (egpt, gpt_rec) with a shared center block.
    if args.iter_sweep and not isinstance(lm.model.center, (type(None), torch.nn.ModuleList)):
        base_n = lm.model.n_center_iter
        sweep_iters = [1, 3, 5, 7, 9, base_n, base_n + 2, base_n + 5, base_n + 10]
        sweep_iters = sorted(set(n for n in sweep_iters if n > 0))
        sweep_results = {}

        # Use a fast subset: wikitext + boolq + piqa (low latency)
        sweep_tasks = [t for t in task_list if t in ("wikitext", "boolq", "piqa", "arc_easy")]
        print(f"\n=== Test-time compute scaling sweep (n_iter sweep on {sweep_tasks}) ===",
              flush=True)

        for n in sweep_iters:
            lm.model.n_center_iter = n
            lm.model.eval()
            r = evaluator.simple_evaluate(
                model=lm, tasks=sweep_tasks,
                num_fewshot=0, batch_size=args.batch_size, device=args.device,
            )
            row = {"n_iter": n}
            for task, res in r["results"].items():
                for k, v in res.items():
                    if not k.endswith("_stderr") and k != "alias":
                        row[f"{task}/{k}"] = v
            sweep_results[n] = row
            wikitext_ppl = row.get("wikitext/word_perplexity,none", "?")
            boolq_acc    = row.get("boolq/acc,none", "?")
            print(f"  n_iter={n:3d}: wikitext_ppl={wikitext_ppl}  boolq_acc={boolq_acc}",
                  flush=True)

        # Restore original
        lm.model.n_center_iter = base_n

        sweep_path = str(Path(args.output_path).parent / "iter_sweep_results.json")
        with open(sweep_path, "w") as f:
            json.dump(sweep_results, f, indent=2, default=str)
        print(f"\nIter sweep saved to {sweep_path}", flush=True)


if __name__ == "__main__":
    # Suppress HF dataset downloads (cluster is offline)
    os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    main()
