"""
Pre-process Megatron BPE shards → flat uint8 byte files.

Converts each .bin (int32 BPE token IDs) to a .bytes file (raw UTF-8 bytes).
This is a one-time offline step.  After it completes, MegatronByteDataset uses
the fast path: pure numpy uint8 slice, zero tokenizer calls during training.

Throughput: ~300-500M tokens/minute on a single CPU core (tiktoken is fast).
Full run (536B tokens): ~20-30 hours on 1 core.
Run with multiple workers (one per shard) to parallelize.

Usage:
  # Process both shards in parallel (submit two jobs):
  python preprocess_bytes_20260501.py --shard 0
  python preprocess_bytes_20260501.py --shard 1

  # Or process a single shard:
  python preprocess_bytes_20260501.py --shard 0 --chunk_tokens 10_000_000

Output:
  /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0.bytes
  /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1.bytes

Each output file is a flat uint8 array: raw UTF-8 bytes of the decoded text.
MegatronByteDataset.bytes_path = shard_path + ".bytes"

bsub submission (one job per shard, normal queue, CPU-only):
  bsub -q normal -G grp_ebm -J preprocess_bytes_s0 -n 1 -M 32G -W 24:00 \\
       -o $HOME/bsub_logs/preprocess_bytes_s0_%J.stdout \\
       -e $HOME/bsub_logs/preprocess_bytes_s0_%J.stderr \\
       <<'EOF'
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  python /proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/scripts/multi-block-ablation/preprocess_bytes_20260501.py --shard 0
  EOF
"""

import argparse
import os
import time
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

SHARD_PATHS = [
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0",
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1",
]
TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"
CHUNK = 1_000_000   # tokens per decode chunk (~3-4MB, fast Python list)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--shard", type=int, required=True, choices=[0, 1],
                   help="Shard index to process (0 or 1)")
    p.add_argument("--chunk_tokens", type=int, default=CHUNK)
    p.add_argument("--resume", action="store_true",
                   help="Skip already-written chunks (resume interrupted run)")
    args = p.parse_args()

    in_path  = SHARD_PATHS[args.shard] + ".bin"
    out_path = SHARD_PATHS[args.shard] + ".bytes"

    print(f"Loading tokenizer from {TOKENIZER_PATH}...", flush=True)
    tok = AutoTokenizer.from_pretrained(TOKENIZER_PATH)

    print(f"Memory-mapping {in_path}...", flush=True)
    tokens = np.memmap(in_path, dtype=np.int32, mode='r')
    n_tokens = len(tokens)
    print(f"  {n_tokens/1e9:.2f}B tokens", flush=True)

    # Estimate output size: ~3.7 bytes/token for NematronCC
    estimated_bytes = int(n_tokens * 3.7)
    print(f"  Estimated output: {estimated_bytes/1e9:.1f}GB", flush=True)

    resume_offset = 0
    if args.resume and os.path.exists(out_path):
        resume_offset = os.path.getsize(out_path)
        print(f"  Resuming from {resume_offset/1e9:.2f}GB already written", flush=True)

    out_file = open(out_path, 'ab' if args.resume else 'wb')
    C = args.chunk_tokens

    # Skip to resume token position (approximate: resume_offset / 3.7 tokens)
    start_token = int(resume_offset / 3.7) if args.resume else 0
    # Align to chunk boundary
    start_token = (start_token // C) * C

    t0 = time.time()
    total_bytes_written = resume_offset
    total_tokens_done   = start_token

    for offset in range(start_token, n_tokens, C):
        toks = tokens[offset:offset + C].tolist()
        try:
            text = tok.decode(toks, skip_special_tokens=True)
            raw  = text.encode("utf-8")
        except Exception as e:
            print(f"  Decode error at offset {offset}: {e}", flush=True)
            raw = b" " * len(toks)  # fallback: spaces

        out_file.write(raw)
        total_bytes_written += len(raw)
        total_tokens_done   += len(toks)

        if (offset // C) % 100 == 0:
            elapsed = time.time() - t0 + 1e-6
            tok_rate = total_tokens_done / elapsed / 1e6
            pct = 100 * total_tokens_done / n_tokens
            eta_min = (n_tokens - total_tokens_done) / (tok_rate * 1e6) / 60
            print(f"  {pct:.1f}%  {total_tokens_done/1e9:.2f}B tokens  "
                  f"{total_bytes_written/1e9:.2f}GB written  "
                  f"{tok_rate:.1f}M tok/s  ETA {eta_min:.0f}min", flush=True)

    out_file.close()
    total_elapsed = time.time() - t0
    print(f"\nDone! {total_bytes_written/1e9:.2f}GB written to {out_path}", flush=True)
    print(f"Total time: {total_elapsed/3600:.1f}h  "
          f"Rate: {total_tokens_done/total_elapsed/1e6:.1f}M tok/s", flush=True)


if __name__ == "__main__":
    main()
