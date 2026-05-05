import contextlib
"""
Byte-level EGPT with w4s2 linear-pool tokenizer.

Architecture:
  bytes → embed(256, d_local) → CausalLinearPool(W=4, S=2) → [B, T/2, D]
        → [2× standard GPT block]
        → [EGPT block × N recurrences]
        → [1× standard GPT block]
        → linear expand → byte logits

Key hypothesis:
  Replacing the large BPE embedding (77M params for 100k vocab) with byte-level
  encoding frees ~77M params to be spent on transformer layers. At 80M total,
  the byte model has 4× more transformer capacity than the iso-param BPE model.

  Two EGPT variants:
  - layernorm:        EGPT update x -= proj_attn(LN(x)) + proj_mlp(LN(x)) [standard]
  - rmsnorm_reileigh: EGPT update with RMSNorm + tangent projection P(v,g)=v-g(g·v)/D

Model size (~84M total for D=1280):
  byte embedding:   256 × 64 = 16K
  linear pool:      4 × 64 × 1280 = 328K
  3 GPT blocks:     3 × (4×1280² + 2×1280×5120) = 3 × 19.7M = 59M
  1 EGPT block:     2×1280² + 2×1280×5120 + 2×1280² = 18.9M
  decoder:          1280 × 2560 + 1280 × 256 = 3.6M
  Total:            ~82M

Compared to V56 (84M, 91.6% in embedding):
  V56: 77M embedding + 7M transformer
  byte: 0.35M embedding + 82M transformer → 10× more transformer capacity

Usage:
  python train_byte_egpt_20260501.py --variant layernorm
  python train_byte_egpt_20260501.py --variant rmsnorm_reileigh
"""

import argparse
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb

# ── Config ─────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--variant", choices=[
                       "layernorm",           # EGPT + LayerNorm (shared center)
                       "rmsnorm_rayleigh",    # EGPT + RMSNorm + Rayleigh projection (shared center)
                       "rmsnorm_reileigh",    # alias for rmsnorm_rayleigh (old spelling)
                       "rec_gpt",             # Recurrent GPT (shared block × N)
                       "deep_gpt",            # Deep GPT (separate blocks, no sharing)
                       "deep_egpt_rayleigh",  # Deep EGPT (separate EGPT blocks, RMSNorm+Rayleigh)
                   ], default="layernorm")
    p.add_argument("--d_model",     type=int,   default=1280)
    p.add_argument("--d_local",     type=int,   default=64,   help="Byte embedding dim")
    p.add_argument("--n_head",      type=int,   default=20,   help="Attention heads (d_model/n_head=64)")
    p.add_argument("--n_pre",       type=int,   default=2,    help="Standard GPT blocks before EGPT")
    p.add_argument("--n_post",      type=int,   default=1,    help="Standard GPT blocks after EGPT")
    p.add_argument("--n_egpt_iter", type=int,   default=10,   help="EGPT recurrence iterations (base count)")
    p.add_argument("--iter_dropout_range", type=int, default=0,
                   help="Vary iteration count ± range per batch during training (0=disabled). "
                        "E.g. n_egpt_iter=10 + range=3 → uniform[7,13] per batch. "
                        "Trains robustness to test-time compute scaling.")
    p.add_argument("--window_size", type=int,   default=4,    help="Byte pool window (w4s2)")
    p.add_argument("--stride",      type=int,   default=2,    help="Byte pool stride (w4s2)")
    p.add_argument("--block_size",  type=int,   default=1024, help="Input byte sequence length")
    p.add_argument("--batch_size",  type=int,   default=32)
    p.add_argument("--grad_accum",  type=int,   default=1)
    p.add_argument("--lr",          type=float, default=2e-3)
    p.add_argument("--min_lr",      type=float, default=2e-4)
    p.add_argument("--steps",       type=int,   default=60000)
    p.add_argument("--warmup_steps",type=int,   default=2000)
    p.add_argument("--eval_interval",type=int,  default=1000)
    p.add_argument("--log_interval",type=int,   default=50)
    p.add_argument("--save_interval",type=int,  default=10000)
    p.add_argument("--dataset",     type=str,   default="megatron",
                   help="megatron (default, same data as V56/V65/V66) or wikitext")
    p.add_argument("--save_dir",    type=str,   default=None)
    p.add_argument("--wandb_project",type=str,  default="energy-inference-large")
    p.add_argument("--device",      type=str,   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compile",     action="store_true", default=True)
    p.add_argument("--no_compile",  action="store_true")
    p.add_argument("--seed",        type=int,   default=42)
    return p.parse_args()


# ── Normalization ───────────────────────────────────────────────────────────
# Use nn.RMSNorm (PyTorch ≥ 2.4) which calls F.rms_norm — a single fused CUDA
# kernel, much faster than manual pow/mean/sqrt/div/mul chain.
RMSNorm = nn.RMSNorm


# ── Byte encoder / decoder ─────────────────────────────────────────────────

class ByteLinearPool(nn.Module):
    """w4s2: stride-2 linear pool of 4 byte embeddings → 1 compressed token.

    Encodes d_local × window_size bytes into d_model via a causal linear projection.
    Causal: position k only sees bytes in window [k*stride-window+1 .. k*stride].
    """
    def __init__(self, vocab_size, d_local, d_model, window_size=4, stride=2):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.embed = nn.Embedding(vocab_size, d_local)
        self.pool_proj = nn.Linear(window_size * d_local, d_model, bias=False)

    def forward(self, x):
        # x: [B, T] (byte indices)
        B, T = x.shape
        W, S = self.window_size, self.stride
        h = self.embed(x)  # [B, T, d_local]
        # Pad left so first window ends at position W-1
        h = F.pad(h, (0, 0, W - S, 0))  # [B, T + W - S, d_local]
        T_pad = h.shape[1]
        # Sliding windows with stride S: output length = T // S
        n_windows = T // S
        windows = h.unfold(1, W, S)  # [B, n_windows, d_local, W]
        # Take only as many windows as we have strides
        windows = windows[:, :n_windows]
        B2, N, D, W2 = windows.shape
        flat = windows.permute(0, 1, 3, 2).reshape(B2, N, W * D)  # [B, N, W*d_local]
        return self.pool_proj(flat)  # [B, N, d_model]


class ByteDecoder(nn.Module):
    """Linear expand: each compressed token predicts the next `stride` bytes."""
    def __init__(self, d_model, vocab_size, stride=2):
        super().__init__()
        self.stride = stride
        self.expand = nn.Linear(d_model, stride * d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, z):
        # z: [B, N, d_model]
        B, N, D = z.shape
        S = self.stride
        h = self.expand(z)  # [B, N, S*D]
        h = h.view(B, N, S, D)  # [B, N, S, D]
        h = self.ln(h)
        return self.head(h)  # [B, N, S, vocab_size]


# ── Attention ──────────────────────────────────────────────────────────────

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.c_attn = nn.Linear(d_model, 3 * d_model, bias=False)
        self.c_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.c_attn(x).split(C, dim=-1)
        q = q.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_head, self.d_head).transpose(1, 2)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


# ── Transformer blocks ─────────────────────────────────────────────────────

class GPTBlock(nn.Module):
    def __init__(self, d_model, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class EGPTBlock(nn.Module):
    """Energy descent block: x = x - proj_attn(attn(ln(x))) - proj_mlp(mlp(ln(x))).

    Rayleigh tangent projection P(v, g) = v - ĝ(ĝ·v) = v - g(g·v)/D
    where g = RMSNorm(x) has ||g||² = D, so ĝ = g/√D is a unit vector.
    This projects v onto the tangent space of the unit sphere at ĝ, equivalent
    to (I - x̂x̂ᵀ)v where x̂ = x/||x||₂.

    Implementation: both attn and mlp projections are done in a single batched
    pass over g to minimise memory reads:
      stacked = [attn_out | mlp_out]  → [B, T, 2, D]
      coeffs  = einsum('btd,btid->bti', g, stacked) / D  → [B, T, 2, 1]
      stacked -= g.unsqueeze(-2) * coeffs.unsqueeze(-1)
    This reads g once instead of twice.
    """
    def __init__(self, d_model, n_head, use_rmsnorm=False, apply_reileigh=False):
        super().__init__()
        self.apply_rayleigh = apply_reileigh  # (correct spelling; alias for legacy param name)
        self.ln = RMSNorm(d_model) if use_rmsnorm else nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_head)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model, bias=False),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model, bias=False),
        )
        self.proj_attn = nn.Linear(d_model, d_model, bias=False)
        self.proj_mlp  = nn.Linear(d_model, d_model, bias=False)
        self._inv_d = None  # cached 1/D (set lazily)

    def forward(self, x):
        g = self.ln(x)
        a = self.attn(g)
        m = self.mlp(g)
        if self.apply_rayleigh:
            # Batch both projections in one pass: read g once, project both a and m.
            # stacked: [B, T, 2, D]; coeffs = g · stacked / D → [B, T, 2, 1]
            inv_d = 1.0 / g.shape[-1]
            stacked = torch.stack([a, m], dim=2)           # [B, T, 2, D]
            coeffs  = (g.unsqueeze(2) * stacked).sum(-1, keepdim=True).mul_(inv_d)
            stacked = stacked - g.unsqueeze(2) * coeffs    # [B, T, 2, D]
            a, m = stacked.unbind(2)
        return x - self.proj_attn(a) - self.proj_mlp(m)


# ── Full model ─────────────────────────────────────────────────────────────

class ByteEGPT(nn.Module):
    """Byte-level model: w4s2 encoder → [pre-blocks] → [center × N] → [post-blocks] → decoder.

    center_block controls the shared center:
      'egpt'    : EGPTBlock × n_center_iter  (energy descent update)
      'gpt_rec' : GPTBlock  × n_center_iter  (weight-shared recurrent GPT, same compute)
      'none'    : no center block (n_pre covers all depth, for deep non-recurrent GPT)
    """

    def __init__(self, d_model, d_local, n_head, vocab_size=256,
                 window_size=4, stride=2, block_size=1024,
                 n_pre=2, n_post=1, n_egpt_iter=10,
                 use_rmsnorm=False, apply_reileigh=False,
                 center_block='egpt', iter_dropout_range=0):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.n_center_iter = n_egpt_iter
        self.iter_dropout_range = iter_dropout_range  # ±range for variable looping
        self.center_type = center_block
        self.block_size = block_size

        compressed_len = block_size // stride

        self.encoder = ByteLinearPool(vocab_size, d_local, d_model, window_size, stride)
        self.pos_emb  = nn.Embedding(compressed_len, d_model)
        self.pre_blocks  = nn.ModuleList([GPTBlock(d_model, n_head) for _ in range(n_pre)])

        if center_block == 'egpt':
            self.center = EGPTBlock(d_model, n_head, use_rmsnorm, apply_reileigh)
        elif center_block == 'gpt_rec':
            self.center = GPTBlock(d_model, n_head)
        elif center_block == 'egpt_deep':
            # Separate (non-shared) EGPT blocks — n_egpt_iter unique weight sets
            self.center = nn.ModuleList([
                EGPTBlock(d_model, n_head, use_rmsnorm, apply_reileigh)
                for _ in range(n_egpt_iter)
            ])
        else:  # 'none' — deep GPT: all depth in pre_blocks
            self.center = None

        self.post_blocks = nn.ModuleList([GPTBlock(d_model, n_head) for _ in range(n_post)])
        self.ln_f    = nn.LayerNorm(d_model)
        self.decoder = ByteDecoder(d_model, vocab_size, stride)

        self.apply(self._init_weights)

    # backward-compat alias used by eval script
    @property
    def egpt_block(self):
        return self.center

    def _init_weights(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, x, targets=None):
        # x: [B, T] byte indices
        B, T = x.shape
        z = self.encoder(x)                    # [B, T//stride, d_model]
        N = z.shape[1]
        pos = torch.arange(N, device=x.device)
        z = z + self.pos_emb(pos)

        for block in self.pre_blocks:
            z = block(z)
        if isinstance(self.center, nn.ModuleList):  # egpt_deep: separate blocks
            for block in self.center:
                z = block(z)
        elif self.center is not None:               # egpt or gpt_rec: shared block
            # Variable looping: during training, sample n_iters ~ Uniform[base-range, base+range].
            # Trains the model to be robust to test-time compute scaling.
            if self.training and self.iter_dropout_range > 0:
                lo = max(1, self.n_center_iter - self.iter_dropout_range)
                hi = self.n_center_iter + self.iter_dropout_range
                n_iters = torch.randint(lo, hi + 1, (1,)).item()
            else:
                n_iters = self.n_center_iter
            for _ in range(n_iters):
                z = self.center(z)
        for block in self.post_blocks:
            z = block(z)
        z = self.ln_f(z)

        # Decoder: z[k] predicts bytes at positions [k*stride+window_size .. k*stride+window_size+stride)
        logits = self.decoder(z)  # [B, N, stride, vocab_size]

        if targets is None:
            return logits, None

        S = self.stride
        W = self.window_size
        offset = W   # shift by window_size (predict next window, not current)
        # targets for z[k]: bytes at [k*S+offset .. k*S+offset+S)
        # Rearrange: [B, N, S, vocab_size] vs [B, N*S] targets
        tgt_start = offset
        tgt_end = offset + N * S
        if tgt_end > T:
            # Trim to available targets (last few windows may not have targets)
            available = T - tgt_start
            n_full = available // S
            logits = logits[:, :n_full]
            tgt = x[:, tgt_start:tgt_start + n_full * S].view(B, n_full, S)
        else:
            tgt = x[:, tgt_start:tgt_end].view(B, N, S)

        n_pred = logits.shape[1]
        loss = F.cross_entropy(
            logits[:, :n_pred].reshape(-1, logits.shape[-1]),
            tgt[:, :n_pred].reshape(-1)
        )
        return logits, loss

    def count_params(self):
        total = sum(p.numel() for p in self.parameters())
        emb   = sum(p.numel() for p in self.encoder.embed.parameters())
        return total, emb


# ── Data loading ────────────────────────────────────────────────────────────

_MEGATRON_PATHS = [
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0",
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1",
]
_TOKENIZER_PATH = "/proj/datasets/tokenizers/granite-4.0-tiktoken"

# Pre-processed byte files (created by preprocess_bytes_20260501.py).
# If present, training uses pure numpy slice (zero tokenizer calls).
# If absent, falls back to streaming BPE→bytes decode.
_BYTE_PATHS = [p + ".bytes" for p in _MEGATRON_PATHS]


class MegatronByteDataset(torch.utils.data.Dataset):
    """Production byte-level dataset for Megatron-format corpora.

    Two backends selected automatically:
    1. Fast (pre-processed): memory-maps flat uint8 byte files created by
       preprocess_bytes_20260501.py.  __getitem__ is a pure numpy slice — zero
       tokenizer overhead, saturates even 8×H100 with 8 workers.
    2. Streaming fallback: reads int32 BPE tokens from the .bin files and decodes
       them on-the-fly using tiktoken.  ~0.5 ms per sample in worker processes;
       fine for single-node training while pre-processing runs in the background.

    In both cases every sample draws from a unique random position in the 536B-token
    corpus (hash-mapped index), so there is zero repetition across 30k-step runs
    and the model never memorises the training set.

    Val samples are drawn from the second half of the corpus and are held out from
    training indices, giving an in-distribution BPC estimate.  A separate WikiText-103
    BPC is also logged for cross-experiment comparison (see build_wikitext_val).
    """

    def __init__(self, data_paths, tokenizer_path, block_size_bytes=1024,
                 seed=42, n_samples=2_000_000):
        import numpy as np
        self.block_size = block_size_bytes
        self.n_samples  = n_samples
        self._seed      = seed
        self.tokens_per_sample = max(block_size_bytes // 3 + 32, 128)

        # ── Try pre-processed uint8 byte files first ──────────────────────────
        byte_paths = [p + ".bytes" for p in data_paths]
        if all(os.path.exists(p) for p in byte_paths):
            print("  Loading pre-processed byte memmaps (fast path)...", flush=True)
            self._mode = "bytes"
            self.byte_shards = []
            self.byte_cum    = [0]
            for p in byte_paths:
                arr = np.memmap(p, dtype=np.uint8, mode='r')
                self.byte_shards.append(arr)
                self.byte_cum.append(self.byte_cum[-1] + len(arr))
                print(f"    {p.split('/')[-1]}: {len(arr)/1e9:.2f}GB bytes", flush=True)
            self.total_bytes = self.byte_cum[-1]
            print(f"  Total: {self.total_bytes/1e12:.2f}TB  (zero-decode fast path)", flush=True)
        else:
            # ── Streaming fallback: decode BPE→bytes in DataLoader workers ────
            from transformers import AutoTokenizer
            print("  Streaming decode fallback (pre-processed byte files not found).", flush=True)
            print("  Run preprocess_bytes_20260501.py in the background for the fast path.", flush=True)
            self._mode = "streaming"
            self.tok = AutoTokenizer.from_pretrained(tokenizer_path)
            self.tok_shards = []
            self.tok_cum    = [0]
            for p in data_paths:
                arr = np.memmap(p + '.bin', dtype=np.int32, mode='r')
                self.tok_shards.append(arr)
                self.tok_cum.append(self.tok_cum[-1] + len(arr))
                print(f"    {p.split('/')[-1]}: {len(arr)/1e9:.2f}B tokens", flush=True)
            self.total_tokens = self.tok_cum[-1]
            # Estimate total bytes for val-set split (≈ 3.7 bytes/BPE token on NematronCC)
            self.total_bytes  = int(self.total_tokens * 3.7)
            print(f"  Total: {self.total_tokens/1e9:.2f}B tokens  "
                  f"(streaming — no buffer, full diversity)", flush=True)

    # ── Internal helpers ─────────────────────────────────────────────────────

    def _read_bytes(self, byte_pos, n_bytes):
        """Fast path: numpy slice from pre-processed uint8 mmap."""
        import numpy as np
        for i, (lo, hi) in enumerate(zip(self.byte_cum[:-1], self.byte_cum[1:])):
            if lo <= byte_pos < hi:
                local = byte_pos - lo
                end   = min(local + n_bytes, len(self.byte_shards[i]))
                chunk = self.byte_shards[i][local:end].tolist()
                if len(chunk) < n_bytes and i + 1 < len(self.byte_shards):
                    chunk += self.byte_shards[i+1][:n_bytes - len(chunk)].tolist()
                return chunk
        return [32] * n_bytes

    def _decode_at(self, token_pos):
        """Streaming path: decode BPE tokens to bytes at random corpus position."""
        import numpy as np
        n = min(self.tokens_per_sample, self.total_tokens - token_pos)
        for i, (lo, hi) in enumerate(zip(self.tok_cum[:-1], self.tok_cum[1:])):
            if lo <= token_pos < hi:
                local = token_pos - lo
                toks  = self.tok_shards[i][local:local + n].tolist()
                break
        else:
            toks = self.tok_shards[0][:n].tolist()
        try:
            raw = list(self.tok.decode(toks, skip_special_tokens=True).encode("utf-8"))
        except Exception:
            raw = [32] * (self.block_size + 1)
        if len(raw) < self.block_size + 1:
            raw += [32] * (self.block_size + 1 - len(raw))
        return raw[:self.block_size + 1]

    def _get_sample(self, idx):
        bs1 = self.block_size + 1
        if self._mode == "bytes":
            max_start = self.total_bytes - bs1 - 1
            pos = (idx * 2_654_435_761 + self._seed) % max_start
            raw = self._read_bytes(pos, bs1)
            if len(raw) < bs1:
                raw += [32] * (bs1 - len(raw))
            return raw
        else:
            max_start = self.total_tokens - self.tokens_per_sample - 1
            token_pos = (idx * 2_654_435_761 + self._seed) % max_start
            return self._decode_at(token_pos)

    # ── Dataset interface ────────────────────────────────────────────────────

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        return torch.tensor(self._get_sample(idx), dtype=torch.long)

    def get_val_batch(self, batch_size, device, seed=0):
        """In-distribution val batch: second half of corpus, held out from training."""
        import numpy as np
        rng = np.random.default_rng(seed + 99_999)
        if self._mode == "bytes":
            half = self.total_bytes // 2
            max_s = self.total_bytes - self.block_size - 2
            positions = rng.integers(half, max_s, size=batch_size)
            seqs = [torch.tensor(self._read_bytes(int(p), self.block_size + 1), dtype=torch.long)
                    for p in positions]
        else:
            half = self.total_tokens // 2
            max_s = self.total_tokens - self.tokens_per_sample - 1
            positions = rng.integers(half, max_s, size=batch_size)
            seqs = [torch.tensor(self._decode_at(int(p)), dtype=torch.long)
                    for p in positions]
        return torch.stack(seqs).to(device)


# Alias used by training code that still references ByteBufferDataset
ByteBufferDataset = MegatronByteDataset


def build_wikitext_val(block_size: int, n_batches: int = 40, batch_size: int = 64):
    """Load WikiText-103 validation set as byte tensor for cross-experiment BPC.

    Returns a list of (batch_size, block_size) tensors ready for model eval.
    WikiText BPC is the standard comparison point with nn-tokenizer and other runs.
    """
    try:
        from datasets import load_dataset
        ds = load_dataset("wikitext", "wikitext-103-raw-v1", split="validation",
                          trust_remote_code=True)
        text = "\n\n".join(r["text"] for r in ds if r["text"].strip())
        data = torch.frombuffer(text.encode("utf-8"), dtype=torch.uint8).long()
        batches = []
        step = max(1, (len(data) - block_size) // (n_batches * batch_size))
        for bi in range(n_batches):
            starts = torch.arange(bi * batch_size, (bi + 1) * batch_size) * step
            starts = starts.clamp(0, len(data) - block_size - 1)
            x = torch.stack([data[s:s + block_size] for s in starts])
            batches.append(x)
        return batches
    except Exception as e:
        print(f"  WikiText-103 val unavailable ({e}), skipping wiki BPC.", flush=True)
        return []


class ByteDataset:
    """Simple in-memory byte dataset (for WikiText-103)."""
    def __init__(self, data: torch.Tensor, block_size: int):
        self.data = data
        self.block_size = block_size

    def get_batch(self, batch_size, device):
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        return x.to(device)

    def get_val_batch(self, batch_size, device, seed=0):
        torch.manual_seed(seed)
        ix = torch.randint(len(self.data) - self.block_size, (batch_size,))
        x = torch.stack([self.data[i:i+self.block_size] for i in ix])
        return x.to(device)


def load_wikitext103(split="train"):
    from datasets import load_dataset
    ds = load_dataset("wikitext", "wikitext-103-raw-v1", split=split,
                      trust_remote_code=True)
    text = "\n\n".join([row["text"] for row in ds if row["text"].strip()])
    return torch.frombuffer(text.encode("utf-8"), dtype=torch.uint8).long()


# ── LR schedule ─────────────────────────────────────────────────────────────

def get_lr(step, warmup, max_steps, lr, min_lr):
    if step < warmup:
        return lr * step / warmup
    if step > max_steps:
        return min_lr
    decay = (step - warmup) / (max_steps - warmup)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * decay))


# ── Network check ────────────────────────────────────────────────────────────

def check_wandb_connectivity(timeout_s: int = 8) -> bool:
    """Return True if wandb.ai is reachable. Exit with code 1 if not.

    Fails fast at job start so the scheduler can reassign to a node with
    working internet instead of wasting GPU-hours on a silent offline run.
    """
    import subprocess
    result = subprocess.run(
        ["curl", "-s", "--max-time", str(timeout_s),
         "--head", "https://api.wandb.ai"],
        capture_output=True,
    )
    if result.returncode != 0:
        print(f"ERROR: wandb.ai unreachable (curl exit {result.returncode}). "
              "Exiting so job scheduler can retry on a connected node.", flush=True)
        sys.exit(1)
    return True


# ── Training ────────────────────────────────────────────────────────────────

def train():
    args = parse_args()
    if args.no_compile:
        args.compile = False

    # ── DDP setup (torchrun sets LOCAL_RANK/RANK/WORLD_SIZE env vars) ─────────
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler

    use_ddp = 'LOCAL_RANK' in os.environ
    if use_ddp:
        local_rank  = int(os.environ['LOCAL_RANK'])
        world_size  = int(os.environ.get('WORLD_SIZE', 1))
        dist.init_process_group('nccl')
        torch.cuda.set_device(local_rank)
        device = torch.device(f'cuda:{local_rank}')
        is_main = (local_rank == 0)
    else:
        local_rank = 0; world_size = 1; is_main = True
        device = torch.device(args.device)

    # Fail fast on offline nodes — only main rank needs internet (for wandb)
    if is_main:
        check_wandb_connectivity()

    torch.manual_seed(args.seed + local_rank)  # different seed per rank for data diversity

    variant = args.variant
    is_rayleigh  = variant in ("rmsnorm_rayleigh", "rmsnorm_reileigh", "deep_egpt_rayleigh")
    use_rmsnorm    = is_rayleigh
    apply_rayleigh = is_rayleigh
    center_block   = {
        "layernorm":          "egpt",
        "rmsnorm_rayleigh":   "egpt",
        "rmsnorm_reileigh":   "egpt",
        "rec_gpt":            "gpt_rec",
        "deep_gpt":           "none",
        "deep_egpt_rayleigh": "egpt_deep",
    }[variant]

    run_name = f"byte_{variant}_w{args.window_size}s{args.stride}_dl{args.d_local}_D{args.d_model}_{args.n_egpt_iter}iter"
    if use_ddp and world_size > 1:
        run_name += f"_ddp{world_size}"
    save_dir = args.save_dir
    if save_dir is None:
        results_root = Path(__file__).parent.parent.parent / "results" / "multi-block-ablation"
        save_dir = results_root / run_name
    save_dir = Path(save_dir)
    if is_main:
        save_dir.mkdir(parents=True, exist_ok=True)

    # Count params (main rank only)
    _model_for_count = ByteEGPT(
        d_model=args.d_model, d_local=args.d_local, n_head=args.n_head,
        vocab_size=256, window_size=args.window_size, stride=args.stride,
        block_size=args.block_size, n_pre=args.n_pre, n_post=args.n_post,
        n_egpt_iter=args.n_egpt_iter,
        use_rmsnorm=use_rmsnorm, apply_reileigh=apply_rayleigh,
        center_block=center_block,
    )
    total_params, emb_params = _model_for_count.count_params()
    del _model_for_count

    init_cfg = {**vars(args),
                "num_parameters":     total_params,   # matches dolomite Vxx key
                "emb_params":         emb_params,
                "transformer_params": total_params - emb_params}
    if is_main:
        wandb.init(project=args.wandb_project,
                   entity="nima-dehmamy-projects",
                   name=run_name,
                   config=init_cfg)
        print(f"  Total params: {total_params/1e6:.2f}M  "
              f"embedding: {emb_params/1e6:.3f}M ({100*emb_params/total_params:.1f}%)  "
              f"transformer: {(total_params-emb_params)/1e6:.2f}M", flush=True)

    n_total_samples = args.steps * args.batch_size * args.grad_accum * world_size + 10000
    if args.dataset == "megatron":
        if is_main:
            print("Loading Megatron Nematron-CC data (byte-level)...", flush=True)
        train_ds = MegatronByteDataset(_MEGATRON_PATHS, _TOKENIZER_PATH,
                                       block_size_bytes=args.block_size,
                                       seed=args.seed + local_rank,
                                       n_samples=n_total_samples)
        val_ds = train_ds
    else:
        print("Loading WikiText-103 (byte-level)...")
        train_data = load_wikitext103("train")
        val_data   = load_wikitext103("validation")
        print(f"  train: {len(train_data)/1e6:.1f}M bytes, val: {len(val_data)/1e6:.1f}M bytes")
        train_ds = ByteDataset(train_data, args.block_size)
        val_ds   = ByteDataset(val_data, args.block_size)

    model = ByteEGPT(
        d_model=args.d_model, d_local=args.d_local, n_head=args.n_head,
        vocab_size=256, window_size=args.window_size, stride=args.stride,
        block_size=args.block_size, n_pre=args.n_pre, n_post=args.n_post,
        n_egpt_iter=args.n_egpt_iter,
        use_rmsnorm=use_rmsnorm, apply_reileigh=apply_rayleigh,
        center_block=center_block,
        iter_dropout_range=args.iter_dropout_range,
    ).to(device)

    # DDP: compile first (per-rank), then wrap with DDP.
    # DDP + torch.compile work correctly together — compile fuses the single-GPU
    # forward/backward, DDP handles gradient all-reduce asynchronously.
    effective_batch = args.batch_size  # per rank; total = batch * world_size * grad_accum
    if args.compile:
        # dynamic=True: trace with symbolic shapes → compilation uses ~1-2GB not 76GB.
        # Without dynamic=True, torch.compile materializes full activation tensors during
        # tracing (256×512×D per pass × 13 passes × optimization rounds = ~76GB OOM).
        # dynamic=True fixes this by tracing symbolically and handling all batch sizes
        # with one compiled kernel.
        # dynamic=True: symbolic tracing (no full tensors during trace)
        # reduce-overhead: skip Triton autotuning (which runs real tensors and OOMs at batch=256)
        # dynamic=True only — no CUDA Graphs (reduce-overhead uses CUDA Graphs which
        # conflict with no_sync() gradient accumulation: graph captures different
        # all-reduce patterns per accumulation step).
        model = torch.compile(model, dynamic=True)
        if is_main:
            print("  torch.compile(dynamic=True) enabled", flush=True)
    if use_ddp and world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
        if is_main:
            print(f"  DDP across {world_size} GPUs + torch.compile", flush=True)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=0.1,
        betas=(0.9, 0.95), eps=1e-8,
    )

    # WikiText-103 val: only load on main rank (avoids redundant downloads)
    wiki_val_batches = []
    if is_main:
        print("Loading WikiText-103 val for BPC benchmark...", flush=True)
        wiki_val_batches = build_wikitext_val(args.block_size, n_batches=40,
                                              batch_size=args.batch_size)
        if wiki_val_batches:
            print(f"  WikiText-103 val: {len(wiki_val_batches)} batches of {args.batch_size}", flush=True)

    scaler = torch.cuda.amp.GradScaler(enabled=(device.type == "cuda"))
    dtype  = torch.bfloat16 if device.type == "cuda" else torch.float32

    # DataLoader with DistributedSampler when using DDP
    use_dataloader = isinstance(train_ds, (MegatronByteDataset, ByteDataset))
    # With DDP+compile: limit workers to avoid torch.compile triggering worker crashes
    # (4 ranks × 8 workers = 32 processes on one node causes instability during tracing)
    if use_ddp and world_size > 1 and args.compile:
        n_workers = 2  # safe: fast-path uses pure numpy, 2 workers is enough
    else:
        n_workers = min(8, max(2, os.cpu_count() // max(1, world_size)))
    if use_dataloader:
        sampler = (DistributedSampler(train_ds, num_replicas=world_size,
                                      rank=local_rank, shuffle=False)
                   if use_ddp and world_size > 1 else None)
        loader = iter(torch.utils.data.DataLoader(
            train_ds,
            batch_size=effective_batch,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=n_workers,
            prefetch_factor=8,
            pin_memory=True,
            persistent_workers=True,
            drop_last=True,
        ))
        if is_main:
            print(f"  DataLoader: {n_workers} workers, prefetch=8, "
                  f"{'DistributedSampler' if sampler else 'shuffle'}", flush=True)

    def next_batch():
        if use_dataloader:
            seq = next(loader)
            return seq[:, :args.block_size].to(device, non_blocking=True)
        else:
            return train_ds.get_batch(effective_batch, device)

    # Resume from checkpoint if one exists in save_dir
    start_step = 1
    best_val_loss = float("inf")
    if is_main:
        latest_ckpt = sorted(save_dir.glob("ckpt_*.pt"))[-1] if list(save_dir.glob("ckpt_*.pt")) else None
        if latest_ckpt:
            ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
            raw_sd = ckpt["model"]
            # Checkpoint was saved with model.module.state_dict() which preserves
            # _orig_mod. prefix from torch.compile. Load directly into the same structure.
            base = model.module if hasattr(model, 'module') else model
            try:
                base.load_state_dict(raw_sd)
            except RuntimeError:
                # Fallback: strip _orig_mod. for loading into non-compiled model
                stripped = {k.replace("_orig_mod.", "").replace("module.", ""): v for k, v in raw_sd.items()}
                base.load_state_dict(stripped)
            optimizer.load_state_dict(ckpt["optimizer"])
            start_step = ckpt["step"] + 1
            best_val_loss = ckpt.get("val_loss", float("inf"))
            print(f"  Resumed from {latest_ckpt.name} (step {ckpt['step']})", flush=True)
    # Broadcast start_step to all ranks so they skip the same steps
    if use_ddp and world_size > 1:
        import torch.distributed as _dist2
        t = torch.tensor([start_step], dtype=torch.long, device=device)
        _dist2.broadcast(t, src=0)
        start_step = int(t.item())
    # Skip data already consumed before the checkpoint
    if start_step > 1 and use_dataloader:
        skip = (start_step - 1) * args.grad_accum
        if is_main:
            print(f"  Skipping {skip} batches to resume at step {start_step}...", flush=True)
        for _ in range(skip):
            try: next(loader)
            except StopIteration: break

    t0 = time.time()

    for step in range(start_step, args.steps + 1):
        # Background buffer refresh (every 5k steps, no GPU stall)
        if use_dataloader and hasattr(train_ds, 'maybe_refresh'):
            train_ds.maybe_refresh(step)

        lr = get_lr(step, args.warmup_steps, args.steps, args.lr, args.min_lr)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        # Gradient accumulation with DDP no_sync: skip all-reduce on non-final steps
        # to avoid N all-reduce calls per update (only need 1 on the last step).
        optimizer.zero_grad()
        accum_loss = 0.0
        for accum_i in range(args.grad_accum):
            x = next_batch()
            # Use no_sync() for all but the last accumulation step when using DDP
            is_last_accum = (accum_i == args.grad_accum - 1)
            ctx = (contextlib.nullcontext() if (not use_ddp or is_last_accum)
                   else model.no_sync())
            with ctx:
                with torch.autocast(device.type, dtype=dtype):
                    _, loss = model(x, targets=x)
                if loss.dim() > 0:
                    loss = loss.mean()
                loss = loss / args.grad_accum
                scaler.scale(loss).backward()
            accum_loss += loss.item()

        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        if is_main and step % args.log_interval == 0:
            dt = time.time() - t0
            bpc = accum_loss / math.log(2)
            t0 = time.time()
            print(f"step={step:6d}  train-loss={accum_loss:.4f}  train-bpc={bpc:.4f}  learning_rate={lr:.2e}  dt={dt:.1f}s", flush=True)
            wandb.log({"train/loss": accum_loss, "train/bpc": bpc,
                       "train/learning_rate": lr, "train/step_time (sec)": dt / args.log_interval,
                       "iteration": step})

        if is_main and step % args.eval_interval == 0:
            with torch.no_grad():
                val_losses = []
                for vi in range(40):
                    x = val_ds.get_val_batch(effective_batch, device, seed=vi)
                    with torch.autocast(device.type, dtype=dtype):
                        _, loss = model(x, targets=x)
                    if loss.dim() > 0:
                        loss = loss.mean()
                    val_losses.append(loss.item())
            val_loss = sum(val_losses) / len(val_losses)
            val_bpc  = val_loss / math.log(2)
            wiki_bpc = None
            if wiki_val_batches:
                wiki_losses = []
                with torch.no_grad():
                    for xw in wiki_val_batches:
                        xw = xw.to(device, non_blocking=True)
                        with torch.autocast(device.type, dtype=dtype):
                            _, lw = model(xw, targets=xw)
                        if lw.dim() > 0:
                            lw = lw.mean()
                        wiki_losses.append(lw.item())
                wiki_bpc = sum(wiki_losses) / len(wiki_losses) / math.log(2)
            log_dict = {"val/loss": val_loss, "val/bpc": val_bpc, "iteration": step}
            if wiki_bpc is not None:
                log_dict["val/wiki_bpc"] = wiki_bpc
            suffix = f"  wiki_bpc={wiki_bpc:.4f}" if wiki_bpc else ""
            print(f"  val-loss={val_loss:.4f}  val-bpc={val_bpc:.4f}{suffix}", flush=True)
            wandb.log(log_dict)
            raw_sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save({"step": step, "model": raw_sd,
                            "val_loss": val_loss, "args": vars(args)},
                           save_dir / "best.pt")

        if is_main and step % args.save_interval == 0:
            raw_sd = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
            torch.save({"step": step, "model": raw_sd,
                        "optimizer": optimizer.state_dict(), "args": vars(args)},
                       save_dir / f"ckpt_{step:06d}.pt")
            print(f"  Saved checkpoint at step {step}", flush=True)

    if use_ddp and world_size > 1:
        dist.destroy_process_group()
    if is_main:
        wandb.finish()
        print(f"Training complete. Best val_bpc = {best_val_loss/math.log(2):.4f}")


if __name__ == "__main__":
    import traceback as _tb
    try:
        train()
    except Exception as _e:
        local_rank = int(os.environ.get('LOCAL_RANK', 0))
        print(f"\n[RANK {local_rank}] UNCAUGHT EXCEPTION: {type(_e).__name__}: {_e}", flush=True)
        _tb.print_exc()
        raise
