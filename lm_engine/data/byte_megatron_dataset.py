"""ByteMegatronDataset: reads pre-processed uint8 byte files for dolomite training.

Replaces the standard MegatronDataset (which reads BPE int32 token IDs) with a
dataset that reads raw UTF-8 bytes from pre-processed .bytes files.

Pre-processing: run preprocess_bytes_20260501.py to convert Megatron .bin files
  (int32 BPE token IDs) to .bytes files (flat uint8 raw bytes).
  Output: /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0.bytes

Usage in YAML config (model_type: byte_energy or energy with vocab_size=256):
  datasets:
    - class_name: ByteMegatronDataset
      data_name: ByteMegatron
      data_sampling_ratio: 1
      class_args:
        byte_paths:
          - /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0.bytes
          - /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1.bytes
        sequence_length: 4096    # bytes per training sample (no compression)
        eval_steps: 2
        split: 99.5,0.5,0
        seed: 42

For w4s2 compression (window=4, stride=2):
  The model receives sequence_length bytes but the ByteLinearPool inside the model
  compresses them to sequence_length//stride compressed tokens before the transformer.
  OR: set sequence_length=4096 for the transformer and use sequence_length*2=8192
  raw bytes per sample (handled by the model's ByteLinearPool).
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


_BYTE_PATHS_DEFAULT = [
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0.bytes",
    "/proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1.bytes",
]


class ByteMegatronDataset(Dataset):
    """Streaming byte dataset backed by pre-processed uint8 memory-mapped files.

    Produces samples of shape [sequence_length+1] containing consecutive raw bytes
    (same convention as Megatron: sample[:-1] = input, sample[1:] = labels).
    Actually for byte models with w4s2, we need sample = [sequence_length] and labels
    are computed internally by the model's decoder offset. For simplicity, we return
    [sequence_length+1] and let the training loop split input/labels.

    When .bytes files don't exist (preprocessing still running), falls back to
    streaming decode from .bin files using the HF tokenizer.
    """

    def __init__(
        self,
        byte_paths: list[str] | None = None,
        sequence_length: int = 4096,
        split: str = "99.5,0.5,0",
        eval_steps: int = 2,
        seed: int = 42,
        data_cache_path: str | None = None,   # unused, for API compat
        data_name: str = "ByteMegatron",      # unused, for API compat
        **kwargs,
    ):
        self.sequence_length = sequence_length
        self.seed = seed
        self._rng = np.random.default_rng(seed)

        paths = byte_paths or _BYTE_PATHS_DEFAULT
        available = [p for p in paths if __import__('os').path.exists(p)]

        if available:
            self._mode = "bytes"
            self._shards = [np.memmap(p, dtype=np.uint8, mode='r') for p in available]
            self._cum = np.cumsum([0] + [len(s) for s in self._shards])
            self._total = int(self._cum[-1])
            print(f"  ByteMegatronDataset: {self._total/1e9:.2f}GB from {len(available)} byte shard(s)")
        else:
            # Fallback: streaming BPE→bytes via tokenizer
            self._mode = "streaming"
            self._setup_streaming_fallback(paths)

    def _setup_streaming_fallback(self, paths: list[str]):
        """Fallback when .bytes files aren't ready: decode BPE→bytes on-the-fly."""
        from transformers import AutoTokenizer
        tok_path = "/proj/datasets/tokenizers/granite-4.0-tiktoken"
        self.tok = AutoTokenizer.from_pretrained(tok_path)
        # Use .bin files
        bin_paths = [p.replace(".bytes", ".bin") for p in paths]
        available = [p for p in bin_paths if __import__('os').path.exists(p)]
        self._shards = [np.memmap(p, dtype=np.int32, mode='r') for p in available]
        self._cum = np.cumsum([0] + [len(s) for s in self._shards])
        self._total_tokens = int(self._cum[-1])
        # ~3.7 bytes per BPE token
        self._total = int(self._total_tokens * 3.7)
        self._tokens_per_sample = self.sequence_length // 3 + 32
        print(f"  ByteMegatronDataset: streaming fallback from {len(available)} .bin shard(s)")

    def _read_bytes(self, pos: int, n: int) -> np.ndarray:
        """Read n bytes starting at global position pos."""
        for i, (lo, hi) in enumerate(zip(self._cum[:-1], self._cum[1:])):
            if lo <= pos < hi:
                local = pos - lo
                end = min(local + n, len(self._shards[i]))
                chunk = self._shards[i][local:end]
                if len(chunk) < n and i + 1 < len(self._shards):
                    chunk = np.concatenate([chunk, self._shards[i+1][:n-len(chunk)]])
                if len(chunk) < n:
                    chunk = np.pad(chunk, (0, n - len(chunk)), constant_values=32)
                return chunk.astype(np.int64)
        return np.full(n, 32, dtype=np.int64)

    def _decode_at(self, token_pos: int) -> np.ndarray:
        """Streaming fallback: decode BPE tokens → bytes at given position."""
        n = min(self._tokens_per_sample, self._total_tokens - token_pos)
        for i, (lo, hi) in enumerate(zip(self._cum[:-1], self._cum[1:])):
            if lo <= token_pos < hi:
                toks = self._shards[i][token_pos - lo : token_pos - lo + n].tolist()
                break
        else:
            toks = self._shards[0][:n].tolist()
        try:
            raw = list(self.tok.decode(toks, skip_special_tokens=True).encode("utf-8"))
        except Exception:
            raw = [32] * (self.sequence_length + 1)
        needed = self.sequence_length + 1
        if len(raw) < needed:
            raw += [32] * (needed - len(raw))
        return np.array(raw[:needed], dtype=np.int64)

    def __len__(self) -> int:
        # Large fixed size — data is effectively infinite (536B bytes)
        return 10_000_000

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        needed = self.sequence_length + 1
        max_start = max(1, self._total - needed - 1)
        pos = int((idx * 2_654_435_761 + self.seed) % max_start)

        if self._mode == "bytes":
            seq = self._read_bytes(pos, needed)
        else:
            tok_pos = int(pos / 3.7)
            seq = self._decode_at(tok_pos)

        # Return in the same format as MegatronDataset
        # input_ids = seq[:-1], labels = seq[1:]  (standard LM convention)
        # But for byte models with internal decoder offset, we return the full seq
        # and let the model compute loss internally.
        tokens = torch.tensor(seq, dtype=torch.long)
        return {
            "input_ids": tokens[:-1],      # [sequence_length]
            "labels":    tokens[1:],       # [sequence_length] (shifted right)
        }
