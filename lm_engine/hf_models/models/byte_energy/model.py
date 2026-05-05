"""ByteEnergyModel: dolomite EnergyModel with w4s2 byte tokenizer.

Drops in place of EnergyModel/EnergyPreTrainedModel. The only changes are:
  1. Input:  Embedding(256, d_local) + ByteLinearPool (w×d_local → D) replaces Embedding(vocab, D)
  2. Output: ByteDecoder (D → stride×D → 256) replaces lm_head (D → vocab)
  3. Loss:   CE over 256-class byte predictions (stride preds per compressed token)
  4. Data:   expects raw uint8 byte sequences (0–255), not BPE token IDs

Everything else (FSDP, DDP, energy blocks, attention, normalization, checkpointing,
wandb, lr scheduling, distributed training) is inherited from the dolomite machinery.

Usage in YAML config:
  model_type: byte_energy
  d_local: 64
  window_size: 4
  stride: 2
  hidden_size: 1600       # transformer width
  vocab_size: 256         # byte alphabet (auto-set by ByteEnergyConfig)
  sequence_length: 4096   # bytes per training sample (before compression)
  ... (all other energy model fields)
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from transformers.modeling_outputs import CausalLMOutputWithPast

from ..energy.base import EnergyModel, EnergyPreTrainedModel
from .config import ByteEnergyConfig


# ── Byte encoder (replaces nn.Embedding + LM head) ────────────────────────────

class ByteLinearPool(nn.Module):
    """Causal w4s2-style linear pool: W byte embeddings → 1 compressed token.

    Pads left by (W-S) zeros so position k only sees bytes ≤ k*S + S - 1.
    Output length = input_length // stride.
    """

    def __init__(self, d_local: int, d_model: int, window_size: int = 4, stride: int = 2):
        super().__init__()
        self.window_size = window_size
        self.stride = stride
        self.embed = nn.Embedding(256, d_local)
        self.pool_proj = nn.Linear(window_size * d_local, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T]  long tensor of byte indices 0-255
        B, T = x.shape
        W, S = self.window_size, self.stride
        h = self.embed(x)                                  # [B, T, d_local]
        h = F.pad(h, (0, 0, W - S, 0))                    # left-pad W-S zeros
        n_windows = T // S
        windows = h.unfold(1, W, S)[:, :n_windows]        # [B, N, d_local, W]
        flat = windows.permute(0, 1, 3, 2).reshape(B, n_windows, W * h.shape[-1])
        return self.pool_proj(flat)                        # [B, N, D]


class ByteDecoder(nn.Module):
    """Linear expand: each compressed token independently predicts stride bytes.

    Note: this is a simple parallel decoder (no autoregression within window).
    Teacher-forced or U-Net decoder variants can be added later.
    """

    def __init__(self, d_model: int, stride: int = 2):
        super().__init__()
        self.stride = stride
        self.expand = nn.Linear(d_model, stride * d_model, bias=False)
        self.ln = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, 256, bias=False)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        # z: [B, N, D]
        B, N, D = z.shape
        h = self.expand(z).view(B, N, self.stride, D)
        return self.head(self.ln(h))                       # [B, N, stride, 256]


# ── ByteEnergyModel ────────────────────────────────────────────────────────────

class ByteEnergyPreTrainedModel(EnergyPreTrainedModel):
    config_class = ByteEnergyConfig


class ByteEnergyModel(ByteEnergyPreTrainedModel, EnergyModel):
    """EnergyModel with byte-level input/output instead of BPE tokens.

    All energy blocks, FSDP sharding, torch.compile, and training infrastructure
    are inherited from EnergyModel. Only the embedding + head are replaced.
    """

    def __init__(self, config: ByteEnergyConfig):
        # Initialize energy model normally (creates standard embedding + blocks)
        super().__init__(config)

        D = config.hidden_size
        d_local = getattr(config, 'd_local', 64)
        W = getattr(config, 'window_size', 4)
        S = getattr(config, 'stride', 2)

        # Replace standard token embedding with byte encoder
        del self.wte                                  # remove nn.Embedding(vocab, D)
        self.byte_encoder = ByteLinearPool(d_local, D, W, S)
        self.byte_decoder = ByteDecoder(D, S)

        # Compressed sequence length (for positional embedding)
        self._stride = S
        self._window = W
        # pos_emb for compressed sequence — register as buffer for FSDP safety
        # (actual pos embedding is rope, handled internally in blocks)

    def get_input_embeddings(self):
        return self.byte_encoder.embed

    def set_input_embeddings(self, value):
        self.byte_encoder.embed = value

    def forward(
        self,
        input_ids: torch.Tensor,           # [B, T] raw bytes 0–255
        attention_mask=None,
        position_ids=None,
        labels=None,                       # [B, T] bytes for loss (same as input_ids shifted)
        past_key_values=None,
        use_cache=False,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=True,
        cu_seqlens=None,
        max_seqlen=None,
    ):
        B, T = input_ids.shape
        W, S = self._window, self._stride

        # ── Encode bytes → compressed tokens ─────────────────────────────────
        compressed = self.byte_encoder(input_ids)   # [B, T//S, D]

        # ── Run energy transformer on compressed sequence ─────────────────────
        # Call parent forward with pre-computed embeddings
        # We bypass the standard embedding by passing hidden_states directly.
        # Use the base model forward (EnergyModel.base_model calls _init_model blocks)
        hidden_states = compressed + self.pos_embedding(compressed)

        # Run through energy blocks (inherits energy/gpt blocks from EnergyModel)
        base_out = self.model(
            inputs_embeds=compressed,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
        )
        z = base_out.last_hidden_state          # [B, N, D]

        # ── Decode compressed tokens → byte logits ────────────────────────────
        logits = self.byte_decoder(z)           # [B, N, S, 256]

        # ── Loss ──────────────────────────────────────────────────────────────
        loss = None
        if labels is not None:
            # Target: bytes at positions [W : W + N*S]  (predict next window)
            tgt_start = W
            N = z.shape[1]
            tgt_end = tgt_start + N * S
            if tgt_end > T:
                n_full = (T - tgt_start) // S
                logits = logits[:, :n_full]
                tgt = labels[:, tgt_start : tgt_start + n_full * S].reshape(B, n_full, S)
            else:
                tgt = labels[:, tgt_start : tgt_end].reshape(B, N, S)
            loss = F.cross_entropy(
                logits.reshape(-1, 256),
                tgt.reshape(-1),
            )

        if not return_dict:
            output = (logits,) + base_out[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=base_out.past_key_values,
            hidden_states=base_out.hidden_states,
            attentions=base_out.attentions,
        )

    def pos_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """Stub — RoPE handles positional info inside the attention blocks."""
        return torch.zeros_like(x)
