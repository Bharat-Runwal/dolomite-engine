# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from typing import Iterable

import torch

from ..config import CommonConfig
from .attention import _SoftmaxAttentionCache
from .mamba2 import _Mamba2Cache
from .rnn import _RNNCache


_CACHE_CLASSES = {
    "causal_convolution": _RNNCache,
    "energy_attention": _SoftmaxAttentionCache,
    "mixed_head_attention": _SoftmaxAttentionCache,
    "energy_grad_mixed_head_attention": _SoftmaxAttentionCache,
    "mixed_head_energy_descent": _SoftmaxAttentionCache,
    "gru": _RNNCache,
    "mamba2": _Mamba2Cache,
    "multihead_latent_attention": _SoftmaxAttentionCache,
    "rnn": _RNNCache,
    "softmax_attention": _SoftmaxAttentionCache,
}

CACHE_TYPE = torch.Tensor | tuple[torch.Tensor, torch.Tensor] | None


class GenerationCache:
    def __init__(self, config: CommonConfig, **kwargs) -> GenerationCache:
        # For iterated energy blocks, each iteration needs its own cache slot.
        # Total slots = sum(layer_iterations) so that each (block, iteration) pair
        # gets a unique cache index via the layer_id counter in the forward loop.
        cache = []
        for i in range(config.num_layers):
            mixer_type = config.sequence_mixer_blocks[i].sequence_mixer_type
            num_iter = max(1, config.layer_iterations[i]) if hasattr(config, 'layer_iterations') else 1
            for _ in range(num_iter):
                cache.append(_CACHE_CLASSES[mixer_type](config, i, **kwargs))
        self.cache = cache

    def __getitem__(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def __iter__(self) -> Iterable[CACHE_TYPE]:
        for layer_idx in range(len(self)):
            yield self.cache[layer_idx].get_cache()

    def update(self, *, layer_idx: int, **kwargs) -> CACHE_TYPE:
        return self.cache[layer_idx].update(**kwargs)

    # TODO remove this function
    def get_cache(self, layer_idx: int) -> CACHE_TYPE:
        return self.cache[layer_idx].get_cache()

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.cache[layer_idx].get_seq_length()

    def reorder_cache(self, beam_idx: torch.Tensor) -> None:
        for cache in self.cache:
            cache.reorder_cache(beam_idx)
