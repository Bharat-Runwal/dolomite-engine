# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import EnergyConfig
from .layer import EnergyBlock


class EnergyPreTrainedModel(PreTrainedModelMixin):
    config_class = EnergyConfig
    layer_class = EnergyBlock

    @property
    def _no_split_modules(self):
        # When shared_backbone=True, all EnergyBlocks share the same attn/ffwd/ln
        # tensors. FSDP2 can't shard parameters that already belong to another
        # fully_shard'd module, so we skip per-block wrapping and let only the
        # top-level model get wrapped (DDP-style with stage=0).
        if getattr(self.config, "shared_backbone", False):
            return []
        return ["EnergyBlock"]

    @_no_split_modules.setter
    def _no_split_modules(self, value):
        # transformers < 5 does `self._no_split_modules = self._no_split_modules or []`
        # in PreTrainedModel.__init__; a read-only property raises there. Accept and
        # discard so this model works on both transformers 4.57.x and 5.x.
        pass


class EnergyModel(EnergyPreTrainedModel, BaseModelMixin): ...
