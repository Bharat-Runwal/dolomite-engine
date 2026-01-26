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
    _no_split_modules = ["EnergyBlock"]


class EnergyModel(EnergyPreTrainedModel, BaseModelMixin): ...
