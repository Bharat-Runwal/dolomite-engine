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

    # NOTE: upstream defines this as a read-only @property, which is
    # incompatible with transformers PreTrainedModel.__init__ (it assigns
    # self._no_split_modules = set(...) and a property has no setter -> crash on
    # both 4.57.1 and 5.1.0). We use a plain class attribute, identical to the
    # property's return value for shared_backbone=False (the only mode this run
    # uses; the config does not set shared_backbone). Zero numeric change.
    _no_split_modules = ["EnergyBlock"]


class EnergyModel(EnergyPreTrainedModel, BaseModelMixin): ...
