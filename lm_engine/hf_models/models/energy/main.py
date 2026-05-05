# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .base import EnergyModel, EnergyPreTrainedModel


class EnergyForCausalLM(EnergyPreTrainedModel, CausalLMModelMixin):
    base_model_class = EnergyModel
