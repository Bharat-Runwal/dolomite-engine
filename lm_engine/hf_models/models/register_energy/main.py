# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from ...mixins import CausalLMModelMixin
from .model import RegisterEnergyModel, RegisterEnergyPreTrainedModel


class RegisterEnergyForCausalLM(RegisterEnergyPreTrainedModel, CausalLMModelMixin):
    base_model_class = RegisterEnergyModel
