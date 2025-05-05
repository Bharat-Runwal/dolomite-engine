from ...mixins import CausalLMModelMixin
from .base import DolomiteMmModel, DolomiteMmPreTrainedModel


class DolomiteMmForCausalLM(DolomiteMmPreTrainedModel, CausalLMModelMixin):
    base_model_class = DolomiteMmModel
