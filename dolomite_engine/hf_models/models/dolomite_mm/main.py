from ...mixins import CausalLMModelMixin
from .base import DolomiteMMModel, DolomiteMMPreTrainedModel


class DolomiteMMForCausalLM(DolomiteMMPreTrainedModel, CausalLMModelMixin):
    base_model_class = DolomiteMMModel
