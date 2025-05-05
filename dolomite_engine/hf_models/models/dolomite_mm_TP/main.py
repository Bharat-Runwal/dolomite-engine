from ...mixins import CausalLMModelMixin_TP
from .base import DolomiteMmModel_TP, DolomiteMmPreTrainedModel_TP
from .weights import get_gpt_dolomite_model_parallel_state_dict


class DolomiteMmForCausalLM_TP(DolomiteMmPreTrainedModel_TP, CausalLMModelMixin_TP):
    base_model_class = DolomiteMmModel_TP
    model_parallel_state_dict_function = get_gpt_dolomite_model_parallel_state_dict
