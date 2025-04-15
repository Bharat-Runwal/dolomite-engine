from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import GPTDolomiteConfig
from .layer import GPTDolomiteBlock, GPTDolomiteMTPBlock


class GPTDolomitePreTrainedModel(PreTrainedModelMixin):
    config_class = GPTDolomiteConfig
    layer_class = GPTDolomiteBlock
    mtp_layer_class= GPTDolomiteMTPBlock
    _no_split_modules = ["GPTDolomiteBlock"]


class GPTDolomiteModel(GPTDolomitePreTrainedModel, BaseModelMixin): ...
