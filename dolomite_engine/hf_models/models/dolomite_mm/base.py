from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import DolomiteMMConfig
from .layer import DolomiteMMBlock
import torch.nn as nn

class DolomiteMMPreTrainedModel(PreTrainedModelMixin):
    config_class = DolomiteMMConfig
    layer_class = DolomiteMMBlock
    _no_split_modules = ["DolomiteMMBlock"]

class DolomiteMMModel(DolomiteMMPreTrainedModel, BaseModelMixin): ...
