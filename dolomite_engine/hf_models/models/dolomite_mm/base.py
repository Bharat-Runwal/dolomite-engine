from ...mixins import BaseModelMixin, PreTrainedModelMixin
from .config import DolomiteMmConfig
from .layer import DolomiteMmBlock
import torch.nn as nn

class DolomiteMmPreTrainedModel(PreTrainedModelMixin):
    config_class = DolomiteMmConfig
    layer_class = DolomiteMmBlock
    _no_split_modules = ["DolomiteMmBlock"]

class DolomiteMmModel(DolomiteMmPreTrainedModel, BaseModelMixin): ...