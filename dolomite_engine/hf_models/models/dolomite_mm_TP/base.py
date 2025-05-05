from ...mixins import BaseModelMixin_TP, PreTrainedModelMixin_TP
from ..dolomite_mm import DolomiteMmConfig
from .layer import DolomiteMmBlock_TP


class DolomiteMmPreTrainedModel_TP(PreTrainedModelMixin_TP):
    config_class = DolomiteMmConfig
    layer_class = DolomiteMmBlock_TP
    _no_split_modules = ["DolomiteMmBlock_TP"]


class DolomiteMmModel_TP(DolomiteMmPreTrainedModel_TP, BaseModelMixin_TP): ...