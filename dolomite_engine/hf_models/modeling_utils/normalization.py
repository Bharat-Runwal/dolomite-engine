# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import torch
import torch.nn as nn
import torch.nn.functional as F

from ...enums import Kernel
from ...kernels import is_kernel_allowed
from ...utils import is_cute_kernels_available
from ..parameter import mark_parameter_as_no_weight_decay


if is_cute_kernels_available():
    from cute_kernels import rmsnorm_cute


class RMSNorm(nn.RMSNorm):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if is_kernel_allowed(Kernel.rmsnorm_cute):
            hidden_states = rmsnorm_cute(x=hidden_states, weight=self.weight, eps=self.eps, memory_efficient=False)
        else:
            hidden_states = super().forward(hidden_states)

        return hidden_states


class GatedRMSNorm(RMSNorm):
    def forward(self, hidden_states: torch.Tensor, gate: torch.Tensor | None = None) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.float()

        if gate is not None:
            hidden_states = hidden_states * F.silu(gate.float())

        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)

        hidden_states = hidden_states.to(input_dtype)

        if self.weight is not None:
            hidden_states = self.weight * hidden_states

        return hidden_states


_NORMALIZATION_FUNCTIONS = {"layernorm": nn.LayerNorm, "rmsnorm": RMSNorm, "silu_gated_rmsnorm": GatedRMSNorm}


def get_normalization_function(
    normalization_function: str, normalized_shape: int, eps: float = 1e-5
) -> nn.LayerNorm | RMSNorm:
    if normalization_function in _NORMALIZATION_FUNCTIONS:
        normalization = _NORMALIZATION_FUNCTIONS[normalization_function](normalized_shape, eps=eps)
    else:
        raise ValueError(f"unexpected `normalization_function` {normalization_function}")

    for parameter in normalization.parameters():
        mark_parameter_as_no_weight_decay(parameter)

    return normalization
