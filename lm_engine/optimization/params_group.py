# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

from __future__ import annotations

import logging
import re
from typing import Any

from ..containers import ModelContainer
from ..enums import ParamsGroupMethod
from ..hf_models import is_parameter_with_mup_learning_rate, is_parameter_with_no_weight_decay
from ..model_wrapper import ModelWrapper
from ..utils import BaseArgs, log_rank_0

_LAYER_INDEX_PATTERN = re.compile(r"\.h\.(\d+)\.")


class _ParamsGroup(BaseArgs):
    name: str
    parameter_name_map: dict
    params_group_kwargs: dict = {}

    def to_param_group(self) -> dict:
        result = {}
        result.update(self.params_group_kwargs)

        # do in a sorted order
        param_names = self.get_param_names()

        result["params"] = []
        for param_name in param_names:
            result["params"].append(self.parameter_name_map[param_name])

        return result

    def get_param_names(self) -> list[str]:
        param_names = list(self.parameter_name_map.keys())
        param_names.sort()
        return param_names

    def __len__(self) -> int:
        return len(self.parameter_name_map)


class _ParamsGroupsList(BaseArgs):
    params_groups: list[_ParamsGroup] = []

    def model_post_init(self, __context: Any) -> None:
        self.params_groups = list(filter(lambda group: len(group) > 0, self.params_groups))
        super().model_post_init(__context)

    def add_params_group(self, params_group: _ParamsGroup) -> None:
        self.params_groups.append(params_group)

    def to_torch_compatible_params_groups(self) -> list[dict]:
        return [group.to_param_group() for group in self.params_groups]

    def get_param_names(self) -> list[str]:
        return {group.name: group.get_param_names() for group in self.params_groups}


def _get_loop_layer_indices(config) -> set[int]:
    """Return the set of physical layer indices that belong to the looped (middle) section.

    For SUT models with num_iterations > 1, the middle layers (between pre and post)
    are executed multiple times. Their gradients accumulate across iterations, so they
    need a reduced learning rate (lr / num_iterations) to compensate.
    """
    num_iterations = getattr(config, "num_iterations", 1)
    if num_iterations <= 1:
        return set()
    num_pre = getattr(config, "num_pre_layers", 0)
    num_post = getattr(config, "num_post_layers", 0)
    num_layers = config.num_layers
    end = num_layers - num_post if num_post > 0 else num_layers
    return set(range(num_pre, end))


def _is_loop_layer_param(param_name: str, loop_layer_indices: set[int]) -> bool:
    """Check if a parameter belongs to a loop (middle) layer based on its name."""
    if not loop_layer_indices:
        return False
    match = _LAYER_INDEX_PATTERN.search(param_name)
    if match:
        return int(match.group(1)) in loop_layer_indices
    return False


def get_normal_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> _ParamsGroupsList:
    config = model.config
    loop_indices = _get_loop_layer_indices(config)
    num_iterations = getattr(config, "num_iterations", 1)

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    normal_loop_params = {}
    no_weight_decay_params = {}
    no_weight_decay_loop_params = {}

    for name, parameter in model.named_parameters():
        is_loop = _is_loop_layer_param(name, loop_indices)
        if is_parameter_with_no_weight_decay(parameter):
            if is_loop:
                no_weight_decay_loop_params[name] = parameter
            else:
                no_weight_decay_params[name] = parameter
        else:
            if is_loop:
                normal_loop_params[name] = parameter
            else:
                normal_params[name] = parameter

    groups = [
        _ParamsGroup(name="normal", parameter_name_map=normal_params),
        _ParamsGroup(
            name="no_weight_decay",
            parameter_name_map=no_weight_decay_params,
            params_group_kwargs={"weight_decay": 0},
        ),
    ]

    if loop_indices:
        loop_lr = optimizer_class_args["lr"] / num_iterations
        log_rank_0(
            logging.INFO,
            f"SUT loop layer LR correction: {len(normal_loop_params) + len(no_weight_decay_loop_params)} params "
            f"in layers {sorted(loop_indices)} get lr={loop_lr:.2e} (base_lr / {num_iterations})",
        )
        groups.extend([
            _ParamsGroup(
                name="normal_loop",
                parameter_name_map=normal_loop_params,
                params_group_kwargs={"lr": loop_lr},
            ),
            _ParamsGroup(
                name="no_weight_decay_loop",
                parameter_name_map=no_weight_decay_loop_params,
                params_group_kwargs={"weight_decay": 0, "lr": loop_lr},
            ),
        ])

    return _ParamsGroupsList(params_groups=groups)


def get_mup_group_with_names(model: ModelWrapper, optimizer_class_args: dict) -> list[_ParamsGroup]:
    config = model.config
    assert (
        config.init_method == "mup"
    ), "both init method for model and params group method for optimizer should be set to mup"

    loop_indices = _get_loop_layer_indices(config)
    num_iterations = getattr(config, "num_iterations", 1)
    m_width = config.m_width
    base_lr = optimizer_class_args["lr"]

    if model.has_teacher_model():
        log_rank_0(logging.WARN, "found a teacher model in the ModelWrapper")
        # this is the student model
        model = model.model

    normal_params = {}
    normal_loop_params = {}
    no_weight_decay_params = {}
    no_weight_decay_loop_params = {}
    mup_params = {}
    mup_loop_params = {}

    for name, parameter in model.named_parameters():
        is_loop = _is_loop_layer_param(name, loop_indices)
        if is_parameter_with_mup_learning_rate(parameter):
            if is_loop:
                mup_loop_params[name] = parameter
            else:
                mup_params[name] = parameter
        elif is_parameter_with_no_weight_decay(parameter):
            if is_loop:
                no_weight_decay_loop_params[name] = parameter
            else:
                no_weight_decay_params[name] = parameter
        else:
            if is_loop:
                normal_loop_params[name] = parameter
            else:
                normal_params[name] = parameter

    groups = [
        _ParamsGroup(name="normal", parameter_name_map=normal_params),
        _ParamsGroup(
            name="no_weight_decay",
            parameter_name_map=no_weight_decay_params,
            params_group_kwargs={"weight_decay": 0},
        ),
        _ParamsGroup(
            name="mup",
            parameter_name_map=mup_params,
            params_group_kwargs={"lr": base_lr / m_width},
        ),
    ]

    if loop_indices:
        loop_lr = base_lr / num_iterations
        mup_loop_lr = base_lr / (m_width * num_iterations)
        num_loop_params = len(normal_loop_params) + len(no_weight_decay_loop_params) + len(mup_loop_params)
        log_rank_0(
            logging.INFO,
            f"SUT loop layer LR correction (muP): {num_loop_params} params "
            f"in layers {sorted(loop_indices)} get lr scaled by 1/{num_iterations} "
            f"(normal_loop={loop_lr:.2e}, mup_loop={mup_loop_lr:.2e})",
        )
        groups.extend([
            _ParamsGroup(
                name="normal_loop",
                parameter_name_map=normal_loop_params,
                params_group_kwargs={"lr": loop_lr},
            ),
            _ParamsGroup(
                name="no_weight_decay_loop",
                parameter_name_map=no_weight_decay_loop_params,
                params_group_kwargs={"weight_decay": 0, "lr": loop_lr},
            ),
            _ParamsGroup(
                name="mup_loop",
                parameter_name_map=mup_loop_params,
                params_group_kwargs={"lr": mup_loop_lr},
            ),
        ])

    return _ParamsGroupsList(params_groups=groups)


_PARAM_GROUPS = {
    None: get_normal_group_with_names,
    ParamsGroupMethod.mup: get_mup_group_with_names,
}


def get_param_groups_list(
    model_container: ModelContainer, optimizer_class_args: dict, params_group_method: ParamsGroupMethod | None
) -> list[list[_ParamsGroup]]:
    if params_group_method not in _PARAM_GROUPS:
        raise ValueError(f"unexpected `params_group_method` {params_group_method}")

    return [_PARAM_GROUPS[params_group_method](model, optimizer_class_args) for model in model_container]
