# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging
import os
from functools import partial
from typing import Callable

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import CPUOffloadPolicy
from torch.distributed._composable.fsdp import MixedPrecisionPolicy as MixedPrecision2
from torch.distributed._composable.fsdp import OffloadPolicy, fully_shard
from torch.distributed._tensor import distribute_tensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.fsdp import CPUOffload
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision as MixedPrecision1
from torch.distributed.fsdp import ShardingStrategy
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.distributed.pipelining import PipelineStage
from torch.distributed.pipelining.schedules import (
    PipelineScheduleMulti,
    PipelineScheduleSingle,
    _PipelineSchedule,
    get_schedule_class,
)

from .arguments import TrainingArgs
from .containers import ModelContainer
from .enums import Kernel
from .gradient_checkpointing import apply_gradient_checkpointing
from .hf_models import CausalLMOutputWithPast
from .hf_models.parameter import _ALL_MARKERS
from .kernels import is_kernel_allowed
from .utils import (
    Accelerator,
    ProcessGroupManager,
    get_module_class_from_name,
    is_torch_xla_available,
    is_torchao_available,
    log_rank_0,
    string_to_torch_dtype,
)


if is_torch_xla_available():
    from torch_xla.distributed.fsdp import XlaFullyShardedDataParallel as XLA_FSDP
    from torch_xla.distributed.fsdp.wrap import transformer_auto_wrap_policy as xla_transformer_auto_wrap_policy


if is_torchao_available():
    from torchao.float8 import ScalingType

    from .fp8 import FP8Manager

# NOTE: this is OURS, not a torch default -- every related torch default is off. It makes
# `_needs_spmd_graph_preservation()` true, which pulls in inductor's `spmd_check` pass. That
# pass calls dist.all_gather_object DURING COMPILE (twice: hashes, then diagnostics on the
# mismatch path). If ranks compile a different NUMBER of graphs they make a different number
# of blocking collective calls and DEADLOCK. Overridden for multi-node below.
torch._inductor.config.reorder_for_compute_comm_overlap = True


_STAGE_FULL_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy.SHARD_GRAD_OP,
    3: ShardingStrategy.FULL_SHARD,
}

_STAGE_HYBRID_SHARDING_STRATEGY_MAP = {
    2: ShardingStrategy._HYBRID_SHARD_ZERO2,
    3: ShardingStrategy.HYBRID_SHARD,
}

_FSDP_1_STRING = "_fsdp_wrapped_module"
_TORCH_COMPILE_STRING = "_orig_mod"
_FSDP_TPU_SHARD_SEPARATOR = "_FSDP_SHARD_SEPARATOR_"
_FSDP_TPU_SHARD = "_fsdp_shard"
_FSDP_TPU_FPW = "_fpw_module"


def _get_pipeline_parallel_schedule(
    pipeline_parallel_schedule: str,
    gradient_accumulation_steps: int,
    pipeline_stages: list[PipelineStage],
    loss_fn: Callable,
) -> _PipelineSchedule:
    try:
        schedule_class = get_schedule_class(pipeline_parallel_schedule)
    except ValueError:
        raise ValueError(
            f"unexpected schedule ({pipeline_parallel_schedule}), expected values are: ['1F1B', "
            "'Interleaved1F1B', 'GPipe', 'FlexibleInterleaved1F1B', 'LoopedBFS', 'InterleavedZeroBubble', "
            "'PipelineScheduleSingle', 'PipelineScheduleMulti']"
        )

    if schedule_class in [PipelineScheduleSingle, PipelineScheduleMulti]:
        raise NotImplementedError()

    if issubclass(schedule_class, PipelineScheduleSingle):
        assert len(pipeline_stages) == 1

    def custom_loss_function(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        loss_dict = loss_fn(output, target)
        return loss_dict["loss"]

    return schedule_class(
        pipeline_stages if issubclass(schedule_class, PipelineScheduleMulti) else pipeline_stages[0],
        n_microbatches=gradient_accumulation_steps,
        loss_fn=custom_loss_function,
    )


def _get_fsdp_mixed_precision(
    dtype: torch.dtype, communication_dtype: torch.dtype | None, fsdp_algorithm: int
) -> MixedPrecision1:
    if communication_dtype is None:
        communication_dtype = dtype

    if fsdp_algorithm == 1:
        mixed_precision = MixedPrecision1(param_dtype=dtype, reduce_dtype=communication_dtype, buffer_dtype=dtype)
    else:
        mixed_precision = MixedPrecision2(param_dtype=dtype, reduce_dtype=communication_dtype)

    return mixed_precision


def _get_parameter_marker_maps(model_container: ModelContainer) -> list[dict]:
    marker_maps = []
    for model in model_container:
        marker_maps.append({})
        for param_name, param in model.named_parameters():
            marker_maps[-1][param_name] = {}
            for marker in _ALL_MARKERS:
                marker_maps[-1][param_name][marker] = getattr(param, marker, False)

    return marker_maps


def _set_parameter_marker_maps(model_container: ModelContainer, marker_maps: list[dict]) -> None:
    for model, _marker_map in zip(model_container, marker_maps):
        for param_name, parameter in model.named_parameters():
            # handle FSDP for TPU
            param_name = param_name.replace(_FSDP_TPU_SHARD_SEPARATOR, ".")
            param_name = param_name.replace(f"{_FSDP_TPU_SHARD}.", "")
            param_name = param_name.replace(f"{_FSDP_TPU_FPW}.", "")

            # handle FSDP-1
            param_name = param_name.replace(f"{_FSDP_1_STRING}.", "")

            # handle torch compile
            param_name = param_name.replace(f"{_TORCH_COMPILE_STRING}.", "")

            for marker, value in _marker_map[param_name].items():
                setattr(parameter, marker, value)


def wrap_model_container_for_distributed_training(
    args: TrainingArgs, model_container: ModelContainer
) -> tuple[ModelContainer, _PipelineSchedule]:
    """converts the model to a ZeRO-DP sharded model

    Args:
        args (TrainingArgs): arguments based on training mode
        model_container (ModelContainer): model container

    Returns:
        tuple[ModelContainer, _PipelineSchedule]: container of parallelized models and pipeline schedule
    """

    stage = args.distributed_args.stage
    cpu_offload = args.distributed_args.cpu_offload
    torch_compile = args.distributed_args.torch_compile
    dtype = args.mixed_precision_args.dtype
    communication_dtype = args.distributed_args.communication_dtype
    efficient_initialization = args.model_args.efficient_initialization
    fsdp_algorithm = args.distributed_args.fsdp_algorithm
    num_pipeline_stages = args.distributed_args.num_pipeline_stages
    data_parallel_sharding_world_size = ProcessGroupManager.get_data_parallel_sharding_world_size()
    data_parallel_replication_world_size = ProcessGroupManager.get_data_parallel_replication_world_size()
    model_name = args.model_args.model_name

    if dtype in ["fp16", "bf16"]:
        if communication_dtype != "fp32":
            log_rank_0(
                logging.WARN,
                f"using ({communication_dtype}) with mixed precision training in ({dtype}), recommended is to use ({torch.float32})",
            )
    elif dtype == "fp8":
        assert is_torchao_available(), "torchao is needed for FP8 training"

        FP8Manager(
            model_container,
            enable_fsdp_fp8_all_gather=ProcessGroupManager.get_data_parallel_sharding_world_size() > 1,
            precompute_fp8_dynamic_scale_for_fsdp=True,
            torch_compile=torch_compile,
            scaling_type_input=ScalingType(args.mixed_precision_args.scaling_type_input),
            scaling_type_weight=ScalingType(args.mixed_precision_args.scaling_type_weight),
            scaling_type_grad_output=ScalingType(args.mixed_precision_args.scaling_type_grad_output),
        )

        dtype = "bf16"

    block_names = model_container[0].model._no_split_modules
    teacher_block_names = (
        model_container[0].teacher_model._no_split_modules if model_container[0].has_teacher_model() else []
    )

    dtype = None if dtype is None else string_to_torch_dtype(dtype)
    communication_dtype = None if communication_dtype is None else string_to_torch_dtype(communication_dtype)

    assert stage in [0, 2, 3]

    dp_mesh = ProcessGroupManager.get_data_parallel_mesh()
    block_classes = [
        get_module_class_from_name(model_container[0], name) for name in block_names + teacher_block_names
    ]

    if args.distributed_args.gradient_checkpointing_method is not None:
        assert len(block_names) == 1

        for model in model_container:
            apply_gradient_checkpointing(
                model,
                args.distributed_args.gradient_checkpointing_method,
                block_name=block_names[0],
                **args.distributed_args.gradient_checkpointing_args,
            )

    marker_maps = _get_parameter_marker_maps(model_container)
    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.tpu:
        assert (
            ProcessGroupManager.get_data_parallel_world_size()
            == ProcessGroupManager.get_data_parallel_sharding_world_size()
        )

        assert num_pipeline_stages == 1
        assert fsdp_algorithm == 1
        assert not torch_compile

        for i, model in enumerate(model_container):
            model_container[i] = XLA_FSDP(
                model,
                compute_dtype=torch.bfloat16,
                buffer_dtype=torch.bfloat16,
                sharding_groups=[
                    torch.distributed.get_process_group_ranks(ProcessGroupManager.get_data_parallel_group())
                ],
                sharding_rank=ProcessGroupManager.get_data_parallel_rank(),
                sharding_world_size=ProcessGroupManager.get_data_parallel_sharding_world_size(),
                auto_wrap_policy=partial(xla_transformer_auto_wrap_policy, transformer_layer_cls=block_classes),
                param_init_fn=_param_init_fsdp_1 if efficient_initialization else None,
            )
    else:
        use_ddp = (stage == 0 or data_parallel_sharding_world_size == 1) and num_pipeline_stages == 1

        mixed_precision_policy = _get_fsdp_mixed_precision(
            dtype=dtype,
            communication_dtype=communication_dtype,
            fsdp_algorithm=2 if use_ddp else fsdp_algorithm,
        )

        if use_ddp or fsdp_algorithm == 2:
            log_rank_0(logging.INFO, "using FSDP-2")
            zero3 = stage == 3

            def _sharding_function(parameter: nn.Parameter) -> Shard:
                dps = (
                    ProcessGroupManager.get_data_parallel_world_size()
                    if data_parallel_sharding_world_size is None
                    else data_parallel_sharding_world_size
                )

                if parameter.size(0) > dps or parameter.dim() == 1:
                    return Shard(0)
                else:
                    for dim in range(1, parameter.dim()):
                        if parameter.size(dim) > dps and parameter.size(dim) % dps == 0:
                            return Shard(dim)

                    log_rank_0(logging.WARN, "sharding along dim=0 since no suitable sharding dimension was found")
                    return Shard(0)

            for i, model in enumerate(model_container):
                if efficient_initialization and model_name is not None:
                    # state dict with Tensors
                    old_state_dict = model.state_dict()

                for module in model.modules():
                    if isinstance(module, tuple(block_classes)):
                        fully_shard(
                            module,
                            mesh=dp_mesh,
                            reshard_after_forward=zero3,
                            shard_placement_fn=_sharding_function,
                            mp_policy=mixed_precision_policy,
                            offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
                        )

                fully_shard(
                    model,
                    mesh=dp_mesh,
                    reshard_after_forward=zero3,
                    shard_placement_fn=None if use_ddp else _sharding_function,
                    mp_policy=mixed_precision_policy,
                    offload_policy=CPUOffloadPolicy(pin_memory=True) if cpu_offload else OffloadPolicy(),
                )

                if efficient_initialization:
                    device = Accelerator.get_current_device()

                    # contributed by Yu Chin Fabian Lim
                    # original reference https://github.com/fabianlim/accelerate/pull/1
                    if model_name is None:
                        model = model.to_empty(device=device)

                        for module in model.modules():
                            if hasattr(module, "reset_parameters"):
                                with torch.device(device):
                                    module.reset_parameters()
                    else:
                        if ProcessGroupManager.get_data_parallel_rank() == 0:
                            model = model.to(device)
                        else:
                            model = model.to_empty(device=device)

                            for module in model.modules():
                                if hasattr(module, "reset_parameters"):
                                    with torch.device(device):
                                        module.reset_parameters()

                        # state dict with DTensors
                        new_state_dict = model.state_dict()

                        for param_name, param in old_state_dict.items():
                            if ProcessGroupManager.get_data_parallel_rank() == 0:
                                full_tensor = param
                            else:
                                full_tensor = torch.empty(param.shape, dtype=param.dtype, device=device)

                            new_state_dict[param_name] = distribute_tensor(
                                full_tensor,
                                device_mesh=new_state_dict[param_name].device_mesh,
                                placements=new_state_dict[param_name].placements,
                            )

                        model.load_state_dict(new_state_dict, assign=True)
                        del old_state_dict, new_state_dict
        elif fsdp_algorithm == 1:
            log_rank_0(logging.INFO, "using FSDP-1")
            assert num_pipeline_stages == 1

            sharding_strategy = (
                _STAGE_FULL_SHARDING_STRATEGY_MAP[stage]
                if data_parallel_replication_world_size == 1
                else _STAGE_HYBRID_SHARDING_STRATEGY_MAP[stage]
            )

            for i, model in enumerate(model_container):
                model_container[i] = FSDP(
                    model,
                    sharding_strategy=sharding_strategy,
                    cpu_offload=CPUOffload(offload_params=True) if cpu_offload else None,
                    mixed_precision=mixed_precision_policy,
                    auto_wrap_policy=partial(transformer_auto_wrap_policy, transformer_layer_cls=block_classes),
                    device_id=Accelerator.get_current_device(),
                    limit_all_gathers=True,
                    use_orig_params=True,
                    # https://github.com/meta-llama/llama-recipes/blob/492455dc080f6c25f356e283e443be0cce86aaeb/src/llama_recipes/finetuning.py#L191
                    sync_module_states=efficient_initialization,
                    param_init_fn=_param_init_fsdp_1 if efficient_initialization else None,
                    device_mesh=dp_mesh,
                )
        else:
            raise ValueError(f"unexpected fsdp_algorithm ({fsdp_algorithm})")

    if torch_compile:
        log_rank_0(logging.INFO, "using torch compile")

        # `_run_block` guards on the layer index `i`, so a model with L DISTINCT blocks
        # produces L specializations of it. torch._dynamo's default recompile_limit is 8:
        # past it dynamo gives up and falls back to EAGER for that function, silently, so
        # torch_compile stops doing anything and any wall-clock number becomes meaningless.
        # Recurrent configs are safe (num_layers 3-7), but a non-recurrent stack is not --
        # the 12-layer port of the colleagues' s8e4 config hits the limit at i == 7.
        # Raise the ceiling to cover the deepest model present, with headroom for the
        # dense->sparse switch (sparse_start_step) which legitimately adds one more.
        num_distinct_blocks = max(
            (len(getattr(m.config, "sequence_mixer_blocks", []) or []) for m in model_container),
            default=0,
        )
        recompile_limit = max(torch._dynamo.config.recompile_limit, 4 * num_distinct_blocks + 8)
        if recompile_limit > torch._dynamo.config.recompile_limit:
            log_rank_0(
                logging.INFO,
                f"raising torch._dynamo recompile_limit "
                f"{torch._dynamo.config.recompile_limit} -> {recompile_limit} "
                f"for {num_distinct_blocks} distinct blocks",
            )
            torch._dynamo.config.recompile_limit = recompile_limit
            torch._dynamo.config.accumulated_recompile_limit = max(
                torch._dynamo.config.accumulated_recompile_limit, 16 * recompile_limit
            )

        # MULTI-NODE COMPILE DEADLOCK (measured 2026-09-17). At 16 GPU / 2 nodes the two hosts
        # compiled structurally DIFFERENT graphs for the same frame -- inductor's spmd_check
        # reported ranks 0-7 with 2 call_function nodes and ranks 8-15 with 27, a split exactly on
        # the host boundary. spmd_check does blocking dist.all_gather_object during compile, so
        # divergent graph counts across hosts deadlock it: the job sat 108 min at 0 steps with
        # cpu_used flat (746 -> 748s across 16 ranks), and with the default 30-min gloo timeout it
        # instead died as `InductorError: Timed out waiting 1800000ms`.
        # The SPMD machinery exists only to protect comm/compute overlap REORDERING, which is a
        # performance optimisation we enable ourselves. Single node is unaffected (all ranks on one
        # host agree), so keep the optimisation there and drop it when hosts could disagree.
        world = int(os.environ.get("WORLD_SIZE", "1"))
        local = int(os.environ.get("LOCAL_WORLD_SIZE", "0")) or torch.cuda.device_count() or 1
        n_nodes = max(1, world // max(1, local))
        # DOLOMITE_SPMD_DIAG=1 re-enables the check and makes a mismatch RAISE instead of hang,
        # so a bisect gets a diagnostic rather than a 108-minute wedge. Debug only.
        spmd_diag = os.environ.get("DOLOMITE_SPMD_DIAG", "0") == "1"
        if spmd_diag:
            torch._inductor.config.aten_distributed_optimizations.spmd_check = True
            torch._inductor.config.aten_distributed_optimizations.spmd_mismatch = "error"
            log_rank_0(logging.INFO, "DOLOMITE_SPMD_DIAG=1: spmd_check ON, mismatch=error (fail fast)")
        elif world > 1:
            # WIDENED 2026-09-19 from `n_nodes > 1` to `world > 1`. The claim above that "single
            # node is unaffected (all ranks on one host agree)" is DISPROVEN: abl_B_134M_6G1x6S
            # (job 1775598, a recurrent Switch block, 1 node x 4 GPUs) died with exactly this
            # signature -- InductorError from post_grad_passes -> spmd_check ->
            # dist.all_gather_object, then a NCCL collective timeout on the other ranks. Its 400M
            # twin at 2 nodes SURVIVED only because the old guard already applied there. So graph
            # divergence is a property of the MODEL (data-dependent structure), not of the host
            # boundary, and any world_size > 1 can hit it.
            # spmd_check is a DIAGNOSTIC: it detects divergence, it does not prevent it. Turning it
            # off costs the comm/compute overlap reordering (a performance optimisation we enable
            # ourselves at :63) and nothing else. DOLOMITE_SPMD_DIAG=1 still restores it, failing
            # fast rather than hanging, for bisects.
            torch._inductor.config.reorder_for_compute_comm_overlap = False
            torch._inductor.config.aten_distributed_optimizations.spmd_check = False
            log_rank_0(
                logging.INFO,
                f"world_size {world} ({n_nodes} node(s)): disabled reorder_for_compute_comm_overlap "
                f"and inductor spmd_check to avoid the compile-time all_gather deadlock",
            )

        for i, model in enumerate(model_container):
            model_container[i] = torch.compile(model)

    _set_parameter_marker_maps(model_container, marker_maps)

    pipeline_stages = []
    pipeline_schedule = None

    if num_pipeline_stages > 1:
        micro_batch_size = args.training_parameters.micro_batch_size
        sequence_length = args.datasets[0].class_args.get("sequence_length")

        pipeline_parallel_schedule = args.distributed_args.pipeline_parallel_schedule
        gradient_accumulation_steps = args.training_parameters.gradient_accumulation_steps

        if pipeline_parallel_schedule == "1F1B":
            assert (
                gradient_accumulation_steps % num_pipeline_stages == 0
            ), f"gradient_accumulation_steps ({gradient_accumulation_steps}) should be divisible by num_pipeline_stages ({num_pipeline_stages})"

        for model in model_container:
            intermediate_dtype = string_to_torch_dtype(args.mixed_precision_args.dtype)

            dummy_input_tensor = model.model.get_dummy_input_tensor(
                micro_batch_size, sequence_length, intermediate_dtype=intermediate_dtype
            )
            dummy_output_tensor = model.model.get_dummy_output_tensor(
                micro_batch_size,
                sequence_length,
                intermediate_dtype=intermediate_dtype,
                output_parallel_lm_logits_if_possible=True,
            )

            stage = PipelineStage(
                model,
                stage_index=model.pipeline_stage_id,
                num_stages=num_pipeline_stages,
                device=Accelerator.get_current_device(),
                input_args=dummy_input_tensor,
                output_args=dummy_output_tensor,
                group=ProcessGroupManager.get_pipeline_parallel_group(),
            )

            pipeline_stages.append(stage)

        lm_loss_multiplier = 1 / (
            args.training_parameters.micro_batch_size * args.datasets[0].class_args.get("sequence_length")
        )

        def _pipeline_parallel_loss(input: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
            use_fused_linear_cross_entropy = is_kernel_allowed(Kernel.fused_linear_cross_entropy)

            if isinstance(input, tuple):
                input, aux_loss = input
            else:
                aux_loss = 0

            output = CausalLMOutputWithPast(
                logits=None if use_fused_linear_cross_entropy else input,
                aux_loss=aux_loss,
                last_hidden_state=input if use_fused_linear_cross_entropy else None,
            )
            loss = model.get_loss(output, target, lm_loss_multiplier)

            return loss

        pipeline_schedule = _get_pipeline_parallel_schedule(
            pipeline_parallel_schedule=args.distributed_args.pipeline_parallel_schedule,
            gradient_accumulation_steps=args.training_parameters.gradient_accumulation_steps,
            pipeline_stages=pipeline_stages,
            loss_fn=_pipeline_parallel_loss,
        )

    return model_container, pipeline_schedule


def _param_init_fsdp_1(module: nn.Module, teacher_block_names: list[str], model_name: str) -> None:
    assert len(teacher_block_names) == 0, "efficient initialization doesn't support distillation"

    device = Accelerator.get_current_device()

    if model_name is None:
        module = module.to_empty(device=device)

        if hasattr(module, "reset_parameters"):
            with torch.no_grad():
                module.reset_parameters()
    else:
        if ProcessGroupManager.get_data_parallel_rank() != 0:
            module = module.to_empty(device=device)
