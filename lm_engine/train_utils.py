# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import logging

import torch
from torch.distributed import ReduceOp
from transformers import AutoConfig

from .enums import GradientCheckpointingMethod
from .hf_models import CommonConfig, is_custom_model
from .hf_models.modeling_utils import is_glu
from .utils import (
    Accelerator,
    ExperimentsTracker,
    MetricsTrackingDict,
    ProcessGroupManager,
    divide_if_divisible,
    log_metrics,
)


def all_reduce_metrics_tracker(metrics_tracker: MetricsTrackingDict) -> MetricsTrackingDict:
    tensor = [metrics_tracker[key] for key in metrics_tracker]
    tensor = torch.stack(tensor)
    # NOTE the cpu() call was to save memory but might not be needed anymore
    # tensor = torch.stack(tensor) / ProcessGroupManager.get_data_parallel_world_size()
    # tensor = tensor.cpu()
    # gloo op doesn't support averaging so we do sum and divide by world size above

    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.tpu:
        torch.distributed.all_reduce(tensor, op=ReduceOp.SUM, group=ProcessGroupManager.get_data_parallel_group())
        tensor = tensor * (1 / ProcessGroupManager.get_data_parallel_world_size())
    else:
        torch.distributed.all_reduce(tensor, op=ReduceOp.AVG, group=ProcessGroupManager.get_data_parallel_group())

    for i, key in enumerate(metrics_tracker):
        metrics_tracker[key] = tensor[i]

    return metrics_tracker


def track_metrics(
    global_step: int, experiments_tracker: ExperimentsTracker, metrics_tracker: MetricsTrackingDict, context: str
) -> None:
    """tracks metrics like training loss, learning rate etc

    Args:
        global_step (int): global step during training
        experiments_tracker (ExperimentsTracker): metrics tracker
        metrics_tracker (float): metrics tracker
        context (str): experiment context
    """

    # experiments tracker
    experiments_tracker.track(metrics_tracker.get_dict(), step=global_step, context=context)

    message = f"step = {global_step}"
    for key in metrics_tracker:
        if key == "learning_rate":
            message += f", {key} = {metrics_tracker[key]:.4e}"
        else:
            message += f", {context}-{key} = {metrics_tracker[key]:.4f}"

    log_metrics(logging.INFO, message)


_PREV_SINK_VALUES: torch.Tensor | None = None


@torch.no_grad()
@torch.compiler.disable
def collect_sink_metrics(model: torch.nn.Module) -> dict:
    """Collect metrics from learnable attention sink parameters.

    Iterates over the model to find 'sinks' parameters (from Attention layers with
    use_attention_sink=True) and returns their value/gradient statistics for logging.
    Tracks value delta between calls as a proxy for gradient activity (FSDP2 zeros
    gradients after optimizer.step(), making direct grad snapshots unreliable).

    Args:
        model: the model (possibly FSDP-wrapped)

    Returns:
        dict with sink metrics, empty if no sink parameters found
    """
    global _PREV_SINK_VALUES

    sink_values = []

    for name, param in model.named_parameters():
        if not name.endswith(".sinks"):
            continue

        # FSDP2 wraps params as DTensors; convert to regular tensors for metric ops
        data = param.data.detach()
        if hasattr(data, "full_tensor"):
            data = data.full_tensor()

        sink_values.append(data)

    if not sink_values:
        return {}

    all_values = torch.cat(sink_values)

    metrics = {
        "sink/mean_value": all_values.mean(),
        "sink/min_value": all_values.min(),
        "sink/max_value": all_values.max(),
    }

    # value delta: how much sinks moved since last logging call
    if _PREV_SINK_VALUES is not None and _PREV_SINK_VALUES.shape == all_values.shape:
        delta = (all_values - _PREV_SINK_VALUES).abs()
        metrics["sink/mean_delta"] = delta.mean()
        metrics["sink/max_delta"] = delta.max()

    _PREV_SINK_VALUES = all_values.clone()

    return metrics


@torch.no_grad()
@torch.compiler.disable
def collect_stable_rank_metrics(model: torch.nn.Module, tokens_per_update: int, skip_large_dim: int = 8192) -> dict:
    """Compute stable rank of per-update gradient matrices for all 2-D Linear weights.

    Must be called AFTER gradient accumulation + clipping and BEFORE optimizer.step(),
    so that param.grad holds the actual update gradient handed to the optimizer.

    Stable rank: SR(G) = ||G||_F^2 / sigma_max(G)^2
    This lies in [1, min(m, n)] and measures the effective number of directions
    contributing to the gradient.

    Matrices with any dimension > skip_large_dim (e.g. lm_head over large vocab)
    are skipped because their SVD is too expensive and their gradients are sparse.

    A token-count cap warning is emitted when tokens_per_update < min(m, n) for
    any tracked layer (the cap artifact is not enforced here, only warned about).

    Args:
        model: the model (possibly FSDP-wrapped)
        tokens_per_update: batch_size * seq_len * grad_accum * dp_world_size
        skip_large_dim: skip matrices with any dimension larger than this

    Returns:
        dict with keys like "stable_rank/<layer_name>" plus aggregate stats,
        empty dict if no gradients are found (e.g. first step, no-grad context)
    """
    sr_values: dict[str, float] = {}

    for name, module in model.named_modules():
        if not isinstance(module, torch.nn.Linear):
            continue
        param = module.weight
        if param.grad is None:
            continue
        if param.grad.ndim != 2:
            continue

        rows, cols = param.grad.shape
        if rows > skip_large_dim or cols > skip_large_dim:
            continue

        # FSDP2 gradients may be DTensors; gather the full tensor for SVD
        G = param.grad.detach()
        if hasattr(G, "full_tensor"):
            G = G.full_tensor()

        G32 = G.float()
        try:
            sv = torch.linalg.svdvals(G32)   # descending
            frob_sq = (sv * sv).sum().item()
            sigma_max_sq = sv[0].item() ** 2
            if sigma_max_sq < 1e-30:
                sr = float("nan")
            else:
                sr = frob_sq / sigma_max_sq
        except Exception:
            sr = float("nan")

        min_dim = min(rows, cols)
        if tokens_per_update < min_dim:
            logging.warning(
                f"stable_rank: tokens_per_update ({tokens_per_update}) < min_dim ({min_dim}) "
                f"for {name}.weight - rank is token-count capped, SR may be misleadingly low"
            )

        sr_values[f"stable_rank/{name}.weight"] = sr

    if not sr_values:
        return {}

    valid = [v for v in sr_values.values() if not (v != v)]  # filter nan
    metrics = dict(sr_values)
    if valid:
        metrics["stable_rank/mean"] = sum(valid) / len(valid)
        metrics["stable_rank/min"] = min(valid)
        metrics["stable_rank/max"] = max(valid)
    return metrics


def _get_linear_flops(m: int, k: int, n: int, gradient_checkpointing: bool = False) -> int:
    forward_flops = 2 * m * k * n
    backward_flops = 2 * forward_flops

    total_flops = forward_flops + backward_flops
    if gradient_checkpointing:
        total_flops += forward_flops

    return total_flops


def _get_attention_flops(batch_size: int, sequence_length: int, hidden_size: int) -> int:
    attention_forward_flops = 2 * batch_size * sequence_length * (sequence_length + 1) * hidden_size
    attention_backward_flops = 5 * attention_forward_flops / 2
    return attention_forward_flops + attention_backward_flops


def get_model_tflops(
    config: AutoConfig | CommonConfig,
    batch_size: int,
    sequence_length: int,
    gradient_checkpointing_method: GradientCheckpointingMethod | None,
    gradient_checkpointing_args: dict,
) -> None:
    if not is_custom_model(config.model_type):
        return 0

    b = batch_size
    s = sequence_length
    h = config.hidden_size
    v = config.vocab_size

    num_layers_checkpointed = (
        gradient_checkpointing_args.get("num_blocks", config.num_layers)
        if gradient_checkpointing_method == GradientCheckpointingMethod.block
        else 0
    )

    total_flops = 0
    for layer_idx in range(config.num_layers):
        block = config.sequence_mixer_blocks[layer_idx]
        sequence_mixer_type = block.sequence_mixer_type
        gradient_checkpointing_enabled = layer_idx < num_layers_checkpointed

        if sequence_mixer_type == "causal_convolution":
            sequence_mixer_flops = _get_linear_flops(
                b * s, h, block.in_channels, gradient_checkpointing=gradient_checkpointing_enabled
            )
            sequence_mixer_flops += divide_if_divisible(
                _get_linear_flops(
                    b * s, block.in_channels, block.out_channels, gradient_checkpointing=gradient_checkpointing_enabled
                ),
                block.num_groups,
                "",
            )
            sequence_mixer_flops += _get_linear_flops(
                b * s, block.out_channels, h, gradient_checkpointing=gradient_checkpointing_enabled
            )
        elif sequence_mixer_type == "softmax_attention":
            # QKV projection FLOPs
            sequence_mixer_flops = _get_linear_flops(
                b * s,
                h,
                h * (1 + 2 * block.num_key_value_heads / block.num_attention_heads),
                gradient_checkpointing=gradient_checkpointing_enabled,
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(
                b * s, h, h, gradient_checkpointing=gradient_checkpointing_enabled
            )

            sequence_mixer_flops += _get_attention_flops(b, s, h)
        elif sequence_mixer_type == "multihead_latent_attention":
            # QKV down and up projection FLOPs
            sequence_mixer_flops = 2 * _get_linear_flops(
                b * s,
                h,
                block.query_compression_size + 2 * block.key_value_compression_size,
                gradient_checkpointing=gradient_checkpointing_enabled,
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(
                b * s, h, h, gradient_checkpointing=gradient_checkpointing_enabled
            )

            sequence_mixer_flops += _get_attention_flops(b, s, h)
        elif sequence_mixer_type == "mamba2":
            # NOTE taken from NexaAI's fork (might be incorrect)
            # Mamba2 FLOP calculation based on its specific architecture
            # Core components: projection, convolution, SSM operations
            # Input projection + convolution + SSM computation + output projection
            # TODO fix this for gradient checkpointing
            projection_flops = 4 * b * s * h * block.intermediate_size
            ssm_flops = 4 * b * s * block.intermediate_size * block.state_size

            sequence_mixer_flops = projection_flops + ssm_flops
            sequence_mixer_flops *= 2
        elif sequence_mixer_type == "rnn":
            num_heads = max(block.num_input_heads, block.num_weight_heads)
            # input projection FLOPs
            sequence_mixer_flops = _get_linear_flops(
                b * s, h, (block.num_input_heads + num_heads) * block.state_head_dim
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(b * s, block.state_head_dim * num_heads, h)

            # sigmoid(Wh + x)
            sequence_mixer_flops += (
                s
                * num_heads
                * (_get_linear_flops(b, block.state_head_dim, block.state_head_dim) + b * block.state_head_dim)
            )
        elif sequence_mixer_type == "gru":
            num_heads = max(
                block.num_input_heads,
                block.num_forget_input_heads,
                block.num_reset_input_heads,
                block.num_weight_heads,
                block.num_forget_weight_heads,
                block.num_reset_weight_heads,
            )

            # input projection FLOPs
            sequence_mixer_flops = _get_linear_flops(
                b * s,
                h,
                (block.num_input_heads + block.num_forget_input_heads + block.num_reset_input_heads + num_heads)
                * block.state_head_dim,
            )
            # output projection FLOPs
            sequence_mixer_flops += _get_linear_flops(b * s, block.state_head_dim * num_heads, h)

            # sigmoid(Wh + x)
            sequence_mixer_flops += (
                3
                * s
                * num_heads
                * (_get_linear_flops(b, block.state_head_dim, block.state_head_dim) + b * block.state_head_dim)
            )
        elif sequence_mixer_type == "gated_deltanet":
            return 0
        else:
            raise NotImplementedError(f"unexpected sequence_mixer_type ({sequence_mixer_type})")

        total_flops += sequence_mixer_flops

        block = config.mlp_blocks[layer_idx]

        # 2x for input and output linear layer
        mlp_flops = 2 * _get_linear_flops(
            b * s, h, block.intermediate_size, gradient_checkpointing=gradient_checkpointing_enabled
        )
        if block.mlp_type == "MoE":
            mlp_flops *= block.num_experts_per_tok

        if is_glu(block.activation_function):
            mlp_flops *= 1.5

        total_flops += mlp_flops

    total_flops += _get_linear_flops(b * s, h, v)
    total_flops /= 10**12

    return total_flops
