# **************************************************
# Copyright (c) 2025, Mayank Mishra
# **************************************************

import glob
import json
import logging
import math
import os
import random
import shutil

import numpy as np
import torch
import torch.distributed.checkpoint as dcp
import yaml
from torch.distributed.checkpoint import FileSystemReader
from torch.distributed.checkpoint.format_utils import _EmptyStateDictLoadPlanner
from torch.distributed.checkpoint.state_dict_loader import _load_state_dict

from ..arguments import DistillationArgs, TrainingArgs, UnshardingArgs, args_dict_to_pydantic_args
from ..containers import LRSchedulerContainer, ModelContainer, OptimizerContainer
from ..data import ResumableDataLoader
from ..hf_models import fix_unsharded_state_dict
from ..model_wrapper import ModelWrapper, get_model_container
from ..utils import (
    Accelerator,
    Communication,
    ExperimentsTracker,
    ProcessGroupManager,
    is_torch_xla_available,
    load_yaml,
    log_rank_0,
    run_rank_n,
    string_to_torch_dtype,
)
from .lr_scheduler import _get_lr_scheduler_path, _LRSchedulerSaver, _resume_learning_rate
from .model import _get_model_path, _ModelSaver
from .model_optimizer import _get_model_optimizer_path, _ModelOptimizerSaver
from .optimizer import _get_optimizer_path, _OptimizerSaver


if is_torch_xla_available():
    from torch_xla.core.xla_model import save as xla_save
    from torch_xla.distributed.fsdp import (
        consolidate_sharded_model_checkpoints as xla_consolidate_sharded_model_checkpoints,
    )


_TRAINING_CONFIG_PREFIX = "training_config"
_KILLSWITCH = "KILLSWITCH"
_FUTURE = None


def ensure_last_checkpoint_is_saved() -> None:
    if _FUTURE is not None:
        _FUTURE.result()


def save_checkpoint(
    args: TrainingArgs,
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer | None,
    lr_scheduler_container: LRSchedulerContainer | None,
    train_dataloader: ResumableDataLoader,
    experiments_tracker: ExperimentsTracker,
    iteration: int,
    metadata: dict = {},
) -> None:
    """save checkpoint during training

    Args:
        args (TrainingArgs): arguments for training
        model_container (ModelContainer): models to save
        optimizer_container (OptimizerContainer): optimizers to save
        lr_scheduler_container (LRSchedulerContainer): learning rate schedulers to save
        train_dataloader (DataLoader): train dataloader to save
        experiments_tracker (ExperimentsTracker): experiment tracker to save
        iteration (int): current iteration
        metadata (dict): extra stuff to store

    Raises:
        ValueError: if unexpected distributed backend is found
    """

    save_path = _get_base_path(args.save_args.save_path, iteration)
    os.makedirs(save_path, exist_ok=True)

    ensure_last_checkpoint_is_saved()

    if lr_scheduler_container is None:
        log_rank_0(
            logging.WARN,
            "lr_scheduler_container is not passed to save_checkpoint. Therefore, the function will not save the lr_scheduler",
        )
    else:
        lr_scheduler_path = _get_lr_scheduler_path(save_path)
        os.makedirs(os.path.dirname(lr_scheduler_path), exist_ok=True)
        torch.save(_LRSchedulerSaver(lr_scheduler_container).state_dict(), _get_lr_scheduler_path(save_path))

    rng_state = {
        "random_rng_state": random.getstate(),
        "np_rng_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "accelerator_rng_state": Accelerator.get_rng_state(),
    }
    rng_state_path = _get_rng_state_path(save_path)
    os.makedirs(os.path.dirname(rng_state_path), exist_ok=True)
    torch.save(rng_state, rng_state_path)

    if train_dataloader is not None:
        dataloader_path = _get_dataloader_path(save_path)
        os.makedirs(os.path.dirname(dataloader_path), exist_ok=True)
        torch.save(train_dataloader.state_dict(), dataloader_path)

    if experiments_tracker is not None:
        run_rank_n(json.dump)(
            experiments_tracker.state_dict(), run_rank_n(open)(_get_experiments_tracker_path(save_path), "w"), indent=4
        )

    metadata["accelerator"] = Accelerator.get_accelerator().value
    run_rank_n(json.dump)(metadata, run_rank_n(open)(_get_metadata_path(save_path), "w"), indent=4)

    save_args(args, save_path)

    if args.save_args.save_optimizer:
        if optimizer_container is None:
            log_rank_0(
                logging.WARN,
                "optimizer_container is not passed to save_checkpoint but save_optimizer is set to True. "
                "Therefore, the function will not save the optimizer",
            )

    if args.save_args.async_checkpointing:
        global _FUTURE
        _FUTURE = dcp.async_save(
            {
                "state": _ModelOptimizerSaver(
                    model_container=model_container,
                    optimizer_container=optimizer_container if args.save_args.save_optimizer else None,
                )
            },
            checkpoint_id=_get_model_optimizer_path(save_path),
        )

        def _f(_future):
            run_rank_n(json.dump)(
                {"latest_checkpointed_iteration": iteration},
                run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
                indent=4,
            )

            log_rank_0(logging.INFO, f"checkpoint saved asynchronously at {iteration}")

            if os.path.exists(os.path.join(args.save_args.save_path, _KILLSWITCH)):
                exit()

        _FUTURE.add_done_callback(_f)
    else:
        accelerator = Accelerator.get_accelerator()

        if accelerator == Accelerator.cuda:
            dcp.save({"state": _ModelSaver(model_container)}, checkpoint_id=_get_model_path(save_path))

            if args.save_args.save_optimizer and optimizer_container is not None:
                dcp.save(
                    {"state": _OptimizerSaver(model_container, optimizer_container)},
                    checkpoint_id=_get_optimizer_path(save_path),
                )
        elif accelerator == Accelerator.tpu:
            assert len(model_container) == 1
            assert len(optimizer_container) == 1

            os.makedirs(_get_model_optimizer_path(save_path), exist_ok=True)

            xla_save(
                {
                    "model": model_container[0].state_dict(),
                    "optimizer": optimizer_container[0].state_dict(),
                    "shard_metadata": model_container[0].get_shard_metadata(),
                },
                os.path.join(_get_model_optimizer_path(save_path), f"{ProcessGroupManager.get_global_rank()}.pt"),
                master_only=False,
            )

        Communication.barrier()

        run_rank_n(json.dump)(
            {"latest_checkpointed_iteration": iteration},
            run_rank_n(open)(_get_latest_checkpointed_iterations_path(args.save_args.save_path), "w"),
            indent=4,
        )

        log_rank_0(logging.INFO, f"checkpoint saved at {iteration}")

        # Preserve token-milestone checkpoints BEFORE pruning can remove them.
        run_rank_n(_preserve_token_milestones)(args, iteration)

        # Prune old checkpoints if max_to_keep is set
        max_to_keep = args.save_args.max_to_keep
        if max_to_keep is not None and max_to_keep > 0:
            base = args.save_args.save_path
            existing = sorted(
                [d for d in os.listdir(base) if d.startswith("global_step")],
                key=lambda d: int(d.split("global_step")[1]),
            )
            to_delete = existing[:-max_to_keep]
            for old_dir in to_delete:
                old_path = os.path.join(base, old_dir)
                run_rank_n(shutil.rmtree)(old_path)
                log_rank_0(logging.INFO, f"pruned old checkpoint: {old_dir}")

        if os.path.exists(os.path.join(args.save_args.save_path, _KILLSWITCH)):
            ProcessGroupManager.destroy_process_groups()
            exit()


def _tokens_per_step(args: TrainingArgs) -> int | None:
    """Exact tokens/step: ``WORLD_SIZE x micro_batch_size x gradient_accumulation_steps x seq``.

    GPUS IS NOT IN THE CONFIG -- it comes from the launcher -- which is why every previous attempt
    to recover a run's token budget after the fact was guesswork. ``WORLD_SIZE`` is set by torchrun,
    so at save time the figure is exact. Mirrors the computation `pretrain.py` logs to wandb.
    """
    seq = args.datasets[0].class_args.get("sequence_length") if args.datasets else None
    if not seq:
        return None
    world = int(os.environ.get("WORLD_SIZE", "1"))
    tp = args.training_parameters
    return world * tp.micro_batch_size * tp.gradient_accumulation_steps * int(seq)


def _preserve_token_milestones(args: TrainingArgs, iteration: int) -> None:
    """Hard-link this checkpoint into ``milestones/`` if it is the first one past a token milestone.

    Called BEFORE pruning, in the same save call that wrote ``global_step{iteration}``, so the
    checkpoint is guaranteed to still exist and no tolerance heuristic is needed: this checkpoint is
    the first at or after milestone ``m`` exactly when
    ``iteration >= want > iteration - save_interval``.
    """
    milestones = args.save_args.token_milestones_b
    if not milestones:
        return
    tps = _tokens_per_step(args)
    if not tps:
        return
    base = args.save_args.save_path
    src = os.path.join(base, f"global_step{iteration}")
    if not os.path.isdir(src):
        return
    mdir = os.path.join(base, "milestones")
    interval = max(1, int(args.save_args.save_interval or 1))

    for m in milestones:
        # FLOOR, not ceil. 32e9/262144 = 122070.3125 while the 134M arms END at step 122070, so
        # ceil() makes the final milestone unreachable and it is silently never captured -- the
        # exact bug the old cron script had to fix in 2026-09-18. Floor costs at most a fraction
        # of one step of tokens.
        want = int(float(m) * 1e9 / tps)
        # only the FIRST checkpoint at or after the crossing, and never re-link one already taken
        if not (iteration >= want > iteration - interval):
            continue
        if glob.glob(os.path.join(mdir, f"tok{float(m):g}B_*")):
            continue
        actual = iteration * tps
        dst = os.path.join(mdir, f"tok{float(m):g}B_step{iteration}_actual{actual / 1e9:.2f}B")
        try:
            os.makedirs(mdir, exist_ok=True)
            # hard links: no extra bytes until pruning would have removed the original
            shutil.copytree(src, dst, copy_function=os.link, dirs_exist_ok=False)
            log_rank_0(
                logging.INFO,
                f"token milestone {float(m):g}B preserved: step {iteration} "
                f"({actual / 1e9:.2f}B actual) hard-linked to {os.path.basename(dst)}",
            )
        except Exception as e:  # never let bookkeeping kill a training run
            log_rank_0(logging.WARN, f"could not preserve {float(m):g}B milestone at {iteration}: {e}")


def load_checkpoint_for_training(
    args: TrainingArgs,
    args_class: type[TrainingArgs | DistillationArgs],
    model_container: ModelContainer,
    optimizer_container: OptimizerContainer,
    lr_scheduler_container: LRSchedulerContainer,
    train_dataloader: ResumableDataLoader,
) -> tuple[int, dict, dict]:
    """load checkpoint for training

    Args:
        args (TrainingArgs): arguments for training
        args_class (type[TrainingArgs | DistillationArgs]): class for arguments
        model_container (ModelContainer): models to save
        optimizer_container (OptimizerContainer): optimizers to save
        lr_scheduler_container (LRSchedulerContainer): learning rate schedulers to save
        train_dataloader (ResumableDataLoader): train dataloader to load

    Raises:
        ValueError: if unexpected distributed backend is found

    Returns:
        tuple[int, dict, dict]: checkpointed iteration, metadata, experiments_tracker state dict
    """

    if args.load_args is None or args.load_args.load_path is None:
        return

    load_optimizer = args.load_args.load_optimizer
    load_rng_state = args.load_args.load_rng_state
    load_dataloader_state = args.load_args.load_dataloader_state
    load_experiments_tracker_state = args.load_args.load_experiments_tracker_state
    load_starting_iteration = args.load_args.load_starting_iteration

    iteration = args.load_args.iteration
    if iteration is None:
        iter_path = _get_latest_checkpointed_iterations_path(args.load_args.load_path)
        if not os.path.exists(iter_path):
            # First fresh-start training run: load_args.load_path is set (so the
            # config can resume cleanly on later restarts) but no checkpoint
            # exists yet. Treat as "nothing to load" instead of crashing.
            log_rank_0(
                logging.INFO,
                f"No checkpoint at {args.load_args.load_path} (file missing: {iter_path}); "
                f"starting fresh from iteration 0",
            )
            return 0, {}, None
        iteration = json.load(open(iter_path, "r"))[
            "latest_checkpointed_iteration"
        ]

    load_path = _get_base_path(args.load_args.load_path, iteration)

    args_file = os.path.join(load_path, f"{_TRAINING_CONFIG_PREFIX}.yml")
    args_from_checkpoint = load_yaml(args_file)
    args_from_checkpoint = args_dict_to_pydantic_args(args_class, **args_from_checkpoint)

    log_rank_0(logging.INFO, f"loading checkpoint saved at {load_path}")

    accelerator = Accelerator.get_accelerator()

    if accelerator == Accelerator.cuda:
        if args_from_checkpoint.save_args.async_checkpointing:
            saver = _ModelOptimizerSaver(model_container, optimizer_container if load_optimizer else None)
            state_dict = {"state": saver.state_dict()}
            dcp.load(state_dict, checkpoint_id=_get_model_optimizer_path(load_path))
            saver.load_state_dict(state_dict["state"])
        else:
            saver = _ModelSaver(model_container)
            state_dict = {"state": saver.state_dict()}
            dcp.load(state_dict, checkpoint_id=_get_model_path(load_path))
            saver.load_state_dict(state_dict["state"])

            if load_optimizer:
                saver = _OptimizerSaver(model_container, optimizer_container)
                state_dict = {"state": saver.state_dict()}
                # dcp.load(state_dict, checkpoint_id=_get_optimizer_path(load_path))
                dcp.load(
                state_dict,
                checkpoint_id=_get_optimizer_path(load_path),
                planner=dcp.DefaultLoadPlanner(allow_partial_load=True),
                    )
                saver.load_state_dict(state_dict["state"])

        del saver, state_dict
    elif accelerator == Accelerator.tpu:
        assert len(model_container) == 1
        assert len(optimizer_container) == 1

        state_dict = torch.load(
            os.path.join(_get_model_optimizer_path(load_path), f"{ProcessGroupManager.get_global_rank()}.pt")
        )

        model_container[0].load_state_dict(state_dict["model"])
        if load_optimizer:
            optimizer_container[0].load_state_dict(state_dict["optimizer"])

    if args.load_args.load_lr_scheduler:
        assert load_optimizer, "load_lr_scheduler requires loading of optimizer"

        _LRSchedulerSaver(lr_scheduler_container).load_state_dict(
            torch.load(_get_lr_scheduler_path(load_path), weights_only=False)
        )
    elif args.load_args.resume_learning_rate:
        for model, optimizer, lr_scheduler in zip(model_container, optimizer_container, lr_scheduler_container):
            _resume_learning_rate(
                args,
                model=model,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                iteration=iteration if load_starting_iteration else None,
            )

    if load_rng_state:
        rng_state = torch.load(_get_rng_state_path(load_path), weights_only=False)
        random.setstate(rng_state["random_rng_state"])
        np.random.set_state(rng_state["np_rng_state"])
        torch.set_rng_state(rng_state["torch_rng_state"])
        Accelerator.set_rng_state(rng_state["accelerator_rng_state"])

    metadata = json.load(open(_get_metadata_path(load_path), "r"))

    if load_dataloader_state and train_dataloader is not None:
        train_dataloader.load_state_dict(torch.load(_get_dataloader_path(load_path), weights_only=False))

    experiments_tracker_json = None
    if load_experiments_tracker_state and os.path.exists(_get_experiments_tracker_path(load_path)):
        experiments_tracker_json = json.load(open(_get_experiments_tracker_path(load_path), "r"))

    if not load_starting_iteration:
        iteration = 0

    return iteration, metadata, experiments_tracker_json


def load_checkpoint_and_unshard(args: UnshardingArgs) -> tuple[ModelWrapper, TrainingArgs, dict]:
    """load checkpoint for inference

    Args:
        args (UnshardingArgs): arguments
    """

    load_path = args.load_args.load_path

    iteration = args.load_args.iteration
    if iteration is None:
        iteration = json.load(open(_get_latest_checkpointed_iterations_path(args.load_args.load_path), "r"))[
            "latest_checkpointed_iteration"
        ]

    log_rank_0(logging.INFO, f"loading checkpoint saved at {_get_base_path(load_path, iteration)}")

    args_file = os.path.join(_get_base_path(load_path, iteration), f"{_TRAINING_CONFIG_PREFIX}.yml")
    args_from_checkpoint = load_yaml(args_file)
    metadata = json.load(open(_get_metadata_path(_get_base_path(load_path, iteration)), "r"))

    accelerator = Accelerator(metadata["accelerator"])

    # turn off distillation for unsharding
    if "teacher_args" in args_from_checkpoint:
        args_from_checkpoint["tuning_args"]["tuning_method"] = "pretraining"
        args_from_checkpoint.pop("teacher_args")

    args_from_checkpoint = args_dict_to_pydantic_args(TrainingArgs, **args_from_checkpoint)

    if args.mixed_precision_args is not None:
        log_rank_0(logging.INFO, "overriding mixed precision args")
        args_from_checkpoint.mixed_precision_args = args.mixed_precision_args

    checkpoint_tp_world_size = args_from_checkpoint.distributed_args.tensor_parallel_world_size
    use_meta = args_from_checkpoint.model_args.model_name is None

    with (
        torch.device("meta") if use_meta else torch.device(Accelerator.get_current_device()),
        ProcessGroupManager.set_dummy_tensor_parallel_rank(0),
        ProcessGroupManager.set_dummy_tensor_parallel_world_size(1),
        ProcessGroupManager.set_dummy_pipeline_parallel_rank(0),
        ProcessGroupManager.set_dummy_pipeline_parallel_world_size(1),
        args_from_checkpoint.distributed_args.temporary_argument_value("num_pipeline_stages", 1),
    ):
        model = get_model_container(args_from_checkpoint, efficient_initialization=False, keep_in_fp32=False)[0]

    if use_meta:
        model = model.to_empty(device="cpu")

    if accelerator == Accelerator.cuda:
        state = {}
        if args_from_checkpoint.save_args.async_checkpointing:
            _load_state_dict(
                state,
                storage_reader=FileSystemReader(_get_model_optimizer_path(_get_base_path(load_path, iteration))),
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )

            state = state["state"]["model"]
        else:
            _load_state_dict(
                state,
                storage_reader=FileSystemReader(_get_model_path(_get_base_path(load_path, iteration))),
                planner=_EmptyStateDictLoadPlanner(),
                no_dist=True,
            )

            state = state["state"]

        if checkpoint_tp_world_size > 1:
            state = fix_unsharded_state_dict(
                model.config, state, tensor_parallel_world_size=checkpoint_tp_world_size, prefix="model."
            )

        dtype = string_to_torch_dtype(model.dtype)
        for key in list(state.keys()):
            state[key] = state[key].to(dtype)
    elif accelerator == Accelerator.tpu:
        state, _ = xla_consolidate_sharded_model_checkpoints(
            f"{_get_model_optimizer_path(_get_base_path(load_path, iteration))}/", "*.pt", "", save_model=False
        )

    # backward-compat: rename W_O → W_O_gpt for EGrad checkpoints trained before commit 3191a28
    # Only applies when checkpoint has W_O and model (per current code) expects W_O_gpt.
    model_keys = set(model.state_dict().keys())
    if (any(".sequence_mixer.W_O." in k for k in state) and
            not any(".sequence_mixer.W_O." in k for k in model_keys) and
            any(".sequence_mixer.W_O_gpt." in k for k in model_keys)):
        state = {k.replace(".sequence_mixer.W_O.", ".sequence_mixer.W_O_gpt."): v for k, v in state.items()}

    # backward-compat: drop load_balance_bias when the model does not expect it.
    # BoltzmannMoEFFEnergy registers that buffer only when balance_rate > 0. Between
    # 2026-09-13 and 2026-09-14 it was registered UNCONDITIONALLY and persistently, so
    # checkpoints written in that window carry the key even for arms with balance_rate = 0,
    # and unsharding them now fails with
    #     Unexpected key(s) in state_dict: "...ffwd.moe.load_balance_bias"
    # It is a zero tensor for those arms and carries no information, so dropping it is
    # lossless. Affected: hop_K16_top2_renorm, slope90k_hyb, big_hop_sandwich. Arms that DO
    # set balance_rate keep the key, because the model then expects it.
    stray = [k for k in state if k.endswith("load_balance_bias") and k not in model_keys]
    if stray:
        log_rank_0(logging.WARN, f"dropping {len(stray)} load_balance_bias key(s) absent from the model")
        state = {k: v for k, v in state.items() if k not in set(stray)}

    model.load_state_dict(state)

    return model, args_from_checkpoint, state


@run_rank_n
def save_args(args: TrainingArgs, save_path: str) -> None:
    """saves training args as a json

    Args:
        args (TrainingArgs): arguments for training or inference
        save_path (str): save location on disk
    """

    save_path = os.path.join(save_path, f"{_TRAINING_CONFIG_PREFIX}.yml")
    yaml.dump(args.to_dict(), open(save_path, "w"), indent=2)


def _get_checkpoint_tag(iteration: int) -> str:
    return f"global_step{iteration}"


def _get_base_path(path: str, iteration: int) -> str:
    return os.path.join(path, _get_checkpoint_tag(iteration))


def _get_dataloader_path(path: str) -> str:
    return os.path.join(path, "dataloader", f"dataloader-{ProcessGroupManager.get_data_parallel_rank()}.pt")


def _get_rng_state_path(path: str) -> str:
    return os.path.join(path, "rng_state", f"rng_state-{ProcessGroupManager.get_global_rank()}.pt")


def _get_latest_checkpointed_iterations_path(path: str) -> str:
    return os.path.join(path, "latest_checkpointed_iteration.json")


def _get_experiments_tracker_path(path: str) -> str:
    return os.path.join(path, "experiments_tracker.json")


def _get_metadata_path(path: str) -> str:
    return os.path.join(path, "metadata.json")
