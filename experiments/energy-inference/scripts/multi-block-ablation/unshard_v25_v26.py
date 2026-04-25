"""Unshard V25/V26 checkpoints with W_O → W_O_gpt key remapping.

V25/V26 were trained before the W_O→W_O_gpt rename (commit 3191a28).
Their FSDP shards have `sequence_mixer.W_O.weight`; current model expects `W_O_gpt`.
This script unshards via FSDP consolidation, remaps keys, then saves as HF safetensors.
"""

import argparse
import os
import sys

REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
sys.path.insert(0, REPO)

import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.checkpoint.format_utils import dcp_to_torch_save

from lm_engine.arguments import UnshardingArgs
from lm_engine.checkpointing._base import (
    _get_model_optimizer_path,
    _get_base_path,
    consolidate_sharded_model_checkpoints,
)
from lm_engine.hf_models import get_model_class


def remap_state_dict(state_dict: dict) -> dict:
    remapped = {}
    n = 0
    for k, v in state_dict.items():
        new_k = k.replace(".sequence_mixer.W_O.", ".sequence_mixer.W_O_gpt.")
        if new_k != k:
            n += 1
        remapped[new_k] = v
    if n:
        print(f"  Remapped {n} W_O → W_O_gpt keys")
    return remapped


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--load_path", required=True, help="Sharded checkpoint directory")
    parser.add_argument("--unsharded_path", required=True, help="Output unsharded directory")
    args = parser.parse_args()

    load_path = args.load_path
    unsharded_path = args.unsharded_path

    import json, glob

    # Find latest iteration
    latest_json = os.path.join(load_path, "latest_checkpointed_iteration.json")
    with open(latest_json) as f:
        iteration = json.load(f)["latest_checkpointed_iteration"]

    print(f"Loading iteration {iteration} from {load_path}")

    model_path = _get_model_optimizer_path(_get_base_path(load_path, iteration))
    print(f"Model shards at: {model_path}")

    # Consolidate FSDP shards
    state, _ = consolidate_sharded_model_checkpoints(
        f"{model_path}/", "*.pt", "", save_model=False
    )
    state = state["state"]

    # Remap keys
    state = remap_state_dict(state)

    # Load config to build model
    import yaml
    config_path = os.path.join(load_path, "global_step{}".format(iteration), "training_config.yml")
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    from transformers import AutoConfig
    from lm_engine.hf_models import CommonConfig, is_custom_model

    model_config_path = os.path.join(load_path, "global_step{}".format(iteration), "model")
    model_config = AutoConfig.from_pretrained(model_config_path)

    from lm_engine.hf_models import get_model_class
    model_cls = get_model_class(model_config, "pretraining")
    model = model_cls(model_config)

    # Cast and load
    for key in list(state.keys()):
        state[key] = state[key].to(torch.bfloat16)

    model.load_state_dict(state)
    print("State dict loaded successfully.")

    os.makedirs(unsharded_path, exist_ok=True)
    model.save_pretrained(unsharded_path)
    print(f"Saved to {unsharded_path}")


if __name__ == "__main__":
    main()
