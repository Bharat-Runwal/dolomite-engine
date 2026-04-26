"""V44: Layer merger for EGPT NxM → (N/2)x(2M) — weight sharing, not depth reduction.

Loads an EGPT checkpoint, averages adjacent pairs (0+1), (2+3), ...,
but sets layer_iterations=[2]*out_n_layers so the merged block runs TWICE per
forward pass. Total compute unchanged; only the number of unique weight sets halves.

This is the correct interpretation of layer merging: we share weights between
adjacent layers rather than removing depth.

Strategy:
  naive       - simple arithmetic mean of weight tensors
  procrustes  - align layer b to layer a via Procrustes rotation before averaging

Usage:
    python merge_layers_v44.py \
        --ckpt_path <path_to_EGPT_result_dir> \
        --out_path  <path_for_merged_HF_model> \
        [--strategy naive|procrustes] \
        [--out_n_layers 6]
"""
import argparse
from pathlib import Path
import copy
import json
import os
import sys

import torch

REPO = str(Path(__file__).resolve().parents[4])
sys.path.insert(0, REPO)

import lm_engine.hf_models  # noqa: F401 — triggers register_model_classes()


def _get_latest_step_dir(base_dir: str) -> str:
    latest_json = os.path.join(base_dir, "latest_checkpointed_iteration.json")
    if os.path.isfile(latest_json):
        with open(latest_json) as f:
            step = json.load(f)["latest_checkpointed_iteration"]
        return os.path.join(base_dir, f"global_step{step}")
    dirs = [d for d in os.listdir(base_dir) if d.startswith("global_step")]
    if not dirs:
        raise FileNotFoundError(f"No global_step dirs in {base_dir}")
    dirs.sort(key=lambda d: int(d.replace("global_step", "")))
    return os.path.join(base_dir, dirs[-1])


def naive_merge(sd_a: dict, sd_b: dict) -> dict:
    out = {}
    for k in sd_a:
        out[k] = (sd_a[k].float() + sd_b[k].float()) / 2.0
    return out


def procrustes_merge(sd_a: dict, sd_b: dict) -> dict:
    """Align sd_b to sd_a via per-weight Procrustes rotation, then average."""
    out = {}
    for k in sd_a:
        wa = sd_a[k].float()
        wb = sd_b[k].float()
        if wa.dim() == 2 and wa.shape == wb.shape:
            M = wa.T @ wb
            try:
                U, _, Vh = torch.linalg.svd(M, full_matrices=False)
                R = Vh.T @ U.T
                wb_aligned = wb @ R
            except Exception:
                wb_aligned = wb
            out[k] = (wa + wb_aligned) / 2.0
        else:
            out[k] = (wa + wb) / 2.0
    return out


def merge_model(model, strategy: str = "naive", out_n_layers: int = 6):
    """Return a new model with pairwise-merged layers and doubled layer_iterations.

    Key difference from V42: layer_iterations sums the original pair's iterations
    so total compute (forward passes) is preserved. E.g. 12x1 → 6x2.
    """
    orig_transformer = model.transformer
    n = len(orig_transformer.h)
    assert n == 2 * out_n_layers, f"Expected {2 * out_n_layers} layers, got {n}"

    orig_iters = list(model.config.layer_iterations)
    assert len(orig_iters) == n, f"layer_iterations length {len(orig_iters)} != num_layers {n}"

    merge_fn = naive_merge if strategy == "naive" else procrustes_merge

    new_config = copy.deepcopy(model.config)
    new_config.num_layers = out_n_layers
    # Sum iterations for each merged pair → preserves total compute
    new_config.layer_iterations = [orig_iters[2*i] + orig_iters[2*i+1] for i in range(out_n_layers)]
    new_config.sequence_mixer_blocks = new_config.sequence_mixer_blocks[:out_n_layers]
    new_config.mlp_blocks = new_config.mlp_blocks[:out_n_layers]

    print(f"  orig layer_iterations: {orig_iters}")
    print(f"  new  layer_iterations: {new_config.layer_iterations}  (total={sum(new_config.layer_iterations)})")

    from transformers import AutoModelForCausalLM
    new_model = AutoModelForCausalLM.from_config(new_config)
    new_model = new_model.to(dtype=torch.bfloat16)

    new_transformer = new_model.transformer
    new_transformer.wte.weight.data.copy_(orig_transformer.wte.weight.data)
    if hasattr(new_transformer, 'ln_f') and hasattr(orig_transformer, 'ln_f'):
        new_transformer.ln_f.weight.data.copy_(orig_transformer.ln_f.weight.data)

    orig_blocks = orig_transformer.h
    new_blocks  = new_transformer.h

    for i in range(out_n_layers):
        a_idx = 2 * i
        b_idx = 2 * i + 1
        sd_a = dict(orig_blocks[a_idx].state_dict())
        sd_b = dict(orig_blocks[b_idx].state_dict())
        merged_sd = merge_fn(sd_a, sd_b)
        merged_sd = {k: v.to(torch.bfloat16) for k, v in merged_sd.items()}
        new_blocks[i].load_state_dict(merged_sd, strict=True)
        print(f"  Merged layers {a_idx}+{b_idx} (iters {orig_iters[a_idx]}+{orig_iters[b_idx]}) → new layer {i} (iters {new_config.layer_iterations[i]})")

    return new_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True)
    parser.add_argument("--out_path", required=True)
    parser.add_argument("--strategy", default="naive", choices=["naive", "procrustes"])
    parser.add_argument("--out_n_layers", type=int, default=6)
    args = parser.parse_args()

    top_unsharded = os.path.join(args.ckpt_path, "unsharded")
    if os.path.isdir(top_unsharded):
        load_dir = top_unsharded
    else:
        ckpt_dir = _get_latest_step_dir(args.ckpt_path)
        step_unsharded = os.path.join(ckpt_dir, "unsharded")
        load_dir = step_unsharded if os.path.isdir(step_unsharded) else ckpt_dir

    from transformers import AutoModelForCausalLM
    print(f"Loading model from {load_dir}")
    model = AutoModelForCausalLM.from_pretrained(load_dir, torch_dtype=torch.bfloat16, device_map="cpu")
    print(f"Loaded {model.config.num_layers}-layer EGPT (layer_iterations={model.config.layer_iterations})")
    print(f"Merging → {args.out_n_layers} unique layers using strategy='{args.strategy}'...")

    merged = merge_model(model, strategy=args.strategy, out_n_layers=args.out_n_layers)
    os.makedirs(args.out_path, exist_ok=True)
    merged.save_pretrained(args.out_path)
    merged.config.save_pretrained(args.out_path)
    print(f"Saved merged model to {args.out_path}")

    total = sum(p.numel() for p in merged.parameters())
    print(f"Merged model params: {total/1e6:.1f}M  "
          f"({merged.config.num_layers} unique layers × {merged.config.layer_iterations[0]} iters)")


if __name__ == "__main__":
    main()
