"""V42: Naive layer merger for EGPT 12-layer → 6-layer.

Loads the V1 EGPT 160M checkpoint (12 energy blocks), averages adjacent pairs
(0+1), (2+3), ..., (10+11) to produce a 6-block model, then saves it as a
standalone HF model directory suitable for fine-tuning with dolomite-engine.

Usage:
    python merge_layers_v42.py \
        --ckpt_path <path_to_v1_checkpoint_dir> \
        --out_path  <path_for_merged_model> \
        [--strategy naive|procrustes]
"""
import argparse
import copy
import json
import os
import sys

import torch

REPO = "/proj/dmfexp/nima/Code/dolomite-engine"
sys.path.insert(0, REPO)


def _load_sharded_model(ckpt_dir: str):
    """Load FSDP-sharded dolomite-engine checkpoint into a single-device model."""
    from transformers import AutoModelForCausalLM
    # dolomite-engine keeps a config.json + model_*.safetensors in each global_step dir
    # We use unsharded if available
    unsharded = os.path.join(ckpt_dir, "unsharded")
    if os.path.isdir(unsharded):
        load_dir = unsharded
    else:
        load_dir = ckpt_dir
    print(f"Loading model from {load_dir}")
    model = AutoModelForCausalLM.from_pretrained(
        load_dir,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
    )
    return model


def _get_latest_step_dir(base_dir: str) -> str:
    latest_json = os.path.join(base_dir, "latest_checkpointed_iteration.json")
    if os.path.isfile(latest_json):
        with open(latest_json) as f:
            step = json.load(f)["latest_checkpointed_iteration"]
        return os.path.join(base_dir, f"global_step{step}")
    # fall back to highest global_step
    dirs = [d for d in os.listdir(base_dir) if d.startswith("global_step")]
    if not dirs:
        raise FileNotFoundError(f"No global_step dirs in {base_dir}")
    dirs.sort(key=lambda d: int(d.replace("global_step", "")))
    return os.path.join(base_dir, dirs[-1])


def naive_merge(sd_a: dict, sd_b: dict) -> dict:
    """Average corresponding tensors from two state-dict slices."""
    out = {}
    for k in sd_a:
        out[k] = (sd_a[k].float() + sd_b[k].float()) / 2.0
    return out


def procrustes_merge(sd_a: dict, sd_b: dict) -> dict:
    """Align sd_b to sd_a via per-weight Procrustes rotation, then average.

    For each weight W in sd_b, find orthogonal R = argmin||W_a - W_b R||_F
    via SVD: U S V^T = W_a^T W_b  →  R = V U^T.
    Then merged = (W_a + W_b R) / 2.
    Only applied to 2D weight matrices; biases and 1D params are averaged directly.
    """
    out = {}
    for k in sd_a:
        wa = sd_a[k].float()
        wb = sd_b[k].float()
        if wa.dim() == 2 and wa.shape == wb.shape:
            # Procrustes: align wb → wa
            M = wa.T @ wb  # (d_out, d_out)
            try:
                U, _, Vh = torch.linalg.svd(M, full_matrices=False)
                R = Vh.T @ U.T  # orthogonal
                wb_aligned = wb @ R
            except Exception:
                wb_aligned = wb  # fall back to naive on SVD failure
            out[k] = (wa + wb_aligned) / 2.0
        else:
            out[k] = (wa + wb) / 2.0
    return out


def merge_model(model, strategy: str = "naive", out_n_layers: int = 6):
    """Return a new model whose layers are pairwise merges of the original."""
    import copy
    n = len(model.model.transformer.h)
    assert n == 2 * out_n_layers, f"Expected {2 * out_n_layers} layers, got {n}"

    merge_fn = naive_merge if strategy == "naive" else procrustes_merge

    # Build new config
    new_config = copy.deepcopy(model.config)
    new_config.num_layers = out_n_layers
    new_config.layer_iterations = [1] * out_n_layers
    new_config.sequence_mixer_blocks = new_config.sequence_mixer_blocks[:out_n_layers]
    new_config.mlp_blocks = new_config.mlp_blocks[:out_n_layers]

    # Build new model (random init, then overwrite with merged weights)
    from transformers import AutoModelForCausalLM
    new_model = AutoModelForCausalLM.from_config(new_config)
    new_model = new_model.to(dtype=torch.bfloat16)

    # Copy embedding and lm_head from original
    new_model.model.transformer.wte.weight.data.copy_(
        model.model.transformer.wte.weight.data)
    if hasattr(new_model.model.transformer, 'ln_f'):
        new_model.model.transformer.ln_f.weight.data.copy_(
            model.model.transformer.ln_f.weight.data)

    orig_blocks = model.model.transformer.h
    new_blocks  = new_model.model.transformer.h

    for i in range(out_n_layers):
        a_idx = 2 * i
        b_idx = 2 * i + 1
        sd_a = {k: v for k, v in orig_blocks[a_idx].state_dict().items()}
        sd_b = {k: v for k, v in orig_blocks[b_idx].state_dict().items()}
        merged_sd = merge_fn(sd_a, sd_b)
        # Cast back to bf16 for storage
        merged_sd = {k: v.to(torch.bfloat16) for k, v in merged_sd.items()}
        new_blocks[i].load_state_dict(merged_sd, strict=True)
        print(f"  Merged layers {a_idx} + {b_idx} → new layer {i}")

    return new_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt_path", required=True,
                        help="Path to V1 EGPT result dir (contains global_step*/unsharded)")
    parser.add_argument("--out_path", required=True,
                        help="Output dir for merged HF model")
    parser.add_argument("--strategy", default="naive", choices=["naive", "procrustes"])
    parser.add_argument("--out_n_layers", type=int, default=6)
    args = parser.parse_args()

    ckpt_dir = _get_latest_step_dir(args.ckpt_path)
    model = _load_sharded_model(ckpt_dir)
    print(f"Loaded {model.config.num_layers}-layer EGPT. Merging → {args.out_n_layers} layers "
          f"using strategy='{args.strategy}'...")

    merged = merge_model(model, strategy=args.strategy, out_n_layers=args.out_n_layers)
    os.makedirs(args.out_path, exist_ok=True)
    merged.save_pretrained(args.out_path)
    merged.config.save_pretrained(args.out_path)
    print(f"Saved merged model to {args.out_path}")

    # Quick sanity: count params
    total = sum(p.numel() for p in merged.parameters())
    print(f"Merged model params: {total/1e6:.1f}M")


if __name__ == "__main__":
    main()
