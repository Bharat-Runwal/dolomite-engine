"""One-shot converter: load a saved model with surrogate_router attributes,
swap BoltzmannMoE layers for SurrogateBoltzmannMoE with use_surrogate=True,
and save back.  Run this on the already-trained b5_with_surrogate_router checkpoint.

Usage:
    python convert_to_surrogate.py \
        --checkpoint /path/to/b5_with_surrogate_router \
        --output     /path/to/b5_surrogate_active
"""
import argparse
import json
import shutil
from pathlib import Path
import torch
import lm_engine.hf_models  # noqa: registers energy model
from transformers import AutoModelForCausalLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading {args.checkpoint}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.checkpoint, torch_dtype=torch.bfloat16,
        trust_remote_code=True
    ).to(device)

    from lm_engine.hf_models.modeling_utils.mlp_blocks.mlp import (
        BoltzmannMoE_Energy_MLP, SurrogateBoltzmannMoE_Energy_MLP
    )
    from safetensors import safe_open

    # Load surrogate_router weights directly from safetensors (they were saved
    # but from_pretrained() drops them as "unexpected" since BoltzmannMoE.__init__
    # doesn't create the surrogate_router attribute)
    surr_weights = {}
    st_path = Path(args.checkpoint) / "model.safetensors"
    if st_path.exists():
        with safe_open(str(st_path), framework="pt") as sf:
            for k in sf.keys():
                if "surrogate_router" in k:
                    surr_weights[k] = sf.get_tensor(k).to(device)
    print(f"Loaded {len(surr_weights)} surrogate_router weight tensors from safetensors.")

    n_converted = 0
    for parent_name, parent_module in list(model.named_modules()):
        for child_name, child_module in list(parent_module.named_children()):
            if not isinstance(child_module, BoltzmannMoE_Energy_MLP):
                continue
            m = child_module
            # Find the surrogate weight for this layer
            weight_key = f"{parent_name}.{child_name}.surrogate_router.weight".lstrip(".")
            if weight_key not in surr_weights:
                print(f"  WARNING: no saved weight for {weight_key} — skipping")
                continue
            surrogate = SurrogateBoltzmannMoE_Energy_MLP(
                hidden_size=m.hidden_size,
                intermediate_size=m.intermediate_size,
                n_experts=m.n_experts,
                temperature=m.temperature,
                repulsion_coef=m.repulsion_coef,
                n_repulsion_pairs=m.n_repulsion_pairs,
                surrogate_coef=0.0,
                use_surrogate=True,
                activation_function="gelu_pytorch_tanh",
                add_bias=False, dropout=0.0,
                init_method="normal", initializer_range=0.02,
                m_width=m.hidden_size, num_layers=1,
            ).to(device).to(torch.bfloat16)
            surrogate.W1.weight.data.copy_(m.W1.weight.data)
            surrogate.W2.weight.data.copy_(m.W2.weight.data)
            surrogate.surrogate_router.weight.data.copy_(surr_weights[weight_key])
            setattr(parent_module, child_name, surrogate)
            n_converted += 1

    print(f"Converted {n_converted} BoltzmannMoE layers → SurrogateBoltzmannMoE (use_surrogate=True)")

    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)

    # Patch config.json directly: change mlp_type and add surrogate fields.
    # Can't use Pydantic (it rejects unknown fields); JSON edit is cleaner.
    cfg_path = out / "config.json"
    cfg = json.load(open(cfg_path))
    for blk in cfg.get("mlp_blocks", []):
        if blk.get("mlp_type") == "BoltzmannMoE_Energy_MLP":
            blk["mlp_type"] = "SurrogateBoltzmannMoE_Energy_MLP"
            blk["surrogate_coef"] = 0.0
            blk["use_surrogate"] = True
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    print(f"  config.json patched: mlp_type → SurrogateBoltzmannMoE_Energy_MLP")

    # Copy tokenizer
    src = Path(args.checkpoint)
    for f in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"):
        if (src / f).exists():
            shutil.copy2(src / f, out / f)
    print(f"Saved to {out}")

if __name__ == "__main__":
    main()
