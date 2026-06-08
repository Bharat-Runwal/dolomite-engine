"""Dump raw per-(block, iteration) energy stats as JSON for all model folders.

For each unsharded model, runs 30 wikitext samples and saves:
  {folder}/energy_raw.json

with keys like "block0_iter0", "block0_iter1", ..., each containing mean, std, n.
"""

import sys
import os
import json
import torch
import numpy as np
from collections import defaultdict

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

from lm_engine.hf_models.register_hf import register_model_classes
register_model_classes()

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


def compute_energies(model, input_ids):
    base_model = model.transformer
    with torch.no_grad():
        hidden_states = base_model.wte(input_ids)
        if base_model.m_emb is not None:
            hidden_states = hidden_states * base_model.m_emb

        rope_cos_sin = None
        if base_model.position_embedding_type == "rope":
            position_ids = torch.arange(input_ids.shape[1], device=input_ids.device).unsqueeze(0)
            rope_cos_sin = base_model._get_rope_cos_sin(
                key_length=input_ids.shape[1], position_ids=position_ids, dtype=hidden_states.dtype)

        energies = []
        for i, num_iter in enumerate(base_model.layer_iterations):
            block = base_model.h[i]
            has_energy = hasattr(block, 'energy_per_token') and block.sequence_mixer_type == "energy_attention"
            for j in range(num_iter):
                hidden_states = block(
                    hidden_states, past_key_values=None, attention_mask=None,
                    rope_cos_sin=rope_cos_sin, cu_seqlens=None, max_seqlen=None, layer_id=None)
                if has_energy:
                    e = block.energy_per_token(hidden_states, rope_cos_sin=rope_cos_sin)
                    energies.append((i, j, e.float().mean().item()))
    return energies


def process_model(model_path, output_dir, tokenizer, texts, device):
    json_path = os.path.join(output_dir, "energy_raw.json")
    if os.path.exists(json_path):
        print(f"  SKIP (exists): {json_path}")
        return

    print(f"  Loading: {model_path}")
    model = AutoModelForCausalLM.from_pretrained(model_path, dtype=torch.bfloat16)
    model.config.use_cache = False
    model.eval()
    model.to(device)

    all_energies = defaultdict(list)

    for idx, text in enumerate(texts):
        input_ids = tokenizer.encode(text, return_tensors='pt', max_length=512, truncation=True).to(device)
        if input_ids.shape[1] < 10:
            continue
        energies = compute_energies(model, input_ids)
        for block_idx, iter_idx, mean_e in energies:
            all_energies[(block_idx, iter_idx)].append(mean_e)
        if (idx + 1) % 10 == 0:
            print(f"    Processed {idx+1}/{len(texts)}")

    result = {}
    for (b, it), vals in sorted(all_energies.items()):
        key = f"block{b}_iter{it}"
        result[key] = {
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "n": len(vals),
        }

    os.makedirs(output_dir, exist_ok=True)
    with open(json_path, 'w') as f:
        json.dump(result, f, indent=2)
    print(f"  Saved: {json_path}")

    del model
    torch.cuda.empty_cache()


def main():
    BASE_UNSHARDED = "/proj/dmfexp/energy-gpt/checkpoints-bsaha/unsharded/egpt_test"
    LANDSCAPE_BASE = "tools/bs/energy_landscape"

    FAMILIES = {
        "EGPT_4b_9_9_9_9": ["30k", "40k", "50k", "70k"],
        "EGPT_5b_9_9_9_9_9": ["30k", "40k", "50k", "60k", "70k", "80k", "90k"],
        "EGPT_6b_9_9_9_9_9_9": ["30k", "40k", "50k", "60k", "70k", "80k", "90k"],
    }

    # 8-block models: each variant has only 30k step, grouped under "8_blocks" folder
    EIGHT_BLOCK_VARIANTS = [
        "EGPT_8_blocks_1_11_12_12_12_12_11_1",
        "EGPT_8_blocks_15_3_9_9_9_9_3_15",
        "EGPT_8_blocks_1_9_9_9_9_9_9_1",
        "EGPT_8_blocks_2_11_11_12_12_11_11_2",
        "EGPT_8_blocks_2_9_9_9_9_9_9_2",
        "EGPT_8_blocks_3_11_11_11_11_11_11_3",
        "EGPT_8_blocks_3_15_3_15_3_15_3_15",
        "EGPT_8_blocks_3_15_9_9_9_9_3_15",
        "EGPT_8_blocks_3_9_9_9_9_9_9_3",
        "EGPT_8_blocks_4_10_11_11_11_11_10_4",
        "EGPT_8_blocks_4_9_9_9_9_9_9_4",
        "EGPT_8_blocks_5_9_9_9_9_9_9_5",
        "EGPT_8_blocks_6_9_9_9_9_9_9_6",
        "EGPT_8_blocks_9_9_9_9_9_9_9_9",
    ]

    tokenizer = AutoTokenizer.from_pretrained('/proj/dmfexp/energy-gpt/data/granite-4.0-tiktoken')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
    texts = [t for t in dataset["text"] if len(t.strip()) > 100][:30]
    print(f"Using {len(texts)} samples\n")

    for family, steps in FAMILIES.items():
        print(f"\n{'='*60}")
        print(f"Family: {family} ({len(steps)} steps)")
        print(f"{'='*60}")

        for step in steps:
            model_name = f"{family}_{step}"
            model_path = os.path.join(BASE_UNSHARDED, model_name)
            output_dir = os.path.join(LANDSCAPE_BASE, family, model_name)

            if not os.path.exists(model_path):
                print(f"  SKIP (no model): {model_path}")
                continue

            process_model(model_path, output_dir, tokenizer, texts, device)

    # Process folder-based variant groups (model name = subfolder name)
    VARIANT_FOLDERS = ["3_blocks", "4_blocks", "4_blocks_dropout", "5_blocks", "5_blocks_dropout", "6_blocks", "8_blocks"]

    for folder in VARIANT_FOLDERS:
        folder_path = os.path.join(LANDSCAPE_BASE, folder)
        if not os.path.isdir(folder_path):
            print(f"\n  SKIP (no folder): {folder_path}")
            continue

        variants = sorted([d for d in os.listdir(folder_path)
                          if os.path.isdir(os.path.join(folder_path, d)) and d.startswith("EGPT")])

        print(f"\n{'='*60}")
        print(f"Folder: {folder} ({len(variants)} variants)")
        print(f"{'='*60}")

        for model_name in variants:
            model_path = os.path.join(BASE_UNSHARDED, model_name)
            output_dir = os.path.join(folder_path, model_name)

            if not os.path.exists(model_path):
                print(f"  SKIP (no model): {model_path}")
                continue

            process_model(model_path, output_dir, tokenizer, texts, device)

    print("\nAll done!")


if __name__ == '__main__':
    main()
