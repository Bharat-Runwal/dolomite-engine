#!/usr/bin/env python3
"""Download WikiText-2 and convert to Megatron binary format for local testing.

Usage:
    python scripts/local/prepare_wikitext.py [--tokenizer gpt2] [--output-dir data/wikitext2]

This script avoids importing the full lm_engine package (which may have
optional dependencies). Instead it directly uses the indexed_dataset builder.
"""

import argparse
import json
import os
import sys
from unittest.mock import MagicMock

# Add repo root to path so we can import the minimal megatron modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../.."))

# Stub optional transformers classes to avoid import errors
for _name in ("granitemoehybrid", "granitemoeshared"):
    sys.modules.setdefault(
        f"lm_engine.hf_models.model_conversion.{_name}", MagicMock()
    )

import torch
from transformers import AutoTokenizer
from datasets import load_dataset

from lm_engine.data.megatron.indexed_dataset import (
    DType,
    MMapIndexedDatasetBuilder,
    get_bin_path,
    get_idx_path,
)


def main():
    parser = argparse.ArgumentParser(description="Prepare WikiText-2 for Megatron training")
    parser.add_argument("--tokenizer", default="gpt2", help="HuggingFace tokenizer name")
    parser.add_argument("--output-dir", default="data/wikitext2", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    output_prefix = os.path.join(args.output_dir, "wikitext2")

    print(f"Loading tokenizer: {args.tokenizer}")
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    print("Downloading WikiText-2 from HuggingFace...")
    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    # Build Megatron binary dataset directly
    key = "text"
    bin_path = get_bin_path(f"{output_prefix}_{key}")
    idx_path = get_idx_path(f"{output_prefix}_{key}")
    builder = MMapIndexedDatasetBuilder(bin_path, dtype=DType.optimal_dtype(tokenizer.vocab_size))

    n_docs = 0
    n_tokens = 0
    for item in ds:
        text = item["text"].strip()
        if not text:
            continue
        ids = tokenizer.encode(text)
        if not ids:
            continue
        ids.append(tokenizer.eos_token_id)
        builder.add_item(torch.IntTensor(ids))
        builder.end_document()
        n_docs += 1
        n_tokens += len(ids)

    builder.finalize(idx_path)

    print(f"Done! {n_docs} documents, {n_tokens} tokens")
    print(f"Output: {bin_path}, {idx_path}")
    print(f"\nUse in config data_path:")
    print(f"  - {output_prefix}_text")


if __name__ == "__main__":
    main()
