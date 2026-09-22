#!/usr/bin/env python3
"""Instantiate every bsaha200m config on a meta device and report total/active params.

Run from the repo root with PYTHONPATH set to this checkout:
  PYTHONPATH=$PWD .venv-nima/bin/python configs/bsaha200m/_verify_params.py

Active params are counted the way the engine does (base.py), which discounts a block's
experts ONLY if that block defines get_num_active_parameters(). EnergyFF_BoltzmannMoE does,
so its count is honest here -- unlike BoltzmannMoE_Energy_MLP on the bsaha fork.
"""
import glob
import os
import sys

import torch
import yaml

from lm_engine.arguments import TrainingArgs, args_dict_to_pydantic_args
from lm_engine.hf_models import EnergyConfig, EnergyForCausalLM


def count(path: str) -> tuple[int, int]:
    cfg = yaml.safe_load(open(path))
    args = args_dict_to_pydantic_args(TrainingArgs, **cfg)
    pc = dict(args.model_args.pretrained_config)
    pc.pop("model_type", None)
    with torch.device("meta"):
        model = EnergyForCausalLM(EnergyConfig(**pc))
    total = sum(p.numel() for p in model.parameters())
    active = [0]

    def walk(module):
        for child in module.children():
            if hasattr(child, "get_num_active_parameters"):
                active[0] += child.get_num_active_parameters()
            else:
                for p in child.parameters(recurse=False):
                    active[0] += p.numel()
                walk(child)

    walk(model)
    return total, active[0]


def check_tokenizer(path: str) -> str | None:
    """The trainer asserts tokenizer.bos_token_id == config.bos_token_id and dies at step 0.

    This cost a full 12-job submission cycle: the configs carried 0/0/0 (copied from a
    template) while granite-4.0-tiktoken uses 100257/100257/100256.
    """
    from transformers import AutoTokenizer

    cfg = yaml.safe_load(open(path))
    args = args_dict_to_pydantic_args(TrainingArgs, **cfg)
    pc = dict(args.model_args.pretrained_config)
    tok = AutoTokenizer.from_pretrained(args.tokenizer_args.tokenizer_name)
    for key, got in (("bos_token_id", tok.bos_token_id),
                     ("eos_token_id", tok.eos_token_id),
                     ("pad_token_id", tok.pad_token_id)):
        if pc.get(key) != got:
            return f"{key}: config={pc.get(key)} tokenizer={got}"
    if pc.get("vocab_size") < len(tok):
        return f"vocab_size {pc.get('vocab_size')} < tokenizer {len(tok)}"
    return None


def main() -> int:
    here = os.path.dirname(os.path.abspath(__file__))
    files = sorted(glob.glob(os.path.join(here, "bs200m_*.yml")))
    if not files:
        print("no configs found", file=sys.stderr)
        return 1
    print(f"{'config':30s} {'total':>12s} {'active':>12s} {'act/tot':>8s}")
    totals = []
    problems = []
    for f in files:
        total, active = count(f)
        totals.append(total)
        bad = check_tokenizer(f)
        flag = f"  <-- TOKENIZER MISMATCH: {bad}" if bad else ""
        if bad:
            problems.append(os.path.basename(f))
        print(f"{os.path.basename(f)[:-4]:30s} {total:>12,} {active:>12,} {100*active/total:>7.1f}%{flag}")
    spread = max(totals) - min(totals)
    print(f"\ntotal spread: {spread:,} ({100*spread/min(totals):.3f}%)")
    if problems:
        print(f"\nFAIL: {len(problems)} config(s) would die at step 0 on the tokenizer assert")
        return 1
    print("tokenizer ids: all OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
