#!/usr/bin/env python3
"""Measure actual FLOPs for any model using torch.utils.flop_counter.

Reports FLOPs per forward pass and per token for given sequence length.
Works with any model architecture (handles hybrid blocks, halting configs, etc.)

Usage:
    python measure_flops.py /path/to/model1 /path/to/model2 ...
    python measure_flops.py /path/to/model --seq_len 4096
"""

import sys
import os
import json
import argparse
import torch
from torch.utils.flop_counter import FlopCounterMode

sys.path.insert(0, '.')
sys.path.insert(0, './accelerated-model-architectures')

from lm_engine.hf_models.register_hf import register_model_classes
register_model_classes()

from transformers import AutoModelForCausalLM


def measure_flops(model_path, seq_len=4096, batch_size=1):
    """Measure FLOPs using torch.utils.flop_counter."""
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map="cuda")
    model.eval()
    model.config.use_cache = False

    input_ids = torch.randint(0, 1000, (batch_size, seq_len), device="cuda")

    # Warmup
    with torch.no_grad():
        model(input_ids)

    # Measure with FlopCounterMode
    flop_counter = FlopCounterMode(display=False)
    with torch.no_grad():
        with flop_counter:
            model(input_ids)

    total_flops = flop_counter.get_total_flops()

    # Get model info
    config_path = os.path.join(model_path, 'config.json')
    with open(config_path) as f:
        cfg = json.load(f)

    layer_iters = cfg.get('layer_iterations', [1] * cfg.get('num_layers', 12))
    total_iters = sum(layer_iters)
    params = sum(p.numel() for p in model.parameters())

    del model
    torch.cuda.empty_cache()

    return {
        'name': os.path.basename(model_path.rstrip('/')),
        'path': model_path,
        'params': params,
        'layer_iterations': layer_iters,
        'total_iters': total_iters,
        'seq_len': seq_len,
        'total_flops': total_flops,
        'flops_per_token': total_flops / seq_len,
        'tflops': total_flops / 1e12,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('models', nargs='+')
    parser.add_argument('--seq_len', type=int, default=4096)
    args = parser.parse_args()

    results = []
    for path in args.models:
        if not os.path.exists(os.path.join(path, 'config.json')):
            print(f"Skipping {path} (no config.json)")
            continue
        name = os.path.basename(path.rstrip('/'))
        print(f"Measuring {name}...", flush=True)
        try:
            r = measure_flops(path, seq_len=args.seq_len)
            results.append(r)
            print(f"  {r['tflops']:.3f} TFLOPs, {r['flops_per_token']/1e9:.2f} GFLOPs/token, iters={r['total_iters']}")
        except Exception as e:
            print(f"  Error: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        return

    ref = min(results, key=lambda r: r['total_flops'])

    print(f"\n{'='*95}")
    print(f"FLOPs Comparison (seq_len={args.seq_len}, measured with torch FlopCounterMode)")
    print(f"{'='*95}")
    print(f"{'Model':<45} {'Params':>8} {'Iters':>6} {'TFLOPs':>9} {'GF/tok':>9} {'vs ref':>7}")
    print('-' * 95)
    for r in sorted(results, key=lambda x: x['total_flops']):
        ratio = r['total_flops'] / ref['total_flops']
        print(f"{r['name']:<45} {r['params']/1e6:7.1f}M {r['total_iters']:>6} "
              f"{r['tflops']:9.3f} {r['flops_per_token']/1e9:8.2f}G {ratio:6.2f}x")

    # Print halting savings if we have baseline + halting variants
    baselines = {r['name']: r for r in results if 'halting' not in r['path'] and 'tts' not in r['path']}
    halting = [r for r in results if 'halting' in r['path'] or 'tts' in r['path'] or 'converge' in r['path']]
    if halting:
        print(f"\n{'='*95}")
        print("Halting FLOPs Savings")
        print(f"{'='*95}")
        print(f"{'Variant':<45} {'Iters':>6} {'TFLOPs':>9} {'Saved':>7}")
        print('-' * 70)
        for r in halting:
            # Try to find parent baseline
            parent = None
            for bname, br in baselines.items():
                if bname in r['path']:
                    parent = br
                    break
            if parent:
                saved = (1 - r['total_flops'] / parent['total_flops']) * 100
                print(f"{r['name']:<45} {r['total_iters']:>6} {r['tflops']:9.3f} {saved:6.1f}%")
            else:
                print(f"{r['name']:<45} {r['total_iters']:>6} {r['tflops']:9.3f}")

    print(f"\nReference: {ref['name']} ({ref['tflops']:.3f} TFLOPs)")


if __name__ == '__main__':
    main()
