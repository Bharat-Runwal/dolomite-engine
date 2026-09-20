"""
Thin wrapper around lm-evaluation-harness for energy-GPT checkpoints.

Handles two cases:
  1. Base energy checkpoint (no proj_state.pt): loads normally.
  2. Structured-proj checkpoint (has proj_state.pt): after from_pretrained creates
     the base arch (which ignores L_attn/U_attn/V_attn keys), this wrapper swaps
     the energy blocks to DualLowRankPortHamiltonianProjection and reloads the
     trained structured weights from proj_state.pt.

Usage
-----
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install lm-eval -q
  python eval_harness.py \\
      --model hf \\
      --model_args "pretrained=<ckpt>,dtype=bfloat16" \\
      --tasks arc_challenge,arc_easy,boolq,copa,hellaswag,lambada_openai,\\
              openbookqa,piqa,race,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot \\
      --device cuda:0 \\
      --batch_size 4 \\
      --output_path <ckpt>/harness_results.json

All standard lm_eval CLI flags are forwarded as-is.
"""

import sys
import os
from pathlib import Path

# Must happen before any lm_eval import so the energy architecture is
# registered in HF's AutoModel registry when lm_eval calls from_pretrained.
import lm_engine.hf_models  # noqa: F401

# ── Patch from_pretrained to swap structured proj after loading ────────────────

from transformers import AutoModelForCausalLM as _AMCLM
_orig_from_pretrained = _AMCLM.from_pretrained.__func__


def _load_without_dispatch(cls, path, args, kwargs):
    """Reload without accelerate's device_map dispatch, then move the model by hand.

    WHY THIS EXISTS (2026-09-20). `surrogate_free_proxy: true` (the default on
    `EnergyFF_SurrogateBoltzmannMoE`) converts `proxy_V/B/V2/B2/bias/scale` into
    NON-PERSISTENT buffers, so they are deliberately absent from the state_dict -- nothing
    reads their values once `surrogate_replaces_proxy` overrides `_proxy_energies`. But when
    lm_eval passes a `device_map`, accelerate builds the model under `init_empty_weights`, so
    those buffers are created on META and no checkpoint entry ever fills them. The subsequent
    `dispatch_model -> model.to(device)` then dies with

        NotImplementedError: Cannot copy out of meta tensor; no data!

    killing BOTH halves of the eval (`ev_` exit 1; `evg_` reported DONE while writing nothing).
    Loading without a device_map skips `init_empty_weights`, the buffers get real storage in
    `__init__`, and 0 meta buffers remain -- verified on
    `cmix_134M_hyb_w1w2_sparse_surr_32B/unsharded_step122070`.

    Only arms that actually hit the error take this path, so no currently-working arm changes.
    Safe for our sizes: every arm here fits on one GPU (largest is 1.0B), which is all
    `device_map` was buying.
    """
    import torch
    dm = kwargs.pop("device_map", None)
    kwargs.pop("low_cpu_mem_usage", None)
    model = _orig_from_pretrained(cls, path, *args, **kwargs)
    dev = None
    if isinstance(dm, dict):
        dev = next((v for v in dm.values() if v is not None), None)
    elif isinstance(dm, str) and dm not in ("auto", "balanced", "balanced_low_0", "sequential"):
        dev = dm
    if dev is None:
        dev = "cuda" if torch.cuda.is_available() else "cpu"
    if isinstance(dev, int):
        dev = f"cuda:{dev}"
    remaining = [n for n, b in model.named_buffers() if b.is_meta]
    assert not remaining, (
        "buffers still on meta after a dispatch-free reload, so the device_map was not the "
        f"cause and moving the model would corrupt it: {remaining}")
    print(f"[eval_harness] reloaded without device_map and moved to {dev} "
          f"(meta-buffer workaround)", flush=True)
    return model.to(dev)


@classmethod  # type: ignore[misc]
def _patched_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs):
    try:
        model = _orig_from_pretrained(cls, pretrained_model_name_or_path, *args, **kwargs)
    except NotImplementedError as exc:
        if "meta tensor" not in str(exc):
            raise
        model = _load_without_dispatch(cls, pretrained_model_name_or_path, args, dict(kwargs))
    proj_state_path = Path(pretrained_model_name_or_path) / "proj_state.pt"
    if not proj_state_path.exists():
        return model

    # Determine rank from saved weight shapes: U_attn has shape (d, rank)
    import torch
    saved = torch.load(proj_state_path, map_location="cpu", weights_only=True)
    # Find rank and dissipation_rank from any layer
    rank = dissipation_rank = None
    for key, val in saved.items():
        if "U_attn" in key and val.ndim == 2:
            rank = val.shape[1]
        if "L_attn" in key and val.ndim == 2:
            dissipation_rank = val.shape[1]
        if rank is not None and dissipation_rank is not None:
            break
    if rank is None or dissipation_rank is None:
        print(f"[eval_harness] Could not infer rank from proj_state.pt — skipping swap", flush=True)
        return model

    # Check for special modes
    learnable_alpha = any("log_alpha_attn" in k for k in saved)
    single_stream = not any("U_ff" in k for k in saved)

    # Determine which block names have structured proj keys saved.
    # Keys look like "transformer.h.10.proj.U_attn".
    # For checkpoints that only trained a subset of blocks (e.g. h.10 only),
    # we must only swap those blocks — leaving the others at their original
    # unconstrained weights so eval is correct.
    swapped_block_names = set()
    for key in saved:
        if ".proj." in key:
            block_key = key.rsplit(".proj.", 1)[0]  # e.g. "transformer.h.10"
            swapped_block_names.add(block_key)

    # Infer num_iters per block from log_alpha_attn shape.
    per_block_num_iters = {}
    for key, val in saved.items():
        if "log_alpha_attn" in key and val.ndim == 1:
            block_key = key.rsplit(".proj.", 1)[0]
            per_block_num_iters[block_key] = val.shape[0]

    print(f"[eval_harness] Swapping to DualLowRankPH (rank={rank}, dr={dissipation_rank}, "
          f"learnable_alpha={learnable_alpha}, single_stream={single_stream})", flush=True)
    print(f"[eval_harness] Blocks to swap: {sorted(swapped_block_names)}", flush=True)
    if per_block_num_iters:
        print(f"[eval_harness] Per-block num_iters: {per_block_num_iters}", flush=True)

    # Import swap function from training script (supports per-block num_iters + block filter)
    sys.path.insert(0, str(Path(__file__).parent))
    from train_structured_proj_20260412 import swap_to_structured_proj  # noqa: E402
    swap_to_structured_proj(
        model, rank, dissipation_rank,
        init_from_weights=False,   # don't warmstart — we'll load saved weights
        learnable_alpha=learnable_alpha,
        single_stream=single_stream,
        per_block_num_iters=per_block_num_iters,
        only_block_names=swapped_block_names,
    )

    # Restore saved structured proj weights
    missing, unexpected = model.load_state_dict(saved, strict=False)
    print(f"[eval_harness] Loaded proj_state.pt  missing={len(missing)}  unexpected={len(unexpected)}", flush=True)
    return model


_AMCLM.from_pretrained = _patched_from_pretrained  # type: ignore[method-assign]

# ── Run lm_eval CLI ────────────────────────────────────────────────────────────

from lm_eval.__main__ import cli_evaluate  # noqa: E402

sys.exit(cli_evaluate())
