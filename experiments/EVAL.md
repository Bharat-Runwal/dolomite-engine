# Standardized eval pipeline

This page documents the canonical evaluation scripts used to benchmark every
trained checkpoint in this fork — both `experiments/energy-inference/` (EGPT
variants) and `experiments/boltzmann-moe/` (Boltzmann-MoE variants). Use it as
a reference if you've forked dolomite-engine and want to evaluate your own
checkpoints with the same task list and protocol.

## TL;DR — already trained a model? Run this

```bash
# 1. Unshard the FSDP checkpoint to HF format (if not already done)
python -m lm_engine.unshard --config <(printf "load_args:\n  load_path: %s\nunsharded_path: %s/unsharded\nmixed_precision_args:\n  dtype: bf16\n" "$SAVE_PATH" "$SAVE_PATH")

# 2. Submit the LM-harness eval (13 tasks, ~1-2 hours, 1 GPU)
bash experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
     "$SAVE_PATH/unsharded" "eval_${RUN_NAME}"

# 3. Optional: Big-Bench Hard (27 tasks, 3-shot, ~3 hours)
bash experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh \
     "$SAVE_PATH/unsharded" "eval_bbh_${RUN_NAME}"
```

Results write to:
- `$SAVE_PATH/unsharded/harness_results.json` — main 13-task suite
- `$SAVE_PATH/unsharded/harness_bbh_results_<timestamp>.json` — BBH

All `run_*.sh` launchers in this repo automatically chain into both scripts at
end-of-training (after the final `lm_engine.unshard` step).

---

## Files

| Purpose | Path |
|---|---|
| Submit main eval (bsub wrapper) | `experiments/energy-inference/scripts/structured-proj/submit_eval.sh` |
| Eval harness (Python; registers `lm_engine.hf_models`, runs `lm-eval`) | `experiments/energy-inference/scripts/structured-proj/eval_harness.py` |
| Submit BBH eval (bsub wrapper) | `experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh` |

`eval_harness.py` is a thin wrapper around `lm-evaluation-harness` that:
1. Imports `lm_engine.hf_models` before `lm_eval` so HF's `AutoModelForCausalLM`
   knows the energy/boltzmann architectures.
2. (Conditional) If `proj_state.pt` is present in the checkpoint, swaps the
   energy projection blocks to `DualLowRankPortHamiltonianProjection` and
   reloads the trained structured weights — used by `structured-proj`
   experiments only; harmless for plain energy / boltzmann-moe checkpoints.
3. Forwards all remaining CLI flags to `lm-eval` directly.

## Standard task list

```
arc_challenge, arc_easy, boolq, copa, hellaswag,
openbookqa, piqa, sciq, wikitext, winogrande,
mmlu, gsm8k, gsm8k_cot
```

13 tasks total. The first 10 are commonly aggregated as **avg10** (mean of
primary `acc`). When `acc_norm` is reported (arc_challenge/easy, hellaswag,
openbookqa, piqa), the **avg10_norm** variant uses `acc_norm` for those and
`acc` for the rest (boolq, copa, sciq, winogrande, mmlu).

**Excluded from the standard list** (cluster-specific):
- `lambada_openai` — Arrow parquet "Repetition level histogram size mismatch"
  on this cluster; works elsewhere.
- `race` — same Arrow parquet corruption (EleutherAI/race high, test split 1045).
- `bbh_fewshot` — dataset not pre-cached; run separately via
  `submit_bbh_eval.sh` if you've staged the SaylorTwift/bbh dataset under
  `$HF_HOME`.

## Submission defaults

`submit_eval.sh` requests:
- 1 GPU (any), `preemptable` queue, `grp_preemptable` group
- 48 GB memory, 2-hour walltime
- bsub logs → `$HOME/bsub_logs/${JOB_NAME}_%J.{stdout,stderr}`

`submit_bbh_eval.sh` requests:
- 1 GPU, `normal` queue, `grp_ebm` group (BBH is heavier/longer)
- 48 GB memory, 3-hour walltime

If your group/queue is different, edit the `bsub` flags in those scripts
(or copy them into your experiment dir and change locally).

## Environment expectations

The bsub heredoc activates `nanoGPT-og`'s `.venv`:
```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
uv pip install accelerate lm-eval -q     # idempotent
```

That venv has torch + safetensors + the dolomite editable install (without
flash-attn — energy code falls back to `F.scaled_dot_product_attention`). If
you fork to a different machine, point `submit_eval.sh` at your venv +
`PYTHONPATH` to your dolomite checkout.

`HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1` are set so `lm-eval` reads
cached datasets and doesn't try to phone Hugging Face Hub. The standard 13
tasks should be cached by the first-ever eval run on this cluster; if you
hit dataset-not-found errors, run once with the env vars unset to populate
the cache.

## Output format — `harness_results.json`

`lm-eval` writes the standard JSON; the relevant fields are:

```json
{
  "results": {
    "arc_challenge": {"acc": 0.245, "acc_stderr": ..., "acc_norm": 0.270, ...},
    "arc_easy":      {"acc": 0.470, "acc_norm": 0.488, ...},
    "boolq":         {"acc": 0.581, ...},
    "wikitext":      {"word_perplexity": 41.6, "byte_perplexity": ..., ...},
    "gsm8k":         {"exact_match,strict-match": 0.011, "exact_match,flexible-extract": 0.021, ...},
    "mmlu":          {"acc": 0.249, ...},
    ...
  },
  "config": {...}
}
```

To compute aggregates from this JSON:

```python
import json
r = json.load(open("harness_results.json"))["results"]
ten = ["arc_challenge","arc_easy","boolq","copa","hellaswag",
       "openbookqa","piqa","sciq","winogrande","mmlu"]
avg10      = sum(r[t]["acc"] for t in ten) / 10
avg10_norm = sum(r[t].get("acc_norm", r[t]["acc"]) for t in ten) / 10
ppl        = r["wikitext"]["word_perplexity"]
gsm8k_flex = r["gsm8k_cot"]["exact_match,flexible-extract"]
print(f"avg10={avg10:.4f}  avg10_norm={avg10_norm:.4f}  WT-PPL={ppl:.2f}  GSM8K-flex={gsm8k_flex:.4f}")
```

## Where eval results live

The launchers chain unshard + eval at the end of training, so each completed
run carries its own results next to the unsharded checkpoint:

- `experiments/energy-inference/results/multi-block-ablation/<run>/unsharded/harness_results.json`
- `experiments/boltzmann-moe/results/<run>/unsharded/harness_results_<timestamp>.json`

Mid-training intermediate evals are typically named
`harness_results_step<N>.json` or saved next to `unsharded_step<N>/`. See e.g.
`experiments/boltzmann-moe/results/scale_gptmoe_12moe_K4I2048_d1280/unsharded_step26000/`
for a pattern.

## Re-running eval on an existing unsharded checkpoint

If you just want to re-eval (e.g. after adding a task or fixing a metric)
without retraining or re-unsharding:

```bash
bash experiments/energy-inference/scripts/structured-proj/submit_eval.sh \
     /path/to/unsharded eval_rerun
```

The script overwrites `harness_results.json` in place. To save iterations to
distinct files, change `--output_path` in the heredoc, or copy the existing
results JSON before re-running.

## Common gotchas

1. **Unsharded path required.** `lm-eval` expects HF format; the FSDP-sharded
   `global_step<N>/` directories aren't directly loadable. Always unshard first.
2. **Energy / Boltzmann arch registration.** `eval_harness.py` imports
   `lm_engine.hf_models` *before* `lm-eval`. If you use vanilla `lm-eval` CLI
   directly, your custom architectures will fail HF's auto-registry lookup.
3. **`trust_remote_code=True`** is set both in `--model_args` and as a top-level
   flag, because the lm-eval version on this cluster requires both.
4. **bsub log path typo in BBH script** — `submit_bbh_eval.sh` writes to
   `$HOME/bsub_logs/bsub_logs/...` (double-nested). Either pre-create that dir
   or fix the script.
5. **GPU memory.** 48 GB is enough for ≤500M-param models at `batch_size=4`.
   For larger checkpoints lower `batch_size` (slower) or up the `-M` flag.
