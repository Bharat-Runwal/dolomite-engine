# Standardized eval pipeline

Canonical evaluation scripts for every trained checkpoint in this fork — both
`experiments/energy-inference/` (EGPT variants) and `experiments/boltzmann-moe/`
(Boltzmann-MoE variants). Use as a reference if you've forked dolomite-engine
and want to evaluate your own checkpoints with the same task list and protocol.

## Files in this directory

| File | Purpose |
|---|---|
| `submit_eval.sh` | Submits the 13-task LM-harness bsub. Optionally chains BBH (default `--with-bbh`). |
| `submit_bbh_eval.sh` | Standalone BBH (3-shot, 27 subtasks). Called by `submit_eval.sh` if `--with-bbh`. |
| `eval_harness.py` | Python wrapper around `lm-evaluation-harness`; registers `lm_engine.hf_models` so HF auto-recognizes EGPT / Boltzmann arches. |
| `compute_aggregates.py` | Reads `harness_results.json` and prints `avg10`, `avg10_norm`, WikiText-PPL, GSM8K, per-task acc/acc_norm. |
| `bench_checkpoint.sh` | One-shot wrapper: unshards a `global_step<N>/` checkpoint then chains `submit_eval.sh`. |

## TL;DR

```bash
# Trained a checkpoint? One-line benchmark (unshard + LM + BBH):
bash experiments/eval_scripts/bench_checkpoint.sh \
     /path/to/save_path  34000  my_run_name
# → unsharded_step34000/, harness_results.json, harness_bbh_results_<ts>.json

# Already unsharded? Just eval (LM + BBH):
bash experiments/eval_scripts/submit_eval.sh \
     /path/to/save_path/unsharded  eval_my_run

# LM-only (no BBH):
bash experiments/eval_scripts/submit_eval.sh \
     /path/to/unsharded  eval_my_run  --no-bbh

# Aggregate after results land:
python experiments/eval_scripts/compute_aggregates.py \
     /path/to/unsharded/harness_results.json
```

All `run_*.sh` training launchers in this repo automatically chain into
`submit_eval.sh` at end-of-training (after the final `lm_engine.unshard` step).

## Standard task list (13 tasks)

```
arc_challenge, arc_easy, boolq, copa, hellaswag, openbookqa,
piqa, sciq, wikitext, winogrande, mmlu, gsm8k, gsm8k_cot
```

The first 10 are aggregated as **avg10** (mean of primary `acc`).
**avg10_norm** uses `acc_norm` for tasks that report it
(arc_challenge, arc_easy, hellaswag, openbookqa, piqa) and `acc` otherwise
(boolq, copa, sciq, winogrande, mmlu).

`compute_aggregates.py` prints both, plus WikiText word-PPL, GSM8K (strict),
GSM8K-CoT (flexible-extract), and a per-task table.

**Excluded by default** (cluster-specific, not protocol-defining):
- `lambada_openai`, `race` — Arrow parquet "Repetition level histogram size mismatch" on this cluster
- `bbh_fewshot` — separate path; runs via `submit_bbh_eval.sh`

## BBH (Big-Bench Hard)

`bbh_fewshot` runs 27 subtasks at 3-shot, exact-match. Heavier than the main
suite (~3-4 hours, 1 GPU). Above-random performance is expected at ≥400M
params with math+web pretraining; sub-random ⇒ check tokenizer / model
registration.

```bash
bash experiments/eval_scripts/submit_bbh_eval.sh /path/to/unsharded eval_bbh_<run>
```

Output: `<unsharded>/harness_bbh_results_<UTC-timestamp>.json` (timestamped so
multiple BBH runs on the same checkpoint don't clobber).

**Dataset caching:** lm-eval-harness expects `SaylorTwift/bbh` cached under
`$HF_HOME`. If not cached, run a one-shot online fetch first (drop the
`HF_HUB_OFFLINE=1` line in `submit_bbh_eval.sh` for one job).

## Submission defaults

| Job | Queue | Group | GPU | Walltime | Mem |
|---|---|---|---:|---:|---:|
| LM (`submit_eval.sh`) | `preemptable` | `grp_preemptable` | 1 | 02:00 | 48 GB |
| BBH (`submit_bbh_eval.sh`) | `preemptable` | `grp_preemptable` | 1 | 04:00 | 48 GB |
| Unshard+chain (`bench_checkpoint.sh`) | `preemptable` | `grp_preemptable` | 1 | 01:00 | 32 GB |

bsub logs land in `$HOME/bsub_logs/${JOB_NAME}_%J.{stdout,stderr}`. Adjust the
`bsub` flags in each script if your group/queue is different.

## Environment expectations

The bsub heredocs activate `nanoGPT-og`'s `.venv`:
```bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
uv pip install accelerate lm-eval -q     # idempotent
```

That venv has torch + safetensors + dolomite editable install (without
flash-attn — energy code falls back to `F.scaled_dot_product_attention`). If
you fork to a different machine, point `submit_eval.sh` at your venv +
`PYTHONPATH` to your dolomite checkout.

`HF_DATASETS_OFFLINE=1` and `HF_HUB_OFFLINE=1` are set so `lm-eval` reads
cached datasets and doesn't try to phone Hugging Face Hub.

## Output format — `harness_results.json`

`lm-eval` writes the standard JSON; the relevant fields:

```json
{
  "results": {
    "arc_challenge": {"acc": 0.245, "acc_stderr": ..., "acc_norm": 0.270, ...},
    "arc_easy":      {"acc": 0.470, "acc_norm": 0.488, ...},
    "boolq":         {"acc": 0.581, ...},
    "wikitext":      {"word_perplexity": 41.6, ...},
    "gsm8k":         {"exact_match,strict-match": 0.011, ...},
    "gsm8k_cot":     {"exact_match,flexible-extract": 0.021, ...},
    "mmlu":          {"acc": 0.249, ...},
    ...
  },
  "config": {...}
}
```

`compute_aggregates.py` parses this and emits headline numbers. Use `--json`
for machine-readable output.

## Where eval results live

The launchers chain unshard + eval at end-of-training, so each completed run
carries its own results next to the unsharded checkpoint:

- `experiments/energy-inference/results/multi-block-ablation/<run>/unsharded/harness_results.json`
- `experiments/boltzmann-moe/results/<run>/unsharded/harness_results_<ts>.json`

Mid-training intermediate evals are typically saved under
`unsharded_step<N>/harness_results.json` (use `bench_checkpoint.sh <save> <N> <name>`).

## Re-running eval on an existing unsharded checkpoint

```bash
bash experiments/eval_scripts/submit_eval.sh /path/to/unsharded eval_rerun
```

Overwrites `harness_results.json` in place. To save iterations to distinct
files, copy the existing results JSON before re-running, or change
`--output_path` in the heredoc.

## Common gotchas

1. **Unsharded path required.** `lm-eval` expects HF format; the FSDP-sharded
   `global_step<N>/` directories aren't directly loadable. Always unshard first
   (or use `bench_checkpoint.sh`).
2. **Energy / Boltzmann arch registration.** `eval_harness.py` imports
   `lm_engine.hf_models` *before* `lm-eval`. If you use vanilla `lm-eval` CLI
   directly, your custom architectures will fail HF's auto-registry lookup.
3. **`trust_remote_code=True`** is set both in `--model_args` and as a top-level
   flag, because the lm-eval version on this cluster requires both.
4. **GPU memory.** 48 GB is enough for ≤500M-param models at `batch_size=4`.
   For larger checkpoints lower `batch_size` (slower) or up the `-M` flag.

## Legacy paths (backwards compat)

The old scripts at
`experiments/energy-inference/scripts/structured-proj/{submit_eval.sh,eval_harness.py}`
and `experiments/energy-inference/scripts/multi-block-ablation/submit_bbh_eval.sh`
are kept for now — older `run_*.sh` launchers reference them. New work should
prefer `experiments/eval_scripts/`.
