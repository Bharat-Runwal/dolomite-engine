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
| `compute_avg11.py` | **CANONICAL headline aggregator (2026-09-14).** Prints the colleague-/paper-consistent 11-task `Avg11` (acc_norm ×6 + acc ×5), with MMLU, GSM8K-CoT and WikiText-PPL reported separately. Refuses to emit `Avg11` for a run missing any of the 11 tasks (prints `INCOMPLETE (k/11)`). Use this for any headline or cross-team number. |
| `restate_to_avg11_20260914.py` | Walks every stored `harness_results_*.json` and emits the `Avg11` table (`--md` for markdown), marking INCOMPLETE runs. Pure stdlib; imports the recipe from `compute_avg11.py`. |
| `compute_aggregates.py` | **DEPRECATED for headlines (2026-09-14).** Prints legacy `avg10` / `avg10_norm` (MMLU-in, race/lambada-out), WikiText-PPL, GSM8K, per-task acc/acc_norm. `avg10` is ~3pp ABOVE `Avg11` for a pure scoring-convention reason — do not quote it next to a colleague's Avg11. Kept only to reproduce old avg10 tables. |
| `restate_avg9_to_avg10_20260912.py` | Legacy: measures the avg9→avg10 gap. Superseded by `restate_to_avg11_20260914.py` for headlines. |
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

# Aggregate after results land (CANONICAL Avg11 headline):
python experiments/eval_scripts/compute_avg11.py \
     /path/to/unsharded/harness_results.json
# (legacy avg10, deprecated for headlines: compute_aggregates.py)
```

All `run_*.sh` training launchers in this repo automatically chain into
`submit_eval.sh` at end-of-training (after the final `lm_engine.unshard` step).

## Standard task list (13 tasks)

```
arc_challenge, arc_easy, boolq, copa, hellaswag, openbookqa,
piqa, sciq, wikitext, winogrande, mmlu, gsm8k, gsm8k_cot
```

**CANONICAL headline aggregate (2026-09-14): `Avg11`** via `compute_avg11.py` — the
11-task mean used by the EGPT-RL / FET colleagues and the paper `tab:scaling`:
`acc_norm` on {arc_challenge, arc_easy, hellaswag, openbookqa, piqa, sciq} and `acc`
on {boolq, copa, winogrande, **race**, **lambada_openai**}, with **MMLU (acc) and
GSM8K-CoT (flex) reported SEPARATELY, never folded in**. `race` + `lambada_openai`
are required, so any checkpoint evaluated before the 2026-08-03 `pyarrow>=20` fix
comes back `INCOMPLETE (9/11)` and must be re-evaluated before an Avg11 can be quoted.

The legacy **avg10** (`compute_aggregates.py`, now deprecated for headlines) instead
aggregates the first 10 tasks (mean of primary `acc`); **avg10_norm** uses `acc_norm`
for tasks that report it (arc_challenge, arc_easy, hellaswag, openbookqa, piqa) and
`acc` otherwise (boolq, copa, sciq, winogrande, mmlu). It INCLUDES MMLU and EXCLUDES
race + lambada, so it sits ~3pp ABOVE Avg11 — never mix the two in one table.

`compute_aggregates.py` prints both avg10 forms, plus WikiText word-PPL, GSM8K
(strict), GSM8K-CoT (flexible-extract), and a per-task table.

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

---

# SWITCH TO THE SHARED HARNESS (2026-09-12)

**We now evaluate with a vendored, version-pinned copy of the same
lm-evaluation-harness our colleagues use**, so that every number in the ICLR
submission is produced by identical code.

    experiments/eval_scripts/lm-evaluation-harness/     <- vendored, USE THIS
    experiments/eval-scripts -> eval_scripts            <- symlink for the hyphen path

Source: `/proj/dmfexp/energy-gpt/lm-evaluation-harness` (colleague bsaha3), copied
without `.git`. Provenance is recorded in
`lm-evaluation-harness/VENDORED_FROM.txt`:

| | |
|---|---|
| upstream | `github.com/EleutherAI/lm-evaluation-harness` |
| commit | `ad3f4d0cad1cfcdb815f1e795f7947e49ed9f2e9` ("increment version", #3433) |
| version | **0.4.9.2** |
| local modifications | **none** — stock upstream, detached HEAD |

To use it, prepend it to `PYTHONPATH`; do **not** `pip install lm-eval`, which would
pull a different version into site-packages:

    export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine/experiments/eval_scripts/lm-evaluation-harness:$PYTHONPATH

## What we were doing before, and what actually differs

We were calling a floating `uv pip install lm-eval`, which had resolved to
**0.4.11**. Diffing 0.4.11 against 0.4.9.2 over everything that can affect a score:

- **Task YAMLs for all 15 tasks we run: identical except `dataset_path`.** 0.4.11
  uses the newer namespaced Hub names (`allenai/sciq`, `allenai/openbookqa`,
  `allenai/winogrande`, `aps/super_glue`, `openai/gsm8k`); 0.4.9.2 uses the older bare
  names (`sciq`, `openbookqa`, ...). **No differences in prompts, `num_fewshot`,
  filters, metrics, stop sequences, or aggregation.**
- `lm_eval/tasks/mmlu/`: 0 differing files.
- `lm_eval/api/metrics.py`: 0.4.11 adds a `likelihood` passthrough metric we do not
  use. `lm_eval/api/model.py`: typing imports and docstring indentation.
  `lm_eval/filters/__init__.py`: PEP-585 typing modernisation. **All cosmetic.**

**Conclusion: the switch should not move any number.** It buys reproducibility (a
pinned commit instead of whatever pip resolves that day) and cross-person
comparability, not a correction.

### One operational gotcha the switch introduces

The older bare dataset names **miss our HF cache**, which is keyed by the newer
namespaced names. Under `HF_DATASETS_OFFLINE=1` this made `sciq`, `openbookqa`,
`winogrande`, `super_glue/boolq` and `race/high` fail outright. Fixed with cache
aliases pointing at the *same underlying data* (so this is not a data change):

    C=~/.cache/huggingface/datasets
    ln -s allenai___sciq        $C/sciq
    ln -s allenai___openbookqa  $C/openbookqa
    ln -s allenai___winogrande  $C/winogrande
    ln -s allenai___ai2_arc     $C/ai2_arc
    ln -s ../aps___super_glue/boolq  $C/super_glue/boolq
    mkdir -p $C/race && ln -s ../ehovy___race/all $C/race/all \
                     && ln -s ../EleutherAI___race/high $C/race/high

Verified: all 15 tasks load offline through the vendored harness after this.

## Task list was inconsistent across our own scripts — now unified

Before the switch there were **three different task lists** in this repo, so "Avg"
was not necessarily over the same tasks between tables:

| script | tasks |
|---|---|
| `submit_eval.sh` | 15 (incl. `race`, `lambada_openai`, `gsm8k`, `gsm8k_cot`) |
| `unshard_eval_progression.sh` | 13 (no `race`, no `lambada_openai`) |
| `collect_flops_wave_20260912.sh` | 11 (also no gsm8k) |

All three now use `EVAL_TASKS` defined in `eval_tasks.sh` (single source of truth).

**The canonical headline aggregate is now `Avg11` from `compute_avg11.py`** (2026-09-14;
see the metric section above) — an 11-task mean that INCLUDES `race` + `lambada_openai`
and EXCLUDES MMLU from the average. The older `avg10` / `avg10_norm` from
`compute_aggregates.py` — a mean over exactly ten tasks: `arc_challenge, arc_easy,
boolq, copa, hellaswag, openbookqa, piqa, sciq, winogrande, mmlu`, with `avg10_norm`
substituting `acc_norm` for the five tasks where it is standard (`arc_challenge,
arc_easy, hellaswag, openbookqa, piqa`) and `acc` for the other five — is **deprecated
for headlines**: it EXCLUDES `race` + `lambada_openai` (evaluated but not averaged,
despite a commit message that says "11-task Avg%") and INCLUDES MMLU, leaving it ~3pp
above Avg11.

## BBH remains a deliberate local deviation

`bbh_fewshot` upstream ships **without a `filter_list`**, so `exact_match` is a
verbatim string compare and every model scores 0.0000 (the continuation carries a
leading space from `target_delimiter`). We rescore post hoc with `bbh_rescore.py`,
porting the per-subtask `flexible-extract` regexes from `bbh_zeroshot/*.yaml`. See
`BBH_FIX.md`. This is unchanged by the switch and **must be disclosed** whenever BBH
numbers are compared to stock-harness results elsewhere.
