# BBH em=0.0000 bug + fix (2026-06-26)

## The bug

Every BBH eval in this project up to 2026-06-26 reported
`exact_match = 0.0000` across all 27 subtasks (math_v9, RecGPT,
FET-1.5e-3, Hopfield-MEAN step 6000). The models were not actually
scoring zero — the `bbh_fewshot` task in `lm-evaluation-harness` ships
without a `filter_list`, so `exact_match` is computed as a verbatim
Python `==` between the raw post-stop continuation and the gold target.

With the upstream prompt `doc_to_text: "Q: {{input}}\nA:"` and
`target_delimiter: " "`, a small model that copies the 3-shot demo
format produces strings like `" False"` (with a leading space from the
delimiter), `" (D)"`, `" Yes"`, etc. Then `" False" == "False"` is
`False` ⇒ em=0 for every example for every subtask.

The parallel `bbh_zeroshot` variants in the same package ship a
per-subtask `flexible-extract` filter chain with regexes like
`\b(True|False)\b`, `\(([A-Z])\)`, `[-0-9]+`, etc. `bbh_fewshot` simply
has no extractor wired up.

## The fix

Post-hoc rescoring, no upstream-package fork:

1. `submit_bbh_eval.sh` now passes `--log_samples` and an
   `--output_path` directory to lm-eval-harness so the raw 3-shot
   generations are saved to `samples_<subtask>_*.jsonl`.
2. `bbh_rescore.py` (new in this directory) ports the per-subtask
   flexible-extract regexes from `lm_eval/tasks/bbh/zeroshot/*.yaml`
   and applies them to both the model prediction and the gold target.
   Generic fallback (strip whitespace, take first line, strip trailing
   punct, casefold) handles `dyck_languages` / `word_sorting`.
3. The original (broken) `harness_bbh_results_*.json` is left in
   place. The rescored numbers land at
   `harness_bbh_results_*.rescored.json` and a flat per-subtask
   summary at `bbh_rescore_summary_*.json`, both next to the
   checkpoint.

## How to re-evaluate any prior checkpoint

```bash
bash /proj/dmfexp/nima/Code/dolomite-engine/experiments/eval_scripts/submit_bbh_eval.sh \
    <unsharded_ckpt_path> <run_name>
```

The launcher will (a) rerun BBH with `--log_samples` and
(b) automatically rescore on completion. Wait ~5–8 min on 1 GPU.

The rescored summary JSON is the file to read.

## Validation

Re-ran BBH on Hopfield-MEAN step 6000:
- Old (broken) lm-eval em: **0.0000** across all 27 subtasks
- New (rescored) macro-avg em: **0.2649**

Top subtasks: `sports_understanding` 0.536, `causal_judgement` 0.519,
`hyperbaton` 0.516, `snarks` 0.500, `formal_fallacies` 0.476,
`boolean_expressions` 0.460. Above-random on 13/27 binary or
low-cardinality multiple-choice subtasks — recovers the model's true
BBH signal.

Bottom subtasks (`multistep_arithmetic_two` 0.004,
`word_sorting` 0.028, `dyck_languages` 0.032) reflect real model
weakness on open-ended generation tasks, not a metric bug.

## Files

- Fix: `experiments/eval_scripts/bbh_rescore.py` (new),
  `experiments/eval_scripts/submit_bbh_eval.sh` (modified).
- Validation job: bsub `1755237` (eval_bbh_hopfield_mean_step6k_rescored,
  DONE 2026-06-26 19:02).
- Result files:
  `.../math_fet_hopfield_mean_8gpt_1egpt6x_d1536_int8k_lra32_itd3_lr1p5e3_33b_16gpu/unsharded_step6000/`
  - `harness_bbh_results_2026-06-26T18-54-59Z_*.rescored.json`
  - `bbh_rescore_summary_2026-06-26T18-54-59Z.json`
