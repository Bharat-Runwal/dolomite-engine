"""
Thin wrapper around lm-evaluation-harness for energy-GPT checkpoints.

Registers lm_engine.hf_models BEFORE lm_eval loads the model, so the
custom 'energy' model type is recognised by AutoModelForCausalLM.

Usage
-----
  source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
  export PYTHONPATH=/proj/dmfexp/nima/Code/dolomite-engine:$PYTHONPATH
  uv pip install lm-eval -q
  python eval_harness.py \\
      --model hf \\
      --model_args "pretrained=<ckpt>,dtype=bfloat16" \\
      --tasks arc_challenge,arc_easy,boolq,copa,hellaswag,lambada_openai,openbookqa,piqa,race,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot \\
      --device cuda:0 \\
      --batch_size 4 \\
      --output_path <ckpt>/harness_results.json

All standard lm_eval CLI flags are forwarded as-is.

Benchmark notes
---------------
  arc_challenge / arc_easy   – ARC (AI2 Reasoning Challenge)
  boolq                      – Yes/no reading comprehension
  copa                       – Commonsense causal reasoning
  hellaswag                  – Sentence completion
  lambada_openai             – Long-range language modelling
  openbookqa                 – Open-book science QA
  piqa                       – Physical intuition QA
  race                       – Reading comprehension (exams)
  sciq                       – Science QA
  wikitext                   – Word-level PPL on WikiText-2
                               (compare to original word_ppl=25)
  winogrande                 – Commonsense pronoun resolution
  mmlu                       – 57-subject knowledge benchmark
  gsm8k                      – Grade-school math (exact match)
  gsm8k_cot                  – Chain-of-thought math
"""

import sys

# Must happen before any lm_eval import so the energy architecture is
# registered in HF's AutoModel registry when lm_eval calls from_pretrained.
import lm_engine.hf_models  # noqa: F401

from lm_eval.__main__ import cli_evaluate

sys.exit(cli_evaluate())
