#!/bin/bash
#BSUB -J dclm-preprocess
#BSUB -n 64
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -W 48:00
#BSUB -o /proj/dmfexp/energy-gpt/logs/preprocess-%J.log
#BSUB -e /proj/dmfexp/energy-gpt/logs/preprocess-%J.err

cd /proj/dmfexp/hopfield++/Projects/dolomite-engine
source .venv/bin/activate

python tools/data/preprocess_data.py \
    --input /proj/dmfexp/energy-gpt/data/dclm_baseline_1_0 \
    --tokenizer gpt2 \
    --output-prefix /proj/dmfexp/energy-gpt/data/dclm-gpt2-tokenized/dclm \
    --json-keys contents \
    --append-eod \
    --max-local-processes 64
