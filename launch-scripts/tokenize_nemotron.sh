#!/bin/bash
#BSUB -J nemotron-tokenize
#BSUB -n 64
#BSUB -G grp_preemptable
#BSUB -q preemptable
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=4G]"
#BSUB -W 72:00
#BSUB -o /proj/dmfexp/energy-gpt/logs/nemotron-tokenize-%J.log
#BSUB -e /proj/dmfexp/energy-gpt/logs/nemotron-tokenize-%J.err

cd /proj/dmfexp/hopfield++/Projects/dolomite-engine
source .venv/bin/activate

python tools/data/preprocess_data.py \
    --input /proj/dmfexp/energy-gpt/data/dmf_tables/nemotron_cc/ \
    --tokenizer ibm-granite/granite-3b-code-base \
    --output-prefix /proj/dmfexp/energy-gpt/data/dmf_tables/nemotron_cc_tokenized \
    --json-keys contents \
    --append-eod \
    --max-local-processes 64
