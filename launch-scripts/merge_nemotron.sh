#!/bin/bash
#BSUB -J nemotron-merge
#BSUB -n 16
#BSUB -R "span[hosts=1]"
#BSUB -R "rusage[mem=8G]"
#BSUB -W 24:00
#BSUB -o /proj/dmfexp/energy-gpt/logs/nemotron-merge-%J.log
#BSUB -e /proj/dmfexp/energy-gpt/logs/nemotron-merge-%J.err

cd /proj/dmfexp/hopfield++/Projects/dolomite-engine
source .venv/bin/activate

python tools/data/merge_data.py \
    --input-directory /proj/dmfexp/energy-gpt/data/dmf_tables/nemotron_cc_tokenized \
    --output-prefix /proj/dmfexp/energy-gpt/data/dmf_tables/nemotron_cc_merged \
    --max-size 250 \
    --workers 8
