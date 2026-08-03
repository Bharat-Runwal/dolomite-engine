#!/bin/bash
# Submit a lm-evaluation-harness job for a given checkpoint.
# Called automatically at the end of each training job.
#
# Usage: bash submit_eval.sh <checkpoint_path> <job_name>
#
# Benchmarks: arc_challenge, arc_easy, boolq, copa, hellaswag, lambada_openai,
#   openbookqa, piqa, race, sciq, wikitext (word_ppl), winogrande,
#   mmlu, gsm8k, gsm8k_cot
#
# Results written to <checkpoint_path>/harness_results.json

CHECKPOINT="${1:?Usage: submit_eval.sh <checkpoint_path> <job_name>}"
JOB_NAME="${2:-eval}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine

TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,race,sciq,wikitext,winogrande,lambada_openai,mmlu,gsm8k,gsm8k_cot"
# race + lambada_openai RE-ENABLED 2026-08-03: the "Repetition level histogram size mismatch" was a
# pyarrow<20 reader bug (their parquet is written by pyarrow>=20); fixed by the pyarrow>=20 pin below.
# bbh_fewshot excluded: SaylorTwift/bbh not cached; use submit_bbh_eval.sh after pre-downloading dataset

bsub \
    -q preemptable \
    -G grp_preemptable \
    -J "${JOB_NAME}" \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 02:00 \
    -o "${HOME}/bsub_logs/${JOB_NAME}_%J.stdout" \
    -e "${HOME}/bsub_logs/${JOB_NAME}_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
export HF_DATASETS_OFFLINE=1
export HF_HUB_OFFLINE=1
uv pip install accelerate lm-eval 'pyarrow>=20' -q   # pyarrow>=20 required to read race/lambada parquet (else 'Repetition level histogram size mismatch')
cd $REPO
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \\
    --model hf \\
    --model_args "pretrained=$CHECKPOINT,dtype=bfloat16,trust_remote_code=True" \\
    --tasks $TASKS \\
    --device cuda:0 \\
    --batch_size 4 \\
    --trust_remote_code \\
    --output_path "$CHECKPOINT/harness_results.json"
EOF

echo "Eval job submitted: ${JOB_NAME}  checkpoint: ${CHECKPOINT}"
