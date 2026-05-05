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

TASKS="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu,gsm8k,gsm8k_cot"
# lambada_openai excluded: Arrow parquet "Repetition level histogram size mismatch" on this cluster
# race excluded: same Arrow parquet corruption (EleutherAI/race high, test split 1045)
# bbh_fewshot excluded: SaylorTwift/bbh not cached; use submit_bbh_eval.sh after pre-downloading dataset

bsub \
    -q normal \
    -G grp_ebm \
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
uv pip install accelerate lm-eval -q
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
