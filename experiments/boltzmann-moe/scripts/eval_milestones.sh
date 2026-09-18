#!/bin/bash
# Evaluate iso-token MILESTONE checkpoints (8B/16B/24B/32B anchors hard-linked by
# milestone_ckpt_backup.sh), highest token count first.
#
# Usage: bash scripts/eval_milestones.sh [max_anchors] [min_tokens_B]
#   e.g. bash scripts/eval_milestones.sh 5 16   -> up to 5 anchors, only >=16B
#
# TWO JOBS PER ANCHOR (2026-09-18). gsm8k_cot generation is ~55-85 min of a >1.5 h 14-task
# eval and preemption kept killing the whole thing inside it. So:
#   evm_<arm>_<N>B      13 likelihood tasks + wikitext  (fast, ~20-30 min)
#   evmg_<arm>_<N>B     gsm8k_cot ALONE                 (slow)
# Each job runs scripts/merge_eval_results.py at the end; whichever finishes last produces
# the combined harness_results_merged_*.json that compute_avg11.py reads (it selects ONE
# newest file and does NOT union across files -- hence the merge).
#
# The gsm8k raw output goes to a SIBLING dir, never inside the unsharded dir, because
# compute_avg11.py globs recursively: a gsm8k-only file inside it could be selected and would
# look like a complete eval missing 11 of 14 tasks.
#
# Each part is skipped independently, so a preempted gsm8k half does not re-run the fast half.
#
# STAGING: lm_engine.unshard wants <load_path>/global_step<iteration>; a milestone dir is that
# content under a different name, so it is symlinked into place (zero bytes).
# OUTPUT LOCATION: under milestones/, because auto_eval_on_finish.sh treats any
# <save_path>/unsharded*/harness_results*.json as "this ARM is fully evaluated".
set -u
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
MAXA=${1:-99}
MINTOK=${2:-0}
LIKE="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,mmlu,wikitext"
GSM="gsm8k_cot"
cd "$EXP" || exit 1
VENV="source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate && export PYTHONPATH=$REPO:\$PYTHONPATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"

n=0
find results -maxdepth 4 -type d -name 'tok*B_step*' 2>/dev/null \
  | awk -F/ '{d=$0; nm=$NF; sub(/^tok/,"",nm); split(nm,a,"B_step"); split(a[2],b,"_actual"); sub(/B$/,"",b[2]);
              printf "%s\t%s\t%s\t%s\n", b[2], a[1], b[1], d}' \
  | sort -k1,1 -nr \
  | while IFS=$'\t' read -r actual nomtok step dir; do
      [ -z "${dir:-}" ] && continue
      awk -v a="$actual" -v m="$MINTOK" 'BEGIN{exit !(a+0 >= m+0)}' || continue
      [ "$n" -ge "$MAXA" ] && break
      n=$((n+1))
      arm=$(echo "$dir" | awk -F/ '{print $(NF-2)}')
      tag="tok${nomtok}B_step${step}"
      base=$(dirname "$dir")
      out="$base/unsharded_${tag}"          # likelihood results + merged result land here
      gout="$base/gsm8k_${tag}"             # gsm8k raw -- deliberately OUTSIDE $out
      O="$EXP/$out"; G="$EXP/$gout"
      stage="$base/.stage_${tag}"
      mkdir -p "$stage"; ln -sfn "$EXP/$dir" "$stage/global_step${step}"
      printf '{\n    "latest_checkpointed_iteration": %s\n}\n' "$step" > "$stage/latest_checkpointed_iteration.json"
      A="$EXP/$stage"
      UNSH="bash $EXP/scripts/unshard_once.sh $A $step $O"
      # rename gsm8k output so it cannot match harness_results_*.json (see merge script)
      RENAME="for f in $G/harness_results_*.json; do [ -e \"\$f\" ] && mv \"\$f\" \"$G/gsm8k_raw_\$(basename \$f .json | sed s/harness_results_//).json\"; done; true"
      MERGE="python $EXP/scripts/merge_eval_results.py $O $G"

      job="evm_${arm}_${nomtok}B"
      if compgen -G "$out/harness_results_2*.json" > /dev/null; then echo "skip $job (done)"
      elif bjobs -noheader -o job_name 2>/dev/null | grep -qx "$job"; then echo "skip $job (queued)"
      else
        bash "$EXP/scripts/bsub/submit_gpu_test.sh" "$job" \
"cd $REPO && $VENV && $UNSH && \
python experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$O,dtype=bfloat16,trust_remote_code=True \
 --tasks $LIKE --device cuda:0 --batch_size 1 --trust_remote_code --output_path $O/harness_results.json && $MERGE" \
 1 03:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | sed "s|^|submitted $job (${actual}B, ${LIKE##*,}+12 tasks): |"
      fi

      jobg="evmg_${arm}_${nomtok}B"
      # NOTE: the gsm8k output is RENAMED to gsm8k_raw_*.json (so it cannot be mistaken for a
      # complete eval), so the done-check must match THAT name. Checking harness_results_*.json
      # here made the gsm part non-idempotent and resubmitted 5 finished jobs on 2026-09-18.
      if compgen -G "$gout/gsm8k_raw_*.json" > /dev/null || compgen -G "$gout/harness_results_*.json" > /dev/null; then echo "skip $jobg (done)"
      elif bjobs -noheader -o job_name 2>/dev/null | grep -qx "$jobg"; then echo "skip $jobg (queued)"
      else
        bash "$EXP/scripts/bsub/submit_gpu_test.sh" "$jobg" \
"cd $REPO && $VENV && $UNSH && \
python experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$O,dtype=bfloat16,trust_remote_code=True \
 --tasks $GSM --device cuda:0 --batch_size 1 --trust_remote_code --output_path $G/harness_results.json && $RENAME && $MERGE" \
 1 04:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | sed "s|^|submitted $jobg (${actual}B, gsm8k_cot only): |"
      fi
    done
