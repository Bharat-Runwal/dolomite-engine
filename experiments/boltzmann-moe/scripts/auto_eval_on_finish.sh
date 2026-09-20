#!/bin/bash
# Submit a full eval for every cmix arm that has FINISHED, and only then.
#
# EDGE-TRIGGERED, not level-triggered. The old collect_flops_wave eval mode resubmitted the
# same eval on every cycle because it only checked "is there a checkpoint"; its own comments
# record that. Here an arm's part is skipped unless ALL of:
#   latest_checkpointed_iteration == num_training_steps   (the run is actually done)
#   that part's results file is absent                    (not already evaluated)
#   no live bsub job for that part                        (not already queued/running)
# so it is safe to run on a short cron.
#
# SPLIT INTO TWO JOBS (2026-09-18). gsm8k_cot generation is ~55-85 min of a >1.5 h 14-task
# eval, and on preemptable the whole job kept dying inside it: ev_cmix_134M_pure lost 51 min
# (792/1319 requests) and then another ~90 min on the retry. Now:
#   ev_<arm>    13 likelihood tasks + wikitext -> <sp>/unsharded_step<N>/harness_results_*.json
#   evg_<arm>   gsm8k_cot ALONE                -> <sp>/gsm8k_step<N>/gsm8k_raw_*.json
# A preemption now costs one task, not the whole eval. Each job runs merge_eval_results.py at
# the end; whichever finishes last writes harness_results_merged_*.json next to the
# likelihood results, because compute_avg11.py selects ONE newest file and does not union.
#
# The gsm8k raw file is deliberately NOT named harness_results_*.json: compute_avg11.py globs
# recursively from the run dir, so a gsm8k-only file with that name could be selected and read
# as a complete eval missing 11 of 14 tasks.
#
# Arms evaluated BEFORE the split already have gsm8k_cot inside their single json; the planner
# detects that and does not queue a redundant gsm8k job for them.
set -u
REPO=/proj/dmfexp/nima/Code/dolomite-engine
EXP=$REPO/experiments/boltzmann-moe
cd "$REPO" || exit 1
LIKE="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,mmlu,wikitext"
GSM="gsm8k_cot"
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null
export PYTHONPATH=$REPO:${PYTHONPATH:-}

python3 - <<'PY' > /tmp/aeof.$$ 2>/dev/null
import yaml, glob, json, os
def newest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None
# DEDUPE ON save_path. The *_4gpu fallback configs deliberately share save_path with their
# 8-GPU sibling (they compensate with gradient_accumulation_steps so tokens/step, step count and
# the LR schedule are identical, and they resume the SAME run). Iterating over config FILES
# therefore submitted two evals for one finished run, racing on the same output files
# (observed 2026-09-18 for cmix_134M_sandwich_32B_sparse). Keep the shortest name = the canonical
# arm, so the '_4gpu' variant never wins.
seen_sp = {}
# GLOB WIDENED 2026-09-19: ablation arms live in configs/iclr_26/ablations/, so the old
# configs/cmix/cmix*.yml glob silently skipped abl_B_134M_6G1x6S -- the FLOP-matched baseline --
# after it completed. Any new config tree must be added here or its arms are never evaluated.
for f in sorted(glob.glob('configs/cmix/cmix*.yml')
                + glob.glob('configs/iclr_26/**/*.yml', recursive=True)):
    n = os.path.basename(f)[:-4]
    # 'probe' catches bsprobe/proxysvd-style throwaways: a 120-step batch probe technically
    # 'finishes', so without this the watcher spends a GPU evaluating a diagnostic.
    if any(t in n for t in ('feas','fixtest','tcptest','bisect','smoke','spmddiag',
                            'probe','cal','diag')):
        continue
    try: c = yaml.safe_load(open(f))
    except Exception: continue
    sp = (c.get('save_args') or {}).get('save_path')
    if not sp: continue
    tot = c['training_parameters']['num_training_steps']
    j = os.path.join(sp, 'latest_checkpointed_iteration.json')
    if not os.path.exists(j): continue
    try: it = json.load(open(j))['latest_checkpointed_iteration']
    except Exception: continue
    if it != tot:                      # NOT finished -- the whole point
        continue
    # Accept the FINAL step's dir, or a legacy plain `unsharded/` -- but NOT some other step.
    # The old glob was `unsharded*`, which matches ANY step. The 1B arm had one stale eval under
    # unsharded_step4000 (2.10B tokens), so this function concluded the arm was already evaluated
    # and NEVER QUEUED the real 32B eval after it finished -- while gen_status_table reported the
    # 2.10B number as the arm's result in the paper. Widening this glob to fix a duplicate-submission
    # bug is what created a silent wrong-number bug; keep both cases explicit instead.
    main = newest(f'{sp}/unsharded_step{it}/harness_results*.json') \
           or newest(f'{sp}/unsharded/harness_results*.json')
    need_like = '0' if main else '1'
    need_gsm = '1'
    if glob.glob(f'{sp}/gsm8k_step*/gsm8k_raw_*.json'):
        need_gsm = '0'
    elif main:                         # pre-split single-json eval already contains gsm8k?
        try:
            if 'gsm8k_cot' in (json.load(open(main)).get('results') or {}):
                need_gsm = '0'
        except Exception: pass
    if need_like == '0' and need_gsm == '0': continue
    prev = seen_sp.get(sp)
    if prev is not None and len(prev) <= len(n): continue    # canonical arm already emitted
    seen_sp[sp] = n
    print(f'{n}\t{sp}\t{it}\t{need_like}\t{need_gsm}')
PY

VENV="source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate && export PYTHONPATH=$REPO:\$PYTHONPATH && export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
while IFS=$'\t' read -r nm sp it need_like need_gsm; do
  [ -z "${nm:-}" ] && continue
  U=$sp/unsharded_step${it}
  G=$sp/gsm8k_step${it}
  # batch_size 1: a checkpoint saved with sparse_start_step>0 reloads with _sparse_active
  # FALSE, so eval runs the DENSE all-K path and single-block arms OOM at batch 4.
  UNSH="bash $EXP/scripts/unshard_once.sh $sp $it $U"
  MERGE="python $EXP/scripts/merge_eval_results.py $U $G"
  RENAME="for f in $G/harness_results_*.json; do [ -e \"\$f\" ] && mv \"\$f\" \"$G/gsm8k_raw_\$(basename \$f .json | sed s/harness_results_//).json\"; done; true"

  if [ "$need_like" = "1" ]; then
    if bjobs -noheader -o job_name 2>/dev/null | grep -qx "ev_${nm}"; then
        echo "skip ev_${nm} (already queued)"
    else
      bash "$EXP/scripts/bsub/submit_gpu_test.sh" "ev_${nm}" \
"cd $REPO && $VENV && $UNSH && \
python experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$U,dtype=bfloat16,trust_remote_code=True \
 --tasks $LIKE --device cuda:0 --batch_size 1 --trust_remote_code --output_path $U/harness_results.json && $MERGE" \
 1 03:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | sed "s|^|submitted ev_${nm} (step ${it}, 13 likelihood tasks): |"
    fi
  fi
  if [ "$need_gsm" = "1" ]; then
    if bjobs -noheader -o job_name 2>/dev/null | grep -qx "evg_${nm}"; then
        echo "skip evg_${nm} (already queued)"
    else
      bash "$EXP/scripts/bsub/submit_gpu_test.sh" "evg_${nm}" \
"cd $REPO && $VENV && $UNSH && \
python experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$U,dtype=bfloat16,trust_remote_code=True \
 --tasks $GSM --device cuda:0 --batch_size 1 --trust_remote_code --output_path $G/harness_results.json && $RENAME && $MERGE" \
 1 04:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | sed "s|^|submitted evg_${nm} (step ${it}, gsm8k_cot only): |"
    fi
  fi
done < /tmp/aeof.$$
rm -f /tmp/aeof.$$
