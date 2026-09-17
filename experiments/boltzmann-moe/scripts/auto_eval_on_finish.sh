#!/bin/bash
# Submit a full eval for every cmix arm that has FINISHED, and only then.
#
# EDGE-TRIGGERED, not level-triggered. The old collect_flops_wave eval mode resubmitted the
# same eval on every cycle because it only checked "is there a checkpoint"; its own comments
# record that. Here an arm is skipped unless ALL of:
#   latest_checkpointed_iteration == num_training_steps   (the run is actually done)
#   no harness_results*.json under unsharded_step<N>      (not already evaluated)
#   no live bsub job named ev_<arm>                       (not already queued/running)
# so it is safe to run on a short cron.
set -u
REPO=/proj/dmfexp/nima/Code/dolomite-engine
cd "$REPO" || exit 1
TASKS="arc_challenge,arc_easy,hellaswag,openbookqa,piqa,sciq,boolq,copa,winogrande,race,lambada_openai,mmlu,gsm8k_cot,wikitext"
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null
export PYTHONPATH=$REPO:${PYTHONPATH:-}

python3 - <<'PY' > /tmp/aeof.$$ 2>/dev/null
import yaml, glob, json, os
for f in sorted(glob.glob('configs/cmix/cmix*.yml')):
    n = os.path.basename(f)[:-4]
    # 'probe' catches bsprobe/proxysvd-style throwaways: a 120-step batch probe technically
    # 'finishes', so without this the watcher spends a GPU evaluating a diagnostic. The
    # milestone script already excluded these; this list had drifted out of sync with it.
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
    # check EVERY unsharded* dir, not just unsharded_step<N>: earlier evals wrote to a plain
    # `unsharded/`, and looking only at the suffixed path queued duplicate jobs for two arms.
    if glob.glob(f'{sp}/unsharded*/harness_results*.json'):
        continue                       # already evaluated
    print(f'{n}\t{sp}\t{it}')
PY

while IFS=$'\t' read -r nm sp it; do
  [ -z "${nm:-}" ] && continue
  if bjobs -noheader -o job_name 2>/dev/null | grep -qx "ev_${nm}"; then
      echo "skip $nm (eval already queued)"; continue
  fi
  U=$sp/unsharded_step${it}
  # batch_size 1: a checkpoint saved with sparse_start_step>0 reloads with _sparse_active
  # FALSE, so eval runs the DENSE all-K path and single-block arms OOM at batch 4.
  bash "$REPO/experiments/boltzmann-moe/scripts/bsub/submit_gpu_test.sh" "ev_${nm}" \
"cd $REPO && source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate && export PYTHONPATH=$REPO:\$PYTHONPATH && \
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True && \
if [ ! -f $U/model.safetensors ]; then C=/tmp/unsh_${nm}_\$\$.yml; printf 'load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n' $sp $it $U > \$C; python -m lm_engine.unshard --config \$C && rm -f \$C; fi && \
python experiments/eval_scripts/eval_harness.py --model hf --model_args pretrained=$U,dtype=bfloat16,trust_remote_code=True \
 --tasks $TASKS --device cuda:0 --batch_size 1 --trust_remote_code --output_path $U/harness_results.json" \
 1 06:00 200G 2>&1 | grep -oE "Job <[0-9]+>" | sed "s|^|submitted ev_${nm} (step ${it}): |"
done < /tmp/aeof.$$
rm -f /tmp/aeof.$$
