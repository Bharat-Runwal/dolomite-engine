#!/bin/bash
# Unshard a checkpoint into <out_dir> exactly once, safe against concurrent callers.
#
# WHY THE LOCK: the eval is split into two independent jobs (likelihood / gsm8k_cot) that both
# need the same unsharded model. A bare `if [ ! -f model.safetensors ]` guard does not prevent
# two jobs that start in the same minute from both running lm_engine.unshard into the same
# directory and interleaving their writes. flock serialises them: the loser waits, then finds
# the file already there and returns immediately.
set -u
LOAD_PATH=$1; ITER=$2; OUT=$3
REPO=/proj/dmfexp/nima/Code/dolomite-engine
LOCK=/tmp/unshard_$(echo "$OUT" | md5sum | cut -c1-16).lock
(
  flock 9 || exit 1
  if [ -f "$OUT/model.safetensors" ]; then
      echo "unshard: already present at $OUT"
      exit 0
  fi
  C=$(mktemp /tmp/unsh_XXXXXX.yml)
  printf 'load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n' \
      "$LOAD_PATH" "$ITER" "$OUT" > "$C"
  cd "$REPO" && python -m lm_engine.unshard --config "$C"
  rc=$?
  rm -f "$C"
  exit $rc
) 9>"$LOCK"
