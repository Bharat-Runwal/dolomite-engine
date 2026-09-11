#!/bin/bash
# Smoke test: launch both reference models to confirm they train on this branch.
#
#   ./configs/boltzman-moe-configs/submit_smoketest.sh              # both, 1 node (8 GPU)
#   ./configs/boltzman-moe-configs/submit_smoketest.sh baseline     # just the baseline
#   ./configs/boltzman-moe-configs/submit_smoketest.sh boltzmoe     # just the BoltzMoE
#   NODES=2 QUEUE=normal GROUP=grp_ebm ./..../submit_smoketest.sh   # override placement
#
# What "passing" looks like: the log reaches `step = 10` with a finite train-loss
# and a non-zero billion_tokens_per_day. That is enough to prove the model builds,
# the data loads, FSDP-2 shards it, and the optimizer steps. It is NOT a quality
# run -- these configs are the full 30k-step recipe, so kill the jobs once you
# have seen a few logged steps.
#
# NOTE: STEPS below overrides num_training_steps so a smoke test cannot silently
# turn into a 30k-step run. Set STEPS=0 to use the config value as-is.
set -euo pipefail

REPO="/proj/dmfexp/bishwajit/Code/dolomite-engine"
CFG_DIR="$REPO/configs/boltzman-moe-configs"
LOG_DIR="${LOG_DIR:-/proj/dmfexp/energy-gpt/logs/boltzmoe-smoketest}"
mkdir -p "$LOG_DIR"
# the trainer does not create save_path's parent; a missing parent only fails at
# the first checkpoint, long after launch, so make it up front.
mkdir -p /proj/dmfexp/energy-gpt/checkpoints-bsaha/boltzman-moe-configs

NODES="${NODES:-1}"
QUEUE="${QUEUE:-normal}"
GROUP="${GROUP:-grp_ebm}"
MEM="${MEM:-900G}"
STEPS="${STEPS:-50}"

# transformers 4.57.1 -- required, see configs/boltzman-moe-configs/README.md
export PRETRAIN_VENV="${PRETRAIN_VENV:-$REPO/.venv-nima}"

declare -A CFG=(
  [baseline]="s12_stdmoe_topk2_400m_30k_2e-3"
  [boltzmoe]="s8e4_fh2_boltzmoe_topk2_400m_30k_2e-3"
)

submit_one () {
  local key="$1" name="${CFG[$1]}"
  local src="$CFG_DIR/${name}.yml"
  [ -f "$src" ] || { echo "FATAL: missing $src" >&2; exit 1; }

  local cfg="$src"
  if [ "$STEPS" != "0" ]; then
    cfg="/tmp/_smoketest_${name}_${STEPS}.yml"
    "$PRETRAIN_VENV/bin/python" - "$src" "$cfg" "$STEPS" <<'PYEOF'
import sys, yaml
src, dst, steps = sys.argv[1], sys.argv[2], int(sys.argv[3])
c = yaml.safe_load(open(src))
c["training_parameters"]["num_training_steps"] = steps
# keep the LR schedule inside the shortened run so nothing divides by zero
c["lr_scheduler_args"]["num_warmup_steps"] = min(c["lr_scheduler_args"]["num_warmup_steps"], max(1, steps // 5))
c["lr_scheduler_args"]["num_decay_steps"]  = max(1, steps - c["lr_scheduler_args"]["num_warmup_steps"])
c["save_args"]["save_interval"] = max(steps, 1)          # don't checkpoint mid-smoketest
c["training_parameters"]["eval_during_training"] = False
c["logging_args"]["log_interval"] = 1                    # see every step
c["logging_args"]["wandb_args"]["name"] = f"smoketest_{c['logging_args']['wandb_args']['name']}"
yaml.dump(c, open(dst, "w"), sort_keys=True, default_flow_style=False, width=200)
print(f"  smoketest config -> {dst} ({steps} steps)")
PYEOF
  fi

  echo "--- $key : $name  (${NODES} node x 8 GPU, queue=$QUEUE group=$GROUP, ${STEPS} steps)"
  bsub \
    -q "$QUEUE" \
    -G "$GROUP" \
    -M "$MEM" \
    -hl \
    -n "$NODES" \
    -J "smoke_${key}" \
    -gpu "num=8/task:mode=exclusive_process" \
    -oo "${LOG_DIR}/${key}.out" \
    -eo "${LOG_DIR}/${key}.err" \
    blaunch bash "$REPO/launch-scripts/pretrain.sh" "$cfg"
}

cd "$REPO"
if [ "$#" -eq 0 ]; then
  submit_one baseline
  submit_one boltzmoe
else
  for k in "$@"; do
    [ -n "${CFG[$k]:-}" ] || { echo "FATAL: unknown target '$k' (use baseline|boltzmoe)" >&2; exit 1; }
    submit_one "$k"
  done
fi

echo
echo "Watch:   bjobs -w | grep smoke_"
echo "Logs:    $LOG_DIR"
echo "Pass if: grep -a 'step = ' $LOG_DIR/*.err | head"
