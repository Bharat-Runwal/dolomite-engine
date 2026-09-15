#!/bin/bash
# run_accel_ab_20260915.sh [submit|status|report]
#
# 4-GPU A/B probe for the boltz-accel knobs. Answers two questions:
#   1. how much of the 400M arm's 6.75 s/step do we recover?
#   2. does the loss curve stay where it was?
#
# Runs from the ISOLATED WORKTREE (/proj/dmfexp/nima/Code/dolomite-accel) so the
# live 32B arms keep reading pristine `main` even if the watchdog resubmits them.
set -uo pipefail
MODE="${1:-submit}"
WT=/proj/dmfexp/nima/Code/dolomite-accel
ARMS=(accel_A_base accel_B_fused accel_C_fused_rep10 accel_D_fused_rep10_proxy8)
mkdir -p "$HOME/bsub_logs"

case "$MODE" in
submit)
  for a in "${ARMS[@]}"; do
    bsub \
      -q preemptable -G grp_preemptable -J "$a" \
      -gpu "num=4/task:mode=exclusive_process" -n 1 -M 64G -W 02:00 \
      -o "$HOME/bsub_logs/${a}_%J.stdout" -e "$HOME/bsub_logs/${a}_%J.stderr" \
      <<INNER
#!/bin/bash
set -uo pipefail
# tempfile falls back to CWD when TMPDIR is unset and /tmp is unwritable inside
# LSF jobs -- that is what littered ~190 pymp-*/ dirs into the results tree.
export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "\$TMPDIR"
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
# PYTHONPATH points at the WORKTREE: this is what makes the probe use the accel
# code while the production arms keep using main.
export PYTHONPATH=${WT}:\${PYTHONPATH:-}
cd ${WT}
bash ${WT}/scripts/common/pretrain.sh ${WT}/configs/boltz_accel/${a}.yml
INNER
  done
  echo "=== submitted ${#ARMS[@]} arms; verify in 60s with: $0 status ==="
  ;;
status)
  bjobs -o "jobid stat job_name run_time" 2>/dev/null | grep -E "JOBID|accel_" || echo "no accel jobs"
  ;;
report)
  python3 "${WT}/experiments/boltzmann-moe/scripts/report_accel_ab_20260915.py"
  ;;
*) echo "usage: $0 {submit|status|report}"; exit 2 ;;
esac
