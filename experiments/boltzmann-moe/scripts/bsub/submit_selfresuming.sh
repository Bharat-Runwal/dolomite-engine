#!/bin/bash
# Submit a long run whose job script resolves load_args AT RUN TIME.
#
#   bash submit_selfresuming.sh <job_name> <config.yml> <n_gpus> <walltime> <mem>
#
# WHY THIS EXISTS. LSF preemption REQUEUES the job on the same jid and re-runs the ORIGINAL
# submitted command. A config with no load_args therefore restarts from step 0 on EVERY
# preemption, indefinitely -- t90k_pure_T12 lost 2670 steps that way, twice, and s90k_pure_T12
# was preempted inside its first few minutes and would have done the same. Deciding resume at
# SUBMIT time cannot work; the job script has to decide it each time it starts.
set -euo pipefail
JOB=${1:?job name}; CFG=${2:?config}; GPUS=${3:-8}; WALL=${4:-24:00}; MEM=${5:-128G}
# Machine-specific paths come from experiments/paths.sh (REPO_ROOT is self-located, so this
# works in any clone from any cwd; VENV/DATA_ROOT default to this cluster and are overridable
# with DOLOMITE_VENV / DOLOMITE_DATA_ROOT).
. "$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)/experiments/paths.sh"
SP=$(grep -E "^\s*save_path:" "$CFG" | head -1 | sed 's/.*save_path:\s*//')
mkdir -p "$HOME/bsub_logs"
bsub -q preemptable -G grp_preemptable -J "$JOB" \
     -gpu "num=${GPUS}/task:mode=exclusive_process" -n 1 -M "$MEM" -W "$WALL" \
     -o "$HOME/bsub_logs/${JOB}_%J.stdout" -e "$HOME/bsub_logs/${JOB}_%J.stderr" <<INNER
#!/bin/bash
unset TMPDIR TEMP TMP
source ${VENV}/bin/activate
export PYTHONPATH=${REPO}:\${PYTHONPATH:-}
CFG="${CFG}"
SP="${SP}"
# Resolve resume HERE, so every requeue re-evaluates it.
if [ -f "\$SP/latest_checkpointed_iteration.json" ]; then
    if grep -qE "^load_args:" "\$CFG"; then
        echo "RESUME: config already carries load_args"
    else
        RC="\$SP/runtime_resume_\$\$.yml"
        cp "\$CFG" "\$RC"
        printf "\\nload_args:\\n  load_path: %s\\n" "\$SP" >> "\$RC"
        CFG="\$RC"
        echo "RESUME: built \$RC"
    fi
    echo "RESUME: latest = \$(cat \$SP/latest_checkpointed_iteration.json)"
else
    echo "RESUME: no checkpoint under \$SP -- starting from step 0 (expected on a FIRST start only)"
fi
bash ${REPO}/scripts/common/pretrain.sh "\$CFG"
INNER
echo "submitted. grep the stderr for 'RESUME:' after any preemption."
