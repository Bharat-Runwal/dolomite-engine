- As you interact with this codebase, record anything that you find useful or will help you improve in the future to this file.

# Training Jobs (Blue Vela)

## Bharat's energy training workflow

Run a command with the `scripts/blue-vela/pretrain.sh` script from the ROOT directory. Tune the following cmd to set resources for your run

```
bsub -q normal -G grp_ebm -M 2000G -hl -n 4 -J energy-v1 -gpu "num=8/task:mode=exclusive_process" -oo /proj/dmfexp/energy-gpt/logs/energy-train-%J.out -eo /proj/dmfexp/energy-gpt/logs/energy-train-%J.err blaunch bash scripts/blue-vela/pretrain.sh configs/
```

- **`-q normal`**: LSF queue: (grp_ebm, grp_preemptable)
- **`-G grp_ebm`**: LSF accounting (grp_ebm, grp_preemptable)
- **`-M 2000G`**: memory limit/request (here 2000 GiB).
- **`-hl`**: host-lock / exclusive host allocation for the job (prevents sharing hosts).
- **`-n 4`**: number of nodes to request.
- **`-J energy-v1`**: job name (shows up in `bjobs`/`bhist`).
- **`-gpu "num=8/task:mode=exclusive_process"`**: request GPUs; here 8 total, with exclusive-process mode per task.
- **`-oo ...energy-train-%J.out`**: stdout log path (`%J` expands to the LSF job id).
- **`-eo ...energy-train-%J.err`**: stderr log path (`%J` expands to the LSF job id).
- **`blaunch ...`**: launcher wrapper used on Blue Vela to start the actual command on allocated resources.
- **`scripts/blue-vela/pretrain.sh configs/`**: training entrypoint script + config directory (adjust to point at a specific config file/dir).

## launch-scripts workflow

```bash
# Submit with default 32-GPU config
REPO_ROOT=$(git rev-parse --show-toplevel) bsub < launch-scripts/train_energy.sh

# Or specify a config explicitly
REPO_ROOT=$(git rev-parse --show-toplevel) CONFIG=configs/energy/other.yml bsub < launch-scripts/train_energy.sh
```

Monitor:
```bash
bjobs                                              # check job status
tail -f /proj/dmfexp/energy-gpt/logs/energy-train-*.out  # watch logs
```