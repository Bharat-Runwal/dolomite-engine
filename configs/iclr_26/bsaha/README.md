# For bsaha — lower-priority long runs, all RESUMABLE, all fine on preemptable

These four are real paper arms that we deprioritised for GPU reasons, not scientific ones. Your
fairshare is better than ours (ours is **0.0030**, against ~0.333 for a fresh user), so these will
actually schedule for you.

**All four have checkpoints and resume automatically** — `submit_train.sh` resolves `load_args` at
run time, so a preemption costs only the steps since the last save.

| config | GPUs | resumes at | target | what it is |
|---|---|---|---|---|
| `abl_G_400M_6G1x6E1x6E_4gpu.yml` | **4** | 8,200 | 61,035 | TWO recurrent energy blocks (12 energy applications/token). **MUST be 4 GPUs single-node** — see below. ~8.5 s/step, so 32B needs ~126 h |
| `abl_G_400M_6G1x6S1x6S.yml` | 8 | 21,400 | 61,035 | its FLOP-matched Switch twin, ~3.1 s/step |
| `abl_C_134M_1G1x6E1G_isototal.yml` | 4 | 60,000 | 122,070 | the TRUE 134M sandwich (`1G1x6E1G`), iso-total |
| `abl_G_400M_6G1x6E1x6E.yml` | 8 | — | — | the 8-GPU variant. **Do not use** unless a whole host is free; kept for reference |

## How to launch

```bash
cd /proj/dmfexp/nima/Code/dolomite-engine
export WANDB__SERVICE_WAIT=300          # the 30 s default has killed a 400M arm at rank 0
bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \
     abl_G_400M_6G1x6E1x6E_4gpu configs/iclr_26/bsaha/abl_G_400M_6G1x6E1x6E_4gpu.yml \
     4 preemptable 24:00 160G
# 8-GPU arms need the node shape forced to 2x4 (a single host with 8 free GPUs rarely places here):
bash experiments/boltzmann-moe/scripts/bsub/submit_train.sh \
     abl_G_400M_6G1x6S1x6S configs/iclr_26/bsaha/abl_G_400M_6G1x6S1x6S.yml \
     8 preemptable 24:00 160G 4
```

## Four things that will otherwise cost you a run

1. **`abl_G_400M_6G1x6E1x6E` MUST be single-node.** It carries `sinkhorn_persist_mu: true` in BOTH
   energy blocks, and persist_mu's data-dependent `.item()` inside the compiled region **wedges on
   multi-node**: job 1798098 compiled fine, then sat at 0 steps for 34 minutes with LSF still
   reporting RUN and flat CPU. The `_4gpu` file is `mbs 2 x ga 16` so 4 GPUs still give the correct
   524,288 tok/step.
2. **GPU count is NOT in the config** and it sets the token budget:
   `tokens/step = GPUS x mbs x ga x sequence_length`. Use the count in the table. A wrong count
   silently halves or doubles the budget — that happened to us three times in one session.
3. **Step lines go to STDERR**, not stdout. `tail -3 $HOME/bsub_logs/<name>_<jobid>.stderr`.
   stdout holds only the launcher echo, `ninja:` and the NCCL banner, so grepping it looks exactly
   like a silent hang.
4. **Sub-60-second deaths are transient, ~50% of multi-node starts.** `gloo ... Connection closed by
   peer`, `lsb_launch(): Failed`, or a wandb `ServiceStartTimeoutError` all mean *resubmit*, not
   *debug*. Only investigate if the same arm fails the same way three times. And a genuinely wedged
   job ignores plain `bkill` — use `bkill -r`.
