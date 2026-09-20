#!/bin/bash
# Preserve checkpoints at fixed TOKEN milestones so iso-token evaluation is always possible.
#
# WHY: max_to_keep is 2, so intermediate checkpoints are pruned within minutes of being written.
# The only reason the iso-token 134M baseline was recoverable at all is that ONE arm accidentally
# had max_to_keep: 95 (and cost 136 GB for it). This makes that capability deliberate and free.
#
# HOW: `cp -al` builds a directory tree of HARD LINKS. The snapshot shares inodes with the
# original, so it costs ~0 bytes, and it SURVIVES deletion of the original by the pruner.
#
# NAMING: dirs are named by the ACTUAL token count reached, not the nominal target, because
# save_interval rarely divides a round token milestone. Arms that share tokens/step and
# save_interval land on the SAME steps, which is what iso-token comparison actually needs --
# the number being exactly 8.000B is not.
set -u
REPO=/proj/dmfexp/nima/Code/dolomite-engine
cd "$REPO" || exit 1
TARGETS_B="${TARGETS_B:-8 16 24 32}"        # token milestones in billions
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate 2>/dev/null
export PYTHONPATH=$REPO:${PYTHONPATH:-}

python3 - "$TARGETS_B" <<'PY'
import yaml, glob, os, sys, csv, subprocess, re, statistics
targets = [float(x)*1e9 for x in sys.argv[1].split()]

ledger = {}
lp = 'experiments/boltzmann-moe/logs/launch_ledger.tsv'
if os.path.exists(lp):
    for row in csv.DictReader(open(lp), delimiter='\t'):
        try: ledger[row['name']] = int(row['gpus'])
        except Exception: pass

def gpus_for(arm, mbs, ga):
    if arm in ledger: return ledger[arm]
    for lg in sorted(glob.glob(f'/u/ndehmamy/bsub_logs/{arm}_*.stderr'), key=os.path.getmtime, reverse=True):
        t = open(lg, errors='ignore').read()
        p = re.findall(r'train-billion_tokens_per_day = ([0-9.]+).*?train-step_time \(sec\) = ([0-9.]+)', t)
        if len(p) >= 10:
            tps = statistics.median(float(b)*1e9*float(s)/86400 for b, s in p[-40:])
            return round(tps/(mbs*ga*4096))
    return None

# GLOB WIDENED 2026-09-20: the abl_* arms live in configs/iclr_26/ablations/ and were getting NO
# milestone anchors at all -- the same omission that made auto_eval skip abl_B and made the health
# check miss abl_B_400M. Any new config tree must be added here too.
for f in sorted(glob.glob('configs/cmix/cmix*.yml')
                + glob.glob('configs/iclr_26/**/*.yml', recursive=True)):
    arm = os.path.basename(f)[:-4]
    if any(t in arm for t in ('bsprobe','feas','fixtest','tcptest','bisect','smoke','spmddiag')):
        continue
    try: c = yaml.safe_load(open(f))
    except Exception: continue
    sp = (c.get('save_args') or {}).get('save_path')
    if not sp or not os.path.isdir(sp): continue
    tp = c['training_parameters']
    g = gpus_for(arm, tp['micro_batch_size'], tp['gradient_accumulation_steps'])
    if not g: continue
    tps = g * tp['micro_batch_size'] * tp['gradient_accumulation_steps'] * 4096
    # one save_interval is the largest legitimate gap between a milestone and its checkpoint
    save_interval = int((c.get('save_args') or {}).get('save_interval') or 0) or 2000
    have = sorted(int(m.group(1)) for d in glob.glob(f'{sp}/global_step*')
                  if (m := re.search(r'global_step(\d+)$', d)))
    if not have: continue
    mdir = f'{sp}/milestones'
    for tgt in targets:
        # FLOOR, not exact: the final milestone is unreachable otherwise. 32e9/262144 =
        # 122070.3125 but the 134M arms END at step 122070, so `s >= 122070.3125` never matched
        # and the 32B anchor was silently never linked (found 2026-09-18). Tolerate being within
        # one save_interval-free step of the target: the run cannot produce a later checkpoint.
        want = int(tgt / tps)                             # step at which the milestone is crossed
        # Dedup on the MILESTONE PREFIX, not the full name. max_to_keep prunes the
        # earliest qualifying checkpoint, so min(cand) DRIFTS UPWARD on later sweeps and a
        # full-name check re-links the same milestone at a later step every cycle
        # (observed 2026-09-17: gptswitch got both tok8B_step32000 and tok8B_step36000).
        # Once a milestone is captured it is frozen.
        if glob.glob(f'{mdir}/tok{tgt/1e9:.0f}B_*'): continue
        # earliest checkpoint at or after the milestone that is STILL on disk
        cand = [s for s in have if s >= want]
        if not cand: continue
        s = min(cand)
        tok = s * tps
        # ---- TOLERANCE GUARD (2026-09-20). WITHOUT THIS THE ANCHOR NAME LIES. ----------------
        # `have` only holds the checkpoints STILL ON DISK, and max_to_keep: 2 prunes everything
        # else, so once an arm runs well past a milestone `min(cand)` is not "the checkpoint at
        # the milestone" -- it is simply the oldest surviving checkpoint. Before this guard the
        # script hard-linked `tok8B_step122000_actual31.98B` and
        # `tok16B_step57000_actual29.88B`: directories NAMED for 8B/16B holding 32B/30B weights.
        # Anything evaluating milestones/tok8B_* would then report a 32B number as an 8B number,
        # which is precisely the class of error that already put a wiki-ppl in a GSM8K column.
        # An unpruned checkpoint can be at most ONE save_interval past the crossing step, so
        # anything beyond that means the real milestone is GONE. Say so; do not invent an anchor.
        if s - want > max(save_interval, 1):
            print(f'  {arm}: milestone {tgt/1e9:.0f}B MISSED -- earliest surviving ckpt is step '
                  f'{s} ({tok/1e9:.2f}B), {s-want} steps past the crossing step {want} '
                  f'(save_interval {save_interval}). Checkpoint was pruned; NOT linking a '
                  f'mislabelled anchor.')
            continue
        dst = f'{mdir}/tok{tgt/1e9:.0f}B_step{s}_actual{tok/1e9:.2f}B'
        os.makedirs(mdir, exist_ok=True)
        r = subprocess.run(['cp','-al',f'{sp}/global_step{s}',dst], capture_output=True, text=True)
        if r.returncode == 0:
            print(f'  {arm}: milestone {tgt/1e9:.0f}B -> step {s} ({tok/1e9:.2f}B actual) hard-linked')
        else:
            print(f'  {arm}: FAILED to link step {s}: {r.stderr.strip()[:90]}')
PY
