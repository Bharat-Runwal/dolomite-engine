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

for f in sorted(glob.glob('configs/cmix/cmix*.yml')):
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
    have = sorted(int(m.group(1)) for d in glob.glob(f'{sp}/global_step*')
                  if (m := re.search(r'global_step(\d+)$', d)))
    if not have: continue
    mdir = f'{sp}/milestones'
    for tgt in targets:
        want = tgt / tps                                  # step at which the milestone is crossed
        # the newest checkpoint at or after the milestone that we actually have
        cand = [s for s in have if s >= want]
        if not cand: continue
        s = min(cand)
        tok = s * tps
        dst = f'{mdir}/tok{tgt/1e9:.0f}B_step{s}_actual{tok/1e9:.2f}B'
        if os.path.exists(dst): continue
        os.makedirs(mdir, exist_ok=True)
        r = subprocess.run(['cp','-al',f'{sp}/global_step{s}',dst], capture_output=True, text=True)
        if r.returncode == 0:
            print(f'  {arm}: milestone {tgt/1e9:.0f}B -> step {s} ({tok/1e9:.2f}B actual) hard-linked')
        else:
            print(f'  {arm}: FAILED to link step {s}: {r.stderr.strip()[:90]}')
PY
