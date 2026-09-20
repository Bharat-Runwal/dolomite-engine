#!/usr/bin/env python3
"""Health check for the 8 cmix paper arms.

Reports ONLY problems / milestones; prints nothing actionable when nominal.

WHY THE STALL CHECK EXISTS (2026-09-18): the earlier inline version alerted only on
"no live job". LSF preemption REQUEUES on the SAME jobid, so a preempted arm goes
back to PEND and still looks live -- both 400M arms (the critical path) sat PEND for
~20 min and the check said QUIET. The user noticed before the monitoring did.
So: RUN is the only healthy state, and a step counter that does not advance between
cycles is reported even while RUN (a wedged job also holds RUN).
"""
import subprocess, re, os, json, statistics, time, yaml

REPO = '/proj/dmfexp/nima/Code/dolomite-engine'
# EVERY arm the paper depends on must be listed here. On 2026-09-20 five arms were found dead at
# once and a SIXTH -- abl_B_400M_6G1x6S, the FLOP-matched Switch baseline at 400M, i.e. the arm
# HANDOFF 14.1's headline comparison requires -- stayed dead through the first relaunch sweep
# purely because it was absent from this list. None of these arms is in watchdog_jobs.conf either,
# so this check is the ONLY thing that notices. Add new arms here the moment they are launched.
ARMS = ['cmix_134M_hybrid_32B_sparse','cmix_134M_sandwich_32B_sparse','cmix_134M_pure_32B_sparse',
        'cmix_134M_gptswitch_32B','cmix_400M_hybrid_sparse','cmix_400M_sandwich_sparse',
        'cmix_400M_baseline_switch','cmix1B_12L_gptDense_32B',
        'cmix_134M_hyb_w1w2_surrMLP_32B','cmix_134M_hyb_w1w2_sparse_surr_32B',
        'abl_B_134M_6G1x6S','abl_B_400M_6G1x6S','abl_C_134M_1G1x6E1G_isototal',
        'abl_D_134M_6G_dense_isototal','abl_E_134M_6G1x6E_baseEGPT']

# Arms no longer live under one directory: the ablations are in configs/iclr_26/ablations/.
# Hardcoding configs/cmix/ is what made the abl_* arms unlistable above.
CFG_DIRS = ['configs/cmix', 'configs/iclr_26/ablations', 'configs/iclr_26']

def cfg_path(name):
    for d in CFG_DIRS:
        c = f'{REPO}/{d}/{name}.yml'
        if os.path.exists(c):
            return c
    raise FileNotFoundError(f'no config for arm {name!r} under {CFG_DIRS}')
STATE = '/tmp/hc_state.json'
SEEN  = '/tmp/hc_reported.txt'

def bj(name, field):
    r = subprocess.run(['bjobs','-noheader','-o',field,'-J',name], capture_output=True, text=True)
    return r.stdout.strip().split('\n')[0].strip()

st_prev = json.load(open(STATE)) if os.path.exists(STATE) else {}
seen = set(open(SEEN).read().split()) if os.path.exists(SEEN) else set()
now = time.time()
st_new, alerts, miles, ratios = {}, [], [], []

for n in ARMS:
    jid, stat = bj(n,'jobid'), (bj(n,'stat') or 'GONE')
    if not jid:
        # A FINISHED arm has no live job either. Reporting that as "relaunch" would restart a
        # completed 32B run (cmix_134M_sandwich_32B_sparse, 2026-09-18) -- which at best wastes
        # GPUs and at worst resumes past the end of the LR schedule. Check completion FIRST.
        try:
            cfg = yaml.safe_load(open(cfg_path(n)))
            tot = cfg['training_parameters']['num_training_steps']
            sp = (cfg.get('save_args') or {}).get('save_path')
            it = json.load(open(f'{sp}/latest_checkpointed_iteration.json'))['latest_checkpointed_iteration']
            if it >= tot:
                alerts.append(f"{n}: COMPLETE at {it:,}/{tot:,} steps -- do NOT relaunch")
                continue
            alerts.append(f"{n}: NO LIVE JOB at {it:,}/{tot:,} -- relaunch")
        except Exception:
            alerts.append(f"{n}: NO LIVE JOB (completion unknown) -- check before relaunching")
        continue
    f = f'/u/ndehmamy/bsub_logs/{n}_{jid}.stderr'
    txt = open(f, errors='ignore').read() if os.path.exists(f) else ''
    pts = [(int(s), float(x)) for s, x in
           re.findall(r'step = (\d+),.*?train-step_time \(sec\) = ([0-9.]+)', txt)]
    step = pts[-1][0] if pts else 0
    st_new[n] = {'jobid': jid, 'stat': stat, 'step': step, 't': now}
    prev = st_prev.get(n, {})

    # --- not-RUN: preempted/requeued arms are PEND but still "live"
    if stat != 'RUN':
        since = prev.get('not_run_since') or now
        st_new[n]['not_run_since'] = since
        mins = (now - since) / 60
        key = f'{n}@{stat}@{jid}'
        if key not in seen or mins > 30:
            alerts.append(f"{n}: STAT={stat} (not training) for {mins:.0f} min, step {step:,}")
            seen.add(key)
    else:
        if prev.get('stat') and prev['stat'] != 'RUN':
            alerts.append(f"{n}: recovered to RUN at step {step:,}")
        # --- wedged while RUN: step counter frozen across cycles
        if prev.get('stat') == 'RUN' and prev.get('step') == step and step > 0:
            mins = (now - prev.get('t', now)) / 60
            if mins > 20:
                alerts.append(f"{n}: RUN but step frozen at {step:,} for {mins:.0f} min")

    if not txt: continue
    oom = len(re.findall('Tried to allocate', txt))
    err = len(re.findall('IBV_WC|ncclRemoteError|InductorError', txt))
    if oom and f'{n}@oom{oom}' not in seen:
        alerts.append(f"{n}: OOM x{oom} {re.findall(r'Tried to allocate [0-9.]+ [GM]iB', txt)[:1]}")
        seen.add(f'{n}@oom{oom}')
    if err and f'{n}@err{err}' not in seen:
        alerts.append(f"{n}: err x{err} {sorted(set(re.findall(r'IBV_WC_[A-Z_]+|ncclRemoteError|InductorError', txt)))[:2]}")
        seen.add(f'{n}@err{err}')
    if not pts: continue

    mk = step // 20000 * 20000
    if mk >= 20000 and f'{n}@{mk}' not in seen:
        miles.append(f"{n}: passed {mk:,} (now {step:,}, {stat})"); seen.add(f'{n}@{mk}')

    cfg = yaml.safe_load(open(cfg_path(n)))
    eb = [b for b in cfg['model_args']['pretrained_config']['mlp_blocks']
          if b['mlp_type'].startswith('EnergyFF')]
    if not eb: continue
    b = eb[0]; sss = b.get('sparse_start_step', 0)
    K = b['n_experts']; Ie = b['intermediate_size'] // K
    pc = (b.get('sparse_candidates') or b['top_k']) + (b.get('sparse_explore') or 0)
    if sss and step > sss + 200 and f'{n}@ratio' not in seen:
        d = [x for s, x in pts if 30 <= s <= sss]; sp = [x for s, x in pts if s >= sss + 150]
        if len(d) >= 15 and len(sp) >= 15:
            md, ms = statistics.median(d), statistics.median(sp)
            ratios.append((n, K, Ie, pc, md, ms, md/ms, K/pc)); seen.add(f'{n}@ratio')

# ---- FAIRSHARE WATCH -------------------------------------------------------------
# LSF dynamic priority = shares / (0.7*cpu_h + 0.7*run_h + 3*started + 1*gpu_run_h).
# HIST_HOURS=5 so only the cpu_time term decays (5 h half-life); GPU_RUN_TIME and STARTED
# fall only when our jobs stop. Being PREEMPTED moves running time into cpu_time, so
# preemption itself lowers our priority -- worth watching, not just inferring.
try:
    out = subprocess.run(['bqueues','-l','preemptable'], capture_output=True, text=True).stdout
    row = [l.split() for l in out.splitlines() if l.strip().startswith('ndehmamy')]
    if row:
        r = row[0]
        prio = float(r[2]); started = int(r[3]); cpu_h = float(r[5])/3600
        run_h = float(r[6])/3600; gpu_h = float(r[8])/3600
        prev_p = st_prev.get('_fairshare', {}).get('prio')
        st_new['_fairshare'] = dict(prio=prio, started=started, cpu_h=cpu_h, gpu_h=gpu_h, t=now)
        if prev_p is None or abs(prio - prev_p) / max(prev_p, 1e-9) > 0.25:
            arrow = '' if prev_p is None else f" (was {prev_p:.4f}, {'UP' if prio > prev_p else 'DOWN'})"
            alerts.append(f"fairshare priority {prio:.4f}{arrow} | started={started} "
                          f"gpu_run={gpu_h:.0f}h cpu_hist={cpu_h:.0f}h  "
                          f"[fresh user = 0.333]")
except Exception:
    pass

json.dump(st_new, open(STATE,'w')); open(SEEN,'w').write('\n'.join(sorted(seen)))
if ratios:
    print("NEW SPARSE-SWITCH RATIO:")
    for n,K,Ie,pc,md,ms,r,ceil in ratios:
        print(f"  {n:32s} K={K:<3d} I_e={Ie:<7,} {md:.4f}s -> {ms:.4f}s = {r:.3f}x  (ceiling {ceil:.1f}x, {100*r/ceil:.0f}%)")
if miles:  print("MILESTONES:");  [print(f"  {m}") for m in miles]
if alerts: print("ALERTS:");      [print(f"  {a}") for a in alerts]
if not (ratios or miles or alerts): print("__QUIET__")
