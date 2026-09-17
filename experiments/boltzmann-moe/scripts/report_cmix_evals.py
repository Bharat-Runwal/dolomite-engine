"""Report Avg11 / MMLU / GSM8K-CoT / wiki-ppl for every evaluated cmix arm.

Two bugs this exists to prevent, both of which already happened once:
  1. `flexible or strict` -- a genuine 0.0 flexible-extract score is FALSY, so the `or`
     silently reported strict-match instead. Read the key explicitly with `is not None`.
  2. Hand-aligning a markdown row -- dropping one cell shifted wiki_ppl 40.58 into the
     GSM8K column. This prints the table itself.
GSM8K-CoT is reported as exact_match,flexible-extract, which is what the colleague's harness
uses (their gsm8k-cot.yaml filters are byte-identical to site-packages lm_eval 0.4.11; the only
diff is dataset_path openai/gsm8k vs gsm8k).
"""
import json, glob, os, yaml

ACC_NORM = ["arc_challenge","arc_easy","hellaswag","openbookqa","piqa","sciq"]
ACC      = ["boolq","copa","winogrande","race","lambada_openai"]

def avg11(res):
    vals, miss = [], []
    for t in ACC_NORM:
        v = res.get(t,{}).get("acc_norm,none");  vals.append(v) if v is not None else miss.append(t)
    for t in ACC:
        v = res.get(t,{}).get("acc,none");       vals.append(v) if v is not None else miss.append(t)
    return (100*sum(vals)/len(vals) if vals else None), len(vals), miss

def get(res, task, key):
    v = res.get(task, {}).get(key)
    return v                                    # never `or` -- 0.0 is a valid score

rows = []
for cfg in sorted(glob.glob('configs/cmix/cmix*.yml')):
    n = os.path.basename(cfg)[:-4]
    try: c = yaml.safe_load(open(cfg))
    except Exception: continue
    sp = (c.get('save_args') or {}).get('save_path')
    if not sp: continue
    tp = c['training_parameters']
    for r in sorted(glob.glob(f'{sp}/unsharded*/harness_results*.json')):
        try: res = json.load(open(r))["results"]
        except Exception: continue
        a, nn, miss = avg11(res)
        step = os.path.basename(os.path.dirname(r)).replace('unsharded','').lstrip('_') or 'final'
        rows.append(dict(arm=n, step=step, steps=tp['num_training_steps'],
                         mbs=tp['micro_batch_size'], ga=tp['gradient_accumulation_steps'],
                         avg11=a, n=nn, miss=miss,
                         mmlu=get(res,'mmlu','acc,none'),
                         gsm_flex=get(res,'gsm8k_cot','exact_match,flexible-extract'),
                         gsm_strict=get(res,'gsm8k_cot','exact_match,strict-match'),
                         wiki=get(res,'wikitext','word_perplexity,none')))

f = lambda v, p=2, s=1: "n/a" if v is None else f"{s*v:.{p}f}"
hdr = ("arm","ckpt","Avg11","n","MMLU","GSM8K flex","GSM8K strict","wiki ppl")
w   = (34,10,7,5,7,11,13,9)
print("".join(h.ljust(x) for h,x in zip(hdr,w)))
print("".join("-"*(x-1)+" " for x in w))
for r in rows:
    print("".join(str(c).ljust(x) for c,x in zip(
        (r['arm'], r['step'], f(r['avg11']), f"{r['n']}/11",
         f(r['mmlu'],2,100), f(r['gsm_flex'],2,100), f(r['gsm_strict'],2,100), f(r['wiki'])), w)))
    if r['miss']: print(f"    INCOMPLETE Avg11 -- missing: {','.join(r['miss'])}")

# GPU count is NOT in the config. Do NOT assume one -- an earlier version hardcoded 4 GPUs and
# mislabelled the 1B arm's 32.0B budget as 16.0B. Resolve it from the launch ledger, else
# EMPIRICALLY from the log (billion_tokens_per_day * 1e9 * step_time / 86400), else say unknown.
import csv, re, statistics
ledger = {}
lp = 'experiments/boltzmann-moe/logs/launch_ledger.tsv'
if os.path.exists(lp):
    for row in csv.DictReader(open(lp), delimiter='\t'):
        try: ledger[row['name']] = int(row['gpus'])
        except Exception: pass

def empirical_tps(arm):
    for lg in sorted(glob.glob(f'/u/ndehmamy/bsub_logs/{arm}_*.stderr'),
                     key=os.path.getmtime, reverse=True):
        t = open(lg, errors='ignore').read()
        pts = re.findall(r'train-billion_tokens_per_day = ([0-9.]+).*?train-step_time \(sec\) = ([0-9.]+)', t)
        if len(pts) >= 10:
            return statistics.median(float(b)*1e9*float(s)/86400 for b, s in pts[-40:])
    return None

print("\ntoken budgets (GPUs from the launch ledger, else inferred empirically from the log):")
seen, unknown = {}, []
for r in {x['arm']: x for x in rows}.values():          # one row per arm
    tps, src = None, ''
    if r['arm'] in ledger:
        tps, src = ledger[r['arm']]*r['mbs']*r['ga']*4096, f"ledger({ledger[r['arm']]}gpu)"
    else:
        e = empirical_tps(r['arm'])
        if e: tps, src = round(e/1024)*1024, f"empirical(~{round(e/(r['mbs']*r['ga']*4096))}gpu)"
    if tps is None:
        unknown.append(r['arm']); print(f"  {r['arm']:34s} {'UNKNOWN GPUs -- budget not verifiable':>44s}"); continue
    seen.setdefault(tps*r['steps'], []).append(r['arm'])
    print(f"  {r['arm']:34s} {tps:>9,} tok/step x {r['steps']:>6} = {tps*r['steps']/1e9:>5.1f}B   [{src}]")
if len(seen) > 1:
    print("  *** NOT ISO-TOKEN: budgets differ across arms above -- do not compare directly ***")
    for tot, arms in sorted(seen.items()):
        print(f"      {tot/1e9:5.1f}B : {', '.join(arms)}")
if unknown:
    print(f"  *** GPU count unresolved for: {', '.join(unknown)} -- launch via submit_train.sh so the ledger records it ***")
