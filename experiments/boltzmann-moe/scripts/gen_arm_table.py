#!/usr/bin/env python3
"""Generate the paper's arm/results table from CONFIGS + STORED EVAL FILES.

Never hand-assemble these rows: a dropped cell once reported wiki-ppl 40.58 as a GSM8K score,
and `flexible or strict` silently reported strict-match because a genuine 0.0 flexible score is
falsy. Everything here is read positionally from the json by explicit key.

Columns: arch notation, expert kind, K, k, I_e, TOTAL params, ACTIVE params/token, then
Avg11 / wiki-ppl / GSM8K-flex / MMLU at whichever token anchor is asked for.

Architecture notation: G = GPT block (softmax attn + dense MLP), S = Switch MoE block,
E = energy block (energy attn + Boltzmann MoE). "1x6E" means ONE energy block applied 6 times
(recurrence), which is NOT the same parameter count as 6 distinct energy blocks.

Usage: python scripts/gen_arm_table.py [--tokens 8|16|24|32|final] [--latex]
"""
import yaml, glob, json, os, sys, argparse, math

REPO = '/proj/dmfexp/nima/Code/dolomite-engine'
EXP = f'{REPO}/experiments/boltzmann-moe'
ACC_NORM = ['arc_challenge','arc_easy','hellaswag','openbookqa','piqa','sciq']
ACC = ['boolq','copa','winogrande','race','lambada_openai']

def newest(pat):
    fs = glob.glob(pat)
    return max(fs, key=os.path.getmtime) if fs else None

def arch(pc):
    """Compress blocks into e.g. 6G+1x6E, 1G+1x4E+1G, 8G+4E, 6G+1S."""
    li = pc.get('layer_iterations') or [1]*len(pc['mlp_blocks'])
    toks = []
    for sm, m, it in zip(pc['sequence_mixer_blocks'], pc['mlp_blocks'], li):
        t = m.get('mlp_type')
        c = 'E' if str(t).startswith('EnergyFF') else ('S' if t == 'MoE' else 'G')
        toks.append((c, int(it)))
    out, i = [], 0
    while i < len(toks):
        c, it = toks[i]; n = 1
        while i+n < len(toks) and toks[i+n] == (c, it): n += 1
        out.append(f"{n}{c}" if it == 1 else f"{n}x{it}{c}")
        i += n
    # Structural name, e.g. 6G1x6E / 5G1x6E1G / 1G1x4E1G / 6G1S. NO separators, because the
    # arm NAMES lie: cmix_134M_sandwich_* is 5G1x6E1G (an energy block in position 6 of 7),
    # not the intended 1G1x6E1G, and every other "sandwich" in the repo is 1G1x4E1G. Report the
    # structure, never the nickname (user instruction 2026-09-18).
    return "".join(out)

def params(pc):
    """(total, active) parameters. hopfield stores ONE matrix per expert, swiglu THREE."""
    d, V = pc['hidden_size'], pc['vocab_size']
    tot = act = V*d                      # tied embeddings
    K = k = Ie = None; kind = 'dense'
    li = pc.get('layer_iterations') or [1]*len(pc['mlp_blocks'])
    for sm, m in zip(pc['sequence_mixer_blocks'], pc['mlp_blocks']):
        tot += 4*d*d; act += 4*d*d       # attention, counted once per DISTINCT block
        t = m.get('mlp_type')
        if t == 'MLP':
            p = 2*d*m['intermediate_size']; tot += p; act += p
        elif t == 'MoE':
            K, k, Ie = m['num_experts'], m['num_experts_per_tok'], m['intermediate_size']
            per = 3*d*Ie; tot += K*per + d*K; act += k*per; kind = 'swiglu'
        elif str(t).startswith('EnergyFF'):
            K, k = m['n_experts'], m['top_k']; Ie = m['intermediate_size']//K
            per = d*Ie; tot += K*per; act += k*per
            kind = m.get('expert_kind', 'hopfield')
    return tot, act, kind, K, k, Ie

def metrics(sp, want):
    """want: '8','16','24' -> milestone anchor; 'final' -> end-of-run eval."""
    if want == 'final':
        f = newest(f'{sp}/unsharded*/harness_results*.json')
    else:
        cands = glob.glob(f'{sp}/milestones/unsharded_tok{want}B_step*/harness_results_merged_*.json') \
             or glob.glob(f'{sp}/milestones/unsharded_tok{want}B_step*/harness_results_2*.json')
        f = max(cands, key=os.path.getmtime) if cands else None
    if not f: return None
    r = json.load(open(f)).get('results', {})
    def m(task, *keys):
        d = r.get(task) or {}
        for kk in keys:
            if kk in d and d[kk] is not None: return d[kk]
        for kk in d:                              # prefix match, e.g. exact_match,flexible-extract
            if any(kk.startswith(x) for x in keys): return d[kk]
        return None
    vals = [m(t,'acc_norm,none','acc_norm') for t in ACC_NORM] + [m(t,'acc,none','acc') for t in ACC]
    ok = [v for v in vals if isinstance(v,(int,float))]
    avg = 100*sum(ok)/len(ok) if len(ok) == 11 else None
    return dict(avg11=avg, complete=len(ok)==11,
                mmlu=(lambda v: 100*v if isinstance(v,(int,float)) else None)(m('mmlu','acc,none','acc')),
                gsm=(lambda v: 100*v if isinstance(v,(int,float)) else None)(m('gsm8k_cot','exact_match,flexible-extract')),
                ppl=m('wikitext','word_perplexity,none','word_perplexity'), src=os.path.basename(f))

ap = argparse.ArgumentParser(); ap.add_argument('--tokens', default='16'); ap.add_argument('--latex', action='store_true')
a = ap.parse_args()
rows = []
for f in sorted(glob.glob(f'{REPO}/configs/cmix/cmix*.yml')):
    n = os.path.basename(f)[:-4]
    if any(t in n for t in ('4gpu','REFERENCE','HANDOFF','probe','cal','diag','feas','fixtest','tcptest','bisect','smoke','spmddiag')):
        continue
    try: c = yaml.safe_load(open(f))
    except Exception: continue
    pc = c['model_args']['pretrained_config']; sp = (c.get('save_args') or {}).get('save_path')
    if not sp: continue
    tot, act, kind, K, k, Ie = params(pc)
    mt = metrics(sp, a.tokens)
    if mt is None: continue
    rows.append(dict(name=n, arch=arch(pc), kind=kind, K=K, k=k, Ie=Ie,
                     tot=tot/1e6, act=act/1e6, **mt))
rows.sort(key=lambda r: -(r['avg11'] or 0))
fmt = lambda v, p=2: ('--' if v is None else f'{v:.{p}f}')
if not a.latex:
    print(f"\n=== anchor: {a.tokens}B tokens" if a.tokens!='final' else "\n=== end-of-run")
    print(f"{'arm':30s} {'arch':16s} {'expert':9s} {'K':>3s} {'k':>2s} {'I_e':>7s} {'TOT':>7s} {'ACT':>7s} {'Avg11':>6s} {'ppl':>7s} {'GSM':>5s} {'MMLU':>5s}")
    for r in rows:
        print(f"{r['name']:30s} {r['arch']:16s} {r['kind']:9s} {str(r['K'] or '--'):>3s} {str(r['k'] or '--'):>2s} "
              f"{(format(r['Ie'], ',') if r['Ie'] else '--'):>7s} {r['tot']:6.0f}M {r['act']:6.0f}M "
              f"{fmt(r['avg11']):>6s} {fmt(r['ppl']):>7s} {fmt(r['gsm']):>5s} {fmt(r['mmlu']):>5s}")
else:
    print(r"% GENERATED by scripts/gen_arm_table.py --tokens " + a.tokens + r" --latex  -- do not hand-edit")
    print(r"\begin{tabular}{llrrrrrrrr}")
    print(r"\toprule")
    print(r"Arm & Arch & $K$ & $k$ & $I_e$ & Total & Active & Avg11 & Wiki ppl & GSM8K \\")
    print(r"\midrule")
    for r in rows:
        nm = r['name'].replace('_',r'\_')
        print(f"{nm} & {r['arch']} & {r['K'] or '--'} & {r['k'] or '--'} & "
              f"{r['Ie']:,} & {r['tot']:.0f}M & {r['act']:.0f}M & "
              f"{fmt(r['avg11'])} & {fmt(r['ppl'])} & {fmt(r['gsm'])} \\\\".replace(',','{,}'))
    print(r"\bottomrule"); print(r"\end{tabular}")
