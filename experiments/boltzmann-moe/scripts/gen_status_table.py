#!/usr/bin/env python3
"""One table of EVERY arm: architecture, parameter budget, training state, benchmark results.

Generated, never hand-assembled (CLAUDE.md: a dropped cell once reported wiki-ppl as a GSM8K
score). Emits a text table and a LaTeX longtable for the ICLR appendix.

Usage: python scripts/gen_status_table.py [--latex]
"""
import yaml, json, glob, os, re, subprocess, statistics, sys, argparse

REPO = '/proj/dmfexp/nima/Code/dolomite-engine'
L = '/u/ndehmamy/bsub_logs'
ACC_NORM = ['arc_challenge','arc_easy','hellaswag','openbookqa','piqa','sciq']
ACC = ['boolq','copa','winogrande','race','lambada_openai']

# arm -> (short label, role). ROLE marks what each arm is FOR.
ROLES = {
 'cmix_134M_hybrid_32B_sparse':        ('134M hybrid',            'main'),
 'cmix_134M_sandwich_32B_sparse':      ('134M block-pos variant', 'main'),
 'cmix_134M_pure_32B_sparse':          ('134M pure recurrent',    'main'),
 'cmix_134M_gptswitch_32B':            ('134M Switch (unmatched)','baseline'),
 'abl_B_134M_6G1x6S':                  ('134M Switch (FLOP-matched)','ABLATION B'),
 'cmix_134M_hyb_w1w2_surrMLP_32B':     ('134M w1w2', 'ABLATION: expert form'),
 'cmix_134M_hyb_w1w2_sparse_surr_32B': ('134M w1w2','ABLATION: expert form + router'),
 'cmix_400M_hybrid_sparse':            ('400M hybrid',            'main'),
 'cmix_400M_sandwich_sparse':          ('400M sandwich',          'main'),
 'cmix_400M_baseline_switch':          ('400M Switch (unmatched)','baseline'),
 'abl_B_400M_6G1x6S':                  ('400M Switch (FLOP-matched)','ABLATION B'),
 'cmix1B_12L_gptDense_32B':            ('1B stacked 8G4E',        'scale'),
 'abl_C_134M_1G1x6E1G_isototal':       ('134M true sandwich, iso-total','ABLATION C'),
 'abl_D_134M_6G_dense_isototal':       ('134M GPT-only dense, iso-total','ABLATION D'),
 'abl_E_134M_6G1x6E_baseEGPT':         ('134M base EGPT, iso-active','ABLATION E'),
}

# WHICH SPARSE MECHANISM -- DERIVED FROM THE CONFIG, NEVER HAND-LABELLED (2026-09-20).
# The paper had been writing "sparse" unqualified, which conflates two different contributions:
#   sparse(proxy)     the router's weight matrices are replaced by a rank-r subspace projection
#                     with a small output dim, so selection costs O(K d r) not O(K d I_e).
#                     hopfield ONLY -- for w1w2 the bilinear energy's terms cancel
#                     (|sum|/sum|term| = 0.0254 vs 1.000), so an m-row subsample needs m = I_e,
#                     where it costs MORE than the mixture it exists to cheapen.
#   sparse(surrogate) a small MLP head is KL-distilled to reproduce the RANKING of the exact
#                     energies and nominates p >= k candidates; the exact energy re-ranks to k.
#                     Works for both expert kinds, and is the only option for w1w2.
# Derived rather than typed because every hopfield arm happens to be proxy and every w1w2 arm
# surrogate -- so a hand-written label would silently encode that confound as if it were a choice.
def sparse_mech(pc):
    blocks = pc.get('mlp_blocks') or []
    for b in blocks:
        t = b.get('mlp_type')
        if t == 'MoE':
            return 'learned gate'
        if t and str(t).startswith('EnergyFF_') and ('BoltzmannMoE' in t):
            if b.get('sparse_forward'):
                return 'sparse(surrogate)' if b.get('surrogate_replaces_proxy') else 'sparse(proxy)'
            # dense: all K experts evaluated per token. If the head supplies the ROUTING WEIGHTS
            # (use_surrogate) that is a different thing again -- say so rather than just "dense".
            return 'dense, surrogate router' if b.get('use_surrogate') else 'dense'
        if t == 'EnergyFF_Hopfield' or t == 'EnergyFF_W1W2':
            return 'no MoE'
    return None

def cfgpath(a):
    for p in (f'{REPO}/configs/cmix/{a}.yml',):
        if os.path.exists(p): return p
    g = glob.glob(f'{REPO}/configs/iclr_26/**/{a}.yml', recursive=True)
    return g[0] if g else None

def arch(pc):
    li = pc.get('layer_iterations') or [1]*len(pc['mlp_blocks'])
    toks=[]
    for sm,m,it in zip(pc['sequence_mixer_blocks'],pc['mlp_blocks'],li):
        t=m.get('mlp_type'); ch='E' if str(t).startswith('EnergyFF') else ('S' if t=='MoE' else 'G')
        toks.append((ch,int(it)))
    out=[];i=0
    while i<len(toks):
        ch,it=toks[i];n=1
        while i+n<len(toks) and toks[i+n]==(ch,it): n+=1
        out.append(f"{n}{ch}" if it==1 else f"{n}x{it}{ch}"); i+=n
    return "".join(out)

def metrics(sp):
    """Newest COMPLETE result for the end-of-run checkpoint."""
    cands = glob.glob(f'{sp}/unsharded*/harness_results_merged_*.json') or \
            glob.glob(f'{sp}/unsharded*/harness_results_2*.json')
    if not cands: return None
    f = max(cands, key=os.path.getmtime)
    r = json.load(open(f)).get('results',{})
    def m(task,*keys):
        d=r.get(task) or {}
        for k in keys:
            if k in d and d[k] is not None: return d[k]
        for k in d:
            if any(k.startswith(x) for x in keys): return d[k]
        return None
    vals=[m(t,'acc_norm,none','acc_norm') for t in ACC_NORM]+[m(t,'acc,none','acc') for t in ACC]
    ok=[v for v in vals if isinstance(v,(int,float))]
    return dict(avg11=100*sum(ok)/len(ok) if len(ok)==11 else None,
                ppl=m('wikitext','word_perplexity,none','word_perplexity'),
                mmlu=(lambda v: 100*v if isinstance(v,(int,float)) else None)(m('mmlu','acc,none','acc')),
                gsm=(lambda v: 100*v if isinstance(v,(int,float)) else None)(m('gsm8k_cot','exact_match,flexible-extract')))

rows=[]
for a,(label,role) in ROLES.items():
    p=cfgpath(a)
    if not p: continue
    c=yaml.safe_load(open(p)); pc=c['model_args']['pretrained_config']; tp=c['training_parameters']
    sp=c['save_args']['save_path']; tot=tp['num_training_steps']
    try: it=json.load(open(f'{sp}/latest_checkpointed_iteration.json'))['latest_checkpointed_iteration']
    except Exception: it=0
    st=subprocess.run(['bjobs','-noheader','-o','stat','-J',a],capture_output=True,text=True).stdout.split('\n')[0].strip()
    state='complete' if it>=tot else (f'{100*it/tot:.0f}% ({st or "no job"})')
    try:
        sys.path.insert(0,REPO)
        from lm_engine.hf_models.modeling_utils.mlp_blocks.energy_ff_paramcount import audit_config
        au=audit_config(p); totp,act,fl=au.total/1e6,au.active/1e6,au.flop_weight/1e6
    except Exception: totp=act=fl=None
    eb=[m for m in pc['mlp_blocks'] if str(m.get('mlp_type','')).startswith('EnergyFF')]
    sb=[m for m in pc['mlp_blocks'] if m.get('mlp_type')=='MoE']
    # .get, not [] -- the non-MoE energy FFNs (EnergyFF_Hopfield / EnergyFF_W1W2, e.g.
    # abl_E) have no n_experts/top_k at all, and indexing raised KeyError on them.
    if eb:
        kind,K,k=eb[0].get('expert_kind'),eb[0].get('n_experts'),eb[0].get('top_k')
        # A non-MoE energy FFN carries no `expert_kind` -- the FORM is the mlp_type itself.
        # Leaving kind=None crashed the text renderer's %s formatting.
        if kind is None:
            _t=str(eb[0].get('mlp_type',''))
            kind='hopfield' if _t.endswith('Hopfield') else ('w1w2' if _t.endswith('W1W2') else '?')
    elif sb: kind,K,k='swiglu',sb[0]['num_experts'],sb[0]['num_experts_per_tok']
    else:   kind,K,k='dense',None,None
    mt=metrics(sp) or {}
    _m = sparse_mech(pc)
    if _m:
        label = f"{label}, {_m}"
    rows.append(dict(arm=a,label=label,role=role,arch=arch(pc),kind=kind,K=K,k=k,
                     tot=totp,act=act,fl=fl,state=state,**mt))

ap=argparse.ArgumentParser(); ap.add_argument('--latex',action='store_true'); args=ap.parse_args()
f=lambda v,p=2: '--' if v is None else f'{v:.{p}f}'
if not args.latex:
    print(f"{'label':30s} {'arch':12s} {'expert':9s} {'K':>3s} {'k':>2s} {'TOT':>7s} {'ACT':>7s} "
          f"{'FLOPwt':>7s} {'state':>14s} {'Avg11':>6s} {'ppl':>7s} {'MMLU':>6s} {'GSM':>5s}  role")
    for r in sorted(rows,key=lambda r:(r['role']!='main',r['label'])):
        print(f"{r['label']:30s} {r['arch']:12s} {r['kind']:9s} {str(r['K'] or '-'):>3s} {str(r['k'] or '-'):>2s} "
              f"{f(r['tot'],1):>7s} {f(r['act'],1):>7s} {f(r['fl'],1):>7s} {r['state']:>14s} "
              f"{f(r.get('avg11')):>6s} {f(r.get('ppl')):>7s} {f(r.get('mmlu')):>6s} {f(r.get('gsm')):>5s}  {r['role']}")
else:
    print(r"% GENERATED by experiments/boltzmann-moe/scripts/gen_status_table.py --latex")
    print(r"\begin{tabular}{llrrrrrrrr}")
    print(r"\toprule")
    print(r"Arm & Arch & $K$ & $k$ & Total & Active & Progress & Avg11 & ppl & GSM8K \\")
    print(r"\midrule")
    last=None
    for r in sorted(rows,key=lambda r:(r['role']!='main',r['label'])):
        if r['role']!=last:
            print(r"\midrule \multicolumn{10}{l}{\emph{" + r['role'].replace('_',' ') + r"}} \\")
            last=r['role']
        print(f"{r['label']} & \\texttt{{{r['arch']}}} & {r['K'] or '--'} & {r['k'] or '--'} & "
              f"{f(r['tot'],0)}M & {f(r['act'],0)}M & {r['state'].replace('%',r'\%')} & "
              f"{f(r.get('avg11'))} & {f(r.get('ppl'))} & {f(r.get('gsm'))} \\\\")
    print(r"\bottomrule"); print(r"\end{tabular}")
