"""Generate all slide figures: main scatter, MoE comparison, register analysis.

Marker size encodes training tokens (log-scaled).
Outputs PDFs to nima/figs/.

Usage: python plot_slides.py
"""
import json, glob, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.lines as mlines
import numpy as np
from pathlib import Path
try:
    from adjustText import adjust_text as _adjust_text
    HAS_ADJUST = True
except ImportError:
    HAS_ADJUST = False

sys.path.insert(0, str(Path(__file__).parent))
from compute_flops import MODELS, get_mflops

BASE  = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
BMOE  = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/results")
NIMA  = Path("/proj/dmfexp/nima/Code/energy/energy-GPT-neurips2026/nima/figs")
NIMA.mkdir(exist_ok=True)

TASKS = ['arc_challenge','arc_easy','boolq','copa','hellaswag','openbookqa','piqa','sciq','winogrande','mmlu']
METS  = ['acc_norm,none','acc_norm,none','acc,none','acc,none','acc_norm,none','acc_norm,none','acc_norm,none','acc,none','acc,none','acc,none']

# Training tokens (in billions) for each model key
TOKENS_B = {
    # 30k steps, 4 GPUs × 4 micro × 4 accum × 4096 = 7.86B
    **{k: 7.86 for k in ['V0 GPT 12×1','V1 EGPT 12×1','V15 EGrad-MixH 12×1',
                          'V41 Sandwich','H1 6GPT+1EGPT×6','H3 6GPT+4EGPT',
                          'H5 6GPT+1EGPT×1','h1-topk-moe','V1+R128','h1-sel-R128',
                          'h1-topk+R128','V9 GPT 24×1','V1-400M EGPT 24×1',
                          'V19 EGrad-MixH 24×1','V31 EGrad-Attn 24×1']},
    'V1-400M+R128': 13.1,
    'V1-400M+R256': 13.1,
    'H3-scale 63B': 63.0,
    'R3 @26B':      26.2,
    'V9 @19B':      18.9,
    'math-GPT @9B': 9.4,
    'math-R3 @10B': 10.5,
    # 36k steps (R-series)
    'V73 6GPT+1EGPT×6': 15.7,   # grad_accum=8 → 2× batch
    # R-series: 36k × 4 GPUs × 4 × 4 × 4096 = 9.43B
}

# Additional models not in MODELS dict
EXTRA_PATHS = {
    # MoE series
    'B1 BoltzMoE orig':      (BASE/'b1_boltz_moe_16x1024_d768_lr2e3',   407, 285, 'moe_iso',     7.86),
    'B4 BoltzMoE rep0.1':    (BASE/'b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3', 407,285,'moe_iso',7.86),
    'C1 TopK full':          (BMOE/'c1_topk_energy_moe_4x2048_top2_d768', 256, 285, 'moe_topk',  7.86),
    # Register series
    'V0 GPT':                (BASE/'v0_gpt_baseline_d768',                162, 321, 'gpt',         7.86),
    'V0+R128':               (BASE/'reg_v0_gpt_12x1_d768_r128',           162, 321, 'reg_gpt',     7.86),
    'V0+R256':               (BASE/'reg_v0_gpt_12x1_d768_r256',           162, 321, 'reg_gpt',     7.86),
    'V1 EGPT':               (BASE/'v1_12x1_d768_lr2e3',                  176, 359, 'egpt',        7.86),
    'V1+R16':                (BASE/'reg_v1_egpt_12x1_d768_r16',            176, 359, 'reg_egpt',   7.86),
    'V1+R128 ★':             (BASE/'reg_v1_egpt_12x1_d768_r128',           176, 359, 'reg_egpt',   7.86),
    'V1+R256':               (BASE/'reg_v1_egpt_12x1_d768_r256',           176, 359, 'reg_egpt',   7.86),
    'H1 hybrid':             (BASE/'h1_6gpt_1egpt6x_d768',                126, 305, 'hybrid',      7.86),
    'H1+R128 full':          (BASE/'reg_h1_6gpt_1egpt6x_d768_r128',       126, 305, 'reg_hybrid',  7.86),
    'H1+R256':               (BASE/'reg_h1_6gpt_1egpt6x_d768_r256',       126, 305, 'reg_hybrid',  7.86),
    'H1+R128 sel. ★':        (BASE/'h1_sel_reg_128_d768',                  126, 305, 'reg_sel',    7.86),
}

MODEL_PATHS = {
    'V0 GPT 12×1':         BASE/'v0_gpt_baseline_d768',
    'V1 EGPT 12×1':        BASE/'v1_12x1_d768_lr2e3',
    'V15 EGrad-MixH 12×1': BASE/'v15_energy_grad_mixed_12x1_d768_lr2e3',
    'V41 Sandwich':        BASE/'v41_sandwich_2gpt8e2gpt_d768_lr2e3',
    'H1 6GPT+1EGPT×6':     BASE/'h1_6gpt_1egpt6x_d768',
    'H3 6GPT+4EGPT':       BASE/'h3_6gpt_4egpt_d768',
    'H5 6GPT+1EGPT×1':     BASE/'h5_6gpt_1egpt1x_d768',
    'h1-topk-moe':         BASE/'h1_topk_egpt_moe_d768',
    'V1+R128':             BASE/'reg_v1_egpt_12x1_d768_r128',
    'h1-sel-R128':         BASE/'h1_sel_reg_128_d768',
    'h1-topk+R128':        BASE/'h1_topk_egpt_moe_r128_d768',
    'V9 GPT 24×1':         BASE/'v9_gpt_baseline_d1024_lr1e3',
    'V1-400M+R128':        BASE/'reg_v1_400m_d1024_r128',
    'V1-400M+R256':        BASE/'reg_v1_400m_d1024_r256',
    'H3-scale 63B':        BASE/'scale_h3_8gpt_4egpt_d1280_63b',
    'R3 @26B':             BASE/'scale_r3_11gpt_1egpt6x_d1280_63b',
    'V9 @19B':             BASE/'scale_v9_gpt_24x1_d1024_126b',
    'math-GPT @9B':        BASE/'math_gpt_24x1_d1024_63b',
    'math-R3 @10B':        BASE/'math_r3_11gpt_1egpt6x_d1280_63b',
    'V1-400M EGPT 24×1':   BASE/'v1_400m_d1024_lr7e4',
    'V19 EGrad-MixH 24×1': BASE/'v19_energy_grad_24x1_d1024_lr1e3',
    'V31 EGrad-Attn 24×1': BASE/'v31_egrad_attn_24x1_d1024_lr1e3',
    'V73 6GPT+1EGPT×6':    BASE/'v73_6gpt_1egpt6x_rmsray_d1280',
}

def get_metrics(path):
    files = sorted(glob.glob(str(path/'unsharded'/'harness_results_2*.json')))
    if not files: return None
    for f in reversed(files):
        r = json.load(open(f))['results']
        if r.get('wikitext'): break
    else: return None
    ppl  = r.get('wikitext',{}).get('word_perplexity,none')
    vals = [r.get(t,{}).get(m,0) for t,m in zip(TASKS,METS)]
    avg  = sum(vals)/len(vals)*100
    gsm  = r.get('gsm8k',{}).get('exact_match,flexible-extract')
    copa = r.get('copa',{}).get('acc,none',0)*100
    return dict(ppl=ppl, avg=avg, gsm=gsm*100 if gsm else None, copa=copa)

# ── styles ──────────────────────────────────────────────────────────────────
FAM_STYLE = {
    'gpt':          dict(c='#2196F3',m='o',z=4),
    'egpt':         dict(c='#F44336',m='s',z=3),
    'mixhead':      dict(c='#9C27B0',m='^',z=3),
    'egrad':        dict(c='#E91E63',m='D',z=4),
    'hybrid':       dict(c='#FF9800',m='*',z=5),
    'hybrid_moe':          dict(c='#FF5722',m='P',z=6),
    'hybrid_moe_register': dict(c='#FF5722',m='X',z=7),
    'register_egpt':       dict(c='#F44336',m='X',z=4),
    'hybrid_register':dict(c='#FF9800',m='X',z=5),
    'moe_iso':      dict(c='#4CAF50',m='h',z=2),
    'moe_topk':     dict(c='#8BC34A',m='P',z=4),
    'reg_gpt':      dict(c='#2196F3',m='X',z=3),
    'reg_egpt':     dict(c='#F44336',m='X',z=4),
    'reg_hybrid':   dict(c='#FF9800',m='X',z=4),
    'reg_sel':      dict(c='#FF5722',m='X',z=5),
}

def token_size(tok_b, base=7.86, smin=55, smax=180):
    """Map training tokens to marker size (log-scaled)."""
    if tok_b is None: return smin
    t = np.log10(max(tok_b, base)) / np.log10(63)
    b = np.log10(base) / np.log10(63)
    frac = (t - b) / (1 - b)
    return smin + frac * (smax - smin)

# Build main dataset
main_data = []
for key, spec in MODELS.items():
    path = MODEL_PATHS.get(key)
    if not path: continue
    m = get_metrics(path)
    if not m or m['ppl'] is None: continue
    tok = TOKENS_B.get(key, 7.86)
    main_data.append(dict(key=key, label=spec.get('label',key).split(' ')[0],
                          fam=spec['family'], params=spec['params_M'],
                          mflops=get_mflops(key,spec), tok=tok, **m))

# Build extra dataset
extra_data = {}
for label, (path, params, mflops, fam, tok) in EXTRA_PATHS.items():
    m = get_metrics(path)
    if not m or m['ppl'] is None: continue
    extra_data[label] = dict(label=label.split(' ')[0], fam=fam,
                              params=params, mflops=mflops, tok=tok, **m)

KEY_LABELS = {'V0 GPT 12×1','V9 GPT 24×1','V1 EGPT 12×1','V31 EGrad-Attn 24×1',
              'V73 6GPT+1EGPT×6','h1-topk-moe','H5 6GPT+1EGPT×1','V1+R128',
              'h1-topk+R128','V1-400M+R128'}

def scatter_main(ax, xs, ys, data, title, xlabel, ylabel,
                 key_labels=None, invert_y=False, annotate=True):
    FAM_ORDER = ['gpt','egpt','mixhead','egrad','hybrid','register_egpt',
                 'hybrid_register','hybrid_moe','hybrid_moe_register']
    texts = []
    for fam in FAM_ORDER:
        for row in [r for r in data if r['fam']==fam]:
            x, y = row.get(xs), row.get(ys)
            if x is None or y is None: continue
            st = FAM_STYLE.get(fam,{'c':'#999','m':'o','z':1})
            sz = token_size(row.get('tok',7.86))
            ax.scatter(x, y, c=st['c'], marker=st['m'], zorder=st['z'],
                       s=sz, edgecolors='white', linewidths=1.0, alpha=0.9)
            if annotate and (key_labels is None or row['key'] in key_labels):
                t = ax.text(x, y, f" {row['label']}", fontsize=9,
                            color=st['c'], fontweight='bold', zorder=10)
                texts.append(t)
    ax.set_xscale('log')
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.22)
    if invert_y: ax.invert_yaxis()
    if HAS_ADJUST and texts:
        _adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle='-', color='#bbb', lw=0.5),
                     expand_text=(1.1, 1.3), force_text=(0.3, 0.4))


# ── Legend elements ──────────────────────────────────────────────────────────
MAIN_LEGEND = [
    mpatches.Patch(color='#2196F3', label='GPT baseline'),
    mpatches.Patch(color='#F44336', label='EGPT / EGrad'),
    mpatches.Patch(color='#9C27B0', label='Mixed-head (MixH)'),
    mpatches.Patch(color='#FF9800', label='Hybrid GPT+EGPT'),
    plt.Line2D([0],[0],marker='P',color='w',markerfacecolor='#FF5722',ms=10,label='Hybrid+MoE'),
    plt.Line2D([0],[0],marker='X',color='w',markerfacecolor='#F44336',ms=9, label='EGPT + Registers'),
    plt.Line2D([0],[0],marker='X',color='w',markerfacecolor='#FF5722',ms=9, label='MoE + Registers'),
]
TOK_LEGEND = [
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#888',ms=6,  label='7.9B tokens'),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#888',ms=9,  label='15.7B tokens'),
    plt.Line2D([0],[0],marker='o',color='w',markerfacecolor='#888',ms=13, label='63B tokens'),
]


# ════════════════════════════════════════════════════════════════════════════
# Fig 1: 6-panel main scatter (PPL inverted)
# ════════════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax,xs,ys,title,xl,yl,inv in [
    (axes[0,0],'params','ppl', 'PPL vs Params',   'Params (M)', 'PPL ↓ (lower=better, top)',  True),
    (axes[0,1],'params','avg', 'Acc vs Params',   'Params (M)', '10-task Avg (%) ↑',           False),
    (axes[0,2],'params','gsm', 'GSM8K vs Params', 'Params (M)', 'GSM8K (%) ↑',                False),
    (axes[1,0],'mflops','ppl', 'PPL vs FLOPs',    'MFLOPs/tok', 'PPL ↓ (lower=better, top)',  True),
    (axes[1,1],'mflops','avg', 'Acc vs FLOPs',    'MFLOPs/tok', '10-task Avg (%) ↑',           False),
    (axes[1,2],'mflops','gsm', 'GSM8K vs FLOPs',  'MFLOPs/tok', 'GSM8K (%) ↑',                False),
]:
    scatter_main(ax, xs, ys, main_data, title, xl, yl, KEY_LABELS, inv)
# Token size legend
tok_leg = fig.legend(handles=TOK_LEGEND, loc='upper right', fontsize=8,
                     bbox_to_anchor=(1.0, 0.99), title='Training tokens', title_fontsize=8)
fig.add_artist(tok_leg)
fig.legend(handles=MAIN_LEGEND, loc='lower center', ncol=6, fontsize=9,
           bbox_to_anchor=(0.5,-0.01), framealpha=0.95)
fig.suptitle('EGPT Ablation: PPL, Accuracy, GSM8K  (marker size ∝ training tokens)',
             fontsize=12, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0,0.04,1,1])
plt.savefig(NIMA/'fig_headline_6panel.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_headline_6panel.pdf")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Fig 2: MoE results comparison
# ════════════════════════════════════════════════════════════════════════════
moe_models = ['B1 BoltzMoE orig','B4 BoltzMoE rep0.1','C1 TopK full',
              'V0 GPT','V1 EGPT','h1-topk-moe']
moe_data = []
for k in moe_models:
    if k in extra_data: moe_data.append(extra_data[k])
    else:
        for row in main_data:
            if row['key'] == k or row['label'] == k.split(' ')[0]: moe_data.append(row); break

fig, axes = plt.subplots(1, 3, figsize=(13, 4.5))
MOE_COLORS = {'moe_iso':'#4CAF50','moe_topk':'#8BC34A','gpt':'#2196F3','egpt':'#F44336',
               'hybrid_moe':'#FF5722','hybrid':'#FF9800'}
MOE_LEGEND = [
    mpatches.Patch(color='#2196F3',label='GPT baseline'),
    mpatches.Patch(color='#F44336',label='EGPT (V1)'),
    mpatches.Patch(color='#4CAF50',label='BoltzMoE iso-param (B-series)'),
    mpatches.Patch(color='#8BC34A',label='C1 TopK full-size'),
    plt.Line2D([0],[0],marker='P',color='w',markerfacecolor='#FF5722',ms=10,label='h1-topk-moe (hybrid)'),
]
for ax, xs, ys, title, xl, yl, inv in [
    (axes[0],'params','ppl','PPL vs Params','Params (M)','PPL ↓',True),
    (axes[1],'params','avg','Accuracy vs Params','Params (M)','10-task Avg (%) ↑',False),
    (axes[2],'params','gsm','GSM8K vs Params','Params (M)','GSM8K (%) ↑',False),
]:
    for row in moe_data:
        x,y = row.get(xs),row.get(ys)
        if x is None or y is None: continue
        c = MOE_COLORS.get(row['fam'],'#999')
        mk = 'P' if row['fam']=='hybrid_moe' else ('X' if 'topk' in row['fam'] else 'o')
        sz = token_size(row.get('tok',7.86))
        ax.scatter(x,y,c=c,marker=mk,s=sz+20,edgecolors='white',linewidths=1.2,alpha=0.95,zorder=4)
        ax.annotate(row['label'],(x,y),textcoords='offset points',xytext=(4,3),fontsize=8.5)
    ax.set_xscale('log'); ax.set_xlabel(xl,fontsize=10); ax.set_ylabel(yl,fontsize=10)
    ax.set_title(title,fontsize=10,fontweight='bold'); ax.grid(True,alpha=0.22)
    if inv: ax.invert_yaxis()
fig.legend(handles=MOE_LEGEND,loc='lower center',ncol=5,fontsize=9,
           bbox_to_anchor=(0.5,-0.02),framealpha=0.95)
fig.suptitle('MoE in EGPT Block: Iso-param fails, full-size wins\n'
             r'Fixing FFN:Attn imbalance (21:1$\to$3:1) recovers V1 EGPT quality; MoE in hybrid beats V0 GPT',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0.06,1,1])
plt.savefig(NIMA/'fig_moe_comparison.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_moe_comparison.pdf")
plt.close()


# ════════════════════════════════════════════════════════════════════════════
# Fig 3: Register effects — effect-on-GSM vs effect-on-PPL scatter
# ════════════════════════════════════════════════════════════════════════════
reg_pairs = [
    ('V0 GPT','V0+R128',  128,'GPT',   '#2196F3'),
    ('V0 GPT','V0+R256',  256,'GPT',   '#1565C0'),
    ('V1 EGPT','V1+R16',   16,'EGPT',  '#EF9A9A'),
    ('V1 EGPT','V1+R128 ★',128,'EGPT', '#F44336'),
    ('V1 EGPT','V1+R256',  256,'EGPT', '#B71C1C'),
    ('H1 hybrid','H1+R128 full',128,'Hybrid','#FFB74D'),
    ('H1 hybrid','H1+R256',     256,'Hybrid','#E65100'),
    ('H1 hybrid','H1+R128 sel. ★',128,'H-Sel','#FF5722'),
]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
ax_ppl_gsm, ax_ppl_avg = axes

for base_key, reg_key, n_reg, ftype, color in reg_pairs:
    b = extra_data.get(base_key)
    r = extra_data.get(reg_key)
    if not b or not r: continue
    dppl = r['ppl'] - b['ppl']    # negative = better
    dgsm = (r['gsm'] or 0) - (b['gsm'] or 0)
    davg = r['avg'] - b['avg']
    mk = 'o' if n_reg==128 else ('s' if n_reg==256 else '^')
    sz = 100 if '★' in reg_key else 60
    for ax, dy, ylabel in [(ax_ppl_gsm, dgsm, 'ΔGSM8K (pp)'),
                            (ax_ppl_avg, davg, 'ΔAvg accuracy (pp)')]:
        ax.scatter(dppl, dy, c=color, marker=mk, s=sz, edgecolors='white',
                   linewidths=1.2, alpha=0.92, zorder=4)
        ax.annotate(f"{ftype}+R{n_reg}", (dppl, dy), textcoords='offset points',
                    xytext=(4,2), fontsize=7.5, color='#333')

for i, (ax, ylabel, title) in enumerate([
    (ax_ppl_gsm, 'ΔGSM8K (pp)', 'Register effect: ΔPPL vs ΔGSM8K'),
    (ax_ppl_avg, 'ΔAvg acc (pp)', 'Register effect: ΔPPL vs ΔAccuracy'),
]):
    ax.axhline(0, color='#ccc', lw=0.8, zorder=0)
    ax.axvline(0, color='#ccc', lw=0.8, zorder=0)
    ax.set_xlabel('ΔPPL (negative = better ←)', fontsize=10)
    ax.set_ylabel(ylabel, fontsize=10)
    ax.set_title(title, fontsize=10, fontweight='bold')
    ax.grid(True, alpha=0.2)
    # "Ideal" annotation only on left plot (GSM); right plot has different story
    if i == 0:
        ax.text(0.03, 0.97, 'Ideal: top-left\n(PPL↓, GSM↑)',
                transform=ax.transAxes, fontsize=8, color='#2e7d32',
                va='top', ha='left', alpha=0.8,
                bbox=dict(boxstyle='round,pad=0.25', fc='white', alpha=0.7, ec='none'))

REG_LEGEND = [
    mpatches.Patch(color='#2196F3', label='GPT + registers'),
    mpatches.Patch(color='#F44336', label='EGPT + registers ★'),
    mpatches.Patch(color='#FF9800', label='Hybrid full registers'),
    mpatches.Patch(color='#FF5722', label='Hybrid selective (EGPT-only ★)'),
    plt.Line2D([0],[0],marker='o',c='w',markerfacecolor='#888',ms=9,label='R=128'),
    plt.Line2D([0],[0],marker='s',c='w',markerfacecolor='#888',ms=9,label='R=256'),
    plt.Line2D([0],[0],marker='^',c='w',markerfacecolor='#888',ms=9,label='R=16'),
]
fig.legend(handles=REG_LEGEND, loc='lower center', ncol=7, fontsize=8.5,
           bbox_to_anchor=(0.5,-0.02), framealpha=0.95)
fig.suptitle('Register tokens: EGPT uniquely improves GSM8K; GPT collapses it\n'
             'Selective (EGPT-only) injection preserves GSM while improving COPA',
             fontsize=11, fontweight='bold')
plt.tight_layout(rect=[0,0.07,1,1])
plt.savefig(NIMA/'fig_register_effect.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_register_effect.pdf")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Fig 5: Select models 6-panel (slide after register delta, before cosreg)
# Shows: GPT baseline, deep EGPT, best hybrid, best MoE (no reg),
#        best register model (V1+R128), MoE+reg, and V1-400M+R128 when ready.
# ════════════════════════════════════════════════════════════════════════════

# Pull selected rows by key from main_data
SELECT_KEYS = [
    'V0 GPT 12×1',      # GPT baseline 160M
    'V9 GPT 24×1',      # GPT baseline 400M
    'V1 EGPT 12×1',     # deep EGPT (also base for V1+R128)
    'V73 6GPT+1EGPT×6', # best hybrid
    'h1-topk-moe',      # best MoE no regs
    'h1-topk+R128',     # MoE + R128
    'V1+R128',          # best register EGPT (V1 EGPT is its base)
    'V1-400M+R128',     # 400M EGPT + R128 (shows when eval is done)
]
SELECT_LABELS = {
    'V0 GPT 12×1':      'V0 GPT',
    'V9 GPT 24×1':      'V9 GPT',
    'V1 EGPT 12×1':     'V1 EGPT',
    'V73 6GPT+1EGPT×6': 'V73 Hybrid',
    'h1-topk-moe':      'h1-MoE',
    'h1-topk+R128':     'MoE+R128',
    'V1+R128':          'V1+R128',
    'V1-400M+R128':     '400M+R128',
}

# Collect data: main_data first, then extra_data overrides
_main_by_key = {r['key']: r for r in main_data}
select_data = []
for k in SELECT_KEYS:
    row = _main_by_key.get(k)
    if row is None:
        # try extra_data by matching label prefix
        for ek, ev in extra_data.items():
            if k in ek or ek.startswith(k.split(' ')[0]):
                row = dict(ev, key=k)
                break
    if row and row.get('ppl') is not None:
        select_data.append(dict(row, label=SELECT_LABELS.get(k, row['label'])))

SELECT_LEGEND = [
    mpatches.Patch(color='#2196F3', label='GPT baseline'),
    mpatches.Patch(color='#F44336', label='EGPT (deep)'),
    mpatches.Patch(color='#FF9800', label='Hybrid GPT+EGPT'),
    plt.Line2D([0],[0],marker='P',color='w',markerfacecolor='#FF5722',ms=11,label='Hybrid+MoE'),
    plt.Line2D([0],[0],marker='X',color='w',markerfacecolor='#F44336',ms=10,label='EGPT+Reg ★'),
    plt.Line2D([0],[0],marker='X',color='w',markerfacecolor='#FF5722',ms=10,label='MoE+Reg'),
]

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for ax, xs, ys, title, xl, yl, inv in [
    (axes[0,0],'params','ppl', 'PPL vs Params',   'Params (M)', 'PPL ↓',  True),
    (axes[0,1],'params','avg', 'Acc vs Params',   'Params (M)', 'Avg (%) ↑', False),
    (axes[0,2],'params','gsm', 'GSM8K vs Params', 'Params (M)', 'GSM8K (%) ↑', False),
    (axes[1,0],'mflops','ppl', 'PPL vs FLOPs',   'MFLOPs/tok', 'PPL ↓',  True),
    (axes[1,1],'mflops','avg', 'Acc vs FLOPs',   'MFLOPs/tok', 'Avg (%) ↑', False),
    (axes[1,2],'mflops','gsm', 'GSM8K vs FLOPs', 'MFLOPs/tok', 'GSM8K (%) ↑', False),
]:
    texts = []
    for row in select_data:
        x, y = row.get(xs), row.get(ys)
        if x is None or y is None: continue
        fam = row.get('fam','gpt')
        st  = FAM_STYLE.get(fam, {'c':'#999','m':'o','z':3})
        sz  = token_size(row.get('tok', 7.86)) * 1.6
        ax.scatter(x, y, c=st['c'], marker=st['m'], zorder=st['z']+2,
                   s=sz, edgecolors='white', linewidths=1.5, alpha=0.95)
        t = ax.text(x, y, f" {row['label']}", fontsize=8, color=st['c'],
                    fontweight='bold', zorder=10)
        texts.append(t)
    ax.set_xscale('log')
    ax.set_xlabel(xl, fontsize=9)
    ax.set_ylabel(yl, fontsize=9)
    ax.set_title(title, fontsize=9, fontweight='bold')
    ax.grid(True, alpha=0.22)
    if inv: ax.invert_yaxis()
    if HAS_ADJUST and texts:
        _adjust_text(texts, ax=ax,
                     arrowprops=dict(arrowstyle='-', color='#aaa', lw=0.5),
                     expand_text=(1.1, 1.3), force_text=(0.4, 0.5))

fig.legend(handles=SELECT_LEGEND, loc='lower center', ncol=6, fontsize=8.5,
           bbox_to_anchor=(0.5, 0.0), framealpha=0.95)
fig.suptitle('Key models: GPT → EGPT → Hybrid → MoE → Registers  (size ∝ tokens)',
             fontsize=11, fontweight='bold', y=1.01)
plt.tight_layout(rect=[0, 0.04, 1, 1])
plt.savefig(NIMA/'fig_select_6panel.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_select_6panel.pdf")
plt.close()


print(f"All figures saved to {NIMA}")
