"""Learning dynamics: loss curves, log-log velocity, and GSM8K vs tokens.

Outputs PDFs to nima/figs/.
Usage: python plot_learning_dynamics.py
"""
import re, json, glob, math
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from pathlib import Path
from scipy.ndimage import uniform_filter1d

LOGS  = Path.home() / "bsub_logs"
BASE  = Path("/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation")
NIMA  = Path("/proj/dmfexp/nima/Code/energy/energy-GPT-neurips2026/nima/figs")
NIMA.mkdir(exist_ok=True)

TOK_PER_STEP = 524_288   # 8 GPU × 4 micro × 4 accum × 4096 seq_len

# ── Colours / styles ──────────────────────────────────────────────────────────
STYLES = {
    'V9 GPT':    dict(c='#2196F3', ls='-',  lw=2.0, label='V9 GPT 24×1 (pure GPT)'),
    'R3 Hybrid': dict(c='#FF9800', ls='-',  lw=2.0, label='R3 11GPT+1EGPT×6 (hybrid)'),
    'H3 Hybrid': dict(c='#E91E63', ls='-',  lw=2.0, label='H3 8GPT+4EGPT (hybrid)'),
    'V73 Hybrid':dict(c='#9C27B0', ls='--', lw=1.8, label='V73 6GPT+1EGPT×6 (hybrid)'),
}

# ── Load training loss from log files ─────────────────────────────────────────
def load_losses(glob_pattern):
    losses = {}
    for f in LOGS.glob(glob_pattern):
        for m in re.finditer(r'step = (\d+), train-loss = ([0-9.]+)', f.read_text(errors='replace')):
            s, l = int(m.group(1)), float(m.group(2))
            if s not in losses:
                losses[s] = l
    return losses

raw = {
    'V9 GPT':    load_losses('scale_v9_gpt_24x1_d1024_126b_*.stderr'),
    'R3 Hybrid': load_losses('scale_r3_11gpt_1egpt6x_d1280_63b_*.stderr'),
    'H3 Hybrid': load_losses('scale_h3_8gpt_4egpt_d1280_63b_*.stderr'),
    'V73 Hybrid':load_losses('scale_v73_6gpt_1egpt6x_d1280_*.stderr'),
}
print("Data points:", {k: len(v) for k,v in raw.items()})

def to_tokens(step): return step * TOK_PER_STEP / 1e9  # in billions

def smooth(steps, losses, window_tok_b=0.5):
    """Return (tokens_b, smoothed_loss) arrays."""
    pts = sorted((to_tokens(s), l) for s, l in losses.items() if s > 0)
    if not pts: return [], []
    xs = np.array([p[0] for p in pts])
    ys = np.array([p[1] for p in pts])
    # window in index units
    step_size = xs[1] - xs[0] if len(xs) > 1 else 0.01
    w = max(3, int(window_tok_b / step_size)) if step_size > 0 else 5
    ys_s = uniform_filter1d(ys, size=min(w, len(ys)//3 or 1), mode='nearest')
    return xs, ys_s

def log_velocity(xs, ys_smooth, window=20):
    """d(log10 loss)/d(log10 tokens) in a rolling window."""
    n = len(xs)
    vels = []
    for i in range(n):
        i0 = max(0, i - window//2)
        i1 = min(n-1, i + window//2)
        if i1 == i0: vels.append(np.nan); continue
        dlogs = (math.log10(ys_smooth[i1]) - math.log10(ys_smooth[i0]))
        dlogt = (math.log10(xs[i1])        - math.log10(xs[i0]))
        vels.append(dlogs / dlogt if dlogt > 1e-9 else np.nan)
    return np.array(vels)

# ── Load GSM8K results from unsharded eval dirs ───────────────────────────────
GSM_KNOWN = {
    # (model_key, tokens_b): gsm_pct
    ('V9 GPT',    7.86): 2.88,   # V9 baseline
    ('V9 GPT',    18.9): 1.44,   # scale eval @36k
    ('R3 Hybrid', 9.4):  2.35,   # R3 baseline
    ('R3 Hybrid', 26.2): 1.82,   # scale eval @50k
    ('H3 Hybrid', 31.5): 2.12,   # H3 final
    ('V73 Hybrid',15.7): 1.44,   # V73 baseline
}

TASKS = ['arc_challenge','arc_easy','boolq','copa','hellaswag','openbookqa','piqa','sciq','winogrande','mmlu']
METS  = ['acc_norm,none','acc_norm,none','acc,none','acc,none','acc_norm,none','acc_norm,none','acc_norm,none','acc,none','acc,none','acc,none']

def load_gsm_from_dir(d):
    files = sorted(glob.glob(str(d / 'harness_results_*.json')))
    if not files: return None
    r = json.load(open(files[-1]))['results']
    gsm = r.get('gsm8k',{}).get('exact_match,flexible-extract')
    return gsm * 100 if gsm is not None else None

# Scan for intermediate eval results
for model_key, model_dir_base, step_range in [
    ('H3 Hybrid', BASE/'scale_h3_8gpt_4egpt_d1280_63b',       [45000,50000,55000,60000]),
    ('R3 Hybrid', BASE/'scale_r3_11gpt_1egpt6x_d1280_63b',    [50000,52000,54000]),
    ('V73 Hybrid',BASE/'v73_6gpt_1egpt6x_rmsray_d1280',       [27000,29000,30000]),
    ('V9 GPT',    BASE/'scale_v9_gpt_24x1_d1024_126b',        [35000,36000,38000]),
]:
    for step in step_range:
        tok = to_tokens(step)
        if (model_key, round(tok, 1)) in GSM_KNOWN: continue
        # check unsharded_STEP dir
        ud = model_dir_base / f'unsharded_{step}'
        gsm = load_gsm_from_dir(ud)
        if gsm is None and step == max(step_range):  # also check plain unsharded for final
            gsm = load_gsm_from_dir(model_dir_base / 'unsharded')
        if gsm is not None:
            GSM_KNOWN[(model_key, round(tok, 1))] = gsm
            print(f"  Loaded GSM: {model_key} @{tok:.1f}B = {gsm:.2f}%")

# ════════════════════════════════════════════════════════════════════════════
# Figure 1: Loss curves (log-log)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 5))
for name, losses in raw.items():
    if not losses: continue
    st = STYLES[name]
    xs, ys = smooth(None, losses, window_tok_b=0.3)
    if len(xs) < 10: continue
    ax.plot(xs, ys, c=st['c'], ls=st['ls'], lw=st['lw'], label=st['label'], alpha=0.9)

ax.set_xscale('log'); ax.set_yscale('log')
ax.set_xlabel('Training tokens (B)', fontsize=11)
ax.set_ylabel('Training loss (log scale)', fontsize=11)
ax.set_title('Training loss vs tokens — GPT vs hybrid architectures', fontsize=11, fontweight='bold')
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
ax.yaxis.set_major_formatter(ticker.FuncFormatter(lambda y, _: f'{y:.2f}'))
ax.grid(True, alpha=0.2, which='both')
# annotate regime bands
ax.axvspan(to_tokens(10000), to_tokens(26000), alpha=0.15, color='#1976D2', label='Plateau (~5-14B tok)')
ax.axvline(to_tokens(26000), color='#333', lw=1.5, ls=':', alpha=0.8)
ax.legend(fontsize=9, loc='upper right')
plt.tight_layout()
plt.savefig(NIMA/'fig_ld_loss_curves.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_ld_loss_curves.pdf")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Figure 2: Log-log velocity (d log loss / d log tokens)
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(8, 4.5))
for name, losses in raw.items():
    if not losses: continue
    st = STYLES[name]
    xs, ys = smooth(None, losses, window_tok_b=0.5)
    if len(xs) < 30: continue
    vel = log_velocity(xs, ys, window=30)
    # smooth velocity
    vel_s = uniform_filter1d(vel, size=40, mode='nearest')
    mask = np.isfinite(vel_s) & (xs > 1.0)   # skip very early noisy steps
    ax.plot(xs[mask], vel_s[mask], c=st['c'], ls=st['ls'], lw=st['lw'], label=st['label'], alpha=0.9)

ax.axhline(-0.05, color='#333', lw=1.8, ls='--', label='Slope = −0.05 (reference)', alpha=0.8)
ax.axvspan(to_tokens(10000), to_tokens(26000), alpha=0.15, color='#1976D2')
ax.axvline(to_tokens(26000), color='#333', lw=1.5, ls=':', alpha=0.8)
ax.set_xscale('log')
ax.set_xlabel('Training tokens (B)', fontsize=11)
ax.set_ylabel(r'Log-log velocity  $\frac{d\log L}{d\log N}$', fontsize=11)
ax.set_title('Log-log descent rate (velocity) — plateau then reacceleration', fontsize=11, fontweight='bold')
ax.set_ylim(-0.22, 0.02)
ax.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f'{x:.0f}'))
ax.grid(True, alpha=0.2)
# annotate phases
for x, label in [(to_tokens(3000),'Phase 1\n(fast)'), (to_tokens(15000),'Plateau'), (to_tokens(30000),'Phase 3\n(reaccel)')]:
    ax.text(x, -0.19, label, ha='center', fontsize=8, color='#555', style='italic')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout()
plt.savefig(NIMA/'fig_ld_velocity.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_ld_velocity.pdf")
plt.close()

# ════════════════════════════════════════════════════════════════════════════
# Figure 3: GSM8K vs training tokens
# ════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(7, 4.5))
for name, st in STYLES.items():
    pts = sorted((tok, gsm) for (n, tok), gsm in GSM_KNOWN.items() if n == name)
    if len(pts) < 1: continue
    xs2 = [p[0] for p in pts]
    ys2 = [p[1] for p in pts]
    ax.plot(xs2, ys2, c=st['c'], ls=st['ls'], lw=1.8, marker='o', ms=7,
            markerfacecolor='white', markeredgewidth=2, label=st['label'], alpha=0.9)
    for x, y in zip(xs2, ys2):
        ax.annotate(f'{y:.2f}%', (x, y), textcoords='offset points', xytext=(3, 5), fontsize=7.5, color=st['c'])

ax.axvspan(to_tokens(10000), to_tokens(26000), alpha=0.15, color='#1976D2', label='Plateau regime')
ax.set_xlabel('Training tokens (B)', fontsize=11)
ax.set_ylabel('GSM8K (flexible-extract, %)', fontsize=11)
ax.set_title('GSM8K degrades with web-only training; math data may help', fontsize=11, fontweight='bold')
ax.grid(True, alpha=0.2)
ax.legend(fontsize=9, loc='upper right')
pending = sum(1 for k in [('H3 Hybrid',23.6),('H3 Hybrid',26.2),('H3 Hybrid',28.8),('R3 Hybrid',27.2),('R3 Hybrid',28.3)] if k not in GSM_KNOWN)
if pending:
    ax.text(0.02, 0.04, f'({pending} intermediate evals pending)', transform=ax.transAxes, fontsize=8, color='gray', style='italic')
plt.tight_layout()
plt.savefig(NIMA/'fig_ld_gsm8k_vs_tokens.pdf', bbox_inches='tight', dpi=150)
print("Saved fig_ld_gsm8k_vs_tokens.pdf")
plt.close()

print(f"\nAll figures saved to {NIMA}")
