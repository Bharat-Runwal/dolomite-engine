"""Collect pre-fix (bypass, buggy) vs post-fix (no_cache) BBH/GSM8K numbers
for all register checkpoints re-evaluated after the decode-bug fix.

See lm_engine/hf_models/models/register_energy/REGISTER_DECODE_BUG.md for the bug.
Reads harness_results*.json (GSM8K etc.) and bbh_rescore_summary*.json (BBH) next
to each checkpoint; picks the latest pre-fix file (no "nocache" in the name) and
the post-fix "nocache" file.
"""
import json
import glob
import os

BASE = "/proj/dmfexp/nima/Code/dolomite-engine/experiments/energy-inference/results/multi-block-ablation"

CKPTS = [
    ("reg_v0_gpt_12x1_d768_r128", "V0 (pure GPT) + R128"),
    ("reg_v0_gpt_12x1_d768_r256", "V0 (pure GPT) + R256"),
    ("reg_v1_egpt_12x1_d768_r16", "V1 (EGPT 12x1) + R16"),
    ("reg_v1_egpt_12x1_d768_r128", "V1 (EGPT 12x1) + R128"),
    ("reg_v1_egpt_12x1_d768_r256", "V1 (EGPT 12x1) + R256"),
    ("reg_v1_400m_d1024_r128", "V1-400M (EGPT) + R128"),
    ("reg_v1_400m_d1024_r256", "V1-400M (EGPT) + R256"),
    ("scale_reg_v1_egpt_d768_r128_126b", "V1 (EGPT) + R128, 126B tok"),
    ("reg_v41_sandwich_2g8e2g_d768_r128", "V41 sandwich (2G8E2G) + R128"),
    ("reg_v56_1x12_d768_r128", "V56 (EGPT 1x12, recurrent) + R128"),
    ("reg_v73_6gpt_1egpt6x_d1280_r128", "V73 (6G+1Ex6, d1280) + R128"),
    ("reg_v73_6gpt_1egpt6x_d1280_r256", "V73 (6G+1Ex6, d1280) + R256"),
    ("reg_h1_6gpt_1egpt6x_d768_r128", "H1 (6G+1Ex6, d768) all-layer + R128"),
    ("reg_h1_6gpt_1egpt6x_d768_r256", "H1 (6G+1Ex6, d768) all-layer + R256"),
    ("h1_sel_reg_128_d768", "H1 selective (EGPT-only) + R128"),
    ("v76_4gpt_1egpt6x_rmsray_d1024_reg128", "V76 (4G+1Ex6, RMSRay) + R128"),
    ("math_fet_hopfield_mean_r256_8gpt_1egpt6x_d1536_int8k_lra32_itd3_lr1p5e3_33b_16gpu",
     "Hopfield-MEAN (8G+1Ex6, d1536) + R256"),
]


def latest(paths):
    if not paths:
        return None
    return max(paths, key=os.path.getmtime)


def load_json(p):
    if p is None:
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def get_gsm8k(d):
    if d is None:
        return None, None
    r = d.get("results", {})
    cot = r.get("gsm8k_cot", {})
    strict = r.get("gsm8k", {})
    return cot.get("exact_match,flexible-extract"), strict.get("exact_match,flexible-extract")


def get_bbh(d):
    if d is None:
        return None
    macro = d.get("__bbh_fewshot_macro_avg")
    if macro is not None:
        return macro.get("exact_match,none")
    return None


rows = []
for ckpt, label in CKPTS:
    d = os.path.join(BASE, ckpt, "unsharded")

    old_main = latest([p for p in glob.glob(f"{d}/harness_results_*.json") if "nocache" not in p])
    new_main = latest(glob.glob(f"{d}/harness_results_nocache_*.json"))
    old_bbh = latest([p for p in glob.glob(f"{d}/bbh_rescore_summary_*.json") if "nocache" not in p])
    new_bbh = latest(glob.glob(f"{d}/bbh_rescore_summary_nocache_*.json"))

    old_cot, old_strict = get_gsm8k(load_json(old_main))
    new_cot, new_strict = get_gsm8k(load_json(new_main))
    old_bbh_em = get_bbh(load_json(old_bbh))
    new_bbh_em = get_bbh(load_json(new_bbh))

    rows.append({
        "ckpt": ckpt, "label": label,
        "old_gsm8k_cot": old_cot, "new_gsm8k_cot": new_cot,
        "old_gsm8k_strict": old_strict, "new_gsm8k_strict": new_strict,
        "old_bbh": old_bbh_em, "new_bbh": new_bbh_em,
    })


def fmt(x):
    return f"{100*x:.2f}%" if isinstance(x, (int, float)) else "—"


print(f"{'Model':45s} {'BBH(bug)':>10} {'BBH(fix)':>10} {'GSM8K-CoT(bug)':>16} {'GSM8K-CoT(fix)':>16}")
for r in rows:
    print(f"{r['label']:45s} {fmt(r['old_bbh']):>10} {fmt(r['new_bbh']):>10} "
          f"{fmt(r['old_gsm8k_cot']):>16} {fmt(r['new_gsm8k_cot']):>16}")

json.dump(rows, open(os.path.join(BASE, "plots", "register_reeval_comparison_20260725.json"), "w"), indent=2)
print(f"\nSaved: {BASE}/plots/register_reeval_comparison_20260725.json")
