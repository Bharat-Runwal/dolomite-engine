"""analyze_cosreg_layer_cosim_20260426.py

Layer cosine-similarity analysis for the cosreg sweep — produces full i,j heatmaps
and consecutive-pair line plots that complement analyze_v52v53_cosim_20260425e.py
(which only computes summary metrics).

Models analysed (12×1 d=768, 176M):
  V0 GPT, V1 EGPT, V39 EGPT+act-cosreg λ=0.01,
  V52 EGPT+act-cosreg λ=0.1 (10× V39),
  V53 EGPT+act-cosreg cosine-ramp 0→1.0,
  V48 EGPT+weight-cosreg

Outputs (in results/multi-block-ablation/plots/):
  layer_sim_cosine_cosreg_full.{pdf,png}
  consecutive_sim_cosreg_full.{pdf,png}
  layer_sim_summary_cosreg_full.{pdf,png}
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _layer_cosim_lib_20260426 import RESULTS_DIR, run  # noqa: E402

MODELS = {
    "V0 GPT":            RESULTS_DIR / "v0_gpt_baseline_d768"                          / "unsharded",
    "V1 EGPT":           RESULTS_DIR / "v1_12x1_d768_lr2e3"                            / "unsharded",
    "V39 act-cosreg\nλ=0.01":  RESULTS_DIR / "v39_egpt_cosreg_12x1_d768_lr2e3"         / "unsharded",
    "V52 act-cosreg\nλ=0.1":   RESULTS_DIR / "v52_egpt_cosreg_lam1e1_12x1_d768_lr2e3"  / "unsharded",
    "V53 act-cosreg\nramp→1.0":RESULTS_DIR / "v53_egpt_cosreg_ramp1p0_12x1_d768_lr2e3" / "unsharded",
    "V48 wt-cosreg":     RESULTS_DIR / "v48_egpt_weight_cosreg_12x1_d768_lr2e3"        / "unsharded",
}

COLORS = {
    "V0 GPT":                       "#4878CF",
    "V1 EGPT":                      "#D65F5F",
    "V39 act-cosreg\nλ=0.01":       "#F4A582",
    "V52 act-cosreg\nλ=0.1":        "#B2182B",
    "V53 act-cosreg\nramp→1.0":     "#762A83",
    "V48 wt-cosreg":                "#2CA02C",
}
MARKERS = {
    "V0 GPT": "s", "V1 EGPT": "o",
    "V39 act-cosreg\nλ=0.01": "^",
    "V52 act-cosreg\nλ=0.1":  "D",
    "V53 act-cosreg\nramp→1.0": "P",
    "V48 wt-cosreg": "X",
}


if __name__ == "__main__":
    run(
        models=MODELS,
        colors=COLORS,
        markers=MARKERS,
        tag="cosreg",
        title_suffix="cosreg sweep — d=768, 12×1 (V0/V1 baselines + V39/V52/V53/V48)",
        summary_title=(
            "Layer alignment vs. cosreg strength  —  "
            "do stronger regularisers actually align layers more?"
        ),
    )
