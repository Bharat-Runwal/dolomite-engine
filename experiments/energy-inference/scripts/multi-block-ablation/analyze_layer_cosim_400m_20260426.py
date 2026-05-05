"""analyze_layer_cosim_400m_20260426.py

Layer cosine-similarity analysis for the 400M deep-EGPT family — full i,j heatmaps
and consecutive-pair line plots. Extends analyze_layer_sim_egrad_edesc_20260425d.py
(which only included V9/V19/V20) to also cover V1 EGPT 24×1 (the headline 400M
EGPT model), V31/V32 (EGrad/EDesc with W_Q^T output), and V40 (act-cosreg 400M).

Models analysed (24×1 d=1024, ~342–400M):
  V9   GPT
  V1_400m  EGPT 24×1
  V19  MixH† EGrad
  V20  MixH† EDesc
  V31  EGrad (W_Q^T output)
  V32  EDesc (W_Q^T output)
  V40  EGPT + act-cosreg λ=0.01

Outputs (in results/multi-block-ablation/plots/):
  layer_sim_cosine_400m_full.{pdf,png}
  consecutive_sim_400m_full.{pdf,png}
  layer_sim_summary_400m_full.{pdf,png}
"""
from pathlib import Path
import sys

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from _layer_cosim_lib_20260426 import RESULTS_DIR, run  # noqa: E402

MODELS = {
    "V9 GPT":      RESULTS_DIR / "v9_gpt_baseline_d1024_lr1e3"      / "unsharded",
    "V1 EGPT 400M": RESULTS_DIR / "v1_400m_d1024_lr7e4"             / "unsharded",
    "V19 MixH†\nEGrad":  RESULTS_DIR / "v19_energy_grad_24x1_d1024_lr1e3" / "unsharded",
    "V20 MixH†\nEDesc":  RESULTS_DIR / "v20_energy_desc_24x1_d1024_lr1e3" / "unsharded",
    "V31 EGrad\n(W_Q^T)": RESULTS_DIR / "v31_egrad_attn_24x1_d1024_lr1e3" / "unsharded",
    "V32 EDesc\n(W_Q^T)": RESULTS_DIR / "v32_edesc_24x1_d1024_lr1e3"      / "unsharded",
    "V40 EGPT\nact-cosreg": RESULTS_DIR / "v40_egpt_cosreg_24x1_d1024_lr7e4" / "unsharded",
}

COLORS = {
    "V9 GPT":           "#4878CF",
    "V1 EGPT 400M":     "#D65F5F",
    "V19 MixH†\nEGrad": "#9467BD",
    "V20 MixH†\nEDesc": "#C5B0D5",
    "V31 EGrad\n(W_Q^T)": "#2CA02C",
    "V32 EDesc\n(W_Q^T)": "#FF7F0E",
    "V40 EGPT\nact-cosreg": "#8C564B",
}
MARKERS = {
    "V9 GPT": "s",
    "V1 EGPT 400M": "o",
    "V19 MixH†\nEGrad": "D",
    "V20 MixH†\nEDesc": "v",
    "V31 EGrad\n(W_Q^T)": "P",
    "V32 EDesc\n(W_Q^T)": "X",
    "V40 EGPT\nact-cosreg": "^",
}


if __name__ == "__main__":
    run(
        models=MODELS,
        colors=COLORS,
        markers=MARKERS,
        tag="400m",
        title_suffix="d=1024, 24×1 — V9/V1_400m + EGrad/EDesc + V40 cosreg",
        summary_title=(
            "Inter-layer alignment at 400M scale  —  "
            "EGPT vs EGrad/EDesc vs cosreg"
        ),
    )
