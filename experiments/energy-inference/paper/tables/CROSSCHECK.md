# Cross-check: paper_v2.tex vs auto-generated tables

Regenerate with:
  python experiments/energy-inference/scripts/multi-block-ablation/make_tables.py

## Summary of root causes

1. **Avg-accuracy column uses different averaging schemes across tables.**
   - tab:egpt_gap, tab:v5_v8 use a 9-task average WITHOUT MMLU
     (V0 → 49.1% in paper, 46.7% with the 10-task scheme).
   - tab:scaling and tab:scaling_full use a 10-task average WITH MMLU
     (V31 → 50.7% in both paper and generator ✓).
   - tab:cosreg_benchmarks uses a 7-task acc-only average (no MMLU/ObQA/SciQ).
   - The auto-generator standardizes on **10-task** (acc_norm where available,
     acc otherwise) — matching the convention used in plot_v27_egrad_edesc.
   - Action: the paper now uses the 10-task scheme via \input{} of
     auto-generated tables. Document the scheme once in the methodology.

2. **Specific genuine errors caught.**
   - V31 ARC-C: paper 27.4 → JSON acc_norm 26.6 (acc 25.3). Neither matches paper.
   - V32 ARC-C: paper 27.4 → JSON acc_norm 28.0.
   - V35/V36 Avg: paper 51.4%/51.6% → JSON 48.8%/48.9%. Substantial.

3. **Wrong bold highlights now corrected by the generator.**
   - tab:scaling: paper bolded V19 (27.7) for ARC-C; correct max is V32 (28.0).
   - tab:mixed_results: paper bolded V11 (47.3%) for Avg; with 10-task scheme
     V0/V10 tie at 46.7% (best now goes to V11 at 46.9%).

## Per-table cell diffs (latest run)

Table                          Paper Rows  Cells with diff
----------------------------------------------------------------------
tab:cosreg_benchmarks                   4               24
  V0             col 1: paper='3.121'               gen='47.5\\%'
  V0             col 2: paper='0.100'               gen='\\textbf{20.1}'
  V0             col 5: paper='0.201'               gen='\\textbf{29.6}'
  V0             col 6: paper='0.516'               gen='\\textbf{25.2}'
  V0             col 7: paper='0.566'               gen='\\textbf{64.0}'
  V0             col 8: paper='0.296'               gen='\\textbf{51.7}'
  V1             col 1: paper='3.271'               gen='\\underline{47.6\\%}'
  V1             col 2: paper='0.536'               gen='18.4'
  V1             col 3: paper='0.034'               gen='47.9'
  V1             col 4: paper='\\textbf{0.496}'     gen='\\textbf{61.1}'
  V1             col 5: paper='0.184'               gen='28.5'
  V1             col 6: paper='0.479'               gen='23.5'
  V1             col 7: paper='0.611'               gen='62.6'
  V1             col 8: paper='0.285'               gen='50.7'
  V39            col 1: paper='3.273'               gen='\\textbf{47.7\\%}'
  V39            col 2: paper='0.529'               gen='\\underline{20.1}'
  V39            col 3: paper='0.021'               gen='\\underline{49.1}'
  V39            col 4: paper='0.490'               gen='58.2'
  V39            col 5: paper='0.201'               gen='\\underline{28.5}'
  V39            col 6: paper='0.491'               gen='23.9'
  V39            col 7: paper='0.582'               gen='\\underline{63.2}'
  V39            col 8: paper='0.285'               gen='\\underline{50.7}'
  V48            col 1: paper='3.287'               gen='46.2\\%'
  V48            col 5: paper='\\multicolumn{8}{c}{\\textit{eval \\'  gen='27.5'
