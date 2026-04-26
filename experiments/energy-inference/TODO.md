# TODO — Energy-Inference Large-Scale

## Paper

This section drives the paper restructuring. The paper is currently
`paper/paper_v2.tex`; tables auto-generate via
`scripts/multi-block-ablation/make_tables.py` into `paper/tables/`.

### Story (target arc)

The paper is part of a larger EGPT (energy-based GPT) research programme;
this submission is the *deep-EGPT analysis* slice. The story should land in
this order, with each stage directly motivating the next:

1. **Why deep EGPT?**
   Classic EGPT was a *single* energy block iterated $L$ times — every
   recurrent step shares one set of millions of parameters. This makes the
   compute spend grow linearly with $L$ but the unique-parameter budget stay
   tiny, so the model has to re-use one pattern at every depth — high
   FLOPs, low capacity.  Deep EGPT (untie weights across the $L$ blocks)
   keeps the "forward pass = gradient steps on an implicit energy" frame
   while letting the model lay down distinct sub-routines per layer; we
   show a slim deep EGPT matches a deep GPT at far lower FLOPs.

2. **The miracle: deep EGPT layers stay aligned.**
   Despite *not* sharing weights, the per-layer outputs of deep EGPT have
   substantially higher functional similarity than deep GPT
   (consecutive-layer cosine $0.176$ vs.\ $0.119$ overall, $\to 0.53$ at
   layers 8–9). This is the empirical anchor of the paper. It is exactly
   what the original "blocks $\approx$ gradient steps on a shared energy"
   intuition predicts, but achieved emergently rather than by weight tying.

3. **Cosine-similarity regularisation.**
   Strategies A/B from the appendix (V39 act-cosreg, V48 wt-cosreg) become
   a main-text section: do we *want* explicit alignment? Light cosreg
   ($\lambda=0.01$) costs no accuracy but only marginally raises alignment.
   V52/V53 (stronger / ramped) test the boundary. This sets up Strategy B.

4. **Strategy B (post-training merger) — moves up to right after alignment.**
   Once we believe layers are functionally aligned, can we merge them?
   V42–V51 (naive vs.\ Procrustes; 6×1 vs.\ 6×2; FT recipes) answer
   *partially yes*. This is now a co-equal main result.

5. **Helmholtz / curl–grad / structural analyses move to appendix.**
   They support point 1 (the projection learns a balanced rotation/grad
   mix) but are not essential to the deep-EGPT vs.\ deep-GPT story.

6. **Mixed-head merger as the killer downstream application.**
   New section right after the alignment + merger results: EGPT outperforms
   GPT on COPA / BoolQ; combining EGPT and GPT *via attention-head merging*
   (mixed-head + EGrad output rule) yields a model that beats GPT on
   the union. V15/V16/V19/V27/V31/V32 anchor this. Frame as the practical
   payoff of deep-EGPT understanding.

### Tables / figures policy

- **Auto-generate every benchmark table.** `make_tables.py` is the single
  source of truth. The paper should `\input{tables/<name>}` and never
  contain numeric cells in `paper_v2.tex` directly. Tables done so far:
  egpt_gap, mixed_results, output_rule, scaling, scaling_v25v26,
  scaling_full, v5_v8, all_small, cosreg_benchmarks.
- **Cross-check report:** `paper/tables/CROSSCHECK.md` lists every cell
  diff against the previous hand-written numbers; address each item.
- **Bar charts preferred for ≤5-model comparisons.**
  Whenever a comparison has ≤5 models on multiple benchmarks, the *main*
  text gets a grouped bar chart and the full table goes to the appendix.
  Targets where this should apply: tab:egpt_gap (4), tab:scaling (5),
  tab:scaling_v25v26 (5), tab:cosreg_benchmarks (4). Keep tables in main
  text only where >5 rows or where exact numbers are part of the
  argument (e.g. tab:output_rule).
- **Highlighting convention** (already implemented in make_tables.py):
  best per column = `\textbf{}`; second best = `\underline{}`; PPL is
  lower-is-better; static columns (Params, MFLOPs, Type) never highlighted.

### Concrete sub-tasks

#### Story rewrite
- [ ] Rewrite `\section{Introduction}` around the deep-EGPT-vs-classic-EGPT
      framing (point 1 above). Drop the existing "EGrad/EDesc as the key"
      hook from the abstract; lead with: "Untying EGPT block weights yields
      deep models that match deep GPT in FLOPs *and* show emergent
      cross-layer alignment that classic GPT lacks."
- [ ] Move "Mechanistic Probes" → "Layer alignment" as the second main-text
      section (currently §4); Helmholtz subsection moves to appendix.
- [ ] Promote "Strategy A: cosreg" (currently appendix) into main text right
      after alignment.
- [ ] Promote "Strategy B: layer merger + FT" (currently appendix) into
      main text right after cosreg.
- [ ] Add new main-text section: **Mixed-head GPT+EGPT merger.** Use V15,
      V16, V19, V27, V31, V32. Lead with the COPA/BoolQ delta. Show that
      mixing improves over both pure-GPT and pure-EGPT on the union.

#### Tables — verify, fix, generate
- [ ] Pick one canonical "Avg" scheme. Recommended: 10-task with
      acc\_norm-where-available (matches plot_v27_egrad_edesc and the
      headline scatter). Update prose anywhere a 9-task or 7-task scheme
      was assumed (CROSSCHECK.md flags these).
- [ ] Fix wrong specific numbers flagged in CROSSCHECK.md:
      V31/V32 ARC-C, V35/V36 Avg, V11 Avg-bold in tab:mixed_results.
- [ ] Add table generators for currently hand-written tables:
      tab:layer_sim, tab:layer_sim_egrad (need activation cos-sim values
      from analyze_layer_activations_*); tab:layer_merger (need training
      loss + activation-sim); loss column of tab:cosreg_benchmarks
      (training loss must come from wandb or a saved metrics file —
      consider `make_metrics.py` that pulls wandb run summaries).
- [ ] Re-run `make_tables.py` after each new model evaluation; commit
      both the regenerated table and any prose changes in the same commit.

#### Bar-chart generator
- [ ] Add `scripts/multi-block-ablation/make_barcharts.py` mirroring
      make_tables.py. One chart per ≤5-model comparison; consume the same
      MODELS registry. Output: `results/multi-block-ablation/plots/bar_<name>.{pdf,png}`.
- [ ] Replace the body of tab:egpt_gap (and friends listed above) in the
      main text with `\includegraphics{plots/bar_<name>}`. Keep the table
      in the appendix as `\input{tables/<name>}`.

#### Models to (re-)run / add for the new story

- [ ] V52, V53 final eval — they test stronger / ramped cosreg (already
      training; eval as soon as 30k done).
- [ ] V54, V55 (parallel-GPT baseline) eval — needed to make
      cos(proj(attn)) vs cos(proj(ffn)) apples-to-apples (RESULTS.md, Apr 25).
- [ ] V40 / V50 400M cosreg — currently incomplete; finish to round out the
      scaling claim.
- [ ] **New** GPT+EGPT *attention-head* merger ablation at 144M:
      a 12×1 model where each block has $k$ EGrad heads + $(12{-}k)$ GPT
      heads with separate output projections (one $W_{O,\mathcal{G}}$
      and the energy heads' $W_{Q,h}^\top$ rule). Sweep $k\in\{0,3,6,9,12\}$.
      This generalises the V15→V31 ladder and gives a clean trade-off curve
      for the merger section.
- [ ] **New** GPT+EGPT merger at 400M: lift the best $k$ from the 144M sweep
      to 24×1 d=1024, train 30k steps, eval. Headline result of section 6.

#### Figures
- [ ] Replace `headline_ppl_acc_vs_params.pdf` with a deep-EGPT-focused
      headline that only includes the models named in the new story:
      V0, V1, V9, V1_400m, V19, V31 (and the new merger model when ready).
      Generate via a new `plot_headline_v3.py` keyed off MODELS_FOR_PAPER.
- [ ] Add a "consecutive-layer cosine sim" figure that compares:
      V0 GPT, V1 EGPT, V31 EGrad, V19 MixH, V39 cosreg. Currently the
      paper shows separate per-pair figures; one panel comparing the
      five would make the alignment story land harder.
- [ ] COPA / BoolQ deltas figure: bar chart of V0, V1, V15, V19, V31, V32
      *only on* COPA + BoolQ — these are where EGPT wins, and where the
      merger picks up most of the gain.

#### Mechanics
- [ ] Add a CI-style script `scripts/regen_paper_artifacts.sh` that runs
      make_tables.py + make_barcharts.py + (later) any plot-script that
      reads from harness_*.json. Run it before every paper recompile.
- [ ] Add `paper/build.sh` that calls pdflatex twice + bibtex once.
- [ ] Add the `tables/` and `plots/` artefacts to `.gitignore`-exception
      list as needed (currently the JSON whitelist suffices because
      tables/*.tex are tracked manually).

## Setup
- [ ] Verify unsharded checkpoints load correctly with HF AutoModelForCausalLM
- [ ] Identify which unsharded checkpoints correspond to best small-scale configs
- [ ] Set up wandb project `energy-inference-large`

## Inference-only experiments (no retraining)
- [ ] Implement alpha sweep at inference time on 400M model (curl vs grad mix)
- [ ] Implement multi-walker Langevin inference (K=4,8 walkers, annealing)
- [ ] Eval: MMLU, BBH, GSM8K on base checkpoint vs alpha-modulated inference
- [ ] Compare inference strategies: greedy vs α-schedule vs multi-walker

## Fine-tuning / grafting experiments
- [ ] Port `train_graft_curriculum_proj_20260408.py` to dolomite-engine format
- [ ] RL fine-tuning of energy dynamics on reasoning tasks (frozen backbone)
- [ ] Split-proj vs single-proj ablation at 400M scale
- [ ] Sweep RL hyperparams: rl_ratio, entropy_coeff, step budget

## Analysis
- [ ] Check if energy correlates with correctness on 400M (vs small-scale result)
- [ ] Plot energy trajectories across recurrence steps on 400M
- [ ] Compare energy dynamics: 400M s1+6×EE+s1 vs 410m hybrid models

## 160M multi-block EGPT ablation (new)

Goal: test whether 12 distinct EGPT blocks can match GPT at 160M scale, and whether they
collapse to fewer effective blocks (motivating weight-sharing / pruning).

- [x] V1: 12 distinct EGPT blocks, dual proj (proj_attn + proj_ff, unconstrained) — job 974324 RUN (step 10, loss=11.7)
- [x] V2: 6 EGPT blocks, 2x recurrence each, dual proj, ~176M — job 974342 PEND
- [x] V3: 12 blocks, shared attn+FF weights, independent proj per layer (~163M, d=1152) — job 974343 PEND
- [x] V4: same as V3 but 4 heads × wider head_dim + 2x FFN (~174M, d=1152) — job 974344 PEND

Post-training analysis for all variants:
- [ ] Measure CKA / weight similarity between blocks to detect functional collapse
- [ ] Try pruning to fewer blocks and evaluate perplexity drop
- [ ] Benchmark vs pure GPT-160M and 410m_hybrid on WikiText / Nematron held-out

Infrastructure:
- [x] Find egpt*.yml example — `configs/energy/baseline_3EGPT_9iter.yml`; nematron: `/proj/datasets/granite-4-datasets-megatron-merged/`
- [x] Token count: 30k steps × 4 GPUs × b=4 × accum=4 × seq=4096 = 7.86B tokens
- [x] Update CLAUDE.md: wandb logging, full run script in RESULTS.md, training time estimation
- [x] Add preemptable queue bsub commands (requires `-G grp_preemptable`)
- [x] Checkpointing + auto-continuation scripts (`scripts/multi-block-ablation/run_v*.sh`)
- [x] `shared_backbone` config option + weight-tying in BaseModelMixin (for V3/V4)
