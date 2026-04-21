# TODO — Energy-Inference Large-Scale

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
