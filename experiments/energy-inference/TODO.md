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
