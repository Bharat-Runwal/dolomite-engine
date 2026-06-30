# BoltzmannMoE scale-up plan — 3B and 7B hybrid configs

Written 2026-06-30. Scales the 680M `h1_boltz_moe_580m_8x4096_d1536` recipe to
two new operating points: ~3B and ~7B total parameters.

---

## Param-counting formula (recap)

Tied embedding, hybrid GPT-prefix + EGPT-tail, all attention is QKVO with no
GQA (KV heads = Q heads). GPT MLP is 3-matrix SwiGLU. EGPT MLP is 2-matrix
BoltzmannMoE.

```
emb              = vocab(100352) * d
per_GPT layer    = 4·d²  +  3·d·I_gpt
per_EGPT block   = 4·d²  +  2·d·I_total       where I_total = K · I_e
total            = emb + num_gpt·per_GPT + num_egpt·per_EGPT
```

Sanity check on the 680M reference
(`d=1536, 11 GPT + 1 EGPT (×6 iterated), I_gpt=6144, K=8, I_e=4096`):

| Block | Params |
|-------|-------:|
| Embedding (tied) | 154M |
| 11 GPT layers | 415M |
| 1 EGPT block | 110M |
| **Total** | **679M** |

(The YAML header comment says "~580M" but it’s a stale label — the actual
count is 679M, which matches what was used to advertise the model as 680M.)

---

## Search results — 3B candidates

Targeting ~3.0B total, with 12-24 GPT layers + 4-6 EGPT blocks (distinct,
not iterated), K=8 experts, GPT FFN at 4·d width.

| d | GPT | EGPT | I_gpt | I_e | emb | GPT | EGPT | Total |
|---:|----:|----:|------:|----:|----:|----:|-----:|------:|
| 2048 | 20 | 4 |  8192 | 4096 | 206M | 1342M |  604M | 2.15B |
| 2048 | 24 | 4 |  8192 | 4096 | 206M | 1611M |  604M | 2.42B |
| 2048 | 20 | 6 |  8192 | 4096 | 206M | 1342M |  906M | 2.45B |
| 2048 | 24 | 6 |  8192 | 4096 | 206M | 1611M |  906M | 2.72B |
| 2304 | 18 | 6 |  9216 | 4096 | 231M | 1529M | 1033M | 2.79B |
| 2304 | 20 | 6 |  9216 | 4096 | 231M | 1699M | 1033M | 2.96B |
| 2560 | 14 | 6 | 10240 | 4096 | 257M | 1468M | 1164M | 2.89B |
| **2560** | **18** | **4** | **10240** | **4096** | **257M** | **1887M** | **776M** | **2.92B** ★ |
| 2560 | 16 | 6 | 10240 | 4096 | 257M | 1678M | 1164M | 3.10B |

### 3B pick: `h2_boltz_moe_3b_d2560_18gpt_4egpt_8x4096.yml`

- **d=2560, 14 GPT + 4 EGPT, K=16, I_e=4096 → 3.17B** (revised from initial
  K=8 design; bumped to K=16 to follow modern MoE scaling).
- 14 GPT layers covers the user-requested 12-24 range; 4 EGPT blocks
  hits the low end of the 4-6 requested range. Dropping 4 GPT layers
  from the initial 18 GPT compensates for the extra ~670M in expanded
  MoE params from K=8→16, keeping total close to 3B.
- head_dim=128, num_heads=20 (d/head_dim = 20, divides cleanly).
- Attention multiplier = 1/sqrt(128) ≈ 0.0884.
- 4 EGPT blocks each iterated 3× with `iter_dropout_range=1` →
  12 effective EGPT passes, training samples [2, 4] iters per block.

---

## Search results — 7B candidates

| d | GPT | EGPT | I_gpt | I_e | emb | GPT | EGPT | Total |
|---:|----:|----:|------:|----:|----:|----:|-----:|------:|
| 3072 | 22 | 6 | 12288 | 4096 | 308M | 3322M | 1434M | 5.06B |
| 3072 | 24 | 6 | 12288 | 4096 | 308M | 3624M | 1434M | 5.37B |
| 3584 | 22 | 4 | 14336 | 4096 | 360M | 4521M | 1145M | 6.03B |
| 3584 | 24 | 4 | 14336 | 4096 | 360M | 4933M | 1145M | 6.44B |
| 4096 | 16 | 4 | 16384 | 4096 | 411M | 4295M | 1342M | 6.05B |
| 4096 | 18 | 4 | 16384 | 4096 | 411M | 4832M | 1342M | 6.59B |
| **4096** | **20** | **4** | **16384** | **4096** | **411M** | **5369M** | **1342M** | **7.12B** ★ |
| 4096 | 18 | 6 | 16384 | 4096 | 411M | 4832M | 2013M | 7.26B |
| 4096 | 16 | 6 | 16384 | 4096 | 411M | 4295M | 2013M | 6.72B |

### 7B pick: `h3_boltz_moe_7b_d4096_20gpt_4egpt_8x4096.yml`

- **d=4096, 16 GPT + 4 EGPT, K=16, I_e=4096 → 7.12B** (revised from initial
  K=8 design; bumped to K=16 to follow modern MoE scaling).
- 16 GPT layers, 4 EGPT, K=16 experts. Trading 4 GPT layers for K=8→16
  happens to leave total exactly at 7.12B.
- head_dim=128, num_heads=32 — matches Llama-2-7B / Mistral-7B head config.
- 4 EGPT blocks each iterated 3× with `iter_dropout_range=1` →
  12 effective EGPT passes, training samples [2, 4] iters per block.
- EGPT-block compute ~2× vs K=8 because Boltzmann soft routing
  activates all K experts per token.

---

## EGPT placement: end-stacked vs interleaved

The configs use the **all-at-end** layout — last 4 layers are EGPT. This is
the cleanest scaffolding (clear GPT prefix vs EGPT tail; YAML anchors stay
trivial) and matches the h1_boltz_moe_580m structure (which puts the single
EGPT block at the end and iterates it 6×).

**Alternative — interleaved**: e.g. for 3B with 18 GPT + 4 EGPT, place EGPT
at layers `[6, 12, 18, 22]` (every 4-6 layers). Pros: lets routing react
to features at multiple depths; matches how Mixture-of-Depths / DenseFormer
papers report best perplexity. Cons: harder to read in YAML, harder to
ablate (e.g. swap EGPT→standard GPT).

Suggest running the end-stacked config first; if it underperforms a same-
param GPT-only baseline, try interleaved as a second pass.

---

## Recurrence knob

`layer_iterations` is exposed as a list of length `num_layers`. The configs
default to `[1]*N` (each block used once). To recover h1-style recurrence:

```yaml
# 3B: make the final EGPT block recurrent ×6
layer_iterations: [1, 1, ..., 1, 6]    # 21 ones + one 6

# 3B: make all 4 EGPT blocks recurrent ×2
layer_iterations: [1]*18 + [2, 2, 2, 2]
```

Recurrence does not change param count (weights are shared across
iterations) but it does increase FLOPs and activation memory roughly
linearly. For the 7B, recurrence on top of d=4096 may push us out of
memory — leave as `[1]*24` for the first run.

---

## LR, tokens, GPUs

| Model | LR (peak) | Tokens | Steps | µbs | accum | seq | GPUs | ETA |
|-------|----------:|-------:|------:|----:|------:|----:|-----:|----:|
| 680M (h1) | 1e-3 | 65B  | 124k | 4 | 4 | 4096 | 32 | ~1.3 d |
| **3B (h2)** | **6e-4** | **80B** | **152k** | 2 | 8 | 4096 | 32 | **~3 d** |
| **7B (h3)** | **3e-4** | **140B** | **134k** | 2 | 8 | 4096 | 32 | **~9 d** |

Rationale:
- **LR scaling**: rough rule `lr ∝ 1/sqrt(d)` (used by GPT-3, Cerebras-GPT,
  μP recipes). From d=1536 → 2560 → 4096, sqrt ratios are 1.29× and 1.63×.
  Round-numbered LRs that respect this: 1e-3 → 6e-4 → 3e-4.
- **Tokens**: Chinchilla-optimal is ~20 tokens/param (60B for 3B, 140B for
  7B). Our 680M run did 65B (~95 t/p, well over Chinchilla). For 3B we
  recommend 80B (~27 t/p, slightly above optimal); for 7B exactly 140B
  (Chinchilla). The 3B can be extended cheaply; the 7B run length is the
  binding constraint.
- **Wall-clock**: 680M trained at ~15B tok/day on 8 H100s (~1.3 days for
  65B). FLOPs/token scales linearly with non-embedding params; on 32 H100s
  with FSDP-2 we expect ~50B tok/day for 3B and ~16B tok/day for 7B. ETAs
  in the table assume those throughputs; double them if intra-node bandwidth
  on a 4-node FSDP-2 setup degrades.
- **Effective batch size**: 32 GPUs × µbs=2 × accum=8 × seq=4096
  = **2.1M tokens/step** (1M for 3B since µbs=2 is the same but we report
  per-step; same formula). This is well above the 680M's 524k tok/step,
  which helps loss curves at the larger model sizes.
- **Activation memory**: At d=4096 we expect µbs=2 with `gradient_checkpointing_method: block`
  to fit in 80GB. If OOM, drop to µbs=1 + accum=16 (same effective batch).

---

## GPU count recommendation

- **3B (h2)**: 32 H100s (4 nodes × 8). Should fit comfortably with FSDP-2.
- **7B (h3)**: 32 H100s minimum; **64 H100s preferred** for wall-clock.
  - The 9-day estimate on 32 GPUs is preemptable-queue territory — make sure
    the auto-resume path is verified (`load_args.load_path` is set; the
    project memory note `feedback_dolomite_resume.md` flags that this is
    silently broken without it).
  - If 64 GPUs are available, double `gradient_accumulation_steps` from 8 → 16
    is unnecessary; instead drop steps from 134k → 67k while keeping the
    same effective batch — same tokens, half the wall-clock.

---

## Files written

```
configs/boltzmann_moe/h2_boltz_moe_3b_d2560_18gpt_4egpt_8x4096.yml
configs/boltzmann_moe/h3_boltz_moe_7b_d4096_20gpt_4egpt_8x4096.yml
experiments/boltzmann-moe/SCALE_UP_PLAN.md                          # this file
```

No jobs submitted. To launch (after smoke-testing on 1 GPU for 100 steps to
verify the new layer_iterations / num_layers wiring):

```bash
REPO=/proj/dmfexp/nima/Code/dolomite-engine
bsub -q preemptable -G grp_preemptable \
    -J egpt_h2_boltz_moe_3b \
    -gpu "num=8/task:mode=exclusive_process" \
    -n 4 -M 64G -W 24:00 \
    -o "$HOME/bsub_logs/egpt_h2_boltz_moe_3b_%J.stdout" \
    -e "$HOME/bsub_logs/egpt_h2_boltz_moe_3b_%J.stderr" \
    <<'EOF'
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:${PYTHONPATH:-}
bash $REPO/scripts/common/pretrain.sh \
    $REPO/configs/boltzmann_moe/h2_boltz_moe_3b_d2560_18gpt_4egpt_8x4096.yml
EOF
```

---

## Caveats

1. **FFN:Attn imbalance (from CLAUDE.md)**: The B-series 422M MoE
   underperformed the 143M V1 baseline because FFN had 21× more params than
   attention. In these new configs:
   - 3B: GPT layers have FFN:Attn = `3·d·10240 / 4·d²` = 30720/10240 = 3.0×;
     EGPT layers have `2·d·32768 / 4·d²` = 65536/10240 = 6.4×.
     Overall block-weighted ≈ 3.7× FFN:Attn — much better than 21×.
   - 7B: GPT = `3·16384/4·4096` = 3.0×; EGPT = `2·32768/4·4096` = 4.0×.
     Block-weighted ≈ 3.2× FFN:Attn. Good.
2. **Recurrence default off**: To match h1's "recurrent EGPT block" design,
   set `layer_iterations: [1]*(N-1) + [k]` with `k > 1`. Not done in this
   first version because (a) the user asked for 4-6 *distinct* EGPT blocks
   and (b) it's a tuneable knob, not a fixed architectural choice.
3. **lm-eval generation**: As of 2026-06-29, register-equipped checkpoints
   have a known bug in cached-decode (see
   `register_energy/REGISTER_DECODE_BUG.md`). These BoltzMoE configs do NOT
   use registers (`model_type: energy`, not `register_energy`), so they are
   unaffected — but worth confirming on the first checkpoint.
4. **Param count vs B-series reporting**: B-series headers used "param
   count w/o emb"; this plan reports total params (including 257M / 411M
   embedding). When comparing to B-series, subtract emb to get "non-embed".
5. **All EGPT at end** is the cleanest scaffold but probably not optimal —
   interleaving EGPT blocks may help. Tackle that as a follow-up if h2/h3
   prove the BoltzMoE recipe works at scale at all.
