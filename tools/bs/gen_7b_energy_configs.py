#!/usr/bin/env python3
"""Generate the 7B energy-MoE configs (for review).

Recipe: GPT layers = standard MoE (the reference 128-expert
block), EGPT layers = energy_attention + dual_unconstrained. d=1536 everywhere
(Bharat's width). Data = math mix. The 4 = {Boltzmann, std} EGPT-MLP x {recursive,
h3 no-recursion}.

  1. boltz_rec   : 20 GPT + 4 EGPT, iters [1]*20+[6,6,9,9], EGPT=BoltzmannMoE  (~3.9B)
  2. boltz_h3    : 32 GPT + 8 EGPT, iters [1]*40,          EGPT=BoltzmannMoE  (~6.3B)
  3. stdmoe_rec  : 20 GPT + 4 EGPT, iters [1]*20+[6,6,9,9], EGPT=std MoE       (~4.1B)
  4. stdmoe_h3   : 32 GPT + 8 EGPT, iters [1]*40,          EGPT=std MoE       (~6.8B)

GPT MoE block mirrors Bharat's 7B exactly EXCEPT use_interleaved_weights=false
(no sonicmoe kernel in the energy build -> the non-kernel MoE path asserts
use_interleaved_weights is false). Run as a review artifact; throughput will be
poor (energy MoE is pure-pytorch, no kernels).
"""
import os

OUT = "configs/boltzmann_moe"
TOK = "/proj/dmfexp/energy-gpt/data/granite-4.0-tiktoken"
CACHE = "/proj/dmfexp/energy-gpt/cache-bsaha"
CKPT = "/proj/dmfexp/energy-gpt/checkpoints-bsaha/boltzmann_sweep"

DATASETS = f"""datasets:
  - class_name: MegatronDataset
    data_name: Megatron
    data_sampling_ratio: 1
    class_args:
      eval_steps: 2
      data_cache_path: {CACHE}
      data_path:
        - 0.35
        - /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_0
        - 0.35
        - /proj/datasets/granite-4-datasets-megatron-merged/web-nemotron-cc-hq-p2_1
        - 0.15
        - /proj/datasets/granite-4-datasets-megatron-merged/megamath-web-pro_0
        - 0.15
        - /proj/datasets/granite-4-datasets-megatron-merged/finemath-3plus-rewritten_0
      split: 99,0.5,0.5
      sequence_length: 4096

tokenizer_args:
  tokenizer_name: {TOK}
"""

# --- Bharat's standard MoE block, on GPT layers (and EGPT for the std variant) ---
# use_interleaved_weights=false: no sonicmoe kernel in this build, so the
# non-kernel MoE path requires it to be false (else assert error on forward).
GPT_ATTN = ("{sequence_mixer_type: softmax_attention, num_attention_heads: 24, "
            "num_key_value_heads: 24, add_bias: false, attention_multiplier: 0.125}")
EGPT_ATTN = """sequence_mixer_type: energy_attention
        num_attention_heads: 24
        num_key_value_heads: 24
        add_bias: false
        attention_multiplier: 0.125"""

def std_moe_block(anchor=None):
    a = f"&{anchor} " if anchor else ""
    return (f"{a}{{mlp_type: MoE, activation_function: swiglu, intermediate_size: 256, "
            f"num_experts: 128, num_experts_per_tok: 8, normalized_topk: true, "
            f"shared_intermediate_size: 1024, use_interleaved_weights: false, "
            f"shared_expert_gating: false, add_bias: false}}")

BOLTZ_BLOCK = """mlp_type: BoltzmannMoE_Energy_MLP
        intermediate_size: 32768
        n_experts: 128
        temperature: 1.0
        repulsion_coef: 0.1
        n_repulsion_pairs: 4
        gelu_grad_method: sigmoid
        dropout: 0.0
        add_bias: false"""

def build(name, n_gpt, n_egpt, iters, egpt_kind, desc):
    # egpt_kind: "boltz" (energy_attn + BoltzmannMoE), "std" (energy_attn + std MoE),
    #            "recgpt" (softmax_attn + std MoE everywhere = pure recurrent GPT baseline)
    n_layers = n_gpt + n_egpt
    assert len(iters) == n_layers, (name, len(iters), n_layers)

    # sequence_mixer_blocks
    smb = [f"      - &gpt_attn {GPT_ATTN}"] + [f"      - *gpt_attn"] * (n_gpt - 1)
    for _ in range(n_egpt):
        if egpt_kind == "recgpt":
            smb.append(f"      - *gpt_attn")          # RecGPT: softmax everywhere, no energy attn
        else:
            smb.append(f"      - {EGPT_ATTN}")        # energy_attention on the recurrent layers

    # mlp_blocks: n_gpt std-MoE (anchor first) + n_egpt (boltz / std-MoE / recgpt std-MoE)
    mlp = [f"      - {std_moe_block('gpt_moe')}"] + [f"      - *gpt_moe"] * (n_gpt - 1)
    if egpt_kind == "boltz":
        for _ in range(n_egpt):
            mlp.append(f"      - {BOLTZ_BLOCK}")
    else:  # "std" and "recgpt" both use standard MoE on the recurrent layers
        for _ in range(n_egpt):
            mlp.append(f"      - *gpt_moe")

    cfg = f"""# {name}
# {desc}
# GPT layers: softmax_attention + standard MoE (128 experts, top-8) --
#   standard-MoE recipe, reference 7B expert geometry (d=1536).
# {"Recurrent layers: softmax_attention + standard MoE (RecGPT baseline -- NO energy attention anywhere)." if egpt_kind=="recgpt" else "EGPT layers: energy_attention + dual_unconstrained + " + ("BoltzmannMoE_Energy_MLP (sigmoid)." if egpt_kind=="boltz" else "standard MoE.")}
# Structure: {n_gpt} GPT + {n_egpt} EGPT = {n_layers} stored layers; layer_iterations sum (effective depth) = {sum(iters)}.
# Data: math mix (nemotron-cc-hq + megamath + finemath). For review.
# NOTE: use_interleaved_weights=false (no sonicmoe kernel here); energy MoE is
#   pure-pytorch (no kernel) -> correct but low throughput, as flagged by Bharat.

{DATASETS}
model_args:
  model_class: AutoModelForCausalLM
  pretrained_config:
    model_type: energy
    num_iterations: 1
    num_pre_layers: 0
    num_post_layers: 0
    layer_iterations: {iters}
    initializer_range: 0.02
    layer_norm_epsilon: 1.0e-05
    normalization_function: rmsnorm
    position_embedding_type: rope
    rope_dim: 64
    hidden_size: 1536
    num_layers: {n_layers}
    init_method: normal
    tie_word_embeddings: true
    energy_proj_type: dual_unconstrained
    bos_token_id: 100257
    eos_token_id: 100257
    pad_token_id: 100256
    vocab_size: 100352
    max_position_embeddings: 4096
    sequence_mixer_blocks:
{chr(10).join(smb)}
    mlp_blocks:
{chr(10).join(mlp)}

tuning_args:
  tuning_method: pretraining

save_args:
  save_path: {CKPT}/{name}
  save_interval: 1000

logging_args:
  log_interval: 10
  experiments_tracker_name: wandb
  wandb_args:
    project: dolomite
    name: {name}

training_parameters:
  num_training_steps: 124000
  eval_interval: 10000
  micro_batch_size: 4
  gradient_accumulation_steps: 4
  eval_during_training: false
  gradient_clipping: 1

optimizer_args:
  class_name: TorchAdamW
  class_args:
    lr: 3e-4
    weight_decay: 0.1
    betas: [0.9, 0.95]
    eps: 1.0e-8

lr_scheduler_args:
  lr_decay_style: cosine
  lr_decay_factor: 0.1
  num_warmup_steps: 2000
  num_constant_steps: 0
  num_decay_steps: 122000

mixed_precision_args:
  dtype: bf16

distributed_args:
  fsdp_algorithm: 2
  gradient_checkpointing_method: block
  torch_compile: false
  stage: 0

load_args:
  load_path: {CKPT}/{name}
"""
    path = os.path.join(OUT, f"{name}.yml")
    with open(path, "w") as f:
        f.write(cfg)
    print(f"wrote {path}  ({n_gpt}GPT+{n_egpt}EGPT, {n_layers} layers, eff depth {sum(iters)}, EGPT={egpt_kind})")

REC = [1]*20 + [6, 6, 9, 9]      # 20 GPT + 4 EGPT, effective 50
H3  = [1]*40                      # 32 GPT + 8 EGPT, no recursion

build("s7b_egpt_boltz_rec_20gpt4egpt6699_d1536",  20, 4, REC, "boltz",
      "7B energy: recursive 20GPT+4EGPTx[6,6,9,9], Boltzmann MoE on EGPT")
build("s7b_egpt_boltz_h3_32gpt8egpt_d1536",       32, 8, H3,  "boltz",
      "7B energy: h3 no-recursion 32GPT+8EGPT, Boltzmann MoE on EGPT")
build("s7b_egpt_stdmoe_rec_20gpt4egpt6699_d1536", 20, 4, REC, "std",
      "7B energy: recursive 20GPT+4EGPTx[6,6,9,9], standard MoE on EGPT (baseline)")
build("s7b_egpt_stdmoe_h3_32gpt8egpt_d1536",      32, 8, H3,  "std",
      "7B energy: h3 no-recursion 32GPT+8EGPT, standard MoE on EGPT (baseline)")

# RecGPT baselines: "same structure, only RecGPT" (per design discussion).
# Pure recurrent GPT: softmax_attention + standard MoE on ALL layers (no energy
# attention anywhere). Param-identical to the stdmoe energy configs (energy attn
# 2d^2 + dual proj 2d^2 == softmax attn 4d^2), so iso-param / iso-FLOP-depth control
# that isolates the entire energy contribution.
build("s7b_recgpt_rec_20l4rec6699_d1536",         20, 4, REC, "recgpt",
      "7B RecGPT baseline: recursive 20+4x[6,6,9,9], softmax+std MoE everywhere (no energy)")
build("s7b_recgpt_h3_40l_d1536",                  32, 8, H3,  "recgpt",
      "7B RecGPT baseline: h3 no-recursion 40 layers, softmax+std MoE everywhere (no energy)")
