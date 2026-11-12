#!/bin/bash
# ICLR27 Boltzmann-MoE rebuttal (reviewer 4PYN, W4a/W4b): 6 sensitivity runs at d=768, 4.00B tokens.
# Base configs are Nima's Table-1 d=768 arms, untouched:
#   G6 = 6S6E Boltzmann-MoE deep (K=4 k=2), trained at lr 1e-3
#   E3 = 6G1x6S Switch-MoE baseline (K=8 k=2), trained at lr 2e-3 (NOTE: not 1e-3 as the paper says)
# All runs: 524288 tok/step (G6_lr5e-4 on 8 GPUs; the other five on 4 GPUs, ga doubled), seq 4096, 524288 tok/step, 7629 steps = 4.00B tokens, seed 42, warmup 500.
#   W4a (learning rate):  G6 @5e-4, E3 @5e-4, E3 @1e-3  -> both arms at {5e-4, 1e-3} (+ existing G6@1e-3, E3@2e-3)
#   W4b (Sinkhorn batch):  G6 micro-batch 1/2/8 at fixed global batch (existing G6 = micro-batch 4)
set -euo pipefail
cd "$(dirname "$0")"
B=../../fwe_sweep
mk() {  # name base lr mbs ga
    local N=$1 BASE=$2 LR=$3 MBS=$4 GA=$5
    sed -e "s#data_cache_path: .*#data_cache_path: /proj/dmfexp/energy-gpt/.cache/megatron_fwe_nemotron_bsaha#" \
        -e "s#save_path: .*#save_path: /proj/dmfexp/energy-gpt/checkpoints-bsaha/rebuttal/$N#" \
        -e "s#^\(    name: \)fineweb_.*#\1$N#" \
        -e "s#^\(  micro_batch_size: \).*#\1$MBS#" \
        -e "s#^\(  gradient_accumulation_steps: \).*#\1$GA#" \
        -e "s#^\(    lr: \).*#\1$LR#" \
        -e "s#LAUNCH: 4 GPUs, tok/step=524288#LAUNCH: 8 GPUs (mbs $MBS x ga $GA), tok/step=524288#" \
        "$B/$BASE.yml" > "$N.yml"
    sed -i "1s#^#\# ICLR27 rebuttal run $N (base: fwe_sweep/$BASE.yml; lr=$LR mbs=$MBS ga=$GA on 8 GPUs)\n#" "$N.yml"
    echo "wrote $N.yml"
}
mk reb_G6_lr5e-4 fineweb_G6_6S6E_isoact_d768 5e-4 4 4
mk reb_E3_lr5e-4 fineweb_E3_switch_K8        5e-4 4 4
mk reb_E3_lr1e-3 fineweb_E3_switch_K8        1e-3 4 4
mk reb_G6_mb1    fineweb_G6_6S6E_isoact_d768 1e-3 1 16
mk reb_G6_mb2    fineweb_G6_6S6E_isoact_d768 1e-3 2 8
mk reb_G6_mb8    fineweb_G6_6S6E_isoact_d768 1e-3 8 2
