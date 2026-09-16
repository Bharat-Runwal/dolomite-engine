#!/bin/bash
# run_avg11_sweep_20260915.sh {verify|submit|status|resubmit|restate}
#
# Drives the full Avg11 re-eval sweep for the two paper tables:
#   tab:headline       (nima/sec/headline_table.tex)
#   tab:all_moe_results(nima/sec/appendices/boltz_moe.tex)
# Every cited checkpoint has the 9 base Avg11 tasks scored but is MISSING
# race + lambada_openai. This adds only those two per checkpoint (see
# reeval_racelambda_20260915.sh) and merges to a complete Avg11 file.
#
# MODES:
#   verify    filesystem check: each path has config.json + a base harness json;
#             flags already-complete (has race+lambada) and merged-present. No bsub.
#   submit    submit a race+lambada job for every checkpoint NOT yet merged.
#   status    how many checkpoints now have a merged *_avg11reeval.json.
#   resubmit  submit only for checkpoints still missing their merged file
#             (use after a wave, for preempted stragglers).
#   restate   run restate_to_avg11_20260914.py (text + --md) over the whole tree.
#
# R3 (headline) is SHARDED-ONLY and is NOT in this list — it needs unsharding
# first and is handled separately.
set -uo pipefail
MODE="${1:-verify}"
REPO=/proj/dmfexp/nima/Code/dolomite-engine
SD="${REPO}/experiments/eval_scripts"
RES="${REPO}/experiments/boltzmann-moe/results"
MBA="${REPO}/experiments/energy-inference/results/multi-block-ablation"

# shortname | absolute unsharded checkpoint dir
#   headline+MoE shared rows (v9, v1_400m, v1, c1) appear once.
CKPTS=(
  # --- baselines (v9/v1_400m/v1 shared with headline) ---
  "v9|${MBA}/v9_gpt_baseline_d1024_lr1e3/unsharded"
  "v1_400m|${MBA}/v1_400m_d1024_lr7e4/unsharded"
  "v1|${MBA}/v1_12x1_d768_lr2e3/unsharded"
  "v58|${MBA}/v58_egpt_1x24_d1024_lr1e3/unsharded"
  # --- B-series ---
  "b1|${RES}/b1_boltz_moe_16x1024_d768_lr2e3/unsharded"
  "b4|${RES}/b4_boltz_moe_repulsion_strong_16x1024_d768_lr2e3/unsharded"
  "b5|${RES}/b5_boltz_moe_rep_strong_dropout_wd_16x1024_d768_lr2e3/unsharded"
  # --- C1 (shared with headline) / d1 ---
  "c1|${RES}/c1_topk_energy_moe_4x2048_top2_d768/unsharded"
  "d1|${RES}/d1_boltz_moe_egpt_12x1_d768/unsharded"
  # --- H-series ---
  "h1_egpt|${RES}/h1_egpt_d768/unsharded"
  "h1_boltz_iso|${RES}/h1_boltz_egpt_moe_d768/unsharded"
  "h1_topk|${RES}/h1_topk_egpt_moe_d768/unsharded"
  "h1_topk_r128|${RES}/h1_topk_egpt_moe_r128_d768/unsharded"
  "h1_boltz_full|${RES}/h1_boltz_moe_fullsize_d768/unsharded"
  "h1_boltz_topk2|${RES}/h1_boltz_topk2_egpt_d768/unsharded"
  "h1_gptmoe_boltz|${RES}/h1_gptmoe_boltz_egpt_d768/unsharded"
  # --- gelu_grad A/B ---
  "h1_tanh_norep|${RES}/h1_boltz_moe_fullsize_tanhexact_norep_d768/unsharded"
  "h1_erf|${RES}/h1_boltz_moe_fullsize_erfexact_d768/unsharded"
  "h1_tanh|${RES}/h1_boltz_moe_fullsize_tanhexact_d768/unsharded"
  # --- h2 ---
  "h2|${RES}/h2_6gpt_2egpt6x_boltz_d768/unsharded"
  # --- 680M Boltzmann (8 steps) ---
  "680m_14k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step14k"
  "680m_18k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step18k"
  "680m_30k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step30k"
  "680m_76k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step76000"
  "680m_102k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step102000"
  "680m_110k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step110000"
  "680m_118k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step118000"
  "680m_124k|${RES}/h1_boltz_moe_580m_8x4096_d1536/unsharded_step124000"
  # --- scale_h3 (3 steps) ---
  "h3_104k|${RES}/scale_h3_8gpt_4egpt_boltz_d1280/unsharded_step104000"
  "h3_120k|${RES}/scale_h3_8gpt_4egpt_boltz_d1280/unsharded_step120000"
  "h3_124k|${RES}/scale_h3_8gpt_4egpt_boltz_d1280/unsharded_step124000"
  # --- 8gpt+4switch (4 steps) ---
  "sw_24k|${RES}/scale_gptmoe_8gpt_4switchmoe_d1280/unsharded_step24000"
  "sw_60k|${RES}/scale_gptmoe_8gpt_4switchmoe_d1280/unsharded_step60000"
  "sw_108k|${RES}/scale_gptmoe_8gpt_4switchmoe_d1280/unsharded_step108000"
  "sw_124k|${RES}/scale_gptmoe_8gpt_4switchmoe_d1280/unsharded_step124000"
  # --- 12moe_K4I2048 (4 steps) ---
  "k4i2048_26k|${RES}/scale_gptmoe_12moe_K4I2048_d1280/unsharded_step26000"
  "k4i2048_65k|${RES}/scale_gptmoe_12moe_K4I2048_d1280/unsharded_step65000"
  "k4i2048_110k|${RES}/scale_gptmoe_12moe_K4I2048_d1280/unsharded_step110000"
  "k4i2048_124k|${RES}/scale_gptmoe_12moe_K4I2048_d1280/unsharded_step124000"
  # --- 12moe_K4I4096 (4 steps) ---
  "k4i4096_14k|${RES}/scale_gptmoe_12moe_K4I4096_d1280/unsharded_step14000"
  "k4i4096_45k|${RES}/scale_gptmoe_12moe_K4I4096_d1280/unsharded_step45000"
  "k4i4096_73k|${RES}/scale_gptmoe_12moe_K4I4096_d1280/unsharded_step73000"
  "k4i4096_96k|${RES}/scale_gptmoe_12moe_K4I4096_d1280/unsharded_step96000"
  # --- gptswitchmoe-680M (3 steps) ---
  "gsw_29k|${RES}/h1_gptswitchmoe_580m_8x4096_d1536/unsharded_step29000"
  "gsw_48k|${RES}/h1_gptswitchmoe_580m_8x4096_d1536/unsharded_step48000"
  "gsw_124k|${RES}/h1_gptswitchmoe_580m_8x4096_d1536/unsharded_step124000"
  # --- headline-only rows ---
  "v0|${MBA}/v0_gpt_baseline_d768/unsharded"
  "v15|${MBA}/v15_energy_grad_mixed_12x1_d768_lr2e3/unsharded"
  "h5|${MBA}/h5_6gpt_1egpt1x_d768/unsharded"
  "h3hl|${MBA}/h3_6gpt_4egpt_d768/unsharded"
  "v41|${MBA}/v41_sandwich_2gpt8e2gpt_d768_lr2e3/unsharded"
  "v76|${MBA}/v76_4gpt_1egpt6x_rmsray_d1024_reg128/unsharded"
  "v73|${MBA}/v73_6gpt_1egpt6x_rmsray_d1280/unsharded"
  "v19|${MBA}/v19_energy_grad_24x1_d1024_lr1e3/unsharded"
  "v31|${MBA}/v31_egrad_attn_24x1_d1024_lr1e3/unsharded"
)

has_merged() { ls "$1"/harness_results_*_avg11reeval.json >/dev/null 2>&1; }
base_json()  { ls "$1"/harness_results_*.json 2>/dev/null | grep -v avg11reeval | grep -v harness_bbh | head -1; }
has_rl_in_base() {
  local b; b=$(base_json "$1"); [ -z "$b" ] && return 1
  python3 -c "import json,sys; r=json.load(open('$b')).get('results',{}); sys.exit(0 if ('race' in r and 'lambada_openai' in r) else 1)" 2>/dev/null
}

case "$MODE" in
  verify)
    ok=0; nocfg=0; nobase=0; complete=0; merged=0
    for row in "${CKPTS[@]}"; do
      name="${row%%|*}"; dir="${row#*|}"
      cfg="no"; base="no"; comp="no"; mrg="no"
      [ -f "$dir/config.json" ] && cfg="yes"
      [ -n "$(base_json "$dir")" ] && base="yes"
      has_rl_in_base "$dir" && comp="yes"
      has_merged "$dir" && mrg="yes"
      printf "  %-16s cfg=%-3s base=%-3s rl_in_base=%-3s merged=%-3s  %s\n" "$name" "$cfg" "$base" "$comp" "$mrg" "$dir"
      [ "$cfg" = yes ] && ok=$((ok+1)) || nocfg=$((nocfg+1))
      [ "$base" = no ] && nobase=$((nobase+1))
      [ "$comp" = yes ] && complete=$((complete+1))
      [ "$mrg" = yes ] && merged=$((merged+1))
    done
    echo "---"
    echo "total=${#CKPTS[@]}  config_ok=${ok}  missing_config=${nocfg}  missing_base_json=${nobase}  already_have_rl=${complete}  already_merged=${merged}"
    ;;
  submit|resubmit)
    n=0
    for row in "${CKPTS[@]}"; do
      name="${row%%|*}"; dir="${row#*|}"
      if [ "$MODE" = resubmit ] && has_merged "$dir"; then continue; fi
      if [ "$MODE" = submit ] && has_merged "$dir"; then echo "skip ${name}: already merged"; continue; fi
      if [ ! -f "$dir/config.json" ]; then echo "SKIP ${name}: no config.json at ${dir}" >&2; continue; fi
      bash "${SD}/reeval_racelambda_20260915.sh" "$dir" "rl_${name}"
      n=$((n+1))
    done
    echo "=== submitted ${n} race+lambda jobs (mode=${MODE}) ==="
    ;;
  status)
    done_n=0; todo=""
    for row in "${CKPTS[@]}"; do
      name="${row%%|*}"; dir="${row#*|}"
      if has_merged "$dir"; then done_n=$((done_n+1)); else todo="${todo} ${name}"; fi
    done
    echo "merged: ${done_n}/${#CKPTS[@]}"
    [ -n "$todo" ] && echo "still missing merged file:${todo}"
    ;;
  restate)
    python "${SD}/restate_to_avg11_20260914.py"
    echo; echo "=== markdown ==="; python "${SD}/restate_to_avg11_20260914.py" --md
    ;;
  *) echo "usage: $0 {verify|submit|status|resubmit|restate}"; exit 2 ;;
esac
