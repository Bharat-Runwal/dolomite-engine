#!/bin/bash
shopt -s extglob 2>/dev/null || true
# collect_flops_wave_20260912.sh — the follow-up for the ICLR FLOPS wave.
#
# RUN THIS when configs/iclr_flops/* have finished (bjobs shows no iclr_* jobs, or
# each save_path's latest_checkpointed_iteration.json reads 30000).
#
#   bash experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh status
#   bash experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh eval
#   bash experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh table
#   bash experiments/boltzmann-moe/scripts/collect_flops_wave_20260912.sh proxy
#
# status : per-arm step count + whether the job is still alive
# eval   : unshard final ckpt + submit lm-eval-harness for every finished arm
#          (1 GPU each on normal/grp_ebm, which schedules immediately)
# table  : build the quality-vs-k/K frontier from the harness JSONs
# proxy  : fit the cheap proxy router on each finished checkpoint and measure the
#          paired-NLL cost of proxy routing + top-k (needs `eval` to have unsharded)
#
# QUEUE NOTE (measured 2026-09-12): preemptable is unusable (17.6k PEND, an 8-GPU
# single-host ask unsatisfiable on 588 hosts). Use `normal` with -G grp_ebm and
# <=4 GPUs per job; those start in seconds. 32-GPU quota.

set -uo pipefail
REPO=/proj/dmfexp/nima/Code/dolomite-engine
# 2026-09-13: was iclr_flops ONLY, so every arm added since (the Switch baselines, the
# single-block attribution pair, the 400M tier, the GPT-MoE baselines, the slope pair)
# was invisible to status/eval/table and would never be auto-evaluated on completion.
# Now scans all ICLR config dirs and derives each arm's results dir from its own
# save_path, so no per-dir RES mapping can drift out of sync.
CFGDIRS="$REPO/configs/iclr_flops $REPO/configs/iclr_moebase $REPO/configs/iclr_1blk \
         $REPO/configs/iclr_big $REPO/configs/iclr_ctrl $REPO/configs/iclr_gptmoe \
         $REPO/configs/iclr_slope"
cfg_of() { for d in $CFGDIRS; do [ -f "$d/$1.yml" ] && { echo "$d/$1.yml"; return; }; done; }
res_of() { grep -E '^\s*save_path:' "$(cfg_of "$1")" 2>/dev/null | head -1 | sed 's/.*save_path:[[:space:]]*//'; }
OUT=$REPO/experiments/boltzmann-moe/results/router_analysis
mkdir -p "$HOME/bsub_logs" "$OUT"
MODE="${1:-status}"

arms() { for d in $CFGDIRS; do ls $d/*.yml 2>/dev/null; done | xargs -n1 basename | sed 's/\.yml$//' | sort -u; }
step_of() {
    local f="$(res_of "$1")/latest_checkpointed_iteration.json"
    [ -f "$f" ] && grep -oE '[0-9]+' "$f" | head -1 || echo 0
}

case "$MODE" in

status)
    printf "%-30s %8s %8s  %s\n" ARM STEP TARGET STATE
    for a in $(arms); do
        s=$(step_of "$a")
        t=$(grep -E '^\s*num_training_steps:' $(cfg_of "$a") | grep -oE '[0-9]+' | head -1)
        alive=$(bjobs -noheader -o "stat" -J "$a" 2>/dev/null | tr -d ' ' | head -1)
        [ -z "$alive" ] && alive="-"
        done_flag=""; [ "$s" -ge "$t" ] 2>/dev/null && done_flag="DONE"
        printf "%-30s %8s %8s  %s %s\n" "$a" "$s" "$t" "$alive" "$done_flag"
    done
    ;;

eval)
    for a in $(arms); do
        s=$(step_of "$a")
        t=$(grep -E '^\s*num_training_steps:' $(cfg_of "$a") | grep -oE '[0-9]+' | head -1)
        if [ "$s" -lt "$t" ] 2>/dev/null; then echo "skip $a (step $s < $t)"; continue; fi
        U="$(res_of "$a")/unsharded"
        # NOTE the glob. lm-eval writes harness_results_<ISO timestamp>.json, never the bare
        # name we pass as --output_path, so testing -f "$U/harness_results.json" NEVER matched
        # and this re-submitted an eval for every already-scored arm on every invocation --
        # 7 surplus GPU jobs that pushed us over the 32-GPU quota. Glob for any of them.
        if compgen -G "$U/harness_results*.json" > /dev/null; then echo "skip $a (already evaluated)"; continue; fi
        # ALSO skip when an eval for this arm is already queued or running. The JSON only
        # appears when the eval FINISHES (~1 h), so the glob above is blind for that whole
        # window and a 10-min babysitter loop submitted the same eval repeatedly --
        # ev_iclr_pure_hop_isoP and ev_iclr_big_hop_pure each got a duplicate GPU job.
        if [ -n "$(bjobs -noheader -o 'jobid' -J "ev_$a" 2>/dev/null | head -1)" ]; then
            echo "skip $a (eval already in flight)"; continue
        fi
        # An eval already queued/running leaves no JSON yet, so without this the
        # level-triggered babysitter resubmits the same arm every cycle.
        if bjobs -noheader -o "job_name" 2>/dev/null | grep -qx "ev_$a"; then
            echo "skip $a (eval already in flight)"; continue
        fi
        bsub -q normal -G grp_ebm -J "ev_$a" -gpu "num=1/task:mode=exclusive_process" \
             -n 1 -M 48G -W 04:00 \
             -o "$HOME/bsub_logs/ev_${a}_%J.stdout" -e "$HOME/bsub_logs/ev_${a}_%J.stderr" <<EOF >/dev/null
#!/bin/bash
set -uo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "\$TMPDIR"
# NO lm-eval install: we use the vendored pinned harness via PYTHONPATH
uv pip install accelerate "pyarrow>=20" -q
source /proj/dmfexp/nima/Code/dolomite-engine/experiments/eval_scripts/eval_tasks.sh
eval_env
eval_assert_pin
cd $REPO
if [ ! -f "$U/model.safetensors" ]; then
  C=/tmp/unshard_${a}_\$\$.yml
  printf "load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
      "$(res_of "$a")" "$s" "$U" > "\$C"
  python -m lm_engine.unshard --config "\$C" && rm -f "\$C"
fi
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \
  --model hf --model_args "pretrained=$U,dtype=bfloat16,trust_remote_code=True" \
  --tasks \$EVAL_TASKS \
  --device cuda:0 --batch_size 4 --trust_remote_code \
  --output_path "$U/harness_results.json"
EOF
        echo "submitted ev_$a"
    done
    ;;

table)
    source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
    python - <<'PY'
import json, glob, os, sys
sys.path.insert(0, "/proj/dmfexp/nima/Code/dolomite-engine")
from lm_engine.arguments import TrainingArgs
from lm_engine.utils import load_yaml
# 2026-09-13: was iclr_flops only, so evaluated arms from every other dir were invisible
# to the table. Now scans all ICLR config dirs and takes each arm's results dir from its
# own save_path, matching the shell helpers above.
CFGDIRS = ["configs/iclr_flops", "configs/iclr_moebase", "configs/iclr_1blk",
           "configs/iclr_big", "configs/iclr_ctrl", "configs/iclr_gptmoe",
           "configs/iclr_slope"]
ROOT = "/proj/dmfexp/nima/Code/dolomite-engine"
ACC=["arc_challenge","arc_easy","boolq","copa","hellaswag","openbookqa","piqa","sciq","winogrande","mmlu"]
rows=[]
cfgs = []
for d in CFGDIRS:
    cfgs += sorted(glob.glob(f"{ROOT}/{d}/*.yml"))
for c in cfgs:
    a=os.path.basename(c)[:-4]
    b=TrainingArgs(**load_yaml(c)).model_args.pretrained_config["mlp_blocks"][-1]
    # Key names differ by class: EnergyFF_BoltzmannMoE / TopK_Energy_MoE_MLP use
    # n_experts + top_k, while the standard Switch-style MoE uses num_experts +
    # num_experts_per_tok. A plain b["n_experts"] raised KeyError on the new GPT-MoE arms.
    K = b.get("n_experts") or b.get("num_experts")
    if K is None:
        continue                      # not a mixture block (e.g. a dense-FFN-only arm)
    tk = b.get("top_k") or b.get("num_experts_per_tok") or K
    mt = b.get("mlp_type","")
    learned = mt in ("TopK_Energy_MoE_MLP", "MoE")
    hop = b.get("expert_kind")=="hopfield"
    save = TrainingArgs(**load_yaml(c)).save_args.save_path
    js=glob.glob(f"{save}/unsharded/harness_results*.json")
    if not js: rows.append((a,K,tk,tk/K,learned,hop,None,None,mt)); continue
    d=json.load(open(sorted(js)[-1])); r=d.get("results",d)
    accs=[]
    for t in ACC:
        v=r.get(t,{})
        m=v.get("acc_norm,none", v.get("acc,none"))
        if m is not None: accs.append(m)
    ppl=r.get("wikitext",{}).get("word_perplexity,none")
    rows.append((a,K,tk,tk/K,learned,hop,100*sum(accs)/len(accs) if accs else None,ppl,mt))
print(f"{'arm':30s} {'router':8s} {'exp':4s} {'K':>3s} {'k':>2s} {'k/K':>6s} "
      f"{'FLOPratio':>9s} {'avg%':>7s} {'wikiPPL':>8s}")
for a,K,tk,kk,learned,hop,avg,ppl,mt in rows:
    # cost of the mixture block relative to dense soft, with a free proxy router
    print(f"{a:30s} {('switch' if mt=='MoE' else 'learned' if learned else 'energy'):8s} {('hop' if hop else 'swiglu' if mt=='MoE' else 'w1w2'):6s} "
          f"{K:3d} {tk:2d} {kk:6.3f} {kk:9.3f} "
          f"{('%7.2f'%avg) if avg else '      -':>7s} {('%8.2f'%ppl) if ppl else '       -':>8s}")
print()
print("FLOPratio = k/K = mixture-block cost vs dense soft, assuming the cheap proxy")
print("router (0.21% of exact). With the EXACT energy router it is 0.5+0.5*k/K instead.")
print("avg% here is avg10_norm (10 tasks, MMLU INCLUDED). The h1 reference points below")
print("are RESTATED onto avg10 -- do NOT use the avg9 figures printed in PROGRESS.md /")
print("CLAUDE.md (0.501 etc.), which are ~1.5pp higher. See AVG10_RESTATED.md.")
print("  h1 w1w2 K=4 I_e=2048 dense : avg10 48.32  ppl 36.48")
print("  h1 w1w2 K=4 I_e=2048 top-2 : avg10 47.14  ppl 36.37")
print("  h1 learned K=4      top-2  : avg10 48.17  ppl 39.79")
print("NOTE those h1 arms use I_e=2048; the arms above use I_e=512-1024, so they are")
print("NOT iso-expert-width with h1 -- compare within this table, not across.")
PY
    ;;

proxy)
    for a in $(arms); do
        U="$(res_of "$a")/unsharded"
        [ -f "$U/model.safetensors" ] || { echo "skip $a (not unsharded; run 'eval' first)"; continue; }
        bsub -q normal -G grp_ebm -J "px_$a" -gpu "num=1/task:mode=exclusive_process" \
             -n 1 -M 48G -W 04:00 \
             -o "$HOME/bsub_logs/px_${a}_%J.stdout" -e "$HOME/bsub_logs/px_${a}_%J.stderr" <<EOF >/dev/null
#!/bin/bash
set -uo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "\$TMPDIR"
cd $REPO
# cache (x, E_k) pairs first, then fit the proxy and measure paired NLL
python experiments/boltzmann-moe/scripts/measure_moe_routing_20260912.py \
    --run "$(res_of "$a")" --n_prompts 400 --max_len 1024 --cache_per_iter 6000 --ranks 8 32
python experiments/boltzmann-moe/scripts/fit_proxy_router_20260912.py \
    --run "$(res_of "$a")" --rank 32 --hidden 64 --steps 4000 --n_eval 250
EOF
        echo "submitted px_$a"
    done
    ;;

*) echo "usage: $0 {status|eval|table|proxy}"; exit 1;;
esac
