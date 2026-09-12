#!/bin/bash
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
RES=$REPO/experiments/boltzmann-moe/results/iclr_flops
CFGDIR=$REPO/configs/iclr_flops
OUT=$REPO/experiments/boltzmann-moe/results/router_analysis
mkdir -p "$HOME/bsub_logs" "$OUT"
MODE="${1:-status}"

arms() { ls $CFGDIR/*.yml 2>/dev/null | xargs -n1 basename | sed 's/\.yml$//'; }
step_of() {
    local f="$RES/$1/latest_checkpointed_iteration.json"
    [ -f "$f" ] && grep -oE '[0-9]+' "$f" | head -1 || echo 0
}

case "$MODE" in

status)
    printf "%-30s %8s %8s  %s\n" ARM STEP TARGET STATE
    for a in $(arms); do
        s=$(step_of "$a")
        t=$(grep -E '^\s*num_training_steps:' $CFGDIR/$a.yml | grep -oE '[0-9]+' | head -1)
        alive=$(bjobs -noheader -o "stat" -J "$a" 2>/dev/null | tr -d ' ' | head -1)
        [ -z "$alive" ] && alive="-"
        done_flag=""; [ "$s" -ge "$t" ] 2>/dev/null && done_flag="DONE"
        printf "%-30s %8s %8s  %s %s\n" "$a" "$s" "$t" "$alive" "$done_flag"
    done
    ;;

eval)
    for a in $(arms); do
        s=$(step_of "$a")
        t=$(grep -E '^\s*num_training_steps:' $CFGDIR/$a.yml | grep -oE '[0-9]+' | head -1)
        if [ "$s" -lt "$t" ] 2>/dev/null; then echo "skip $a (step $s < $t)"; continue; fi
        U="$RES/$a/unsharded"
        if [ -f "$U/harness_results.json" ]; then echo "skip $a (already evaluated)"; continue; fi
        bsub -q normal -G grp_ebm -J "ev_$a" -gpu "num=1/task:mode=exclusive_process" \
             -n 1 -M 48G -W 04:00 \
             -o "$HOME/bsub_logs/ev_${a}_%J.stdout" -e "$HOME/bsub_logs/ev_${a}_%J.stderr" <<EOF >/dev/null
#!/bin/bash
set -uo pipefail
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\${PYTHONPATH:-}
export HF_DATASETS_OFFLINE=1 HF_HUB_OFFLINE=1
export TMPDIR=/proj/dmfexp/nima/.cache/tmp && mkdir -p "\$TMPDIR"
uv pip install accelerate lm-eval "pyarrow>=20" -q
cd $REPO
if [ ! -f "$U/model.safetensors" ]; then
  C=/tmp/unshard_${a}_\$\$.yml
  printf "load_args:\n  load_path: %s\n  iteration: %s\nunsharded_path: %s\nmixed_precision_args:\n  dtype: bf16\n" \
      "$RES/$a" "$s" "$U" > "\$C"
  python -m lm_engine.unshard --config "\$C" && rm -f "\$C"
fi
python experiments/energy-inference/scripts/structured-proj/eval_harness.py \
  --model hf --model_args "pretrained=$U,dtype=bfloat16,trust_remote_code=True" \
  --tasks arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,sciq,wikitext,winogrande,mmlu \
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
RES="/proj/dmfexp/nima/Code/dolomite-engine/experiments/boltzmann-moe/results/iclr_flops"
CFG="/proj/dmfexp/nima/Code/dolomite-engine/configs/iclr_flops"
ACC=["arc_challenge","arc_easy","boolq","copa","hellaswag","openbookqa","piqa","sciq","winogrande","mmlu"]
rows=[]
for c in sorted(glob.glob(f"{CFG}/*.yml")):
    a=os.path.basename(c)[:-4]
    b=TrainingArgs(**load_yaml(c)).model_args.pretrained_config["mlp_blocks"][-1]
    K=b["n_experts"]; tk=b.get("top_k") or K
    learned = b["mlp_type"]=="TopK_Energy_MoE_MLP"
    hop = b.get("expert_kind")=="hopfield"
    js=glob.glob(f"{RES}/{a}/unsharded/harness_results*.json")
    if not js: rows.append((a,K,tk,tk/K,learned,hop,None,None)); continue
    d=json.load(open(sorted(js)[-1])); r=d.get("results",d)
    accs=[]
    for t in ACC:
        v=r.get(t,{})
        m=v.get("acc_norm,none", v.get("acc,none"))
        if m is not None: accs.append(m)
    ppl=r.get("wikitext",{}).get("word_perplexity,none")
    rows.append((a,K,tk,tk/K,learned,hop,100*sum(accs)/len(accs) if accs else None,ppl))
print(f"{'arm':30s} {'router':8s} {'exp':4s} {'K':>3s} {'k':>2s} {'k/K':>6s} "
      f"{'FLOPratio':>9s} {'avg%':>7s} {'wikiPPL':>8s}")
for a,K,tk,kk,learned,hop,avg,ppl in rows:
    # cost of the mixture block relative to dense soft, with a free proxy router
    print(f"{a:30s} {'learned' if learned else 'energy':8s} {'hop' if hop else 'w1w2':4s} "
          f"{K:3d} {tk:2d} {kk:6.3f} {kk:9.3f} "
          f"{('%7.2f'%avg) if avg else '      -':>7s} {('%8.2f'%ppl) if ppl else '       -':>8s}")
print()
print("FLOPratio = k/K = mixture-block cost vs dense soft, assuming the cheap proxy")
print("router (0.21% of exact). With the EXACT energy router it is 0.5+0.5*k/K instead.")
print("Reference points already banked: h1 w1w2 K=4 dense avg 50.10 ppl 36.48;")
print("h1 w1w2 K=4 top2 avg 48.56 ppl 36.37; h1 learned K=4 top2 avg 49.90 ppl 39.79.")
PY
    ;;

proxy)
    for a in $(arms); do
        U="$RES/$a/unsharded"
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
    --run "$RES/$a" --n_prompts 400 --max_len 1024 --cache_per_iter 6000 --ranks 8 32
python experiments/boltzmann-moe/scripts/fit_proxy_router_20260912.py \
    --run "$RES/$a" --rank 32 --hidden 64 --steps 4000 --n_eval 250
EOF
        echo "submitted px_$a"
    done
    ;;

*) echo "usage: $0 {status|eval|table|proxy}"; exit 1;;
esac
