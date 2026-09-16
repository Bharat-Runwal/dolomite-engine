set -uo pipefail   # NOT -e: we want every arm attempted, but see FAILED below
FAILED=0
D=$1; CK=$2; TAG=$3
# 12.16c: batch size is a PARAMETER and defaults to 2, not 4. At 400M with I_e=15872 the
# PROXYSEL arm OOMed at 4 (15.50 GiB requested, 10.39 free): proxy_route with
# sparse_forward:false runs the dense all-K path PLUS the proxy heads. 4 is fine at 134M.
BS=${4:-2}
R=$D/ablate
echo "======== MODEL $TAG  ($D, ckpt $CK)"
python -u experiments/boltzmann-moe/scripts/fit_subspace_proxy_20260916.py \
    --run_dir $D --ckpt_name $CK --ranks 16 --out_dims 512 --per_iter \
    --refine_steps 600 --batches 8 --per_call 1024 --recall_at 2 3 4 6 8 --write
python -u experiments/boltzmann-moe/scripts/ablate_sparse_20260916.py \
    --src $D/unsharded_sparse_r16m512 --out_root $R
T="arc_challenge,arc_easy,boolq,copa,hellaswag,openbookqa,piqa,race,sciq,wikitext,winogrande,lambada_openai"
for V in DENSE:$D/$CK PROXYSEL:$R/C_proxysel; do
  n=${V%%:*}; pth=${V##*:}
  echo "-------- $TAG / $n"
  python -u experiments/eval_scripts/eval_harness.py --model hf \
      --model_args pretrained=$pth,dtype=bfloat16,trust_remote_code=True \
      --tasks "$T" --device cuda:0 \
      --use_cache $HOME/lmeval_cache/$TAG-$n --batch_size $BS --trust_remote_code \
      --output_path $pth/harness_results.json || { echo "$TAG/$n HARNESS FAILED"; FAILED=1; }
  # 12.16c: verify the file LANDED. A missing file here means any downstream number is not this arm's.
  ls -l $pth/harness_results*.json >/dev/null 2>&1 || { echo "$TAG/$n NO RESULTS FILE"; FAILED=1; }
  python -u experiments/eval_scripts/compute_avg11.py $pth || { echo "$TAG/$n AVG11 FAILED"; FAILED=1; }
done

# 12.16c: exit nonzero if ANY arm failed, so LSF does not report "Successfully completed"
# for a job that produced no usable number.
if [ "$FAILED" -ne 0 ]; then echo "=== ONE OR MORE ARMS FAILED -- DO NOT READ ANY DELTA ==="; exit 1; fi
echo "=== all arms produced results ==="
