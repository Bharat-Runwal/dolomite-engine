set -e
D=$1; CK=$2; TAG=$3
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
      --tasks "$T" --device cuda:0 --batch_size 4 --trust_remote_code \
      --output_path $pth/harness_results.json || echo "$TAG/$n HARNESS FAILED"
  python -u experiments/eval_scripts/compute_avg11.py $pth || echo "$TAG/$n AVG11 FAILED"
done
