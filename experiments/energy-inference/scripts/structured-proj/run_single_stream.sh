#!/bin/bash
# Single-stream port-Hamiltonian: one K=(J-R) applied to combined (attn+scale_ff*ff) input.
#
# Direct comparison to original single-W architecture but with structured J-R.
# Tests whether the split (dual-stream) design adds value over applying one K
# to the combined input — matching the structure of the trained W exactly.
#
# Usage: bash experiments/energy-inference/scripts/structured-proj/run_single_stream.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J struct_proj_single_stream \
    -gpu "num=1" \
    -n 1 \
    -M 48G \
    -W 01:00 \
    -o "${HOME}/struct_proj_single_stream_%J.stdout" \
    -e "${HOME}/struct_proj_single_stream_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate datasets wandb tqdm -q
cd $REPO
python experiments/energy-inference/scripts/structured-proj/train_structured_proj_20260412.py \
    --steps 3000 \
    --batch_size 8 \
    --grad_accum 8 \
    --seq_len 512 \
    --lr 3e-4 \
    --residual \
    --single_stream \
    --alpha_min 0.35 \
    --alpha_max 0.65 \
    --save_interval 1000 \
    --log_interval 25 \
    --norm_log_interval 100 \
    --wandb_name "410m_struct_single_stream"
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
