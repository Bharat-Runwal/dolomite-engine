#!/bin/bash
# Run per-layer alpha sweep (inference-only, ~10 min on 1 GPU).
# Usage: bash experiments/energy-inference/scripts/perlayer-alpha/run_perlayer_alpha.sh

REPO=/proj/dmfexp/nima/Code/dolomite-engine

bsub \
    -q normal \
    -G grp_ebm \
    -J perlayer_alpha \
    -gpu "num=1" \
    -n 1 \
    -M 32G \
    -W 00:30 \
    -o "${HOME}/perlayer_alpha_%J.stdout" \
    -e "${HOME}/perlayer_alpha_%J.stderr" \
    <<EOF
#!/bin/bash
source /proj/dmfexp/nima/Code/nanoGPT-og/.venv/bin/activate
export PYTHONPATH=$REPO:\$PYTHONPATH
uv pip install accelerate -q
cd $REPO
python experiments/energy-inference/scripts/perlayer-alpha/perlayer_alpha_20260412.py
EOF

echo "Submitted. Monitor: bjobs / bpeek <jobid>"
