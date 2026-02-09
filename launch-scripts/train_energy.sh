#!/bin/bash
#BSUB -J energy-train
#BSUB -n 4
#BSUB -R "span[ptile=1]"
#BSUB -G grp_ebm
#BSUB -q normal
#BSUB -M 120G
#BSUB -gpu "num=8:mode=exclusive_process"
#BSUB -W 72:00
#BSUB -o /proj/dmfexp/energy-gpt/logs/energy-train-%J.out
#BSUB -eo /proj/dmfexp/energy-gpt/logs/energy-train-%J.err

set -euo pipefail

# ========== Blue Vela NCCL/InfiniBand Settings ==========
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_MODULE_LOADING=LAZY

# InfiniBand tuning (validated for Blue Vela H100)
export NCCL_IB_PCI_RELAXED_ORDERING=2
export NCCL_IB_QPS_PER_CONNECTION=4
export NCCL_IB_SPLIT_DATA_ON_QPS=0
export NCCL_IB_ADAPTIVE_ROUTING=1
export NCCL_IB_DISABLE=0
export NCCL_NVLS_ENABLE=1

# Exclude storage IB NICs
export NCCL_IB_HCA="^=mlx5_1,mlx5_6"
export NCCL_SOCKET_IFNAME="=ibp26s0,ibp60s0,ibp77s0,ibp94s0,ibp156s0,ibp188s0,ibp204s0,ibp220s0"

# Debug (comment after initial runs work)
export NCCL_DEBUG_SUBSYS=NET,ENV,INIT

export OMP_NUM_THREADS=64
export TOKENIZERS_PARALLELISM=false

# ========== Config ==========
CONFIG_PATH="${1:-/proj/dmfexp/hopfield++/Projects/dolomite-engine/configs/energy/energy_32gpu.yml}"

# ========== Pre-flight ==========
if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "ERROR: Config not found: $CONFIG_PATH" >&2
  exit 1
fi

mkdir -p /proj/dmfexp/energy-gpt/logs

# ========== Launch ==========
cd /proj/dmfexp/hopfield++/Projects/dolomite-engine
source .venv/bin/activate

# Discover nodes from LSF
mapfile -t HOSTS < <(echo "$LSB_MCPU_HOSTS" | awk '{for(i=1;i<=NF;i+=2)print $i}')
NNODES=${#HOSTS[@]}
MASTER_ADDR="${HOSTS[0]}"
MASTER_PORT=$((10000 + (${LSB_JOBID:-12345} % 40000)))

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  GPUS_PER_NODE=$(echo "$CUDA_VISIBLE_DEVICES" | tr "," "\n" | wc -l)
else
  GPUS_PER_NODE=8
fi

export MASTER_ADDR MASTER_PORT NNODES GPUS_PER_NODE CONFIG_PATH

echo "[INFO] Nodes: ${HOSTS[*]}"
echo "[INFO] MASTER=${MASTER_ADDR}:${MASTER_PORT} NNODES=${NNODES} GPUS=${GPUS_PER_NODE}"
echo "[INFO] Config: ${CONFIG_PATH}"

# Launch on each node
pids=()
for i in $(seq 0 $((NNODES-1))); do
  host="${HOSTS[$i]}"
  echo "[INFO] Launching node_rank=${i} on ${host}"

  lsrun -m "$host" bash -s -- "$i" << 'EOS' & pids+=($!)
#!/bin/bash
set -euo pipefail

NODE_RANK="$1"
cd /proj/dmfexp/hopfield++/Projects/dolomite-engine
source .venv/bin/activate

echo "[node_rank=${NODE_RANK}] host=$(hostname) master=${MASTER_ADDR}:${MASTER_PORT}"

exec torchrun \
  --nnodes="${NNODES}" \
  --nproc_per_node="${GPUS_PER_NODE}" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  --rdzv_id="${LSB_JOBID:-energy}" \
  --node_rank="${NODE_RANK}" \
  -m lm_engine.pretrain \
  --config "${CONFIG_PATH}"
EOS
done

for p in "${pids[@]}"; do
  wait "$p"
done
