#!/bin/bash
# Launch Gear 2 GKD training in Docker across two nodes
# Usage: ./scripts/launch_gear2_multinode_docker.sh [node_rank]
#   node_rank: 0 for Node 1 (master), 1 for Node 2

set -euo pipefail

NODE_RANK=${1:-0}
MASTER_ADDR=${MASTER_ADDR:-192.168.100.10}
MASTER_PORT=${MASTER_PORT:-29500}
CONTAINER_NAME="gkd_gear2_node${NODE_RANK}"

# Get hostname for verification
HOSTNAME=$(hostname)
echo "Launching Gear 2 multi-node training on: $HOSTNAME (rank=$NODE_RANK)"

# Stop existing container if running
docker stop $CONTAINER_NAME 2>/dev/null || true
docker rm $CONTAINER_NAME 2>/dev/null || true

# Get absolute paths
WORKSPACE_DIR=$(pwd)
CACHE_DIR="$HOME/.cache/huggingface"

# Build volume mounts with absolute paths
VOLUMES="-v $WORKSPACE_DIR:/workspace -v $CACHE_DIR:/root/.cache/huggingface"

# On Node 2, also mount temp location for synced teacher model
if [ "$NODE_RANK" = "1" ]; then
    VOLUMES="$VOLUMES -v /tmp/hf_cache_sync:/tmp/hf_cache_sync"
    # Create symlink in container if temp model exists
    PRE_CMD="mkdir -p /root/.cache/huggingface/hub && ln -sf /tmp/hf_cache_sync/models--Qwen--Qwen2.5-14B-Instruct /root/.cache/huggingface/hub/models--Qwen--Qwen2.5-14B-Instruct 2>/dev/null || true; "
else
    PRE_CMD=""
fi

# Handle .env file - use it if exists, otherwise pass WANDB_API_KEY from environment
ENV_ARGS=""
if [ -f "$WORKSPACE_DIR/.env" ]; then
    ENV_ARGS="--env-file $WORKSPACE_DIR/.env"
else
    # Fallback: pass WANDB_API_KEY if set
    if [ -n "${WANDB_API_KEY:-}" ]; then
        ENV_ARGS="-e WANDB_API_KEY=$WANDB_API_KEY"
    fi
fi

# Install dependencies if needed (check if accelerate is available)
# Launch Docker container
docker run --rm -d --name $CONTAINER_NAME --gpus all --ipc=host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  $VOLUMES \
  -w /workspace -e PYTHONPATH=/workspace $ENV_ARGS \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -c "${PRE_CMD}pip install -q accelerate deepspeed transformers datasets hydra-core wandb && bash scripts/launch_gear2_multinode.sh $NODE_RANK"

echo "Container $CONTAINER_NAME started"
echo "Monitor logs: docker logs -f $CONTAINER_NAME"

