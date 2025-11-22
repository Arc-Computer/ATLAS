#!/bin/bash
# Launch multi-node GKD training in Docker using DeepSpeed ZeRO-3
#
# This script launches containerized distributed training across multiple nodes.
# Each node runs the training script in an isolated Docker container with GPU access.
#
# Usage: ./scripts/launch_multinode_gkd_docker.sh <node_rank>
#
# Prerequisites:
#   - Docker installed on all nodes
#   - NVIDIA Container Toolkit (nvidia-docker2)
#   - GPU(s) available on each node
#   - Network connectivity between nodes
#
# Required Environment Variables:
#   MASTER_ADDR - IP address of the master node
#
# Optional Environment Variables:
#   MASTER_PORT     - Port for communication (default: 29500)
#   WANDB_API_KEY   - WandB API key for experiment tracking
#   HF_HOME         - HuggingFace cache directory (default: ~/.cache/huggingface)
#
# Arguments:
#   node_rank - Rank of this node (0 for master, 1+ for workers)
#
# Example:
#   # On master node:
#   MASTER_ADDR=192.168.1.10 ./scripts/launch_multinode_gkd_docker.sh 0
#
#   # On worker node:
#   MASTER_ADDR=192.168.1.10 ./scripts/launch_multinode_gkd_docker.sh 1

set -euo pipefail

# Validate arguments
if [ $# -ne 1 ]; then
    echo "Error: Missing required argument <node_rank>"
    echo "Usage: $0 <node_rank>"
    echo ""
    echo "Example:"
    echo "  MASTER_ADDR=192.168.1.10 $0 0  # Master node"
    echo "  MASTER_ADDR=192.168.1.10 $0 1  # Worker node"
    exit 1
fi

NODE_RANK=$1

# Validate environment variables
if [ -z "${MASTER_ADDR:-}" ]; then
    echo "Error: MASTER_ADDR environment variable must be set"
    echo ""
    echo "MASTER_ADDR should be the IP address of the master node."
    echo "Example: MASTER_ADDR=192.168.1.10 $0 $NODE_RANK"
    exit 1
fi

MASTER_PORT=${MASTER_PORT:-29500}
CONTAINER_NAME="atlas_gkd_multinode_node${NODE_RANK}"
HOSTNAME=$(hostname)

echo "=========================================="
echo "Docker Multi-Node GKD Training"
echo "=========================================="
echo "Node: $HOSTNAME (rank=$NODE_RANK)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Container: $CONTAINER_NAME"
echo "=========================================="

# Validate Docker and GPU availability
if ! command -v docker &> /dev/null; then
    echo "Error: Docker is not installed"
    echo "Install Docker: https://docs.docker.com/get-docker/"
    exit 1
fi

if ! docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi &> /dev/null; then
    echo "Error: NVIDIA Docker runtime not available or no GPUs found"
    echo "Install NVIDIA Container Toolkit: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
    exit 1
fi

# Stop existing container if running
echo "Cleaning up existing containers..."
docker stop "$CONTAINER_NAME" 2>/dev/null || true
docker rm "$CONTAINER_NAME" 2>/dev/null || true

# Prepare paths
WORKSPACE_DIR=$(pwd)
CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$CACHE_DIR"

echo "Workspace: $WORKSPACE_DIR"
echo "HF Cache: $CACHE_DIR"

# Build volume mounts
VOLUMES="-v $WORKSPACE_DIR:/workspace -v $CACHE_DIR:/root/.cache/huggingface"

# Environment configuration
ENV_ARGS="-e MASTER_ADDR=$MASTER_ADDR -e MASTER_PORT=$MASTER_PORT -e PYTHONPATH=/workspace"

# Handle WandB API key
if [ -f "$WORKSPACE_DIR/.env" ]; then
    echo "Using .env file for environment variables"
    ENV_ARGS="$ENV_ARGS --env-file $WORKSPACE_DIR/.env"
elif [ -n "${WANDB_API_KEY:-}" ]; then
    echo "Using WANDB_API_KEY from environment"
    ENV_ARGS="$ENV_ARGS -e WANDB_API_KEY=$WANDB_API_KEY"
else
    echo "Warning: WANDB_API_KEY not set. WandB logging may not work."
fi

echo ""
echo "Launching training container..."
echo ""

# Launch training container
# Note: Dependencies are installed on first run. For faster startup,
# consider building a custom image with pre-installed dependencies.
docker run --rm -d \
  --name "$CONTAINER_NAME" \
  --gpus all \
  --ipc=host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  $VOLUMES \
  -w /workspace \
  $ENV_ARGS \
  nvcr.io/nvidia/pytorch:25.10-py3 \
  bash -c "pip install -q accelerate deepspeed transformers datasets hydra-core wandb && bash scripts/launch_multinode_gkd.sh $NODE_RANK"

echo ""
echo "=========================================="
echo "Container started successfully!"
echo "=========================================="
echo "Monitor logs: docker logs -f $CONTAINER_NAME"
echo "Stop container: docker stop $CONTAINER_NAME"
echo "=========================================="
