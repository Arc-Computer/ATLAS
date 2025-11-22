#!/bin/bash
# Launch multi-node GKD training using DeepSpeed ZeRO-3
#
# This script coordinates distributed training across multiple nodes using
# Accelerate and DeepSpeed. It's optimized for high-quality distillation runs
# with conservative hyperparameters for reliability.
#
# Usage: ./scripts/launch_multinode_gkd.sh <node_rank>
#
# Prerequisites:
#   - Accelerate and DeepSpeed installed on all nodes
#   - Network connectivity between nodes on the specified port
#   - Shared or synced model cache across nodes
#
# Required Environment Variables:
#   MASTER_ADDR - IP address of the master node (e.g., 10.0.0.1)
#
# Optional Environment Variables:
#   MASTER_PORT     - Port for multi-node communication (default: 29500)
#   OUTPUT_DIR      - Training output directory (default: outputs/gkd_gsm8k_multinode)
#   WANDB_PROJECT   - WandB project name (default: atlas_gkd_gsm8k)
#   WANDB_RUN_NAME  - WandB run name (default: gkd_multinode_<timestamp>)
#
# Arguments:
#   node_rank - Rank of this node (0 for master, 1+ for workers)
#
# Example:
#   # On master node (192.168.1.10):
#   MASTER_ADDR=192.168.1.10 ./scripts/launch_multinode_gkd.sh 0
#
#   # On worker node:
#   MASTER_ADDR=192.168.1.10 ./scripts/launch_multinode_gkd.sh 1

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

# Set defaults
MASTER_PORT=${MASTER_PORT:-29500}
OUTPUT_DIR=${OUTPUT_DIR:-outputs/gkd_gsm8k_multinode}
WANDB_PROJECT=${WANDB_PROJECT:-atlas_gkd_gsm8k}
WANDB_RUN_NAME=${WANDB_RUN_NAME:-gkd_multinode_$(date +%Y%m%d_%H%M%S)}

# Get hostname for logging
HOSTNAME=$(hostname)

echo "=========================================="
echo "Multi-Node GKD Training Configuration"
echo "=========================================="
echo "Node: $HOSTNAME (rank=$NODE_RANK)"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Output: $OUTPUT_DIR"
echo "WandB: $WANDB_PROJECT/$WANDB_RUN_NAME"
echo "=========================================="

# Prepare accelerate config with runtime values
ACCELERATE_CONFIG="accelerate/deepspeed_zero3_multinode.yaml"

if [ ! -f "$ACCELERATE_CONFIG" ]; then
    echo "Error: Accelerate config not found at $ACCELERATE_CONFIG"
    exit 1
fi

TEMP_CONFIG=$(mktemp)

# Update config with runtime values (portable sed syntax)
sed "s/machine_rank:.*/machine_rank: $NODE_RANK/" "$ACCELERATE_CONFIG" > "$TEMP_CONFIG"
sed "s|main_process_ip:.*|main_process_ip: $MASTER_ADDR|" "$TEMP_CONFIG" > "$TEMP_CONFIG.tmp" && mv "$TEMP_CONFIG.tmp" "$TEMP_CONFIG"
sed "s/main_process_port:.*/main_process_port: $MASTER_PORT/" "$TEMP_CONFIG" > "$TEMP_CONFIG.tmp" && mv "$TEMP_CONFIG.tmp" "$TEMP_CONFIG"

echo ""
echo "Accelerate Configuration:"
grep -E "(machine_rank|main_process_ip|main_process_port|num_machines|num_processes)" "$TEMP_CONFIG"
echo ""

# Launch GKD training with high-reliability hyperparameters
# These parameters are optimized for quality over speed:
# - temperature=0.6: Lower temperature for more focused generation
# - max_steps=2500: Extended training for better convergence
# - learning_rate=3e-6: Conservative learning rate to avoid instability
# - per_device_train_batch_size=2: Memory-efficient batch size
# - gradient_accumulation_steps=4: Effective batch size = 2 * 4 * num_processes
# - max_new_tokens=128: Constrained generation length
# - eval_steps=50: Frequent evaluation checkpoints

echo "Starting training..."
echo ""

accelerate launch \
  --config_file "$TEMP_CONFIG" \
  train.py \
  --config-name run/teacher_gkd \
  +data@_global_=gsm8k_gkd \
  trainer.temperature=0.6 \
  trainer.max_steps=2500 \
  trainer.learning_rate=3e-6 \
  trainer.per_device_train_batch_size=2 \
  trainer.gradient_accumulation_steps=4 \
  trainer.max_new_tokens=128 \
  trainer.eval_steps=50 \
  trainer.model_init_kwargs.device_map=null \
  trainer.teacher_model_init_kwargs.device_map=null \
  output_dir="$OUTPUT_DIR" \
  wandb_project="$WANDB_PROJECT" \
  wandb_run_name="$WANDB_RUN_NAME"

# Cleanup
rm -f "$TEMP_CONFIG" "$TEMP_CONFIG.tmp"

echo ""
echo "Training completed successfully!"
echo "Output saved to: $OUTPUT_DIR"
