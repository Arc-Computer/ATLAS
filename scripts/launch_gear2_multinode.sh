#!/bin/bash
# Launch Gear 2 GKD training across two nodes using accelerate
# Usage: ./scripts/launch_gear2_multinode.sh [node_rank]
#   node_rank: 0 for Node 1 (master), 1 for Node 2
#
# Note: If using Docker, mount both ~/.cache/huggingface and /tmp/hf_cache_sync
# to access models synced to temp location

set -euo pipefail

NODE_RANK=${1:-0}
MASTER_ADDR=${MASTER_ADDR:-192.168.100.10}
MASTER_PORT=${MASTER_PORT:-29500}

# Get hostname for verification
HOSTNAME=$(hostname)
echo "Launching on node: $HOSTNAME (rank=$NODE_RANK)"

# Update accelerate config with correct machine rank
ACCELERATE_CONFIG="accelerate/deepspeed_zero3_multinode.yaml"
TEMP_CONFIG=$(mktemp)
sed "s/machine_rank:.*/machine_rank: $NODE_RANK/" "$ACCELERATE_CONFIG" > "$TEMP_CONFIG"
sed -i "s/main_process_ip:.*/main_process_ip: $MASTER_ADDR/" "$TEMP_CONFIG"
sed -i "s/main_process_port:.*/main_process_port: $MASTER_PORT/" "$TEMP_CONFIG"

echo "Using accelerate config:"
cat "$TEMP_CONFIG" | grep -E "(machine_rank|main_process_ip|main_process_port|num_machines|num_processes)"

# Launch training with Gear 2 overrides
# Note: HF_HUB_OFFLINE not set - allows model download on Node 2 if needed
accelerate launch \
  --config_file "$TEMP_CONFIG" \
  train.py \
  --config-name teacher_gkd \
  data@_global_=gsm8k_gkd \
  trainer.temperature=0.6 \
  trainer.max_steps=2500 \
  trainer.learning_rate=3e-6 \
  trainer.per_device_train_batch_size=2 \
  trainer.gradient_accumulation_steps=4 \
  trainer.max_new_tokens=128 \
  trainer.eval_steps=50 \
  trainer.model_init_kwargs.device_map=null \
  trainer.teacher_model_init_kwargs.device_map=null \
  output_dir=outputs/gkd_gsm8k_reliability_opt \
  wandb_project=atlas_gkd_gsm8k \
  wandb_run_name=gear2_reliability_$(date +%Y%m%d_%H%M%S)

# Cleanup
rm "$TEMP_CONFIG"

