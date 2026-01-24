#!/bin/bash

offload_found=false
zero1_found=false
args=()
for arg in "$@"; do
  if [ "$arg" = "offload" ]; then
    offload_found=true
  elif [ "$arg" = "zero1" ]; then
    zero1_found=true
  else
    args+=("$arg")
  fi
done

nproc=${args[0]}

if [ "$offload_found" = true ]; then
  config="accelerate/deepspeed_zero3_cpu_offloading.yaml"
elif [ "$zero1_found" = true ]; then
  config="accelerate/deepspeed_zero1.yaml"
else
  config="accelerate/deepspeed_zero3.yaml"
fi
arg2=${args[1]:-"default"}

config_root="src/atlas_core/configs/recipe"
prefix="${config_root}/"
if [[ "$arg2" == $prefix* ]]; then
    arg2="${arg2#$prefix}"
fi

extra_args=("${args[@]:2}")

# generate random port
RND_PORT=$(($RANDOM % 1000 + 12000))
echo $RND_PORT

accelerate launch --num_processes "$nproc" \
  --main_process_port "$RND_PORT" \
  --config_file "$config" \
  -m atlas_core.cli.train recipe@_global_="$arg2" "${extra_args[@]}"
