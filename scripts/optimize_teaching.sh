#!/bin/bash

if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
    echo "Loaded environment variables from .env"
fi

CONFIG_FILE="${1:-configs/optimize/default.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo "Usage: $0 [config_file]"
    echo "Available configs:"
    ls -1 configs/optimize/*.yaml 2>/dev/null | sed 's/^/  - /'
    exit 1
fi

echo "Loading config from: $CONFIG_FILE"

function extract_yaml_value() {
    local key=$1
    local file=$2
    grep "^$key:" "$file" | head -1 | sed 's/^[^:]*: *//' | sed 's/^"\(.*\)"$/\1/'
}

TRAINSET=$(extract_yaml_value "trainset" "$CONFIG_FILE")
VALSET=$(extract_yaml_value "valset" "$CONFIG_FILE")
STUDENT_MODEL=${STUDENT_MODEL:-$(extract_yaml_value "student_model" "$CONFIG_FILE")}
TEACHER_MODEL=${TEACHER_MODEL:-$(extract_yaml_value "teacher_model" "$CONFIG_FILE")}
REFLECTION_LM=${REFLECTION_LM:-$(extract_yaml_value "reflection_lm" "$CONFIG_FILE")}
MAX_CALLS=$(extract_yaml_value "max_metric_calls" "$CONFIG_FILE")
TRACE_STORAGE=$(extract_yaml_value "trace_storage" "$CONFIG_FILE")
OUTPUT=$(extract_yaml_value "output" "$CONFIG_FILE")
USE_VLLM=$(extract_yaml_value "use_vllm_client" "$CONFIG_FILE")
VLLM_HOST=$(extract_yaml_value "vllm_host" "$CONFIG_FILE")
VLLM_PORT=$(extract_yaml_value "vllm_port" "$CONFIG_FILE")

CMD="python -u optimize_teaching.py"
if [ ! -z "$TRAINSET" ]; then
    CMD="$CMD --trainset $TRAINSET"
fi
CMD="$CMD --student-model ${STUDENT_MODEL}"
CMD="$CMD --teacher-model ${TEACHER_MODEL}"
CMD="$CMD --reflection-lm ${REFLECTION_LM:-gpt-4}"
CMD="$CMD --max-metric-calls ${MAX_CALLS:-150}"
CMD="$CMD --trace-storage ${TRACE_STORAGE:-traces/optimize_traces.jsonl}"
CMD="$CMD --output ${OUTPUT:-optimized_prompts.json}"
CMD="$CMD --config $CONFIG_FILE"

if [ ! -z "$VALSET" ] && [ "$VALSET" != "null" ]; then
    CMD="$CMD --valset $VALSET"
fi

if [ "$USE_VLLM" = "true" ]; then
    CMD="$CMD --use-vllm-client"
    CMD="$CMD --vllm-host ${VLLM_HOST:-localhost}"
    CMD="$CMD --vllm-port ${VLLM_PORT:-8765}"
fi

echo "Running command:"
echo "$CMD"
echo ""

eval $CMD