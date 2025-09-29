#!/bin/bash

if [ -f .env ]; then
    while IFS="=" read -r raw_key raw_value; do
        key=$(echo "$raw_key" | xargs)
        value=$(echo "$raw_value" | sed 's/[#].*$//' | xargs)
        if [ -z "$key" ] || [[ "$key" == \#* ]]; then
            continue
        fi
        if [ -z "${!key}" ]; then
            export "${key}=${value}"
        fi
    done < .env
    echo "Loaded environment variables from .env"
fi

CONFIG_FILE="${1:-configs/wrappers/openai_existing_agent.yaml}"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "Error: Config file not found: $CONFIG_FILE"
    echo ""
    echo "Usage: $0 [config_file]"
    echo ""
    echo "Available wrapper configs:"
    ls -1 configs/wrappers/*.yaml 2>/dev/null | sed 's/^/  - /'
    echo ""
    echo "Available optimization configs:"
    ls -1 configs/optimize/*.yaml 2>/dev/null | sed 's/^/  - /'
    exit 1
fi

echo "Loading config from: $CONFIG_FILE"

function extract_yaml_value() {
    local key=$1
    local file=$2
    grep "^$key:" "$file" | head -1 | sed 's/^[^:]*: *//' | sed 's/^"\(.*\)"$/\1/'
}

function extract_nested_value() {
    local parent=$1
    local key=$2
    local file=$3
    awk "/^$parent:/{flag=1; next} /^[^ ]/{flag=0} flag && /^  $key:/" "$file" | head -1 | sed 's/^[^:]*: *//' | sed 's/^"\(.*\)"$/\1/'
}

COMPATIBILITY_MODE=$(extract_yaml_value "compatibility_mode" "$CONFIG_FILE")
TRAINSET=$(extract_yaml_value "trainset" "$CONFIG_FILE")
VALSET=$(extract_yaml_value "valset" "$CONFIG_FILE")
TEACHER_MODEL=${TEACHER_MODEL:-$(extract_yaml_value "teacher_model" "$CONFIG_FILE")}
STUDENT_MODEL=${STUDENT_MODEL:-$(extract_yaml_value "student_model" "$CONFIG_FILE")}
REFLECTION_LM=${REFLECTION_LM:-$(extract_yaml_value "reflection_lm" "$CONFIG_FILE")}

export TEACHER_MODEL
export STUDENT_MODEL
MAX_CALLS=$(extract_yaml_value "max_metric_calls" "$CONFIG_FILE")
TRACE_STORAGE=$(extract_yaml_value "trace_storage" "$CONFIG_FILE")
OUTPUT=$(extract_yaml_value "output" "$CONFIG_FILE")
USE_VLLM=$(extract_yaml_value "use_vllm_client" "$CONFIG_FILE")
VLLM_HOST=$(extract_yaml_value "vllm_host" "$CONFIG_FILE")
VLLM_PORT=$(extract_yaml_value "vllm_port" "$CONFIG_FILE")

if [ "$COMPATIBILITY_MODE" = "true" ]; then
    echo ""
    echo "=== COMPATIBILITY MODE ==="
    echo "Testing your existing agent with ATLAS teaching"

    AGENT_TYPE=$(extract_nested_value "user_agent" "type" "$CONFIG_FILE")
    INTEGRATION_TYPE=$(awk '/^user_agent:/{flag=1} flag && /integration_type:/' "$CONFIG_FILE" | head -1 | sed 's/.*: *//' | sed 's/^"\(.*\)"$/\1/')

    if [ "$INTEGRATION_TYPE" = "http_api" ]; then
        ENDPOINT=$(awk '/^user_agent:/{flag=1} flag && /endpoint:/' "$CONFIG_FILE" | head -1 | sed 's/.*: *//' | sed 's/^"\(.*\)"$/\1/')
        echo "Agent type: HTTP API at $ENDPOINT"

        if [ -z "$OPENAI_API_KEY" ] && grep -q "Authorization.*Bearer" "$CONFIG_FILE"; then
            echo "Warning: No OPENAI_API_KEY in .env. Update config or set in .env"
        fi
    elif [ "$INTEGRATION_TYPE" = "python_function" ]; then
        MODULE_PATH=$(awk '/^user_agent:/{flag=1} flag && /module_path:/' "$CONFIG_FILE" | head -1 | sed 's/.*: *//' | sed 's/^"\(.*\)"$/\1/')
        echo "Agent type: Python function from $MODULE_PATH"
    elif [ "$INTEGRATION_TYPE" = "cli_command" ]; then
        echo "Agent type: CLI command"
    fi
else
    echo ""
    echo "=== NEW AGENT MODE ==="
    echo "Creating new agent with ATLAS teaching"

    if [ -z "$STUDENT_MODEL" ]; then
        echo "Error: student_model not specified in config"
        exit 1
    fi
fi

echo ""
echo "Teacher model: ${TEACHER_MODEL}"
if [ "$COMPATIBILITY_MODE" != "true" ]; then
    echo "Student model: ${STUDENT_MODEL}"
fi
echo "Reflection LM: ${REFLECTION_LM:-gpt-4}"
echo "Dataset: ${TRAINSET:-arc-atlas-rl}"
echo "Max iterations: ${MAX_CALLS:-150}"
echo ""

CMD="python optimize_teaching.py"
CMD="$CMD --trainset ${TRAINSET:-arc-atlas-rl}"

if [ "$COMPATIBILITY_MODE" != "true" ]; then
    CMD="$CMD --student-model ${STUDENT_MODEL}"
fi

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