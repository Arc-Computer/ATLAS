#!/bin/bash

if [ -f .env ]; then
    while IFS='=' read -r raw_key raw_value; do
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

CMD=(python optimize_teaching.py --config "$CONFIG_FILE")

if [ -n "$TEACHER_MODEL" ]; then
    CMD+=(--teacher-model "$TEACHER_MODEL")
fi

if [ -n "$STUDENT_MODEL" ]; then
    CMD+=(--student-model "$STUDENT_MODEL")
fi

if [ -n "$REFLECTION_LM" ]; then
    CMD+=(--reflection-lm "$REFLECTION_LM")
fi

if [ -n "$TRAINSET" ]; then
    CMD+=(--trainset "$TRAINSET")
fi

if [ -n "$VALSET" ]; then
    CMD+=(--valset "$VALSET")
fi

echo "Running command:"
echo "${CMD[@]}"
echo ""

"${CMD[@]}"
