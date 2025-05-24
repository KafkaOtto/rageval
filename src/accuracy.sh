#!/bin/bash

echo "Running pre-hooks before committing..."

export ACCURACY_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=sk-20e9f2df70c6442eba7eaf351222e061
export PYTHONPATH=$(pwd):$PYTHONPATH

# Check for required arguments
if [ -z "$1" ]; then
  echo "Usage: $0 <input-file> <output-file>"
  exit 1
fi

INPUT_FILE="$1"
OUTPUT_FILE="$1"

python3 llmperf/rag_accuracy.py \
    --model "deepseek-chat" \
    --num-concurrent-requests 5 \
    --timeout 9000000 \
    --input-dir "$INPUT_FILE" \
    --output-dir "$OUTPUT_FILE" \
    --metadata "" \
    --llm-api "openai_acc"

echo "======FORMAT====="
