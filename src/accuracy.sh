#!/bin/bash

echo "Running pre-hooks before committing..."

export ACCURACY_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=sk-20e9f2df70c6442eba7eaf351222e061
export PYTHONPATH=$(pwd):$PYTHONPATH


if [ -z "$TREATMENT_ID" ] || [ -z "$1" ]; then
  echo "Usage: TREATMENT_ID=<treatment-id> $0 <num-runs>"
  exit 1
fi

OUTPUT_BASE_DIR="output_with_ram/${TREATMENT_ID}"
LOG_DIR="logs/accuracy"

mkdir -p "${LOG_DIR}"

for (( i=1; i<=NUM_RUNS; i++ ))
do
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/run_$i"
  INPUT_DIR="${OUTPUT_BASE_DIR}/run_$i"
  LOG_FILE="${LOG_DIR}/${TREATMENT_ID}_run_${i}_accuracy.log"

  echo "[$(date)] Starting run $i of $NUM_RUNS..."
  python3 llmperf/rag_accuracy.py \
      --model "deepseek-chat" \
      --num-concurrent-requests 5 \
      --timeout 9000000 \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --metadata "" \
      --llm-api "openai_acc" \
      > "${LOG_FILE}" 2>&1


echo "======FORMAT====="
