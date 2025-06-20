#!/bin/bash

echo "Running pre-hooks before committing..."

export ACCURACY_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=sk-20e9f2df70c6442eba7eaf351222e061
export PYTHONPATH=$(pwd):$PYTHONPATH


if [ -z "$2" ] || [ -z "$1" ] || [ -z "$3" ]; then
  echo "Usage: $0 <num-runs> $1 <treatment_id>"
  exit 1
fi

TREATMENT_ID="$1"
OUTPUT_BASE_DIR="/Users/zhinuanguo/Documents/thesis/projects/internship/results/output_with_ram/${TREATMENT_ID}"
LOG_DIR="logs/accuracy"
START_NUM_RUNS="$2"
END_NUM_RUNS="$3"

mkdir -p "${LOG_DIR}"

ADDITIONAL_SAMPLING_PARAMS="{\"response_file\": \"/Users/zhinuanguo/Documents/thesis/projects/internship/results/output_with_ram/t6_caching_prefix/run_1/RAG_batch_t6_caching_prefix_prod_responses.json\", \"accuracy_file\": \"/Users/zhinuanguo/Documents/thesis/projects/internship/results/output_with_ram/t6_caching_prefix/run_1/RAG_batch_t6_caching_prefix_prod_accuracies.json\"}"

for (( i=START_NUM_RUNS; i<=END_NUM_RUNS; i++ ))
do
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/run_$i"
  INPUT_DIR="${OUTPUT_BASE_DIR}/run_$i"
  LOG_FILE="${LOG_DIR}/${TREATMENT_ID}_run_${i}_accuracy.log"

  echo "[$(date)] Starting run $i of $END_NUM_RUNS..."
  python3 llmperf/rag_accuracy.py \
      --model "deepseek-chat" \
      --num-concurrent-requests 5 \
      --timeout 9000000 \
      --input-dir "$INPUT_DIR" \
      --output-dir "$OUTPUT_DIR" \
      --metadata "" \
      --llm-api "openai_acc" \
      --additional-sampling-params "$ADDITIONAL_SAMPLING_PARAMS" \
      > "${LOG_FILE}" 2>&1
done

echo "======FORMAT====="
