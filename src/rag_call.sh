#!/bin/bash
if [ -z "$TREATMENT_ID" ]; then
  echo "Usage: $0 <treatment-id>"
  exit 1
fi

OUTPUT_DIR="output/${TREATMENT_ID}"
LOG_DIR="logs"
LOG_FILE="${LOG_DIR}/${TREATMENT_ID}.log"

export RAG_URL="http://172.18.0.2:30815/chat-api/external/rag"
export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p "${OUTPUT_DIR}" "${LOG_DIR}"

echo "Starting job with treatment ID: ${TREATMENT_ID}..."
nohup python3 llmperf/rag_evaluation_clients.py \
  --model "RAG" \
  --num-concurrent-requests 1 \
  --timeout 9000000 \
  --warmup-input-dir "../dataset/crag_task_1_and_2_dev_v4_warmup.jsonl.bz2" \
  --prod-input-dir "../dataset/crag_task_1_and_2_dev_v4_prod.jsonl.bz2" \
  --output-dir "${OUTPUT_DIR}" \
  --metadata "name=benchmark,version=1" \
  --treatment-id "${TREATMENT_ID}" \
  --llm-api "RAG" \
  > "${LOG_FILE}" 2>&1 &

echo "Prod job started. Logs: ${LOG_FILE}"
