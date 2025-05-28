#!/bin/bash

if [ -z "$TREATMENT_ID" ] || [ -z "$1" ]; then
  echo "Usage: TREATMENT_ID=<treatment-id> $0 <num-runs>"
  exit 1
fi

NUM_RUNS="$1"
COOLDOWN_SECONDS=500

variable="${TREATMENT_ID##*_}"
technique="${TREATMENT_ID%_*}"
RUNNING_BASE_DIR="/home/otto/thesis/projects/thesis_intern/running/${technique}/${variable}"
DB_INIT_SCRIPT="${RUNNING_BASE_DIR}/db_init.sh"
INSTALL_SCRIPT="${RUNNING_BASE_DIR}/install.sh"
UNINSTALL_SCRIPT="${RUNNING_BASE_DIR}/un_install.sh"

OUTPUT_BASE_DIR="output_with_ram/${TREATMENT_ID}"
LOG_DIR="logs"
mkdir -p "${OUTPUT_BASE_DIR}" "${LOG_DIR}"

BACKEND_PORT=$(kubectl get svc chat-backend-springboot-helm-chart -o jsonpath='{.spec.ports[0].nodePort}')
NODE_IP=$(kubectl get nodes -o jsonpath='{.items[0].status.addresses[?(@.type=="InternalIP")].address}')
export RAG_URL="http://${NODE_IP}:${BACKEND_PORT}/chat-api/external/rag"

export PYTHONPATH="$(pwd):$PYTHONPATH"

for (( i=1; i<=NUM_RUNS; i++ ))
do
  OUTPUT_DIR="${OUTPUT_BASE_DIR}/run_$i"
  LOG_FILE="${LOG_DIR}/${TREATMENT_ID}_run_${i}.log"

  mkdir -p "${OUTPUT_DIR}"

  bash "$INSTALL_SCRIPT"
  bash "$DB_INIT_SCRIPT"

  echo "[$(date)] Starting run $i of $NUM_RUNS..."
  python3 llmperf/rag_evaluation_clients.py \
    --model "RAG" \
    --num-concurrent-requests 1 \
    --timeout 9000000 \
    --warmup-input-dir "../dataset/crag_task_1_and_2_dev_v4_warmup.jsonl.bz2" \
    --prod-input-dir "../dataset/crag_task_1_and_2_dev_v4_prod.jsonl.bz2" \
    --output-dir "${OUTPUT_DIR}" \
    --metadata "name=benchmark,version=1" \
    --treatment-id "${TREATMENT_ID}" \
    --llm-api "RAG" \
    > "${LOG_FILE}" 2>&1

  echo "[$(date)] Run $i finished. Logs: ${LOG_FILE}"

  if [ "$i" -lt "$NUM_RUNS" ]; then
    echo "Cooling down for ${COOLDOWN_SECONDS} seconds..."
    sleep "$COOLDOWN_SECONDS"
  fi

  bash "$UNINSTALL_SCRIPT"
done

echo "[$(date)] All $NUM_RUNS runs completed."
