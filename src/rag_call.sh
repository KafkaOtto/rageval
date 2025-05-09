#!/bin/bash
echo "Running pre-hooks before committing..."

export RAG_URL="http://172.18.0.2:30815/chat-api/external/rag"
export PYTHONPATH="$(pwd):$PYTHONPATH"

mkdir -p logs

# Run the first job in foreground via nohup
echo "Starting warmup job..."
nohup python3 llmperf/rag_evaluation_clients.py \
  --model "RAG" \
  --num-concurrent-requests 1 \
  --timeout 9000000 \
  --input-dir "../dataset/crag_task_1_and_2_dev_v4_warmup.jsonl.bz2" \
  --output-dir "output" \
  --metadata "name=benchmark,version=1" \
  --batch-size 20000 \
  --treatment-id "base_model_warmup_2" \
  --llm-api "RAG" \
  > logs/base_model_warmup_2.log 2>&1

echo "Warmup job completed. Starting prod job..."

# Run the second job after the first finishes
nohup python3 llmperf/rag_evaluation_clients.py \
  --model "RAG" \
  --num-concurrent-requests 1 \
  --timeout 9000000 \
  --input-dir "../dataset/crag_task_1_and_2_dev_v4_prod.jsonl.bz2" \
  --output-dir "output" \
  --metadata "name=benchmark,version=1" \
  --batch-size 20000 \
  --treatment-id "base_model_prod_2" \
  --llm-api "RAG" \
  > logs/base_model_prod_2.log 2>&1 &

echo "Prod job started in background. Check logs in ./logs/"
