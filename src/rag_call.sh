#!/bin/bash
echo "Running pre-hooks before committing..."
export RAG_URL=http://localhost:8080/chat-api/external/rag
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 llmperf/rag_evaluation.py --config rag_config.json
echo "======FORMAT====="
