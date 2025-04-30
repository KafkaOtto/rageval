#!/bin/bash
echo "Running pre-hooks before committing..."
export ACCURACY_BASE_URL=https://api.deepseek.com
export OPENAI_API_KEY=sk-20e9f2df70c6442eba7eaf351222e061
export PYTHONPATH=$(pwd):$PYTHONPATH
python3 llmperf/rag_accuracy.py --config accuracy_config.json
echo "======FORMAT====="
