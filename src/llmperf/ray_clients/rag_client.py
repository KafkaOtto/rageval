import json
import os
import time
from typing import Any, Dict

import ray
import requests

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics


@ray.remote
class RAGClient(LLMClient):
    """Client for VertexAI API."""

    def __init__(self):
        self.te = 1

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        RAG_URL = os.environ.get("RAG_URL")
        if not RAG_URL:
            raise ValueError("the environment variable RAG_URL must be set.")
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        time_to_next_token = []
        generated_text = ""
        total_request_time = 0

        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        try:
            # Define the URL for the request
            url = RAG_URL
            # Define the headers
            headers = {
                "Content-Type": "application/json",
            }

            # Define the data payload
            data = {"stream": False, "model": request_config.model, "messages": [{"content": prompt, "role": "user"}]}
            print(f"send request: {json.dumps(data)}")
            # Make the POST request
            start_time = time.monotonic()
            response = requests.post(url, headers=headers, data=json.dumps(data), timeout=200000)
            generated_text = response.json()["choices"][0]["delta"]["content"]
            print(f"generated_text: {generated_text}")
            total_request_time = time.monotonic() - start_time
            response_code = response.status_code
            response.raise_for_status()
            # add accuracy evaluate
            # output from the endpoint is in the form:
            # {"predictions": ["Input: ... \nOutput:\n ..."]}
            # generated_text = parsed_data["data"]["choices"][0]["delta"]["content"]
            # print(f"generated text: {generated_text}")

        except Exception as e:
            metrics[common_metrics.ERROR_MSG] = str(e)
            metrics[common_metrics.ERROR_CODE] = response_code
            print(f"Warning Or Error: {e}")
            print(response_code)

        metrics[common_metrics.INTER_TOKEN_LAT] = time_to_next_token
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config
if __name__ == "__main__":
    # Run these before hand:

    # gcloud auth application-default login
    # gcloud config set project YOUR_PROJECT_ID
    # export RAG_URL=http://localhost:8080/rag/chat/completions

    client = RAGClient.remote()
    request_config = RequestConfig(
        prompt=("how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?", 10),
        model="RAG"
    )
    ray.get(client.llm_request.remote(request_config))
