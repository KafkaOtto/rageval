import json
import os
import time
from typing import Any, Dict

import ray
import requests
from loguru import logger

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from openai import APIConnectionError, OpenAI, RateLimitError
import re
from llmperf.prompts.templates import INSTRUCTIONS, IN_CONTEXT_EXAMPLES

@ray.remote
class OpenaiAccuracyClient(LLMClient):
    """Client for VertexAI API."""

    def __init__(self):
        BASE_URL = os.environ.get("ACCURACY_BASE_URL")
        OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
        self.openai = OpenAI(base_url=BASE_URL, api_key=OPENAI_KEY)

    def parse_response(self, response: str):
        """
        Return a tuple of (explanation, score) from the response,
        where score is 0 if the prediction is wrong, 1 if the prediction is correct.

        Need to handle
        Corner case 1:
            {"explanation": ...}
            Wait, no! I made a mistake. The prediction does not exactly match the ground truth. ...
            {...}

        Corner case 2:
            {"score": 0, "explanation": "The prediction does not contain item, nick "goose" bradshaw, that is in the ground truth."}
            return a tuple of (explanation, score)
        """
        matches = re.findall(r"{([^}]*)}", response)
        text = ""
        for match in matches:
            text = "{" + match + "}"
        try:
            # Pattern to match the score
            score_pattern = r'"score"\s*:\s*(\d+)'
            score_match = re.search(score_pattern, text)
            if score_match:
                score = int(score_match.group(1))
                if score != 0 and score != 1:
                    raise Exception("bad score: " + response)
            else:
                return "Parse Err: Score not found", -1

            # Pattern to match the explanation
            explanation_pattern = r'"explanation"\s*:\s*"(.+)"'
            explanation_match = re.search(explanation_pattern, text)
            if explanation_match:
                explanation = explanation_match.group(1)
                return explanation, score
            else:
                return text, score
        except Exception as e:
            print(f"Parsing Error with resp: {response}")
            print(f"Error: {e}")
            return response, -1

    def get_system_message(self):
        """Returns the system message containing instructions and in context examples."""
        return INSTRUCTIONS + "\n" + IN_CONTEXT_EXAMPLES

    def construct_message(self,
            prompt: str
    ):
        system_message = self.get_system_message()
        messages = [
            {"role": "system", "content": system_message},
            {
                "role": "user",
                "content": f"{prompt}",
            },
        ]
        return messages

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        max_retries = 5
        prompt = request_config.prompt
        prompt, prompt_len = prompt
        explanation, score = "LLM request failed", -1
        # if request_config.prompt == "I don't know":
        #     return "It is I don't know", 0, request_config
        for attempt in range(max_retries):
            try:
                response = self.openai.chat.completions.create(
                    model=request_config.model,
                    messages=self.construct_message(prompt),
                    response_format={"type": "json_object"},
                    temperature=0.0,
                )
                accuracy_response = response.choices[0].message.content
                explanation, score = self.parse_response(accuracy_response)
            except (APIConnectionError, RateLimitError) as e:
                logger.warning(f"API call failed on attempt {attempt + 1}, retrying...: {e}")
            except Exception as e:
                logger.error(f"Unexpected error: {e}")
                break
        return explanation, score, request_config

if __name__ == "__main__":
    # Run these before hand:

    # gcloud auth application-default login
    # gcloud config set project YOUR_PROJECT_ID
    # export RAG_URL=http://localhost:8080/rag/chat/completions

    client = OpenaiAccuracyClient.remote()
    model = 'deepseek-chat'
    query = "how many 3-point attempts did steve nash average per game in seasons he made the 50-40-90 club?"
    ground_truth = "4 3-points attempts per game"
    prediction = "I don't know"
    prompt = f"Question: {query}\n Ground truth: {ground_truth}\n Prediction: {prediction}\n"
    request_config = RequestConfig(
        prompt=(prompt, 10),
        model=model
    )
    response = ray.get(client.llm_request.remote(request_config))
    logger.info(f"test response: {response}")
