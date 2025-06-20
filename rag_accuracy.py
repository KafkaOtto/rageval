import argparse
import glob
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import ray
from loguru import logger
from openai import APIConnectionError, RateLimitError
from tqdm import tqdm

from llmperf.common import construct_clients
from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import load_answer_score_mapping, check_existing_score


def load_from_result_dir(input_dir: str):
    json_files = glob.glob(f"{input_dir}/*_prod_responses.json")
    print(f"Found {len(json_files)} json files in {input_dir}")
    return json_files

def generate_accuracy_filename(json_filename: str):
    dir_name, base_name = os.path.split(json_filename)
    file_stem, ext = os.path.splitext(base_name)
    accuracy_filename = file_stem.replace("responses", "accuracies") + ext
    return accuracy_filename

def attempt_api_call(client, model_name, messages, max_retries=10):
    """Attempt an API call with retries upon encountering specific errors."""
    # todo: add default response when all efforts fail
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model_name,
                messages=messages,
                response_format={"type": "json_object"},
                temperature=0.0,
            )
            return response.choices[0].message.content
        except (APIConnectionError, RateLimitError):
            logger.warning(f"API call failed on attempt {attempt + 1}, retrying...")
        except Exception as e:
            logger.error(f"Unexpected error: {e}")
            break
    return None

def load_raw_results(complete_requests):
    raw_results = []
    for out in complete_requests:
        explanation, score, completed_request_config = out
        raw_results.append(
            {
                "score": score,
                "explanation": explanation,
                "request_config": dict(completed_request_config),
            }
        )
    return raw_results

def save_results(filename: str,
                 output_dir: str,
                 raw_results):
    # generate *_responses.json to *_accuracies.json
    accuracy_filename = generate_accuracy_filename(filename)
    try:
        output_dir = Path(output_dir)
        with open(output_dir / f"{accuracy_filename}", "w") as f:
            json.dump(raw_results, f, indent=4)
    except Exception as e:
        logger.error("write json file error:", e)
        raise e



def run_file(model: str,
             additional_sampling_params: Optional[Dict[str, Any]] = None,
             num_concurrent_requests: int = 1,
             llm_api="RAG",
             input_dir: str = None,
             output_dir: str = None,
             ):

    responses_files = load_from_result_dir(input_dir)
    clients = construct_clients(llm_api=llm_api, num_clients=num_concurrent_requests)
    req_launcher = RequestsLauncher(clients)

    start_time = time.monotonic()
    iteration = 0
    answer_score = load_answer_score_mapping(additional_sampling_params["response_file"], additional_sampling_params["accuracy_file"])
    # answer_score= {"a": []}
    for _idx, json_filename in enumerate(tqdm(responses_files
            , total=len(responses_files), desc="Loading Files"
    )):
        iteration = iteration + 1
        completed_requests = []
        with open(json_filename, "r", encoding="utf-8") as json_file:
            responses = json.load(json_file)
            for _idy, response in enumerate(tqdm(responses
                    , total=len(responses), desc="Loading Responses"
                                                      )):
                query = response['request_config']['prompt']
                prediction = response['generated_text']
                prediction_lowercase = str(prediction).strip().rstrip('.').lower()
                ground_truth = response['request_config']['metadata']['answer']
                ground_truth_lowercase = str(ground_truth).strip().rstrip('.').lower()
                prompt = f"Question: {query}\n Ground truth: {ground_truth_lowercase}\n Prediction: {prediction_lowercase}\n"
                request_config = RequestConfig(
                    model=model,
                    prompt=(prompt, len(prompt)),
                    llm_api=llm_api,
                )
                existing_outs = check_existing_score(query[0], prediction_lowercase, answer_score)
                if existing_outs is not None:
                    outs = [existing_outs]
                elif prediction_lowercase == "i don't know." or prediction_lowercase == "i don't know":
                    explanation = "The prediction is not sure about the answer."
                    score = -1.0
                    outs = [(explanation, score, request_config)]
                elif prediction_lowercase == ground_truth_lowercase:
                    explanation = "The prediction is correct."
                    score = 1.0
                    outs = [(explanation, score, request_config)]
                elif "invalid" in prediction_lowercase and "invalid" in ground_truth_lowercase:
                    explanation = "The prediction is correct in hallucination."
                    score = 1.0
                    outs = [(explanation, score, request_config)]
                elif "invalid" in prediction_lowercase and "invalid" not in ground_truth_lowercase:
                    # hallucination
                    explanation = "The prediction is incorrect in hallucination."
                    score = 0.0
                    outs = [(explanation, score, request_config)]
                elif "invalid" not in prediction_lowercase and "invalid" in ground_truth_lowercase:
                    # hallucination
                    explanation = "The prediction is incorrect in hallucination."
                    score = 0.0
                    outs = [(explanation, score, request_config)]
                else:
                    req_launcher.launch_requests(request_config)
                    outs = req_launcher.get_next_ready()
                logger.info(f"outs: {outs}")
                completed_requests.extend(outs)
        raw_results = load_raw_results(completed_requests)
        save_results(json_filename, output_dir=output_dir, raw_results=raw_results)

def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

if __name__ == '__main__':
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})

    parser = argparse.ArgumentParser(description="Run a token throughput and latency benchmark.")

    # Define arguments
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--model", type=str, help="The model to use for this load test.")
    parser.add_argument("--num-concurrent-requests", type=int, help="The number of concurrent requests to send.")
    parser.add_argument("--timeout", type=int, help="The amount of time to run the load test for.")
    parser.add_argument("--input-dir", type=str, help="The directory to read the dataset from.")
    parser.add_argument("--output-dir", type=str, help="The directory to save the results to.")
    parser.add_argument("--llm-api", type=str, help="The name of the LLM API to use.")
    parser.add_argument("--metadata", type=str, help="Metadata for the test, e.g. name=benchmark,version=1")
    parser.add_argument("--batch-size", type=str, default=-1, help="Batch size to separate the dataset")
    parser.add_argument("--treatment-id", type=str, help="For output filename id")
    parser.add_argument("--rerun", type=bool, help="If rerun the experiment")
    parser.add_argument(
        "--additional-sampling-params", type=str, default="{}", help=(
            "Additional sampling params to send with the each request to the LLM API. "
            "(default: %(default)s) No additional sampling params are sent."
        ),
    )
    args = parser.parse_args()
    if args.config:
        config_data = load_config(args.config)
        parser.set_defaults(**config_data)
    args = parser.parse_args()
    args.additional_sampling_params = json.loads(args.additional_sampling_params)

    run_file(model=args.model,
             additional_sampling_params=args.additional_sampling_params,
             num_concurrent_requests=args.num_concurrent_requests,
             llm_api=args.llm_api,
             input_dir=args.input_dir,
             output_dir=args.output_dir,
    )