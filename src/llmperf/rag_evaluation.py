import threading
import argparse
from collections.abc import Iterable
import json
import os
from pathlib import Path
import re
import time
import random
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
import ray

from llmperf import common_metrics
from llmperf.common import SUPPORTED_APIS, construct_clients

from llmperf.models import RequestConfig
from llmperf.requests_launcher import RequestsLauncher
from llmperf.utils import (
    LLMPerfResults,
    sample_random_positive_int,
)
from tqdm import tqdm

from data_loader import load_data


def get_accuracies_latencies(
        model: str,
        dataset_path: str,
        additional_sampling_params: Optional[Dict[str, Any]] = None,
        num_concurrent_requests: int = 1,
        test_timeout_s=90,
        llm_api="RAG",
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Get the token throughput and latencies for the given model.

    Args:
        model: The name of the model to query.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        test_timeout_s: The amount of time to run the test for before reporting results.
        llm_api: The name of the llm api to use. Either "openai" or "litellm".

    Returns:
        A summary of the performance metrics collected across all completed requests
        (e.g. throughput, latencies, etc.)
        The individual metrics for each request.
    """
    random.seed(11111)

    if not additional_sampling_params:
        additional_sampling_params = {}

    completed_requests_lock = threading.Lock()
    completed_requests = []
    num_completed_requests = 0
    # make up prompts outside of send loop for faster benchmarking loop
    query_answers = []

    dataset = load_data(dataset_path)
    prompts = dataset["query"]
    print(f"Loading {len(prompts)} queries...")
    max_num_completed_requests = 2

    start_time = time.monotonic()
    pbar = tqdm(total=max_num_completed_requests)

    def launch_request(thread_index):
        nonlocal num_completed_requests
        clients = construct_clients(llm_api=llm_api, num_clients=1)
        req_launcher = RequestsLauncher(clients)
        request_index = thread_index % max_num_completed_requests

        while (
                time.monotonic() - start_time < test_timeout_s
                and num_completed_requests < max_num_completed_requests
        ):
            interaction_id = dataset['interaction_id'][request_index]
            prompt = prompts[request_index]
            request_config = RequestConfig(
                model=model,
                prompt=(prompt, 0),
                sampling_params=additional_sampling_params,
                llm_api=llm_api,
            )
            req_launcher.launch_requests(request_config)

            outs = req_launcher.get_next_ready()
            print(f"outs: {outs}")
            all_metrics = []
            for out in outs:
                request_metrics, gen_text, _ = out
                with completed_requests_lock:
                    if num_completed_requests < max_num_completed_requests:
                        all_metrics.append(request_metrics)
                        query_answers.append({"query": prompt, "answer": gen_text, "interaction_id": interaction_id})
                        completed_requests.extend(all_metrics)
                        pbar.update(len(all_metrics))
                        num_completed_requests += len(all_metrics)
                        request_index = (request_index + num_concurrent_requests) % max_num_completed_requests

    threads = []
    for i in range(num_concurrent_requests):
        thread = threading.Thread(target=launch_request, args=(i,))
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    pbar.close()
    end_time = time.monotonic()
    if end_time - start_time >= test_timeout_s:
        print("Test timed out before all requests could be completed.")

    # check one last time that there are no remaining results to collect.
    clients = construct_clients(llm_api=llm_api, num_clients=1)
    req_launcher = RequestsLauncher(clients)
    outs = req_launcher.get_next_ready()

    for out in outs:
        request_metrics, gen_text, _ = out
        with completed_requests_lock:
            if num_completed_requests < max_num_completed_requests:
                completed_requests.extend(request_metrics)

    print(f"Results for token benchmark for {model} queried with the {llm_api} api.\n")
    ret = metrics_summary(completed_requests, start_time, end_time)

    metadata = {
        "model": model,
        "num_concurrent_requests": num_concurrent_requests,
        "additional_sampling_params": additional_sampling_params,
    }

    metadata["results"] = ret

    return metadata, completed_requests


def metrics_summary(
        metrics: List[Dict[str, Any]], start_time: int, end_time: int
) -> Dict[str, Any]:
    """Generate a summary over metrics generated from potentially multiple instances of this client.

    Args:
        metrics: The metrics to summarize.
        start_time: The time the test started.
        end_time: The time the test ended.

    Returns:
        A summary with the following information:
            - Overall throughput (generated tokens / total test time)
            - Number of completed requests
            - Error rate
            - Error code frequency
            - Quantiles (p25-p99) for the following metrics:
                - Inter token latency
                - Time to first token
                - User total request time
                - Number of tokens processed per request
                - Number of tokens generated per request
                - User throughput (tokens / s)
    """
    ret = {}

    def flatten(item):
        for sub_item in item:
            if isinstance(sub_item, Iterable) and not isinstance(sub_item, str):
                yield from flatten(sub_item)
            else:
                yield sub_item

    df = pd.DataFrame(metrics)
    print(f"loading df: {metrics}")
    df_without_errored_req = df[df[common_metrics.ERROR_CODE].isna()]

    for key in [
        common_metrics.E2E_LAT,
    ]:
        print(key)
        ret[key] = {}
        series = pd.Series(list(flatten(df_without_errored_req[key]))).dropna()
        quantiles = series.quantile([0.25, 0.5, 0.75, 0.9, 0.95, 0.99]).to_dict()
        quantiles_reformatted_keys = {}
        for quantile, value in quantiles.items():
            reformatted_key = f"p{int(quantile * 100)}"
            print(f"    {reformatted_key} = {value}")
            quantiles_reformatted_keys[reformatted_key] = value
        ret[key]["quantiles"] = quantiles_reformatted_keys
        mean = series.mean()
        print(f"    mean = {mean}")
        ret[key]["mean"] = mean
        print(f"    min = {series.min()}")
        ret[key]["min"] = series.min()
        print(f"    max = {series.max()}")
        ret[key]["max"] = series.max()
        print(f"    stddev = {series.std()}")
        ret[key]["stddev"] = series.std()

    ret[common_metrics.NUM_REQ_STARTED] = len(metrics)

    error_codes = df[common_metrics.ERROR_CODE].dropna()
    num_errors = len(error_codes)
    ret[common_metrics.ERROR_RATE] = num_errors / len(metrics) if len(metrics) else 0
    ret[common_metrics.NUM_ERRORS] = num_errors
    print(f"Number Of Errored Requests: {num_errors}")
    error_code_frequency = dict(error_codes.value_counts())
    if num_errors:
        error_code_frequency = dict(error_codes.value_counts())
        print("Error Code Frequency")
        print(error_code_frequency)
    ret[common_metrics.ERROR_CODE_FREQ] = str(error_code_frequency)

    num_completed_requests = len(df_without_errored_req)
    num_completed_requests_per_min = (
            num_completed_requests / (end_time - start_time) * 60
    )
    print(f"Number Of Completed Requests: {num_completed_requests}")
    print(f"Completed Requests Per Minute: {num_completed_requests_per_min}")

    ret[common_metrics.NUM_COMPLETED_REQUESTS] = num_completed_requests
    ret[common_metrics.COMPLETED_REQUESTS_PER_MIN] = num_completed_requests_per_min

    return ret


def run_token_benchmark(
        llm_api: str,
        model: str,
        test_timeout_s: int,
        num_concurrent_requests: int,
        additional_sampling_params: str,
        input_dir: str,
        results_dir: str,
        user_metadata: Dict[str, Any],
):
    """
    Args:
        llm_api: The name of the llm api to use.
        model: The name of the model to query.
        max_num_completed_requests: The number of requests to complete before finishing the test.
        test_timeout_s: The amount of time to run the test for before reporting results.
        num_concurrent_requests: The number of concurrent requests to make. Increase
            this to increase the amount of load and vice versa.
        mean_input_tokens: The mean number of tokens to send in the prompt for the request.
        stddev_input_tokens: The standard deviation of the number of tokens to send in the prompt for the request.
        mean_output_tokens: The mean number of tokens to generate per request.
        stddev_output_tokens: The standard deviation of the number of tokens to generate per request.
        additional_sampling_params: Additional sampling parameters to send with the request.
            For more information see the LLM APIs documentation for the completions.
        results_dir: The directory to save the results to.
        user_metadata: Additional metadata to include in the results.
    """

    summary, individual_responses = get_accuracies_latencies(
        model=model,
        llm_api=llm_api,
        test_timeout_s=test_timeout_s,
        num_concurrent_requests=num_concurrent_requests,
        additional_sampling_params=json.loads(additional_sampling_params),
        dataset_path=input_dir,
    )

    if results_dir:
        filename = "ttttestt"
        summary_filename = f"{filename}_summary"
        individual_responses_filename = f"{filename}_individual_responses"

        # Update to metadata.
        summary.update(user_metadata)

        results = LLMPerfResults(name=summary_filename, metadata=summary)
        results_dir = Path(results_dir)
        if not results_dir.exists():
            results_dir.mkdir(parents=True)
        elif not results_dir.is_dir():
            raise ValueError(f"{results_dir} is not a directory")

        try:
            with open(results_dir / f"{summary_filename}.json", "w") as f:
                json.dump(results.to_dict(), f, indent=4, default=str)
        except Exception as e:
            print(results.to_dict())
            raise e

        try:
            with open(results_dir / f"{individual_responses_filename}.json", "w") as f:
                json.dump(individual_responses, f, indent=4)
        except Exception as e:
            print(individual_responses)
            raise e
def load_config(config_path):
    with open(config_path, "r") as f:
        return json.load(f)

if __name__ == "__main__":
    env_vars = dict(os.environ)
    ray.init(runtime_env={"env_vars": env_vars})

    parser = argparse.ArgumentParser(description="Run a token throughput and latency benchmark.")

    # Define arguments
    parser.add_argument("--config", type=str, help="Path to configuration file (JSON)")
    parser.add_argument("--model", type=str, help="The model to use for this load test.")
    parser.add_argument("--num-concurrent-requests", type=int, help="The number of concurrent requests to send.")
    parser.add_argument("--timeout", type=int, help="The amount of time to run the load test for.")
    parser.add_argument("--input-dir", type=str, help="The directory to read the dataset from.")
    parser.add_argument("--results-dir", type=str, help="The directory to save the results to.")
    parser.add_argument("--llm-api", type=str, help="The name of the LLM API to use.")
    parser.add_argument("--metadata", type=str, help="Metadata for the test, e.g. name=benchmark,version=1")
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
    # Parse user metadata.
    user_metadata = {}
    if args.metadata:
        for item in args.metadata.split(","):
            key, value = item.split("=")
            user_metadata[key] = value
    print(f"args: {args}")
    run_token_benchmark(
        llm_api=args.llm_api,
        model=args.model,
        test_timeout_s=args.timeout,
        num_concurrent_requests=args.num_concurrent_requests,
        additional_sampling_params=args.additional_sampling_params,
        input_dir=args.input_dir,
        results_dir=args.results_dir,
        user_metadata=user_metadata,
    )
